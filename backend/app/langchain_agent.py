from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple
import json

from langchain_community.chat_models import ChatOllama

from .config import get_settings
from .langchain_rag import answer_with_context_langchain
from .langchain_tools import create_event_tool, create_todo_tool
from .mcp_clients import registry as mcp_registry


settings = get_settings()


async def classify_intent(message: str) -> str:
    """Classify message intent as TODO, EVENT, or QA using the LLM."""
    system_prompt = (
        "You are an intent classifier for a personal assistant.\n"
        "Given a single user message, decide if the primary intent is:\n"
        "- TODO: creating or updating a personal todo/reminder/task.\n"
        "- EVENT: scheduling or modifying a calendar event/meeting.\n"
        "- QA: asking a question or chatting (no tool call).\n"
        "Reply with exactly one word: TODO, EVENT, or QA."
    )
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
    )
    messages = [
        ("system", system_prompt),
        ("user", message),
    ]
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    intent = raw.strip().upper().split()[0]
    if intent not in {"TODO", "EVENT", "QA"}:
        return "QA"
    return intent


async def extract_todo(message: str) -> Tuple[str, str | None]:
    """Use the LLM to extract todo text and optional due datetime."""
    system_prompt = (
        "You extract todo information from natural language and create a clear, concise todo title.\n"
        "Given one user message, output ONLY a JSON object with keys:\n"
        '{ "text": string, "due": string | null }.\n'
        "- 'text' should be a natural, concise todo title that includes ALL relevant details.\n"
        "  CRITICAL: Include times, dates, locations, and contextual information in the title.\n"
        "  Examples:\n"
        "  - Input: 'add todo to buy a bus ticket to airport (stansted at 23:20)'\n"
        "    Output text: 'Buy bus ticket to Stansted for 23:20 flight'\n"
        "  - Input: 'add todo \"buy bus ticket to stansted for tomorrow 23:20\"'\n"
        "    Output text: 'Buy bus ticket to Stansted for tomorrow 23:20'\n"
        "  - Input: 'remind me to call John tomorrow'\n"
        "    Output text: 'Call John'\n"
        "  - Input: 'todo: finish the report by Friday'\n"
        "    Output text: 'Finish the report'\n"
        "  Remove command phrases like 'add todo', 'remind me to', 'todo:', etc.\n"
        "  Keep ALL contextual information: flight times, locations, dates mentioned in context.\n"
        "- 'due' should ONLY be set if the user explicitly mentions a DEADLINE or DUE DATE.\n"
        "  Examples of due dates/deadlines: 'due tomorrow', 'by Friday', 'deadline next week', 'due on 2025-11-15'.\n"
        "  IMPORTANT: Times and dates that are CONTEXTUAL (like flight times, meeting times) should be in 'text', NOT in 'due'.\n"
        "  Example: 'buy ticket for flight at 23:20' → text includes '23:20', due is null (not a deadline).\n"
        "  Example: 'finish report by Friday' → text is 'Finish report', due is Friday (this IS a deadline).\n"
        "- If 'due' is set, it should be an ISO 8601 datetime (e.g. 2025-11-15T09:00:00).\n"
        "  Interpret relative dates like 'today', 'tomorrow', or weekdays relative to the current date.\n"
        "- If no explicit due date/deadline is mentioned, set 'due' to null.\n"
        "Do not include any explanation text, only the JSON."
    )
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
    )
    messages = [
        ("system", system_prompt),
        ("user", message),
    ]
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx == -1 or end_idx == -1:
            return message, None
        data = json.loads(raw[start_idx : end_idx + 1])
        text = str(data.get("text") or message).strip()
        due = data.get("due")
        return text, str(due) if due else None
    except Exception:
        return message, None


async def extract_event(message: str) -> Tuple[str, str, str]:
    """Use the LLM to extract event details as ISO datetimes."""
    now = datetime.utcnow()
    today_str = now.date().isoformat()

    system_prompt = (
        "You extract calendar event details from natural language.\n"
        "Given one user message, output ONLY a JSON object with keys:\n"
        '{ "title": string, "start": string, "end": string }.\n'
        "- 'start' and 'end' must be full ISO 8601 datetimes "
        "(e.g. 2025-11-15T09:00:00).\n"
        f"- Today is {today_str}. Interpret relative dates like 'today', "
        "'tomorrow', or weekdays relative to this date.\n"
        "- If the user does not specify an end time, set 'end' to exactly "
        "1 hour after 'start'.\n"
        "Do not include any explanation text, only the JSON."
    )
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
    )
    messages = [
        ("system", system_prompt),
        ("user", message),
    ]
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON found")
        data = json.loads(raw[start_idx : end_idx + 1])
        title = str(data["title"]).strip()
        start_str = str(data["start"]).strip()
        start_dt = datetime.fromisoformat(start_str)
        end_str = data.get("end")
        if end_str:
            end_dt = datetime.fromisoformat(str(end_str).strip())
        else:
            end_dt = start_dt + timedelta(hours=1)
        return title, start_dt.isoformat(), end_dt.isoformat()
    except Exception:
        # Fallback: treat the whole message as the title and schedule a 1-hour event tomorrow
        start_dt = now + timedelta(days=1)
        end_dt = start_dt + timedelta(hours=1)
        return message.strip(), start_dt.isoformat(), end_dt.isoformat()


async def run_agent(message: str) -> Tuple[str, List[str], List[str]]:
    """LangChain-based agent that combines RAG with tool calling."""
    used_tools: List[str] = []
    retrieved_doc_ids: List[str] = []

    intent = await classify_intent(message)

    # Log the detected intent
    import sys
    print(f"DEBUG: Detected intent: {intent} for message: '{message[:50]}...'", file=sys.stderr, flush=True)

    if intent == "TODO":
        # Try to use Notion MCP tools first, fall back to DB if not available
        from .config import get_settings
        settings = get_settings()

        # Explicit logging to see what's happening
        import sys
        print(f"DEBUG: TODO intent detected", file=sys.stderr, flush=True)
        print(f"DEBUG: notion_integration_token is set: {bool(settings.notion_integration_token)}", file=sys.stderr, flush=True)
        if settings.notion_integration_token:
            print(f"DEBUG: Token value starts with: {settings.notion_integration_token[:10] if settings.notion_integration_token else 'None'}...", file=sys.stderr, flush=True)
            # Notion is configured, try to use it via MCP registry
            notion_tools = await mcp_registry.get_tools("notion")

            if notion_tools:
                # Log available tools for debugging
                import sys
                tool_names = [tool.name for tool in notion_tools]
                print(f"DEBUG: Found {len(notion_tools)} Notion MCP tools: {tool_names}", file=sys.stderr, flush=True)

                # Find the create page tool - we want to create a PAGE in a database
                # The correct tool is "API-post-page" (HTTP POST = create)
                create_tool = None

                # Priority 1: Look for "post-page" or "post_page" (HTTP POST = create)
                for tool in notion_tools:
                    tool_name_lower = tool.name.lower()
                    if ("post" in tool_name_lower and "page" in tool_name_lower):
                        create_tool = tool
                        print(f"DEBUG: Using Notion MCP tool (post-page): {tool.name}", file=sys.stderr, flush=True)
                        break

                # Priority 2: Look for tools with "create" AND "page" but NOT "comment" or "database" (as verb)
                if not create_tool:
                    for tool in notion_tools:
                        tool_name_lower = tool.name.lower()
                        # Must have "create" and "page"
                        # Must NOT have "comment", "database" (as in creating a database), "update", "delete"
                        if ("create" in tool_name_lower and
                            "page" in tool_name_lower and
                            "comment" not in tool_name_lower and
                            "update" not in tool_name_lower and
                            "delete" not in tool_name_lower):
                            # Check if it's about creating a database (bad) vs creating a page (good)
                            if "create" in tool_name_lower and "database" in tool_name_lower and "page" not in tool_name_lower:
                                continue  # Skip tools that create databases
                            create_tool = tool
                            print(f"DEBUG: Using Notion MCP tool (create page): {tool.name}", file=sys.stderr, flush=True)
                            break

                # Priority 3: Exact match for known tool names
                if not create_tool:
                    exact_matches = ["api-post-page", "post-page", "create-page", "create_page"]
                    for tool in notion_tools:
                        if tool.name.lower() in exact_matches:
                            create_tool = tool
                            print(f"DEBUG: Using Notion MCP tool (exact match): {tool.name}", file=sys.stderr, flush=True)
                            break

                # If still no tool found, log all create/post-related tools for debugging
                if not create_tool:
                    create_related = [t.name for t in notion_tools if "create" in t.name.lower() or "post" in t.name.lower()]
                    print(f"DEBUG: No suitable create-page tool found. Create/post-related tools: {create_related}", file=sys.stderr, flush=True)

                if create_tool:
                    text, due_iso = await extract_todo(message)
                    # Debug logging
                    import sys
                    print(f"DEBUG: Extracted todo text: '{text}'", file=sys.stderr, flush=True)
                    print(f"DEBUG: Extracted due_iso: '{due_iso}'", file=sys.stderr, flush=True)
                    
                    # Notion API format for creating a page in a database:
                    # parent: { database_id: "..." }
                    # properties: { "Name": { title: [{ text: { content: "..." } }] } }
                    # Note: Property name might be "Name", "Title", or whatever the database title property is

                    if not settings.notion_database_id:
                        raise ValueError("NOTION_DATABASE_ID must be set in .env to create pages")

                    # Ensure text is not empty
                    if not text or not text.strip():
                        text = message.strip()  # Fallback to original message if extraction failed
                        print(f"DEBUG: Text was empty, using original message: '{text}'", file=sys.stderr, flush=True)

                    # Use correct Notion API format
                    # Set status to "To Do" by default
                    # Note: Property names may vary - common names: "Status", "Task Status", "Todo Status"
                    # If this doesn't work, check your database schema and update the property name
                    tool_args = {
                        "parent": {
                            "database_id": settings.notion_database_id
                        },
                        "properties": {
                            "Name": {  # Most common property name for title
                                "title": [{"text": {"content": text.strip()}}]
                            },
                            "Status": {  # Set status to "To Do" - adjust property name if needed
                                "select": {
                                    "name": "To Do"
                                }
                            }
                        }
                    }

                    # Try alternative property names if "Status" doesn't work
                    # Uncomment and adjust if needed:
                    # tool_args["properties"]["Task Status"] = {"select": {"name": "To Do"}}
                    # tool_args["properties"]["Todo Status"] = {"select": {"name": "To Do"}}

                    # Only add Due Date property if a valid due date was extracted
                    # Note: This assumes your database has a "Due Date" property
                    # If your database doesn't have this property, remove this block or adjust the property name
                    if due_iso and due_iso.strip():
                        try:
                            # Validate it's a proper ISO date/datetime
                            # Parse ISO datetime to date if needed
                            due_date = due_iso.split("T")[0] if "T" in due_iso else due_iso
                            # Only add if it's a valid date format (basic check)
                            if len(due_date) >= 10 and due_date[4] == "-" and due_date[7] == "-":
                                tool_args["properties"]["Due Date"] = {
                                    "date": {"start": due_date}
                                }
                        except Exception:
                            # If date parsing fails, just skip adding the Due Date property
                            pass

                    import sys
                    print(f"DEBUG: Calling Notion MCP tool '{create_tool.name}' with parent.database_id: {settings.notion_database_id[:10]}...", file=sys.stderr, flush=True)
                    print(f"DEBUG: Tool args: {json.dumps(tool_args, indent=2)}", file=sys.stderr, flush=True)
                    try:
                        result = await create_tool.ainvoke(tool_args)
                        used_tools.append("create_notion_todo")
                        
                        # Debug: log the result type and content
                        print(f"DEBUG: Result type: {type(result)}", file=sys.stderr, flush=True)
                        print(f"DEBUG: Result (first 500 chars): {str(result)[:500]}", file=sys.stderr, flush=True)
                        
                        # Extract URL from result - handle various formats
                        url = None
                        result_dict = None
                        
                        if isinstance(result, str):
                            # Try to parse as JSON string
                            try:
                                result_dict = json.loads(result)
                            except json.JSONDecodeError:
                                # Maybe it's a string representation, try to find JSON in it
                                start_idx = result.find("{")
                                end_idx = result.rfind("}")
                                if start_idx != -1 and end_idx != -1:
                                    try:
                                        result_dict = json.loads(result[start_idx:end_idx+1])
                                    except json.JSONDecodeError:
                                        pass
                        
                        if result_dict is None and isinstance(result, dict):
                            result_dict = result
                        
                        if result_dict:
                            # Check if this is an error response
                            if result_dict.get("object") == "error":
                                error_msg = result_dict.get("message", "Unknown error")
                                error_code = result_dict.get("code", "")
                                print(f"DEBUG: Notion API returned error: {error_code} - {error_msg}", file=sys.stderr, flush=True)
                                # This will be caught by the outer exception handler
                                raise Exception(f"Notion API error: {error_msg}")
                            
                            url = result_dict.get("url")
                            # Also check if url is nested
                            if not url and "object" in result_dict:
                                url = result_dict.get("url")
                        
                        print(f"DEBUG: Extracted URL: {url}", file=sys.stderr, flush=True)
                        
                        if url:
                            reply = f"Created todo: {text.strip()}\nGo to Notion: {url}"
                        else:
                            # If no URL but result exists, still report success but note the issue
                            print(f"DEBUG: WARNING - No URL found in result but call succeeded", file=sys.stderr, flush=True)
                            reply = f"Created todo: {text.strip()}"
                    except Exception as e:
                        error_str = str(e)
                        # If error is about a property not existing (e.g., "Due Date"), retry without that property
                        if "is not a property that exists" in error_str or ("validation_error" in error_str.lower() and "property" in error_str.lower()):
                            import sys
                            print(f"DEBUG: Property error detected, retrying without Due Date property", file=sys.stderr, flush=True)
                            # Remove Due Date property and retry
                            tool_args_no_due = {
                                "parent": tool_args["parent"],
                                "properties": {
                                    "Name": tool_args["properties"]["Name"],
                                    "Status": tool_args["properties"]["Status"]
                                }
                            }
                            try:
                                result = await create_tool.ainvoke(tool_args_no_due)
                                used_tools.append("create_notion_todo")
                                
                                # Extract URL from result - same logic as above
                                url = None
                                result_dict = None
                                
                                if isinstance(result, str):
                                    try:
                                        result_dict = json.loads(result)
                                    except json.JSONDecodeError:
                                        start_idx = result.find("{")
                                        end_idx = result.rfind("}")
                                        if start_idx != -1 and end_idx != -1:
                                            try:
                                                result_dict = json.loads(result[start_idx:end_idx+1])
                                            except json.JSONDecodeError:
                                                pass
                                
                                if result_dict is None and isinstance(result, dict):
                                    result_dict = result
                                
                                if result_dict:
                                    # Check if this is an error response
                                    if result_dict.get("object") == "error":
                                        error_msg = result_dict.get("message", "Unknown error")
                                        raise Exception(f"Notion API error: {error_msg}")
                                    
                                    url = result_dict.get("url")
                                
                                if url:
                                    reply = f"Created todo: {text.strip()}\nGo to Notion: {url}"
                                else:
                                    reply = f"Created todo: {text.strip()}"
                            except Exception as e2:
                                # If retry also fails, fall back to DB
                                import sys
                                import traceback
                                print(f"DEBUG: Notion MCP retry also failed: {e2}", file=sys.stderr, flush=True)
                                text, due_iso = await extract_todo(message)
                                result = create_todo_tool(text=text, due_iso=due_iso)
                                used_tools.append("create_todo")
                                reply = f"Notion MCP failed ({e2}), created in local DB: {result}"
                        else:
                            # Other errors - fall back to DB
                            import sys
                            import traceback
                            print(f"DEBUG: Notion MCP tool call failed: {e}", file=sys.stderr, flush=True)
                            print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                            text, due_iso = await extract_todo(message)
                            result = create_todo_tool(text=text, due_iso=due_iso)
                            used_tools.append("create_todo")
                            reply = f"Notion MCP failed ({e}), created in local DB: {result}"
                else:
                    # Notion MCP available but no create tool found
                    import sys
                    print(f"DEBUG: Notion MCP tools available but no create tool found. Available: {tool_names}", file=sys.stderr, flush=True)
                    text, due_iso = await extract_todo(message)
                    result = create_todo_tool(text=text, due_iso=due_iso)
                    used_tools.append("create_todo")
                    reply = f"Notion MCP configured but no create tool found. Created in local DB: {result}"
            else:
                # Notion token set but tools couldn't be retrieved
                import sys
                print("DEBUG: INTERNAL_INTEGRATION_TOKEN set but get_notion_mcp_tools() returned empty list", file=sys.stderr, flush=True)
                text, due_iso = await extract_todo(message)
                result = create_todo_tool(text=text, due_iso=due_iso)
                used_tools.append("create_todo")
                reply = f"Notion MCP configured but connection failed. Created in local DB: {result}"
        else:
            # No Notion MCP configured, use DB
            text, due_iso = await extract_todo(message)
            result = create_todo_tool(text=text, due_iso=due_iso)
            used_tools.append("create_todo")
            reply = result if isinstance(result, str) else str(result)
    elif intent == "EVENT":
        title, start_iso, end_iso = await extract_event(message)
        result = create_event_tool(
            title=title,
            start_iso=start_iso,
            end_iso=end_iso,
            description=None,
        )
        used_tools.append("create_event")
        reply = result if isinstance(result, str) else str(result)
    else:
        reply, retrieved_doc_ids = answer_with_context_langchain(message)

    return reply, used_tools, retrieved_doc_ids



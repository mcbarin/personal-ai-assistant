from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple
import json

from langchain_community.chat_models import ChatOllama

from .config import get_settings
from .langchain_rag import answer_with_context_langchain
from .langchain_tools import create_event_tool, create_todo_tool


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
        "You extract todo information from natural language.\n"
        "Given one user message, output ONLY a JSON object with keys:\n"
        '{ "text": string, "due": string | null }.\n'
        "- 'due' should be an ISO 8601 datetime (e.g. 2025-11-15T09:00:00) or null.\n"
        "Interpret relative dates like 'today', 'tomorrow', or weekdays "
        "relative to the current date.\n"
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

    if intent == "TODO":
        text, due_iso = await extract_todo(message)
        # Call the todo helper (which uses our existing DB-backed implementation).
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



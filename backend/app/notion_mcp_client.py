"""Notion MCP client using langchain-mcp-adapters.

This module connects to the Notion MCP server running in Docker via stdio transport
and exposes LangChain-compatible tools for creating and listing todos.
"""
from typing import List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import get_settings


async def get_notion_mcp_tools():
    """Get LangChain tools from the Notion MCP server.

    Returns a list of LangChain tools that can be used in agents.
    If Notion is not configured, returns an empty list.
    """
    settings = get_settings()
    if not settings.notion_integration_token:
        return []

    # Log token info (first 10 chars only for security)
    import sys
    token = settings.notion_integration_token
    if token:
        token = token.strip()  # Remove any leading/trailing whitespace

    token_preview = token[:10] + "..." if token else "None"
    print(f"DEBUG: Using Notion token starting with: {token_preview}", file=sys.stderr, flush=True)
    print(f"DEBUG: Token length: {len(token) if token else 0}", file=sys.stderr, flush=True)

    # Check token format - Notion tokens typically start with 'secret_' but MCP might use 'ntn_'
    if token:
        if not (token.startswith("secret_") or token.startswith("ntn_")):
            print(f"WARNING: Token doesn't start with 'secret_' or 'ntn_'. Format might be incorrect.", file=sys.stderr, flush=True)
        has_spaces = " " in token
        has_newlines = "\n" in token or "\r" in token
        print(f"DEBUG: Token has spaces: {has_spaces}, has newlines: {has_newlines}", file=sys.stderr, flush=True)

    # Configure MCP client to connect to Notion server via Docker stdio
    # Notion MCP server uses OPENAPI_MCP_HEADERS instead of INTERNAL_INTEGRATION_TOKEN
    import json
    openapi_headers = json.dumps({
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28"
    })

    docker_args = [
        "run",
        "--rm",
        "-i",
        "-e", "OPENAPI_MCP_HEADERS",
        "mcp/notion",
    ]
    print(f"DEBUG: Docker command: docker {' '.join(docker_args)}", file=sys.stderr, flush=True)
    print(f"DEBUG: OPENAPI_MCP_HEADERS: {openapi_headers[:50]}...", file=sys.stderr, flush=True)

    # Create client with the Docker command and environment variables
    # The env dict is passed separately to MultiServerMCPClient
    client = MultiServerMCPClient(
        {
            "notion": {
                "transport": "stdio",
                "command": "docker",
                "args": docker_args,
                "env": {
                    "OPENAPI_MCP_HEADERS": openapi_headers
                },
            }
        }
    )

    try:
        import sys
        print(f"DEBUG: Attempting to connect to Notion MCP server via Docker...", file=sys.stderr, flush=True)
        print(f"DEBUG: Using OPENAPI_MCP_HEADERS format for authentication", file=sys.stderr, flush=True)

        tools = await client.get_tools()
        print(f"DEBUG: Successfully retrieved {len(tools)} tools from Notion MCP", file=sys.stderr, flush=True)
        if tools:
            print(f"DEBUG: Tool names: {[t.name for t in tools]}", file=sys.stderr, flush=True)
        return tools
    except Exception as e:
        # If MCP server fails to start or connect, return empty list
        import traceback
        import sys
        print(f"ERROR: Could not connect to Notion MCP server: {e}", file=sys.stderr, flush=True)
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return []


async def create_notion_todo_via_mcp(title: str, due_iso: Optional[str] = None) -> str:
    """Create a todo in Notion via MCP.

    This is a convenience wrapper that calls the MCP tool directly.
    For use in LangChain agents, use get_notion_mcp_tools() instead.
    """
    settings = get_settings()
    if not settings.notion_integration_token:
        return "Notion MCP not configured. Set NOTION_INTEGRATION_TOKEN in .env"

    # Use OPENAPI_MCP_HEADERS format
    import json
    token = settings.notion_integration_token.strip() if settings.notion_integration_token else None
    if not token:
        return "Notion MCP not configured. Set NOTION_INTEGRATION_TOKEN in .env"

    openapi_headers = json.dumps({
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28"
    })

    client = MultiServerMCPClient(
        {
            "notion": {
                "transport": "stdio",
                "command": "docker",
                "args": [
                    "run",
                    "--rm",
                    "-i",
                    "-e", "OPENAPI_MCP_HEADERS",
                    "mcp/notion",
                ],
                "env": {
                    "OPENAPI_MCP_HEADERS": openapi_headers
                },
            }
        }
    )

    tools = await client.get_tools()
    # Find the create_todo or similar tool from Notion MCP
    # Tool names depend on what the mcp/notion server exposes
    create_tool = None
    for tool in tools:
        if "create" in tool.name.lower() and ("todo" in tool.name.lower() or "page" in tool.name.lower()):
            create_tool = tool
            break

    if not create_tool:
        return "Could not find create todo tool in Notion MCP server"

    # Call the tool with appropriate arguments
    # Exact args depend on what mcp/notion expects
    try:
        result = await create_tool.ainvoke(
            {
                "title": title,
                "due_date": due_iso,
            }
        )
        return f"Created Notion todo: {result}"
    except Exception as e:
        return f"Error creating Notion todo: {e}"


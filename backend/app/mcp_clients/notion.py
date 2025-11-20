"""Notion MCP client using langchain-mcp-adapters.

This module connects to the Notion MCP server running in Docker via stdio transport
and exposes LangChain-compatible tools for creating and managing todos.
"""
import json
import logging
from typing import List

from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config import get_settings

logger = logging.getLogger(__name__)


async def get_notion_mcp_tools() -> List:
    """Get LangChain tools from the Notion MCP server.

    Returns a list of LangChain tools that can be used in agents.
    If Notion is not configured, returns an empty list.

    Returns:
        List of LangChain tools from the Notion MCP server
    """
    settings = get_settings()
    token = settings.notion_integration_token
    db_id = settings.notion_database_id

    if not token or not db_id:
        logger.debug("Notion MCP not fully configured (token or DB ID missing). Skipping.")
        return []

    token = token.strip()
    token_preview = token[:10] + "..." if token else "None"
    logger.debug(f"Using Notion token starting with: {token_preview}")
    logger.debug(f"Token length: {len(token) if token else 0}")
    logger.debug(f"Notion Database ID: {db_id}")

    # Check token format
    if not (token.startswith("secret_") or token.startswith("ntn_")):
        logger.warning("Token doesn't start with 'secret_' or 'ntn_'. Format might be incorrect.")

    # Use OPENAPI_MCP_HEADERS for authentication as required by mcp/notion
    openapi_headers = json.dumps({
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28"
    })

    docker_args = [
        "run",
        "--rm",
        "-i",
        "-e",
        f"OPENAPI_MCP_HEADERS={openapi_headers}",
        "mcp/notion",
    ]
    logger.debug(f"Docker command: docker {' '.join(docker_args[:4])} OPENAPI_MCP_HEADERS=***")

    client = MultiServerMCPClient(
        {
            "notion": {
                "transport": "stdio",
                "command": "docker",
                "args": docker_args,
                "env": {
                    "NOTION_DATABASE_ID": db_id  # Pass DB ID as env var to MCP server
                }
            }
        }
    )

    try:
        logger.debug("Attempting to connect to Notion MCP server via Docker...")
        tools = await client.get_tools()
        logger.debug(f"Successfully retrieved {len(tools)} tools from Notion MCP")
        if tools:
            logger.debug(f"Tool names: {[t.name for t in tools]}")
        return tools
    except Exception as e:
        logger.error(f"Could not connect to Notion MCP server: {e}", exc_info=True)
        return []


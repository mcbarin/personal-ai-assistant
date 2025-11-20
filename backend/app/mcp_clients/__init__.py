"""MCP (Model Context Protocol) clients registry.

This module provides a dynamic registry for MCP clients, making it easy to add
new MCP integrations as the project grows.
"""
from typing import Dict, List, Optional, Any

# Type hint for LangChain tools (avoid import if not available for linting)
try:
    from langchain_core.tools import BaseTool
except ImportError:
    BaseTool = Any  # Fallback for type hints


class MCPClientRegistry:
    """Registry for managing multiple MCP clients dynamically."""

    def __init__(self):
        self._clients: Dict[str, callable] = {}

    def register(self, name: str, client_factory: callable):
        """Register an MCP client factory.

        Args:
            name: Unique identifier for the MCP client (e.g., 'notion', 'slack')
            client_factory: Async function that returns a list of LangChain tools
        """
        self._clients[name] = client_factory

    async def get_all_tools(self) -> List[BaseTool]:
        """Get all tools from all registered MCP clients.

        Returns:
            List of LangChain tools from all registered MCP clients
        """
        all_tools: List[BaseTool] = []
        for name, factory in self._clients.items():
            try:
                tools = await factory()
                if tools:
                    all_tools.extend(tools)
            except Exception as e:
                # Log error but continue with other clients
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to get tools from MCP client '{name}': {e}")
        return all_tools

    async def get_tools(self, client_name: str) -> List[BaseTool]:
        """Get tools from a specific MCP client.

        Args:
            client_name: Name of the MCP client

        Returns:
            List of LangChain tools from the specified client, or empty list if not found
        """
        if client_name not in self._clients:
            return []
        try:
            return await self._clients[client_name]()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to get tools from MCP client '{client_name}': {e}")
            return []

    def list_clients(self) -> List[str]:
        """List all registered MCP client names."""
        return list(self._clients.keys())


# Global registry instance
registry = MCPClientRegistry()

# Auto-register available MCP clients
from .notion import get_notion_mcp_tools

registry.register("notion", get_notion_mcp_tools)

__all__ = ["registry", "MCPClientRegistry", "get_notion_mcp_tools"]


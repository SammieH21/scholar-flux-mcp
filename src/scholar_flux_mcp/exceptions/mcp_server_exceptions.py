"""Defines the core exceptions raised when unexpected errors occur during MCP server execution."""


class MCPServerException(Exception):
    """Exception raised when MCP server execution fails due to an unexpected error."""


class MCPServerInitializationException(MCPServerException):
    """Exception raised when MCP server execution fails due to initialize."""


__all__ = ["MCPServerException", "MCPServerInitializationException"]

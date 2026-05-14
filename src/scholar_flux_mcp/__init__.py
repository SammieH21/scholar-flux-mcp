"""ScholarFlux MCP Server.

A Model Context Protocol server that enables LLMs to search academic databases and synthesize research findings with
ScholarFlux.

Example MCP configuration:
    {
        "mcpServers": {
            "scholar_flux": {
                "command": "python",
                "args": ["-m", "scholar_flux_mcp"]
            }
        }
    }

Or with Docker:
    {
        "mcpServers": {
            "scholar_flux_docker": {
                "command": "docker",
                "args": ["run", "-i", "--rm", "scholar_flux_mcp"]
            }
        }
    }

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scholar_flux_mcp.package_metadata import __version__
from scholar_flux_mcp.server import _load_mcp
from scholar_flux_mcp.utils.initializer import initialize_mcp_package

if TYPE_CHECKING:
    from scholar_flux_mcp.server import create_server, mcp  # noqa: TCH004


config_settings, logger, masker = initialize_mcp_package()


def __getattr__(name: str) -> Any:
    """Retrieves `scholar_flux_mcp` attributes, loading `scholar_flux_mcp.server` MCP components only when needed.

    This internal implementation delegates the retrieval of `create_server` and `mcp` to the `scholar_flux_mcp.server`.

    When the `mcp[cli]` package is missing, attempting to retrieve these components will raise an `MCPImportError`.
    Otherwise, they will be loaded only when explicitly requested by the user or by starting the server via
    `python -m scholar_flux_mcp`.

    """
    if name in ("create_server", "mcp"):
        return _load_mcp(name)  # load MCP components from the server only when you need them
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", "create_server", "mcp"]

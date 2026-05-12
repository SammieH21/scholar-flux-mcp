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

from scholar_flux_mcp.package_metadata import __version__
from scholar_flux_mcp.server import create_server, mcp

__all__ = ["__version__", "create_server", "mcp"]

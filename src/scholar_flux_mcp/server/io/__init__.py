"""Helpers for the ScholarFlux MCP server.

Each module contains a set of models and formatters required to run the MCP server via the `main.py` entrypoint.
- health_check: Utilities for verifying the status of the ScholarFlux MCP server
- search: Multi-provider academic paper search
- synthesize: AI-powered research synthesis
- providers: Provider information and management
- history: Persistent local MCP output storage and retrieval via SQLModel

"""

from scholar_flux_mcp.server.io.health_check import (
    HealthCheckFormatter,
    HealthCheckToolInput,
)
from scholar_flux_mcp.server.io.providers import ListProvidersInput
from scholar_flux_mcp.server.io.relevance_search import RelevanceSearchFormatter, RelevanceSearchToolInput
from scholar_flux_mcp.server.io.search import SearchFormatter, SearchToolInput
from scholar_flux_mcp.server.io.synthesis import SynthesisFormatter, SynthesisToolInput

__all__ = [
    "HealthCheckToolInput",
    "HealthCheckFormatter",
    "ListProvidersInput",
    "RelevanceSearchFormatter",
    "RelevanceSearchToolInput",
    "SearchFormatter",
    "SearchToolInput",
    "SynthesisFormatter",
    "SynthesisToolInput",
]

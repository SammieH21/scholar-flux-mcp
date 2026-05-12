"""Services layer for ScholarFlux MCP server.

Each service has a single responsibility:
- SearchService: Multi-provider academic record search
- CacheService: MongoDB-backed caching management
- SynthesisService: PydanticAI-powered literature synthesis

"""

from scholar_flux_mcp.services.cache_service import CacheService
from scholar_flux_mcp.services.grounding_service import GroundingService
from scholar_flux_mcp.services.history_service import HistoryService
from scholar_flux_mcp.services.provider_service import ProviderService
from scholar_flux_mcp.services.relevance_search_service import RelevanceSearchService
from scholar_flux_mcp.services.search_service import SearchService
from scholar_flux_mcp.services.synthesis_service import SynthesisService

__all__ = [
    "ProviderService",
    "CacheService",
    "RelevanceSearchService",
    "SearchService",
    "GroundingService",
    "SynthesisService",
    "HistoryService",
]

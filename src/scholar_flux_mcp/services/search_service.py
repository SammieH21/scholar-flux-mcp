"""Search service for ScholarFlux MCP server.

Provides multi-provider academic record search with normalization. Single responsibility: Coordinating searches across
academic databases.

"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import zip_longest
from typing import TYPE_CHECKING, Any

from scholar_flux_mcp.exceptions import ScholarFluxImportError
from scholar_flux_mcp.models import (
    PaginationInfo,
    SearchInput,
    SearchOutput,
    SearchResponseSummary,
)
from scholar_flux_mcp.services.base_research_service import BaseResearchService
from scholar_flux_mcp.services.cache_service import CacheService
from scholar_flux_mcp.utils.preprocessing_utils import SearchRecordPreprocessingUtils

if TYPE_CHECKING:
    from scholar_flux import MultiSearchCoordinator, SearchCoordinator
    from scholar_flux.utils.record_types import NormalizedRecordList
else:
    try:
        from scholar_flux import MultiSearchCoordinator, SearchCoordinator
    except ImportError:
        MultiSearchCoordinator = None
        SearchCoordinator = None

logger = logging.getLogger(__name__)


class SearchService(BaseResearchService):
    """Multi-provider academic record search service.

    Wraps ScholarFlux's SearchCoordinator and MultiSearchCoordinator to provide unified search across academic
    databases.

    """

    SORT_BY_YEAR: bool = False
    DEFAULT_HISTORY_TTL: int | float | None = 600

    def __init__(self, cache_service: CacheService, **kwargs: Any) -> None:
        """Initialize search service.

        Args:
            cache_service: Initialized cache service for session management.

        """
        self.cache_service = cache_service
        super().__init__(**kwargs)

    @property
    def cache_service(self) -> CacheService:
        """Returns the cache service that stores information on the DataCacheManager and SessionManager instances."""
        if not self._cache_service:
            raise ValueError("A cache service has not yet been assigned.")
        return self._cache_service

    @cache_service.setter
    def cache_service(self, cache_service: CacheService) -> None:
        """Sets the CacheService with validation.

        Used on initialization to verify the structure of the Service.

        """
        self._validate_service_dependency(cache_service, CacheService, allow_missing=True)
        self._cache_service = cache_service

    def sort_records_by_provider(self, records: NormalizedRecordList) -> NormalizedRecordList:
        """Helper for sorting normalized records by record index followed by provider name."""
        records_by_provider: dict[str, NormalizedRecordList] = defaultdict(list)

        for record in records:
            if provider_name := record.get("provider_name"):
                records_by_provider[provider_name].append(record)

        zipped_records = zip_longest(*records_by_provider.values())
        return [record for provider_records in zipped_records for record in provider_records if record is not None]

    async def search(
        self,
        params: SearchInput,
        from_history_cache: bool = False,
        store_history_cache: bool = False,
        force_refresh: bool | None = None,
        **kwargs: Any,
    ) -> SearchOutput:
        """Executes a multi-provider search, retrieving from history if the history cache is enabled.

        Args:
            params (SearchInput):
                The validated record search parameters preprocessed from a `SearchToolInput` and used to customize the
                current record search.
            from_history_cache (bool):
                Indicates whether the history cache should be used if available.
            store_history_cache (bool):
                Indicates whether the output should be stored within the history cache when retrieved.
            force_refresh:
                No-op: Added for compatibility with the API other ScholarFlux MCP research tools.

            **kwargs:
                Additional keyword arguments to pass to `search_pages`. Useful for debugging. possible parameters
                include the following:

                - `cache_only`: Returns a `NonResponse` instead if a request doesn't exist in cache.
                - `from_request_cache`: Whether the session cache should be used when available.
                - `from_processing_cache`: Whether the session cache should be used when available.
                - `parameters`: API-specific fields (note that parameters that don't apply are ignored).
                - `normalize_records`: Flag indicating whether records should be immediately normalized on retrieval.

        Returns:
            SearchOutput with normalized records and metadata.

        """
        try:
            # ScholarFlux is a dependency for performing searches
            if SearchCoordinator is None or MultiSearchCoordinator is None:
                raise ScholarFluxImportError()

            self._validate_service_dependency(params, SearchInput, allow_missing=False)

            # Cache retrieval
            cached_search_output = (
                await self.retrieve_history_cache(params, raise_on_error=False)
                if from_history_cache and self.history_enabled
                else None
            )

            if cached_search_output is not None:
                return cached_search_output

            # Create coordinators for each provider
            coordinators = self._create_coordinators(params)

            # Execute multi-provider search
            multisearch_coordinator = MultiSearchCoordinator()
            multisearch_coordinator.add_coordinators(coordinators)

            logger.info(f"Searching {len(set(params.providers))} providers with {len(set(params.queries))} queries.")

            # Search specified number of pages
            page_start = 1 + params.page_offset
            page_end = page_start + params.pages
            page_range = range(page_start, page_end)
            results = multisearch_coordinator.search_pages(page_range, **kwargs)

            # Filter successful results and normalize
            successful = results.filter()
            if unsuccessful := results.filter(invert=True):
                logger.warning("The following searches were unsuccessful: %s", unsuccessful)
            normalized_records = successful.normalize(include={"query", "page", "cached", "retrieval_timestamp"})

            if self.SORT_BY_YEAR:
                normalized_records = sorted(
                    normalized_records, reverse=True, key=lambda record: record.get("year") or -1
                )
            else:
                normalized_records = self.sort_records_by_provider(normalized_records)
            logger.info("Received the following records: %s", successful)

            # Convert to SearchRecord models
            search_records = SearchRecordPreprocessingUtils.convert_records(
                normalized_records,
                params.max_records,
                open_access_only=params.open_access_only,
                year_from=params.year_from,
                year_to=params.year_to,
            )

            logger.info(f"Retrieved {len(search_records or [])} complete records")

            # Build pagination info
            pagination = PaginationInfo(
                total_records=successful.record_count,
                providers_queried=len(params.providers),
                providers_successful=len({result.provider_name for result in successful}),
                pages_successful=len(successful),
                pages_retrieved=len(results),
            )

            response_summaries = [
                SearchResponseSummary(
                    provider_name=result.provider_name,
                    query=result.query,
                    page=result.page,
                    success=bool(result),
                    status_code=result.status_code,
                    error=result.error,
                    message=result.message,
                    record_count=result.record_count,
                    cached=result.cached,
                    retrieval_timestamp=result.retrieval_timestamp,
                )
                for result in results
            ]
            search_output = SearchOutput(
                search_input=params,
                records=search_records,
                pagination=pagination,
                response_summaries=response_summaries,
            )

            # Store the results within the history cache when not already cached.
            if store_history_cache and self.history_enabled:
                await self.store_history_cache(search_output, raise_on_error=False)

            return search_output

        except ImportError as e:
            logger.error(f"ScholarFlux import error: {e}")
            raise RuntimeError("ScholarFlux not available. Install with: pip install scholar-flux") from e
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _create_coordinators(self, params: SearchInput) -> list[Any]:
        """Create SearchCoordinator instances for each query × provider combination.

        Args:
            params: Search parameters with a list of SearchCoordinatorConfig instances

        Returns:
            List of configured SearchCoordinator instances.

        """
        from scholar_flux import SearchCoordinator

        coordinators = []
        for config in params.search_coordinator_config:
            try:
                coordinator = SearchCoordinator(
                    provider_name=config.provider_name,
                    query=config.query,
                    cache_manager=self._cache_service.data_cache_manager,
                    session=self._cache_service.create_session(),
                    annotate_records=True,
                    **config.api_specific_fields,
                )
                coordinators.append(coordinator)
            except Exception as e:
                logger.warning(f"Failed to create coordinator for {config.provider_name}/{config.query}: {e}")

        return coordinators


__all__ = ["SearchService"]

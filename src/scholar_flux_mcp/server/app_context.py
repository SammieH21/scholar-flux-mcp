"""Defines the app context used as a container for each individual ScholarFluxMCP service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scholar_flux_mcp.services import (
    CacheService,
    HistoryService,
    ProviderService,
    RelevanceSearchService,
    SearchService,
    SynthesisService,
)


@dataclass
class AppContext:
    """Class that contains the lazily initialized services used to interact with ScholarFluxMCP."""

    search_service: SearchService
    relevance_search_service: RelevanceSearchService
    synthesis_service: SynthesisService
    provider_service: ProviderService = field(default_factory=lambda: ProviderService())
    history_service: HistoryService = field(default_factory=lambda: HistoryService(enable=False))

    def __post_init__(self) -> None:
        """Post initialization validation step to ensure that assignments are the correct service types."""
        self.validate(self.search_service, SearchService)
        self.validate(self.relevance_search_service, RelevanceSearchService)
        self.validate(self.synthesis_service, SynthesisService)
        self.validate(self.provider_service, ProviderService)
        self.validate(self.history_service, HistoryService)

    @classmethod
    def create(
        cls,
        search_service: SearchService,
        relevance_search_service: RelevanceSearchService | None = None,
        synthesis_service: SynthesisService | None = None,
        history_service: HistoryService | None = None,
        **kwargs: Any,
    ) -> AppContext:
        """Constructor initializing a new AppContext with the received services."""
        if not isinstance(search_service, SearchService):
            raise TypeError("A valid SearchService must be received to initialize the AppContext")

        # Prefer explicit history services, followed by a shared history service if available. otherwise no-op:
        app_history_service = history_service or search_service._history_service or HistoryService(enable=False)

        init_relevance_search_service = relevance_search_service or RelevanceSearchService(
            search_service=search_service, history_service=app_history_service
        )

        init_synthesis_service = synthesis_service or SynthesisService(
            relevance_search_service=init_relevance_search_service, history_service=app_history_service
        )

        return cls(
            search_service=search_service,
            relevance_search_service=init_relevance_search_service,
            synthesis_service=init_synthesis_service,
            history_service=app_history_service,
            **kwargs,
        )

    @property
    def cache_service(self) -> CacheService:
        """Returns the cache service used by the `SearchService` to retrieve academic records from cache."""
        return self.search_service.cache_service

    @cache_service.setter
    def cache_service(self, cache_service: CacheService) -> None:
        """Verifies that the received cache service contains at least one valid value."""
        if not isinstance(cache_service, CacheService):
            raise TypeError(f"Expected a valid cache service to be assigned but instead received type {CacheService}")
        self.search_service.cache_service = cache_service

    @classmethod
    def validate(cls, obj: object, expected_type: type) -> None:
        """Validates the service assignments received by the `AppContext`.

        Args:
            value (object): The value to validate.
            expected_type (type): The expected type.

        Raises:
            TypeError: If the value is not of the specified type.

        """
        if not isinstance(obj, expected_type):
            context_name = cls.__name__
            expected_type_name = expected_type.__name__
            raise TypeError(
                f"The {context_name} expected a {expected_type_name}, but instead received {type(obj).__name__}"
            )


__all__ = ["AppContext"]

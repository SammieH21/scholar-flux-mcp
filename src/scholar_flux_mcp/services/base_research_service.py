"""Defines the `BaseResearchService` base class for record retrieval and synthesis in ScholarFlux MCP server."""

from __future__ import annotations

import logging
from typing import TypeVar, overload

from scholar_flux_mcp.exceptions import (
    HistoryCacheInitializationException,
    HistoryCacheRetrievalException,
    HistoryCacheStorageException,
)
from scholar_flux_mcp.models import (
    RelevanceSearchInput,
    RelevanceSearchOutput,
    SearchInput,
    SearchOutput,
    SynthesisInput,
    SynthesisOutput,
)
from scholar_flux_mcp.models.type_aliases import ResearchToolInput, ResearchToolOutput
from scholar_flux_mcp.services.history_service import HistoryService

T = TypeVar("T", bound=object)

logger = logging.getLogger(__name__)


class BaseResearchService:
    """Base class for core ScholarFlux MCP research services.

    This base class provides integration with core history storage and retrieval in addition to dependency validation.
    The `SearchService`, `RelevanceSearchService`, and `SynthesisService` each override the `BaseResearchService` while
    defining the `history_service` as a keyword parameter and core positional parameters specific to each service
    normally.

    Attributes:
        history_service (HistoryService):
            The history service used to both store and retrieve previously stored outputs from cache.

    Class Attributes:
        DEFAULT_HISTORY_TTL (int | float | None):
            Defines the default TTL for storage retrieval. Services can override this class attribute for targeted
            cache expiration for each tool.

    """

    DEFAULT_HISTORY_TTL: int | float | None = None

    def __init__(self, *, history_service: HistoryService | None = None) -> None:
        """Initializes the research service with optional history tracking.

        Args:
            history_service (HistoryService | None):
                The history service used for retrieving previous tool outputs from cache.

        """
        self._history_service: HistoryService | None = history_service

    @property
    def history_service(self) -> HistoryService:
        """Returns the history service used to record tool output history."""
        if self._history_service is None:
            class_name = self.__class__.__name__
            raise HistoryCacheInitializationException(
                f"A history service has not yet been assigned to the {class_name}."
            )
        return self._history_service

    @history_service.setter
    def history_service(self, history_service: HistoryService | None) -> None:
        """Sets the HistoryService with validation.

        Args:
            history_service: A history service instance or None to disable history.

        Raises:
            TypeError: If the assigned value is not a HistoryService.

        """
        self._history_service = self._validate_service_dependency(history_service, HistoryService)

    @overload
    async def retrieve_history_cache(
        self,
        input: SynthesisInput,
        raise_on_error: bool = True,
        *,
        ttl: int | float | None = None,
    ) -> SynthesisOutput | None:
        """The latest synthesis output is returned when available if a synthesis input is provided."""
        ...

    @overload
    async def retrieve_history_cache(
        self,
        input: RelevanceSearchInput,
        raise_on_error: bool = True,
        *,
        ttl: int | float | None = None,
    ) -> RelevanceSearchOutput | None:
        """The latest relevance search output is returned when available if a relevance search input is provided."""
        ...

    @overload
    async def retrieve_history_cache(
        self,
        input: SearchInput,
        raise_on_error: bool = True,
        *,
        ttl: int | float | None = None,
    ) -> SearchOutput | None:
        """The latest search output is returned when available if a search input is provided."""
        ...

    async def retrieve_history_cache(
        self,
        input: ResearchToolInput,
        raise_on_error: bool = True,
        *,
        ttl: int | float | None = None,
    ) -> ResearchToolOutput | None:
        """Helper for retrieving previous output given the current tool input."""
        try:
            cached_output = await self.history_service.retrieve(
                input,
                successful_only=True,
                ttl=ttl if ttl is not None else self.DEFAULT_HISTORY_TTL,
                raise_on_error=True,
            )
            if cached_output:
                logger.info(f"Successfully retrieved {type(input).__name__} from cache.")

            return cached_output
        except (HistoryCacheInitializationException, HistoryCacheRetrievalException) as e:
            msg = (
                f"Couldn't retrieve the output associated with the {type(input).__name__} within the output history "
                f"cache: {e}"
            )
            if raise_on_error:
                logger.error(f"{msg}")
                raise
            logger.warning(f"{msg} Skipping output retrieval...")
        return None

    async def store_history_cache(
        self,
        output: ResearchToolOutput,
        raise_on_error: bool = True,
    ) -> None:
        """Helper for storing the current tool output within the history cache."""
        try:
            await self.history_service.store(output, raise_on_error=True)
        except (HistoryCacheInitializationException, HistoryCacheStorageException) as e:
            msg = f"Couldn't store the {type(output).__name__} within the output history cache: {e}"
            if raise_on_error:
                logger.error(f"{msg}")
                raise
            logger.warning(f"{msg} Skipping output storage...")

    @property
    def history_enabled(self) -> bool:
        """Property indicating whether the history service is enabled."""
        return self._history_service is not None and self._history_service.enabled

    @classmethod
    def _validate_service_dependency(cls, obj: T, expected_type: type, *, allow_missing: bool = True) -> T:
        """Validate a service dependency type.

        Args:
            value (object): The value to validate.
            expected_type (type): The expected type.
            allow_missing (type): Whether a value of `None` is acceptable.

        Returns:
            object: The successfully validated object when an error is not raised.

        Raises:
            TypeError: If the value is not of the specified type.

        """
        class_name = cls.__name__
        expected_type_name = expected_type.__name__

        missing_required = obj is None and not allow_missing
        unexpected_type = obj is not None and not isinstance(obj, expected_type)

        if missing_required or unexpected_type:
            raise TypeError(f"{class_name} expected a {expected_type_name}, but received {type(obj).__name__}")
        return obj

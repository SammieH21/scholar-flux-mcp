"""Defines the RelevanceSearchService for retrieving across APIs and reranking records by topic embedding similarity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from scholar_flux_mcp.agents import RecordTopicSimilarityEmbedder
from scholar_flux_mcp.exceptions import EmbedderException, PydanticAIImportError, RapidFuzzImportError
from scholar_flux_mcp.models.schemas import (
    IndexedSearchRecord,
    RecordTopicSimilarityOutput,
    RelevanceSearchInput,
    RelevanceSearchOutput,
    SearchInput,
    SearchRecordList,
    SynthesisInput,
)
from scholar_flux_mcp.services.base_research_service import BaseResearchService
from scholar_flux_mcp.services.search_service import SearchService
from scholar_flux_mcp.utils import SearchRecordPreprocessingUtils

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class RelevanceSearchService(BaseResearchService):
    """Service for performing relevance search and ranking of academic records.

    Orchestrates the relevance search workflow:
    1. Search for relevant records via SearchService
    2. Filter and rank by relevance using embeddings
    3. Return ranked records with similarity scores

    """

    def __init__(
        self,
        search_service: SearchService,
        *,
        record_topic_similarity_embedder: RecordTopicSimilarityEmbedder | None = None,
        **kwargs: Any,
    ):
        """Initializes the relevance search service.

        Args:
            search_service: Initialized search service for record retrieval.

        """
        self.search_service = (
            search_service.search_service if isinstance(search_service, RelevanceSearchService) else search_service
        )
        self.record_topic_similarity_embedder = record_topic_similarity_embedder or RecordTopicSimilarityEmbedder()
        super().__init__(**kwargs)

    @property
    def search_service(self) -> SearchService:
        """Returns the search service that is used to retrieve data from Academic Databases."""
        if not self._search_service:
            raise ValueError("A search service has not yet been assigned.")
        return self._search_service

    @search_service.setter
    def search_service(self, search_service: SearchService) -> None:
        """Sets the SearchService with validation.

        Used on initialization to verify the structure of the service.

        Args:
            search_service: A search service to validate and assign to the `search_service` property if valid.

        """
        self._validate_service_dependency(search_service, SearchService, allow_missing=False)
        self._search_service = search_service

    @property
    def record_topic_similarity_embedder(self) -> RecordTopicSimilarityEmbedder:
        """Returns the `RecordTopicSimilarityEmbedder` used for cosine similarity-based record reranking."""
        if not self._record_topic_similarity_embedder:
            raise ValueError("A valid record-topic similarity embedder has not yet been created.")
        return self._record_topic_similarity_embedder

    @record_topic_similarity_embedder.setter
    def record_topic_similarity_embedder(self, embedder: RecordTopicSimilarityEmbedder) -> None:
        """Validates that the received value is a `RecordTopicSimilarityEmbedder` before assignment."""
        self._validate_service_dependency(embedder, RecordTopicSimilarityEmbedder, allow_missing=False)
        self._record_topic_similarity_embedder = embedder

    async def embed_records(
        self, records: SearchRecordList, params: RelevanceSearchInput | SynthesisInput, raise_on_error: bool = False
    ) -> tuple[RecordTopicSimilarityOutput | None, Sequence[IndexedSearchRecord]]:
        """Embeds a set of records using the configured `RecordTopicSimilarityEmbedder`."""
        try:
            embedding_similarity_output = await self.record_topic_similarity_embedder.embedding_similarity(
                records, params.question, params.queries, params.categories
            )
            indexed_records = self.record_topic_similarity_embedder.filter_similarity_scores(
                embedding_similarity_output.record_similarity_scores,
                threshold=params.similarity_threshold,
                max_records=params.max_records,
            )
        except EmbedderException as e:
            err = (
                "Couldn't use embeddings to filter the resulting record set "
                f"(embedder={self.record_topic_similarity_embedder.model_name}). {e}"
            )
            if raise_on_error:
                raise e.__class__(err) from e

            logger.warning(f"{err}. Skipping...")

            embedding_similarity_output = None
            # Further filtered via params.max_records at the next step
            indexed_records = [IndexedSearchRecord.from_search_record(record, i) for i, record in enumerate(records)]
        return embedding_similarity_output, indexed_records

    async def relevance_search(
        self,
        params: RelevanceSearchInput,
        from_history_cache: bool = False,
        store_history_cache: bool = False,
        raise_on_error: bool = False,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> RelevanceSearchOutput:
        """Executes a relevance search and ranks of academic records by similarity to the current topic.

        Args:
            params (RelevanceSearchInput):
                The parameters used to customize the relevance search. Includes the question, queries, categories, and
                basic SearchInput options used to configure the record search that precedes the reranking step.
            from_history_cache (bool):
                Indicates whether the history cache should be used if available.
            store_history_cache (bool):
                Indicates whether the output should be stored within the history cache when retrieved.
            force_refresh (bool):
                Whether cached results should be refreshed while enabling record search cache retrieval.
            raise_on_error (bool): Indicates whether an error should be raised if the embedding step fails.
            **kwargs: Additional parameters passed to `MultiCoordinator.search_pages()` via the `SearchService`.

        Returns:
            Structured relevance search output with ranked records and similarity scores.

        """
        try:
            self._validate_service_dependency(params, RelevanceSearchInput, allow_missing=False)

            ### Cache Retrieval (If available) ###
            cached_relevance_search_output = (
                await self.retrieve_history_cache(params, raise_on_error=False)
                if from_history_cache and not force_refresh and self.history_enabled
                else None
            )

            if cached_relevance_search_output is not None:
                return cached_relevance_search_output

            ### Data Preparation Phase ###
            search_input = SearchInput.from_search_params(params)
            logger.info(f"Performing relevance search with {len(params.queries)} queries: {params.queries}")
            search_results = await self.search_service.search(
                search_input, from_history_cache=from_history_cache, store_history_cache=False, **kwargs
            )

            if not search_results.records:
                logger.warning("No records found after search. Skipping ranking by relevance..")
                return RelevanceSearchOutput(relevance_search_input=params, search_output=search_results)

            # deduplicate, and filter records based on embedding similarity to the original question
            records = SearchRecordPreprocessingUtils.deduplicate_records(search_results.records)
            embedding_similarity_output, indexed_records = await self.embed_records(
                records, params, raise_on_error=raise_on_error
            )

            # Stores the generated outputs and indexed records from the relevance search
            relevance_search_output = RelevanceSearchOutput(
                relevance_search_input=params,
                indexed_records=indexed_records,
                search_output=search_results,
                record_topic_similarity_output=embedding_similarity_output,
            )

            # Store the results within the history cache when not already cached.
            if store_history_cache and self.history_enabled:
                await self.store_history_cache(relevance_search_output, raise_on_error=False)

            return relevance_search_output

        ### Error Handling ###
        except (RuntimeError, PydanticAIImportError, RapidFuzzImportError):
            # Re-raises uncaught runtime/import errors for inspection
            raise
        except Exception as e:
            logger.error(f"Relevance Search failed: {e}")
            raise


__all__ = ["RelevanceSearchService"]

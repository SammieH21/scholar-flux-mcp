"""Tests for the relevance service and integration with cache and search services."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scholar_flux_mcp.agents.record_topic_similarity_embedder import RecordTopicSimilarityEmbedder
from scholar_flux_mcp.exceptions import DocumentEmbeddingFailedException, PydanticAIImportError
from scholar_flux_mcp.models import (
    JSONDataModel,
    RelevanceSearchOutput,
    SearchInput,
    SearchOutput,
)
from scholar_flux_mcp.services import HistoryService
from tests.testing_utilities import raise_error

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture()
def history_cache(tmp_path, cleanup) -> HistoryService:
    """Temporary cache used to store results for caching search service for later steps when required."""
    url = f"sqlite:///{tmp_path}/relevance_search_cache.db"
    # ensure the history cache doesn't cause errors when unavailable - we're testing the relevance search service
    return HistoryService(url, raise_on_error=False)


@pytest.fixture()
def mock_record_topic_similarity_embedder(
    patch_ollama_available, mock_embedding_model
) -> Iterator[RecordTopicSimilarityEmbedder]:
    """Creates a mock embedder for later relevance searches with mocked embeddings."""
    record_topic_similarity_embedder = RecordTopicSimilarityEmbedder()
    record_topic_similarity_embedder.get_or_create_embedder()
    with record_topic_similarity_embedder.embedder.override(model=mock_embedding_model):
        yield record_topic_similarity_embedder


@pytest.fixture
def computer_literacy_relevance_search_service(
    mock_computer_literacy_api_searches,
    mock_computer_literacy_relevance_search_app_context,
    mock_record_topic_similarity_embedder,
    monkeypatch,
    history_cache,
):
    """Prepares the current relevance search service to enable cache retrieval and mocked embedding outputs."""
    monkeypatch.setattr(
        mock_computer_literacy_relevance_search_app_context.search_service, "_history_service", history_cache
    )

    # temporarily assign a mocked embedding model (doesn't conflict with live relevance searches)
    monkeypatch.setattr(
        mock_computer_literacy_relevance_search_app_context.relevance_search_service,
        "record_topic_similarity_embedder",
        mock_record_topic_similarity_embedder,
    )
    return mock_computer_literacy_relevance_search_app_context.relevance_search_service


@pytest.mark.asyncio
async def test_computer_literacy_multi_provider_multisearch_integration(
    computer_literacy_relevance_search_service,
    mock_computer_literacy_relevance_search_input,
):
    """Verifies that the search service correctly retrieves records from PLOS and OpenAlex providers.

    This test verifies that queries on `computer literacy` correctly retrieve corresponding records from the filesystem
    cache via `scholar-flux`.

    All other tests require that results are received for the RelevanceSearchService to deduplicate and rerank the
    retrieved records.

    """
    search_service = computer_literacy_relevance_search_service.search_service
    mock_search_input = SearchInput.from_search_params(mock_computer_literacy_relevance_search_input)
    search_results = await search_service.search(mock_search_input, store_history_cache=True)
    assert search_results and isinstance(search_results, SearchOutput)
    assert search_results.cache_hit  # all records should originate from cache
    assert search_results.pagination.pages_retrieved == len(mock_search_input.providers) * mock_search_input.pages


@pytest.mark.asyncio
async def test_computer_literacy_basic_relevance_search(
    computer_literacy_relevance_search_service,
    mock_computer_literacy_relevance_search_input,
):
    """Verifies that the relevance search service correctly uses an embedding model for reranking when available."""
    # retrieve cache from only the previous step
    relevance_search_output = await computer_literacy_relevance_search_service.relevance_search(
        mock_computer_literacy_relevance_search_input, from_history_cache=True, force_refresh=True
    )
    assert isinstance(relevance_search_output, RelevanceSearchOutput) and relevance_search_output.successful


@pytest.mark.asyncio
async def test_computer_literacy_relevance_search_fails(
    computer_literacy_relevance_search_service,
    mock_computer_literacy_relevance_search_input,
    monkeypatch,
):
    """Verifies that relevance searches return records without embedding similarity on embedding errors by default."""
    # retrieve cache from only the previous step
    monkeypatch.setattr(
        computer_literacy_relevance_search_service.record_topic_similarity_embedder.embedder,
        "embed_documents",
        raise_error(DocumentEmbeddingFailedException),
    )
    relevance_search_output = await computer_literacy_relevance_search_service.relevance_search(
        mock_computer_literacy_relevance_search_input,
        from_history_cache=True,
        force_refresh=True,
        raise_on_error=False,
    )
    assert isinstance(relevance_search_output, RelevanceSearchOutput) and not relevance_search_output.successful


@pytest.mark.asyncio
async def test_computer_literacy_relevance_search_reraises_when_enabled(
    computer_literacy_relevance_search_service,
    mock_computer_literacy_relevance_search_input,
    monkeypatch,
):
    """Verifies that relevance searches reraise embedding errors when `raise_on_error=True`."""
    # retrieve cache from only the previous step
    monkeypatch.setattr(
        computer_literacy_relevance_search_service.record_topic_similarity_embedder.embedder,
        "embed_documents",
        raise_error(DocumentEmbeddingFailedException, "directly raised exception"),
    )

    with pytest.raises(
        DocumentEmbeddingFailedException,
        match=".*Couldn't use embeddings to filter the resulting record set.*",
    ):
        _ = await computer_literacy_relevance_search_service.relevance_search(
            mock_computer_literacy_relevance_search_input,
            from_history_cache=True,
            force_refresh=True,
            raise_on_error=True,
        )


@pytest.mark.asyncio
async def test_computer_literacy_relevance_search_reraises_pydantic_ai_import_errors_when_enabled(
    computer_literacy_relevance_search_service,
    mock_computer_literacy_relevance_search_input,
    monkeypatch,
):
    """Verifies that relevance searches reraise errors for missing PydanticAI dependencies."""
    # retrieve cache from only the previous step
    monkeypatch.setattr(
        computer_literacy_relevance_search_service.record_topic_similarity_embedder.embedder,
        "embed",
        raise_error(PydanticAIImportError, "directly raised exception"),
    )

    with pytest.raises(
        PydanticAIImportError,
    ):
        _ = await computer_literacy_relevance_search_service.relevance_search(
            mock_computer_literacy_relevance_search_input,
            from_history_cache=True,
            force_refresh=True,
        )


@pytest.mark.skip(reason="Live relevance search test: Trigger directly if needed.")
async def test_computer_literacy_embedding_relevance_search_with_cached_data(
    mock_computer_literacy_api_searches,
    mock_computer_literacy_relevance_search_app_context,
    mock_computer_literacy_relevance_search_input,
    mock_computer_literacy_relevance_search_output_path,
):
    """Verifies that the relevance search correctly embeds and reranks cached records with embedding model defaults."""
    from scholar_flux.utils import JsonFileUtils

    relevance_search_service = mock_computer_literacy_relevance_search_app_context.relevance_search_service
    relevance_search_service.record_topic_similarity_embedder.get_or_create_embedder()

    search_results = await relevance_search_service.relevance_search(
        mock_computer_literacy_relevance_search_input, cache_only=True
    )

    assert search_results and isinstance(search_results, RelevanceSearchOutput)
    assert search_results.question == mock_computer_literacy_relevance_search_input.question
    assert search_results.queries == mock_computer_literacy_relevance_search_input.queries
    assert len(search_results.indexed_records) > 0
    assert search_results.embedding_model_name and search_results.record_topic_similarity_output
    assert search_results

    with JSONDataModel.serialization_context(core_fields_only=True):
        JsonFileUtils.save_as(
            search_results.model_dump(mode="json"), mock_computer_literacy_relevance_search_output_path
        )

    json_data = JsonFileUtils.load_data(mock_computer_literacy_relevance_search_output_path)
    assert RelevanceSearchOutput.model_validate(json_data)

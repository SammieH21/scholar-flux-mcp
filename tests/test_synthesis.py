"""Tests for the synthesis service and integration with cache and search services."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scholar_flux_mcp.models.schemas import (
    JSONDataModel,
    RelevanceSearchInput,
    RelevanceSearchOutput,
    SearchInput,
    SearchOutput,
    SynthesisOutput,
)
from scholar_flux_mcp.server.io import SynthesisFormatter
from scholar_flux_mcp.services import HistoryService, SynthesisService
from tests.testing_utilities import raise_error

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.mark.asyncio
async def test_scholar_flux_mcp_multi_provider_multisearch_integration(
    mock_ai_synthesis_app_context, mock_ai_search_input
):
    """Verifies that coordinated API record searches operate correctly when called via scholar-flux-mcp search service.

    This test runs the full search/processing pipeline with the base scholar-flux pipeline and retrieves pre-cached raw
    responses from CORE, Crossref, and PLOS to verify the structure of the output after orchestration via the
    SearchService.

    """

    search_service = mock_ai_synthesis_app_context.search_service
    assert search_service

    search_results = await search_service.search(mock_ai_search_input, cache_only=True)
    assert search_results and isinstance(search_results, SearchOutput)
    assert search_results.cache_hit  # all records should originate from cache
    assert search_results.pagination.pages_retrieved == len(mock_ai_search_input.providers) * mock_ai_search_input.pages


@pytest.fixture
def mock_ai_synthesis_service(
    mock_ai_synthesis_app_context, mock_embedding_model, mock_pydantic_ai_agent
) -> Iterator[SynthesisService]:
    """Mocks the synthesis service agent and embedder to return dummy outputs without using external APIs or models."""
    synthesis_service = mock_ai_synthesis_app_context.synthesis_service
    assert (
        synthesis_service
        and synthesis_service.relevance_search_service is mock_ai_synthesis_app_context.relevance_search_service
    )
    synthesis_service.record_topic_similarity_embedder.get_or_create_embedder()
    synthesis_service.synthesis_agent.get_or_create_agent()

    with (
        synthesis_service.record_topic_similarity_embedder.embedder.override(model=mock_embedding_model),
        synthesis_service.synthesis_agent.agent.override(model=mock_pydantic_ai_agent),
    ):
        yield synthesis_service


@pytest.fixture
def mock_ai_synthesis_service_with_history(mock_ai_synthesis_service, monkeypatch) -> Iterator[SynthesisService]:
    """Mocks the synthesis service agent and embedder to return dummy outputs without using external APIs or models."""
    history_service = HistoryService(enable=True)
    monkeypatch.setattr(mock_ai_synthesis_service, "_history_service", history_service)
    monkeypatch.setattr(mock_ai_synthesis_service.relevance_search_service, "_history_service", history_service)
    monkeypatch.setattr(
        mock_ai_synthesis_service.relevance_search_service.search_service, "_history_service", history_service
    )
    yield mock_ai_synthesis_service


@pytest.mark.asyncio
async def test_scholar_flux_mcp_multi_provider_relevance_search(mock_ai_synthesis_input, mock_ai_synthesis_service):
    """Verifies that the RelevanceSearchService from the SynthesisService correctly embeds and reranks records."""
    relevance_search_input = RelevanceSearchInput.from_synthesis_params(mock_ai_synthesis_input)
    search_results = await mock_ai_synthesis_service.relevance_search_service.relevance_search(relevance_search_input)

    assert search_results and isinstance(search_results, RelevanceSearchOutput)
    assert search_results.question == mock_ai_synthesis_input.question
    assert search_results.queries == mock_ai_synthesis_input.queries
    assert len(search_results.indexed_records) > 0
    assert search_results.record_topic_similarity_output


@pytest.mark.asyncio
async def test_scholar_flux_mcp_multi_provider_result_synthesis(mock_ai_synthesis_input, mock_ai_synthesis_service):
    """Verifies that the SynthesisService correctly calls pydantic AI to both embed and calculate query similarity.

    Note: This test overrides the base models for the `RecordTopicSimilarityEmbedder` and the `SynthesisAgent` to verify
    the behavior of the synthesis across the entire pipeline. The SynthesisAgent will then return the default
    `SynthesisAgentOutput` which should be further processed without error.

    """
    synthesis_results = await mock_ai_synthesis_service.synthesize(mock_ai_synthesis_input)

    assert synthesis_results and isinstance(synthesis_results, SynthesisOutput)
    assert synthesis_results.synthesis == "Synthesis not available"
    assert synthesis_results.question == mock_ai_synthesis_input.question
    assert synthesis_results.queries == mock_ai_synthesis_input.queries
    assert synthesis_results.records_analyzed > 50  # should be at least 100+ in the final output after filtering
    assert isinstance(synthesis_results.key_findings, list)
    assert isinstance(synthesis_results.grounded_evidence, list)
    assert synthesis_results.confidence_score == 0.5
    assert synthesis_results.successful

    import json

    with SynthesisOutput.serialization_context(core_fields_only=True):
        dumped = json.loads(synthesis_results.model_dump_json())
        SynthesisOutput.model_validate(dumped)


@pytest.mark.asyncio
async def test_scholar_flux_mcp_multi_provider_result_synthesis_with_history(
    mock_ai_synthesis_input, mock_ai_synthesis_service_with_history, monkeypatch
):
    """Verifies that previously stored Synthesis inputs are retrievable via the history cache when enabled."""
    synthesis_service = mock_ai_synthesis_service_with_history
    synthesis_results = await synthesis_service.synthesize(mock_ai_synthesis_input, store_history_cache=True)
    assert synthesis_results.successful

    cached_results = await synthesis_service.retrieve_history_cache(mock_ai_synthesis_input)

    assert synthesis_results == cached_results

    # Shouldn't execute if the SynthesisOutput is retrieved from the history cache
    monkeypatch.setattr(
        synthesis_service.relevance_search_service,
        "relevance_search",
        raise_error(RuntimeError, "Couldn't retrieve the synthesis output from cache"),
    )

    cached_results2 = await synthesis_service.synthesize(mock_ai_synthesis_input, from_history_cache=True)

    assert cached_results == cached_results2


@pytest.mark.asyncio
async def test_scholar_flux_mcp_multi_provider_result_synthesis_propagates_relevance_search_retrieval(
    mock_ai_synthesis_input, mock_ai_synthesis_service_with_history, monkeypatch
):
    """Verifies that the research synthesis also stores intermediate search outputs for later retrieval."""
    synthesis_service = mock_ai_synthesis_service_with_history
    relevance_search_service = mock_ai_synthesis_service_with_history.relevance_search_service
    search_service = relevance_search_service.search_service
    synthesis_results = await synthesis_service.synthesize(mock_ai_synthesis_input, store_history_cache=True)
    assert synthesis_results.successful

    relevance_search_input = RelevanceSearchInput.from_synthesis_params(mock_ai_synthesis_input)

    relevance_search_results = await relevance_search_service.relevance_search(
        relevance_search_input, from_history_cache=True
    )

    # Shouldn't execute if the RelevanceSearchOutput is retrieved from the history cache
    with monkeypatch.context() as m:
        m.setattr(
            relevance_search_service.search_service,
            "search",
            raise_error(RuntimeError, "Couldn't retrieve the relevance search output from cache"),
        )

        relevance_search_results = await relevance_search_service.relevance_search(
            relevance_search_input, from_history_cache=True
        )
    assert synthesis_results.relevance_search_output == relevance_search_results

    search_input = SearchInput.from_search_params(relevance_search_input)
    with monkeypatch.context() as m:
        m.setattr(
            search_service,
            "_create_coordinators",
            raise_error(RuntimeError, "Couldn't retrieve the search output from cache"),
        )
        search_results = await search_service.search(search_input, from_history_cache=True)

    assert relevance_search_results.search_output == search_results


@pytest.mark.skip(reason="Live test for research synthesis: Trigger directly if needed.")
async def test_synthesis_with_cached_data(
    mock_ai_synthesis_json_input, mock_ai_synthesis_app_context, mock_ai_synthesis_output_path
):
    """Verifies that the SynthesisService.synthesis method correctly returns synthesis output with model defaults.

    Note: This test uses the `mock_ai_synthesis_app_context` fixture to synthesize cached records after cache retrieval
    from the `/tests/mocks/ai-learning_structure-testing` filesystem cache.

    This test uses the system-level configured embedding model/llm defaults for testing live synthesis and validates
    that core `SynthesisOutput` fields are populated (e.g., `synthesis`, `suggested_queries`, `evidence`).

    If successfully created, the synthesis writes to the file path configured via the `mock_ai_synthesis_output_path`
    fixture.

    """
    from scholar_flux.utils import JsonFileUtils

    synthesis_service = mock_ai_synthesis_app_context.synthesis_service
    assert synthesis_service

    synthesis_input = mock_ai_synthesis_json_input.as_synthesis_input()
    synthesis_output = await synthesis_service.synthesize(synthesis_input)
    assert isinstance(synthesis_output, SynthesisOutput)

    assert synthesis_output.synthesis
    assert synthesis_output.grounded_evidence
    assert synthesis_output.suggested_queries

    synthesis_markdown = SynthesisFormatter.format_synthesis_markdown(synthesis_output)
    assert [evidence.finding in synthesis_markdown for evidence in synthesis_output.grounded_evidence]
    assert [query in synthesis_markdown for query in synthesis_output.suggested_queries]
    assert synthesis_output.synthesis in synthesis_markdown

    with JSONDataModel.serialization_context(core_fields_only=True):
        JsonFileUtils.save_as(synthesis_output.model_dump(mode="json"), mock_ai_synthesis_output_path)

    json_data = JsonFileUtils.load_data(mock_ai_synthesis_output_path)
    assert SynthesisOutput.model_validate(json_data)

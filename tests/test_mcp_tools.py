"""Tests for MCP Tools.

This module tests the MCP tool implementations including:
- Search tools used to perform academic record searches
- Synthesis tools for grounded research exploration and synthesis
- Provider tools to explore the range of supported providers.

Tests use mock services to ensure that the integration of higher level MCP components works without issue.

"""

import json
import re

import pytest
from pydantic import ValidationError

from scholar_flux_mcp.models import (
    APIProviders,
    ProviderMetadataDescriptions,
    ResearchCategory,
    SearchInput,
    SearchRecord,
    SearchResponseSummary,
    SynthesisInput,
)
from scholar_flux_mcp.server.app_context import AppContext
from scholar_flux_mcp.server.io.base import BaseFormatter
from scholar_flux_mcp.server.io.search import SearchFormatter, SearchToolInput
from scholar_flux_mcp.server.io.synthesis import SynthesisFormatter, SynthesisToolInput


@pytest.mark.asyncio
async def test_search_records_tool_input_validation():
    """Test scholar_flux_search_records input validation."""

    # Valid input
    params = SearchToolInput(
        queries="depression treatment",  # type: ignore
        providers=["pubmed", "plos"],
        max_records=25,
        pages=3,
    )

    assert params.queries == ["depression treatment"]
    assert len(params.providers) == 2
    assert params.max_records == 25


@pytest.mark.asyncio
async def test_app_context_with_creation(mock_computer_literacy_relevance_search_app_context):
    """Verifies that the `AppContext.create()` method correctly initializes with partial services."""
    search_service = mock_computer_literacy_relevance_search_app_context.search_service
    app_context = AppContext.create(search_service)
    assert not app_context.history_service.enabled
    assert app_context.search_service is search_service
    assert app_context.relevance_search_service
    assert app_context.synthesis_service
    assert app_context.provider_service
    assert app_context.cache_service is search_service.cache_service


@pytest.mark.asyncio
async def test_search_records_tool_defaults():
    """Test scholar_flux_search_records has correct defaults."""

    params = SearchToolInput(queries=["test query"])

    assert set(params.providers) == {"pubmed", "plos", "openalex"}
    assert params.max_records == 25
    assert params.pages == 3
    assert params.response_format == "markdown"


@pytest.mark.asyncio
async def test_search_records_tool_with_mock(
    mock_output_search_service, mock_search_record_list, mock_search_response_summary_list
):
    """Test search tool execution with mock service."""

    search_input = SearchInput.create(
        queries="depression treatment",
        providers=[APIProviders.PUBMED, APIProviders.PLOS],
        max_records=10,
        pages=2,
    )

    result = await mock_output_search_service.search(search_input)

    assert result.queries == ["depression treatment"]

    assert len(result.records) == len(mock_search_record_list)
    assert all(isinstance(record, SearchRecord) for record in result.records)

    assert len(result.response_summaries) == len(mock_search_response_summary_list)
    assert all(isinstance(response_summary, SearchResponseSummary) for response_summary in result.response_summaries)

    formatted_summary = SearchFormatter.format_record_search_markdown(result)
    assert isinstance(formatted_summary, str) and formatted_summary


@pytest.mark.asyncio
async def test_search_records_json_output(mock_output_search_service):
    """Test search tool JSON output format."""

    search_input = SearchInput.create(
        queries="anxiety biomarkers",
        providers=[APIProviders.PUBMED],
    )

    result = await mock_output_search_service.search(search_input)

    # Convert to JSON as tool would
    output = result.model_dump()

    assert "queries" in output["search_input"]
    assert "providers" in output["search_input"]
    assert "records" in output
    assert "response_summaries" in output
    assert "pagination" in output


@pytest.mark.asyncio
async def test_synthesis_tool_input_validation():
    """Test scholar_flux_synthesize_research_summary input validation."""

    params = SynthesisToolInput(
        question="What is the efficacy of CBT for depression?",
        categories=["depression", "intervention"],
        max_records=30,
    )

    assert params.question == "What is the efficacy of CBT for depression?"
    assert "depression" in params.categories
    assert params.max_records == 30


@pytest.mark.asyncio
async def test_synthesis_tool_defaults():
    """Verifies that scholar_flux_synthesize_research_summary defaults is constrained to valid/reasonable values."""

    params = SynthesisToolInput(question="Test research question for synthesis?", categories=["general_wellbeing"])

    assert params.categories == ["general_wellbeing"]
    assert isinstance(params.max_records, int) and params.max_records >= 50
    assert isinstance(params.year_from, int)
    assert params.providers and all(APIProviders.get(provider) for provider in params.providers)


@pytest.mark.asyncio
async def test_synthesis_tool_with_missing_query():
    """Verifies that scholar_flux_synthesize_research_summary defaults is constrained to valid/reasonable values."""

    err = "A valid query has not been specified and could not be inferred from the provided research categories"

    with pytest.raises(ValidationError) as excinfo:
        _ = SynthesisToolInput(question="Test research question for synthesis?")

    assert err in str(excinfo.value)


@pytest.mark.asyncio
async def test_synthesis_tool_with_mock(mock_output_synthesis_service):
    """Test synthesis tool execution with mock service."""

    synthesis_input = SynthesisInput(
        question="What is the efficacy of CBT for depression?",
        categories=[ResearchCategory.DEPRESSION],
        providers=[APIProviders.PUBMED],
    )

    result = await mock_output_synthesis_service.synthesize(synthesis_input)

    assert result.question == "What is the efficacy of CBT for depression?"
    assert len(result.key_findings) > 0
    assert result.confidence_score > 0
    assert len(result.grounded_evidence) > 0


@pytest.mark.asyncio
async def test_synthesis_markdown_output(mock_output_synthesis_service):
    """Verifies that synthesis markdown output formatting contains the expected categories."""

    synthesis_input = SynthesisInput(
        question="Test question?",
        categories=[ResearchCategory.DEPRESSION],
    )

    result = await mock_output_synthesis_service.synthesize(synthesis_input)
    markdown = SynthesisFormatter.format_synthesis_markdown(result)

    assert "# Research Synthesis" in markdown
    assert "## Research Question" in markdown
    assert result.question in markdown
    assert "## Key Findings" in markdown

    has_embedder_warning = "Embedding-based record relevance reranking was unavailable" in markdown
    assert bool(not result.record_topic_similarity_output) is has_embedder_warning


@pytest.mark.asyncio
async def test_list_providers_tool(mock_output_provider_service):
    """Verifies that the scholar_flux_list_providers output matches the expected supported provider list."""

    providers = await mock_output_provider_service.get_providers()
    provider_names = [APIProviders.normalize_name(provider_info.name) for provider_info in providers]

    assert len(providers) > 0
    unknown_providers = [p for p in APIProviders if APIProviders.normalize_name(p) not in provider_names]
    assert not unknown_providers


def test_provider_info_completeness():
    """Verifies that that the provider metadata descriptions have all required fields."""
    for provider in APIProviders:
        if provider in ProviderMetadataDescriptions:
            info = ProviderMetadataDescriptions(provider).value
            assert info.name
            assert info.display_name
            assert info.description
            assert info.coverage


def test_search_input_query_min_length():
    """Test search input enforces minimum query length."""

    with pytest.raises(ValidationError):
        SearchToolInput(queries=["a"])  # Too short


def test_search_input_max_records_bounds():
    """Test search input enforces max_records bounds."""

    with pytest.raises(ValidationError):
        SearchToolInput(queries=["test"], max_records=500)  # Too high

    with pytest.raises(ValidationError):
        SearchToolInput(queries=["test"], max_records=0)  # Too low


def test_synthesis_input_question_min_length():
    """Test synthesis input enforces minimum question length."""

    with pytest.raises(ValidationError):
        SynthesisToolInput(question="Short?")  # Too short


def test_base_formatter_invalid_record_serialization():
    """Test BaseFormatter record serialization fails with invalid types."""
    with pytest.raises(TypeError, match=re.escape("Expected a pydantic model or class that supports `model_dump()`")):
        _ = BaseFormatter.format_json_string("Non serializable json")  # type: ignore


def test_search_record_json_serialization():
    """Test search records are JSON serializable."""
    record = SearchRecord(
        page=1,
        provider_name="pubmed",
        query="test",
        doi="10.1000/test",
        title="Test Record",
        abstract="Test abstract",
        year=2024,
    )

    # Should be JSON serializable: BaseFormatter dumps and serializes as a JSON string
    json_str = BaseFormatter.format_json_string(record)
    assert isinstance(json_str, str) and json_str

    parsed = json.loads(json_str)
    assert SearchRecord.model_validate(parsed) == record

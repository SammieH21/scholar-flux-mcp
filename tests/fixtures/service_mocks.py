"""ScholarFluxMCP fixtures used to test functionality of each service/MCP integration.

This module provides reusable fixtures for testing the ScholarFlux MCP server, including temporary databases, mock
services, and test data generators.

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from scholar_flux_mcp.agents import PydanticAIEmbeddingModelFactory, PydanticAIModelFactory
from scholar_flux_mcp.models import (
    AgentEvidenceItem,
    APIProviders,
    EvidenceGroundingOutput,
    EvidenceGroundingStats,
    GroundedEvidenceItem,
    IndexedSearchRecord,
    PaginationInfo,
    RejectedEvidenceItem,
    RelevanceSearchInput,
    RelevanceSearchOutput,
    ResearchCategory,
    SearchInput,
    SearchOutput,
    SearchRecord,
    SearchResponseSummary,
    SynthesisAgentOutput,
    SynthesisInput,
    SynthesisOutput,
)
from scholar_flux_mcp.services.cache_service import CacheService
from scholar_flux_mcp.services.provider_service import ProviderService
from scholar_flux_mcp.services.search_service import SearchService
from scholar_flux_mcp.services.synthesis_service import SynthesisService

if TYPE_CHECKING:
    from collections.abc import Generator

    from pydantic_ai.embeddings.test import TestEmbeddingModel
    from pydantic_ai.models.test import TestModel


# =============================================================================
# CACHE SERVICE FIXTURES
# =============================================================================


@pytest.fixture
def mock_output_cache_service() -> CacheService:
    """Provide a mock CacheService for testing without real caching."""
    service = MagicMock(spec=CacheService)
    service._initialized = True
    service.data_cache_manager = MagicMock()
    service.session_manager = MagicMock()
    service.create_session = MagicMock(return_value=MagicMock())
    service.get_stats = AsyncMock(
        return_value={
            "session_cache_backend": "sqlite",
            "response_cache_storage": "inmemory",
            "namespace": "test",
            "initialized": True,
        }
    )
    return service


# =============================================================================
# SEARCH SERVICE FIXTURES
# =============================================================================


@pytest.fixture
def mock_output_provider_service() -> ProviderService:
    """Mock provider service for simply retrieving a list of all available records."""
    return ProviderService()


@pytest.fixture
def mock_search_record_list() -> list[SearchRecord]:
    """Defines a list of SearchRecords used to verify search, synthesis, and related functionality."""
    # Create mock search results
    return [
        SearchRecord(
            page=1,
            provider_name="pubmed",
            query="test",
            doi="10.1000/test.123",
            title="Cognitive Behavioral Therapy for Depression: A Meta-Analysis",
            abstract="This meta-analysis examines the efficacy of CBT...",
            authors=["Smith, J.", "Jones, K."],
            journal="Journal of Clinical Psychology",
            year=2023,
            keywords=["CBT", "depression", "meta-analysis"],
        ),
        SearchRecord(
            page=1,
            provider_name="plos",
            query="test",
            doi="10.1371/test.456",
            title="Neural Correlates of Anxiety Disorders",
            abstract="We investigated the neural basis of anxiety...",
            authors=["Brown, A.", "Wilson, B."],
            journal="PLOS ONE",
            year=2024,
            keywords=["anxiety", "neuroimaging", "fMRI"],
        ),
        SearchRecord(
            page=1,
            provider_name="crossref",
            query="test",
            doi="10.1328/test.798",
            title="Computational learning success precursors",
            abstract="Factories influencing successful outcomes...",
            authors=["Wilma, B.", "Fred, B."],
            journal="Journal of Computational Science and Education",
            year=2024,
            keywords=["computational methods", "optimization theory", "statistical computing"],
        ),
    ]


@pytest.fixture
def mock_search_response_summary_list() -> list[SearchResponseSummary]:
    """Defines a list of SearchResponseSummary instances used to record response metadata across searches."""
    api_providers = ["pubmed", "plos", "crossref"]
    return [
        SearchResponseSummary(
            provider_name=provider,
            query="test",
            page=1,
            success=True,
            cached=False,
            record_count=1,
        )
        for provider in api_providers
    ]


@pytest.fixture
def mock_indexed_search_record_list(mock_search_record_list: list[SearchRecord]) -> list[IndexedSearchRecord]:
    """Defines a list of indexed search records to verify functionality requiring indices assigned to each record."""
    scores: list[float] = [0.95, 0.75, 0.5]
    return [
        IndexedSearchRecord.from_search_record(record, index=i, topic_similarity_score=score)
        for i, (score, record) in enumerate(zip(scores, mock_search_record_list, strict=True))
    ]


@pytest.fixture
def mock_output_search_service(
    mock_search_record_list: list[SearchRecord], mock_search_response_summary_list: list[SearchResponseSummary]
) -> SearchService:
    """Provide a mock SearchService for testing without real API calls."""
    service = MagicMock(spec=SearchService)
    search_input = SearchInput.create(queries=["depression treatment"], providers=["arXiv", "CORE"])
    mock_output = SearchOutput(
        search_input=search_input,
        records=mock_search_record_list,
        response_summaries=mock_search_response_summary_list,
        pagination=PaginationInfo(
            total_records=2,
            providers_queried=2,
            providers_successful=2,
            pages_successful=9,
            pages_retrieved=2,
        ),
    )

    service.search = AsyncMock(return_value=mock_output)
    service.get_providers = AsyncMock(
        return_value=[
            {"name": "pubmed", "display_name": "PubMed", "requires_api_key": False},
            {"name": "plos", "display_name": "PLOS ONE", "requires_api_key": False},
        ]
    )

    return service


# =============================================================================
# SYNTHESIS SERVICE FIXTURES
# =============================================================================


@pytest.fixture
def mock_synthesis_search_records() -> list[SearchRecord]:
    """Defines a listed of mock SearchRecord instances for synthesis testing."""
    return [
        SearchRecord(
            page=1,
            query="CBT",
            provider_name="test_provider",
            title="CBT Meta-Analysis",
            doi="10.1000/test.123",
            year=2023,
        ),
        SearchRecord(
            page=1,
            query="Anxiety",
            provider_name="test_provider_two",
            title="Anxiety Treatment Literature Review",
            doi="10.2000/test.422",
            year=2024,
        ),
        SearchRecord(
            page=1,
            query="Anxiety",
            provider_name="test_provider_two",
            title="Comparison of CBT and Psycotropic Medication Literature Review",
            doi="10.2000/test.422",
            year=2024,
        ),
    ]


@pytest.fixture
def mock_synthesis_response_summary_list() -> list[SearchResponseSummary]:
    """Defines a list of SearchResponseSummary instances used to record response metadata across syntheses."""
    return [
        SearchResponseSummary(
            provider_name="test_provider",
            query="CBT",
            page=1,
            success=True,
            cached=False,
            record_count=1,
        ),
        SearchResponseSummary(
            provider_name="test_provider_two",
            query="Anxiety",
            page=1,
            success=True,
            cached=False,
            record_count=2,
        ),
    ]


@pytest.fixture
def mock_synthesis_indexed_search_records(mock_synthesis_search_records) -> list[IndexedSearchRecord]:
    """Defines a list of mock IndexedSearchRecord instances for synthesis testing."""
    return [
        IndexedSearchRecord.from_search_record(record, index=i, topic_similarity_score=1.0 - 0.03 * (i + 1))
        for i, record in enumerate(mock_synthesis_search_records)
    ]


@pytest.fixture
def mock_synthesis_evidence_item_list() -> list[AgentEvidenceItem]:
    """Defines a list of AgentEvidenceItem instances for synthesis testing."""
    return [
        AgentEvidenceItem(record_index=0, finding="Large effect size for CBT in depression", relevance_score=0.9),
        AgentEvidenceItem(
            record_index=1, finding="Studies support CBT and medication as treatments for anxiety.", relevance_score=0.8
        ),
        AgentEvidenceItem(
            record_index=2, finding="A combination of CBT and medication is often effective.", relevance_score=0.7
        ),
    ]


@pytest.fixture
def mock_grounded_synthesis_evidence_output(
    mock_synthesis_evidence_item_list: list[AgentEvidenceItem], mock_synthesis_search_records: list[IndexedSearchRecord]
) -> EvidenceGroundingOutput:
    """Defines a mock EvidenceGroundingOutput for synthesis testing."""
    grounded_evidence_items = [
        GroundedEvidenceItem(
            **evidence_item.model_dump(), record=mock_synthesis_search_records[evidence_item.record_index]
        )
        for evidence_item in mock_synthesis_evidence_item_list
    ]
    return EvidenceGroundingOutput(
        grounded_evidence_items=grounded_evidence_items,
        grounding_stats=EvidenceGroundingStats(references_grounded=len(grounded_evidence_items)),
    )


@pytest.fixture
def mock_grounded_synthesis_with_rejected_evidence_output(
    mock_synthesis_evidence_item_list: list[AgentEvidenceItem], mock_synthesis_search_records: list[SearchRecord]
) -> EvidenceGroundingOutput:
    """Defines a mock EvidenceGroundingOutput containing two RejectedEvidenceItems for synthesis testing."""
    grounded_evidence_items = [
        GroundedEvidenceItem(
            **evidence_item.model_dump(),
            record=mock_synthesis_search_records[evidence_item.record_index],  # coerced into an indexed record list
        )
        for evidence_item in mock_synthesis_evidence_item_list[:1]
    ]
    rejected_evidence_items = [
        RejectedEvidenceItem(
            **evidence_item.model_dump(),
            record=None,
            error="Index out of bounds",
        )
        for evidence_item in mock_synthesis_evidence_item_list[1:]
    ]
    return EvidenceGroundingOutput(
        grounded_evidence_items=grounded_evidence_items,
        rejected_evidence_items=rejected_evidence_items,
        grounding_stats=EvidenceGroundingStats(
            references_grounded=len(grounded_evidence_items),
            references_rejected=len(rejected_evidence_items),
        ),
    )


@pytest.fixture
def mock_synthesis_agent_output(mock_synthesis_evidence_item_list: list[AgentEvidenceItem]) -> SynthesisAgentOutput:
    """Defines a mock SynthesisAgentOutput for synthesis testing."""
    return SynthesisAgentOutput(
        synthesis="Based on the available evidence, CBT demonstrates significant efficacy...",
        key_findings=[
            "CBT shows effect size d=0.8 compared to control",
            "Response rates are approximately 60%",
            "Effects are maintained at 6-month follow-up",
        ],
        evidence_summaries=mock_synthesis_evidence_item_list,
        confidence_score=0.85,
        limitations=["Limited to English-language papers"],
        suggested_queries=["CBT vs medication for depression"],
    )


@pytest.fixture
def mock_synthesis_search_output(
    mock_synthesis_search_records: list[SearchRecord], mock_synthesis_response_summary_list: list[SearchResponseSummary]
) -> SearchOutput:
    """Defines a mock SearchOutput for synthesis testing."""
    synthesis_input = SynthesisInput(
        question="What is the efficacy of CBT for depression?",
        providers=[APIProviders.ARXIV, APIProviders.CORE],
        categories=[ResearchCategory.DEPRESSION, ResearchCategory.INTERVENTION],
    )
    search_input = SearchInput.create(providers=synthesis_input.providers, queries=synthesis_input.queries)
    return SearchOutput(
        search_input=search_input,
        records=mock_synthesis_search_records,
        response_summaries=mock_synthesis_response_summary_list,
        pagination=PaginationInfo(
            total_records=3, providers_queried=1, providers_successful=3, pages_successful=3, pages_retrieved=3
        ),
    )


@pytest.fixture
def mock_synthesis_relevance_search_output(
    mock_synthesis_search_output: SearchOutput, mock_synthesis_indexed_search_records: list[IndexedSearchRecord]
) -> RelevanceSearchOutput:
    """Defines a mock RelevanceSearchOutput for synthesis testing."""
    synthesis_input = SynthesisInput(
        question="What is the efficacy of CBT for depression?",
        providers=[APIProviders.ARXIV, APIProviders.CORE],
        categories=[ResearchCategory.DEPRESSION, ResearchCategory.INTERVENTION],
    )
    relevance_search_input = RelevanceSearchInput(
        question=synthesis_input.question, providers=synthesis_input.providers, categories=synthesis_input.categories
    )
    return RelevanceSearchOutput(
        search_output=mock_synthesis_search_output,
        relevance_search_input=relevance_search_input,
        indexed_records=mock_synthesis_indexed_search_records,
    )


@pytest.fixture
def mock_synthesis_output_with_rejected_evidence_output(
    mock_synthesis_agent_output: SynthesisAgentOutput,
    mock_synthesis_relevance_search_output: RelevanceSearchOutput,
    mock_grounded_synthesis_with_rejected_evidence_output: EvidenceGroundingOutput,
) -> SynthesisOutput:
    """Defines a synthesis output with rejected evidence items for testing error handling."""
    synthesis_input = SynthesisInput(
        question="What is the efficacy of CBT for depression?",
        providers=[APIProviders.ARXIV, APIProviders.CORE],
        categories=[ResearchCategory.DEPRESSION, ResearchCategory.INTERVENTION],
    )
    return SynthesisOutput(
        synthesis_input=synthesis_input,
        agent_output=mock_synthesis_agent_output,
        grounding_output=mock_grounded_synthesis_with_rejected_evidence_output,
        relevance_search_output=mock_synthesis_relevance_search_output,
        model_name="llama3:latest",
    )


@pytest.fixture
def mock_synthesis_output(
    mock_synthesis_agent_output: SynthesisAgentOutput,
    mock_synthesis_relevance_search_output: RelevanceSearchOutput,
    mock_grounded_synthesis_evidence_output: EvidenceGroundingOutput,
) -> SynthesisOutput:
    """Defines the synthesis output used to verify synthesis-related functionality."""
    synthesis_input = SynthesisInput(
        question="What is the efficacy of CBT for depression?",
        providers=[APIProviders.ARXIV, APIProviders.CORE],
        categories=[ResearchCategory.DEPRESSION, ResearchCategory.INTERVENTION],
    )
    return SynthesisOutput(
        synthesis_input=synthesis_input,
        agent_output=mock_synthesis_agent_output,
        grounding_output=mock_grounded_synthesis_evidence_output,
        relevance_search_output=mock_synthesis_relevance_search_output,
        model_name="llama3:latest",
    )


@pytest.fixture
def mock_output_synthesis_service(mock_synthesis_output: SynthesisOutput) -> SynthesisService:
    """Provide a mock SynthesisService for testing without real AI calls."""
    service = MagicMock(spec=SynthesisService)
    service.synthesize = AsyncMock(return_value=mock_synthesis_output)
    return service


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================


@pytest.fixture
def sample_search_input() -> SearchInput:
    """Generate sample search input parameters."""
    return SearchInput.create(
        queries="depression treatment CBT",
        providers=[APIProviders.PUBMED, APIProviders.PLOS],
        max_records=10,
        pages=2,
        year_from=2020,
    )


@pytest.fixture
def sample_synthesis_input() -> SynthesisInput:
    """Generate sample synthesis input parameters."""
    return SynthesisInput(
        question="What are the most effective treatments for depression?",
        categories=[ResearchCategory.DEPRESSION, ResearchCategory.INTERVENTION],
        max_records=30,
        year_from=2021,
        providers=[APIProviders.PUBMED, APIProviders.PLOS],
    )


# =============================================================================
# PYDANTIC AI MOCKS
# =============================================================================


@pytest.fixture
def mock_embedding_model(monkeypatch) -> Generator[TestEmbeddingModel, None, None]:
    """Creates a mock TestEmbeddingModel for use with the SynthesisService to test the synthesis pipeline.

    The settings for the PydanticAIEmbeddingModelFactory are defined to use the `embeddinggemma`  model
    as a fallback in case `with embedding.override(model=mock_embedding_model): ...` isn't set. `embeddinggemma`
    should still be displayed as the model being used by the synthesis agent.

    """
    from pydantic_ai.embeddings.test import TestEmbeddingModel

    with monkeypatch.context() as m:
        # Constrain the output to OLLAMA (this is later patched with embedding.override)
        m.setenv("SCHOLAR_FLUX_MCP_DEFAULT_EMBEDDING_PROVIDER", "OLLAMA")  # in case the embedding model isn't mocked
        m.setenv("SCHOLAR_FLUX_MCP_DEFAULT_OLLAMA_EMBEDDING_MODEL", "embeddinggemma:latest")
        m.setattr(PydanticAIEmbeddingModelFactory, "_check_ollama_available", lambda: True)
        yield TestEmbeddingModel()


@pytest.fixture
def mock_pydantic_ai_agent(monkeypatch) -> Generator[TestModel, None, None]:
    """Creates a mock pydantic TestModel for use with the SynthesisService to test the synthesis pipeline.

    The `SynthesisAgentOutput` should be assigned its default settings when successfully assigned via
    `agent.override(model=mock_pydantic_ai_agent)`

    """
    from pydantic_ai.models.test import TestModel

    with monkeypatch.context() as m:
        # Constrain the output to OLLAMA (this is later patched with embedding.override)
        m.setenv("SCHOLAR_FLUX_MCP_DEFAULT_PROVIDER", "OLLAMA")
        m.setenv("SCHOLAR_FLUX_MCP_DEFAULT_OLLMA_MODEL", "rnj-1:latest")  # also in case the agent isn't mocked
        m.setenv("SCHOLAR_FLUX_MCP_GROUNDING_TEXT_TOKEN_LIMIT", "30000")
        m.setattr(PydanticAIModelFactory, "_check_ollama_available", lambda: True)
        yield TestModel()


# =============================================================================
# MCP CONTEXT MOCK
# =============================================================================

__all__ = [
    "mock_search_record_list",
    "mock_indexed_search_record_list",
    "mock_search_response_summary_list",
    "mock_synthesis_response_summary_list",
    "mock_output_cache_service",
    "mock_output_search_service",
    "mock_output_synthesis_service",
    "mock_synthesis_search_records",
    "mock_synthesis_indexed_search_records",
    "mock_synthesis_evidence_item_list",
    "mock_grounded_synthesis_evidence_output",
    "mock_grounded_synthesis_with_rejected_evidence_output",
    "mock_synthesis_agent_output",
    "mock_synthesis_search_output",
    "mock_synthesis_relevance_search_output",
    "mock_synthesis_output_with_rejected_evidence_output",
    "mock_embedding_model",
    "mock_pydantic_ai_agent",
    "sample_search_input",
    "sample_synthesis_input",
]

"""Tests for the synthesis service and integration with cache and search services."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests_mock
from scholar_flux import CachedSessionManager, config_settings

import tests.mocks
from scholar_flux_mcp.models import (
    APIProviders,
    RelevanceSearchInput,
    ResearchCategory,
    SearchInput,
    SynthesisInput,
    SynthesisOutput,
)
from scholar_flux_mcp.server.app_context import AppContext
from scholar_flux_mcp.server.io.relevance_search import RelevanceSearchToolInput
from scholar_flux_mcp.server.io.search import SearchToolInput
from scholar_flux_mcp.server.io.synthesis import SynthesisToolInput
from scholar_flux_mcp.services import (
    CacheService,
    ProviderService,
    RelevanceSearchService,
    SearchService,
    SynthesisService,
)

# from scholar_flux_mcp.models.history import SynthesisOutputHistory, SynthesisInputHistory, SearchRecords

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mocks_ai_session_cache_backend() -> str:
    """Returns the cache used for storing mocks and simulating record retrieval from cache."""
    return "filesystem"


@pytest.fixture
def mocks_ai_directory() -> str:
    """Returns the directory where mocks are stored."""
    return str(tests.mocks.__path__[0])


@pytest.fixture
def mocks_ai_db_filename() -> str:
    """Returns the name of the file-based database where mocks are stored."""
    return "ai-learning-structure-testing"


@pytest.fixture(autouse=True)
def mock_ai_api_searches(monkeypatch, restore_config) -> Generator[requests_mock.Mocker, None, None]:
    """Helper for ensuring that searches to the first 3 pages of Core, PLOS, and OpenAlex are mocked for query Learning
    AI."""
    # Removing system-by-system env variables for consistent retrieval from session cache and turning off all outgoing responses
    with monkeypatch.context() as m, requests_mock.Mocker(real_http=False) as mocker:
        m.setenv(
            "SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND", "filesystem"
        )  # remove this for consistent cache retrieval

        if os.getenv("CORE_API_KEY"):
            m.delenv("CORE_API_KEY", raising=False)
        if os.getenv("SCHOLAR_FLUX_DEFAULT_MAILTO"):
            m.delenv("SCHOLAR_FLUX_DEFAULT_MAILTO", raising=False)

        config_settings.set("CORE_API_KEY", None)  # Optional CORE
        config_settings.set("SCHOLAR_FLUX_DEFAULT_MAILTO", None)  # Optional OpenAlex parameter

        # returned for potential use cases that require direct mocking
        yield mocker


@pytest.fixture
def mock_ai_search_input() -> SearchInput:
    """Search input for verifying the SearchService and cached record retrieval."""
    return SearchInput.create(providers=["openalex", "core", "plos"], queries=["Learning AI"])


@pytest.fixture
def mock_ai_search_markdown_input() -> SearchToolInput:
    """Search input for verifying the full MCP record search tool call pipeline and markdown output."""
    return SearchToolInput(
        providers=["openalex", "core", "plos"],
        queries=["Learning AI"],
        max_records=25,
        pages=3,
        response_format="markdown",
    )


@pytest.fixture
def mock_ai_search_json_input() -> SearchToolInput:
    """Search input for verifying the full MCP record search tool call pipeline."""
    return SearchToolInput(
        providers=["openalex", "core", "plos"],
        queries=["Learning AI"],
        max_records=25,
        pages=3,
        response_format="json",
    )


@pytest.fixture
def mock_ai_synthesis_markdown_input() -> SynthesisToolInput:
    """Synthesis input for verifying the full MCP research synthesis tool call pipeline."""
    return SynthesisToolInput(
        question="What are the key factors that foster AI literacy in the workplace?",
        providers=["openalex", "core", "plos"],
        queries=["Learning AI"],
        max_records=25,
        pages=3,
        year_from=2021,
        year_to=2100,
        open_access_only=False,
        similarity_threshold=0.5,
        categories=["GENERAL"],
        response_format="markdown",
    )


@pytest.fixture
def mock_ai_synthesis_json_input(mock_ai_synthesis_markdown_input) -> SynthesisToolInput:
    """Synthesis input for verifying the full MCP research synthesis tool call pipeline."""
    return mock_ai_synthesis_markdown_input.model_copy(update={"response_format": "json"})


@pytest.fixture
def mock_ai_relevance_search_markdown_input(mock_ai_synthesis_markdown_input) -> RelevanceSearchToolInput:
    """Synthesis input for verifying the full MCP research synthesis tool call pipeline."""
    return RelevanceSearchToolInput.model_validate(mock_ai_synthesis_markdown_input.model_dump(), extra="ignore")


@pytest.fixture
def mock_ai_synthesis_input() -> SynthesisInput:
    """Synthesis input for testing and verifying key functionality with mocked input data retrieved from cache."""
    return SynthesisInput.create(
        question="What are the precursors to success in fostering Computer Science and AI Literacy?",
        providers=[APIProviders.OPENALEX, APIProviders.CORE, APIProviders.PLOS],
        categories=ResearchCategory.COMPUTATION,
        max_records=200,
        pages=3,
        similarity_threshold=0.5,
        year_from=2021,
        year_to=2026,
        queries=["Learning AI"],
    )


@pytest.fixture
def mock_ai_synthesis_output_path() -> Path:
    """Path to the sample AI-generated synthesis output used for mocking."""
    path = Path(__file__).parent.parent / "mocks" / "ai-learning-structure-testing-synthesis-output.json"
    assert path.parent.exists()
    return path


@pytest.fixture
def mock_ai_synthesis_output(mock_ai_synthesis_output_path) -> SynthesisOutput:
    """Synthesis output generated via reranking and synthesis on cached records."""
    with open(mock_ai_synthesis_output_path, "rb") as f:
        json_output = json.load(f)

    return SynthesisOutput.model_validate(json_output)


@pytest.fixture
def mock_ai_relevance_search_input() -> RelevanceSearchInput:
    """Relevance search input for verifying reranking functionality with mocked input data retrieved from cache."""
    return RelevanceSearchInput.create(
        question="What are the precursors to success in fostering Computer Science and AI Literacy?",
        providers=[APIProviders.OPENALEX, APIProviders.CORE, APIProviders.PLOS],
        categories=ResearchCategory.COMPUTATION,
        max_records=200,
        pages=3,
        similarity_threshold=0.5,
        year_from=2021,
        year_to=2026,
        queries=["Learning AI"],
    )


@pytest.fixture
def mocks_ai_session_cache_context(
    mocks_ai_session_cache_backend, mocks_ai_db_filename, mocks_ai_directory, restore_config
):
    """Sets the env context to reused the cache AI-literacy data for testing."""
    config_settings.set("SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND", mocks_ai_session_cache_backend)
    config_settings.set("SCHOLAR_FLUX_SESSION_CACHE_NAME", mocks_ai_db_filename)
    config_settings.set("SCHOLAR_FLUX_SESSION_CACHE_DIRECTORY", mocks_ai_directory)
    yield


@pytest.fixture
async def mock_ai_synthesis_app_context(mocks_ai_session_cache_context, patch_ollama_available) -> AppContext:
    """Context for mocking searches to PLOS, OpenAlex, and CORE to verify that synthesis operates as intended."""
    provider_service = ProviderService()
    cache_service = CacheService()
    await cache_service.initialize()
    search_service = SearchService(cache_service)
    relevance_search_service = RelevanceSearchService(search_service)
    synthesis_service = SynthesisService(relevance_search_service)
    session_manager = cache_service.session_manager
    if not isinstance(session_manager, CachedSessionManager):
        raise TypeError(
            "Error configuring the CachedSessionManager: expected the `cache_service.session_manager` to be a "
            f"`CachedSessionManager` but received type {type(session_manager)}"
        )

    return AppContext(
        search_service=search_service,
        relevance_search_service=relevance_search_service,
        synthesis_service=synthesis_service,
        provider_service=provider_service,
    )


__all__ = [
    "mocks_ai_directory",
    "mocks_ai_session_cache_backend",
    "mocks_ai_db_filename",
    "mocks_ai_session_cache_context",
    "mock_ai_search_input",
    "mock_ai_search_markdown_input",
    "mock_ai_search_json_input",
    "mock_ai_synthesis_input",
    "mock_ai_synthesis_output",
    "mock_ai_synthesis_output_path",
    "mock_ai_synthesis_markdown_input",
    "mock_ai_synthesis_json_input",
    "mock_ai_relevance_search_input",
    "mock_ai_relevance_search_markdown_input",
    "mock_ai_search_markdown_input",
    "mock_ai_api_searches",
    "mock_ai_synthesis_app_context",
]

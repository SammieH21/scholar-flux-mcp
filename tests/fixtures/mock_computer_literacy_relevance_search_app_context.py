"""Tests for the synthesis service and integration with cache and search services."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from scholar_flux import CachedSessionManager, config_settings

import tests.mocks
from scholar_flux_mcp.models import (
    RelevanceSearchInput,
    RelevanceSearchOutput,
)
from scholar_flux_mcp.server.app_context import AppContext
from scholar_flux_mcp.server.io.relevance_search import RelevanceSearchToolInput
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
def mocks_computer_literacy_session_cache_backend() -> str:
    """Returns the cache used for storing mocks and simulating record retrieval from cache."""
    return "filesystem"  # transitioning to the filesystem for storage


@pytest.fixture
def mocks_computer_literacy_directory() -> str:
    """Returns the directory where mocks are stored."""
    return str(tests.mocks.__path__[0])


@pytest.fixture
def mocks_computer_literacy_cache_name() -> str:
    """Returns the name of the file-based database where mocks are stored."""
    return "computer-literacy"


@pytest.fixture
def mock_computer_literacy_api_searches(
    mocks_computer_literacy_session_cache_backend, monkeypatch, restore_config
) -> Generator[None, None, None]:
    """Helper for ensuring that searches to the first 3 pages of Core, PLOS, and OpenAlex are mocked for query Learning
    AI."""
    # Removing system-by-system env variables for consistent retrieval from session cache and turning off all outgoing responses
    with monkeypatch.context() as m:  # , requests_mock.Mocker(real_http=False) as mocker:
        m.setenv(
            "SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND", mocks_computer_literacy_session_cache_backend
        )  # remove this for consistent cache retrieval
        if os.getenv("SCHOLAR_FLUX_DEFAULT_MAILTO"):
            m.delenv("SCHOLAR_FLUX_DEFAULT_MAILTO", raising=False)
        config_settings.set("SCHOLAR_FLUX_DEFAULT_MAILTO", None)  # Optional OpenAlex parameter

        # returned for potential use cases that require direct mocking: requests_mock.Mocker
        yield  # mocker


@pytest.fixture
def mock_computer_literacy_relevance_search_markdown_input() -> RelevanceSearchToolInput:
    """RelevanceSearchToolInput for verifying the output structure of relevance-based searches."""
    return RelevanceSearchToolInput(
        question="Student success computer programming academics college",
        providers=["openalex", "plos"],
        categories=["COMPUTATION"],
        queries=["Student Computer Literacy"],
        max_records=100,
        pages=3,
        response_format="markdown",
        open_access_only=True,
    )


@pytest.fixture
def mock_computer_literacy_relevance_search_json_input() -> RelevanceSearchToolInput:
    """RelevanceSearchToolInput for verifying the output structure of relevance-based searches."""
    return RelevanceSearchToolInput(
        question="Student success computer programming academics college",
        providers=["openalex", "plos"],
        categories=["COMPUTATION"],
        queries=["Student Computer Literacy"],
        max_records=100,
        pages=3,
        response_format="json",
        open_access_only=True,
    )


@pytest.fixture
def mock_computer_literacy_relevance_search_output_path() -> Path:
    """Path to the sample relevance-search output used for mocking."""
    path = Path(__file__).parent.parent / "mocks" / "mock-computer-literacy-relevance-search-output.json"
    assert path.parent.exists()
    return path


@pytest.fixture
def mock_computer_literacy_relevance_search_output(
    mock_computer_literacy_relevance_search_output_path,
) -> RelevanceSearchOutput:
    """Relevance Search output generated via embeddings and reranking on cached records."""
    with open(mock_computer_literacy_relevance_search_output_path, "rb") as f:
        json_output = json.load(f)

    return RelevanceSearchOutput.model_validate(json_output)


@pytest.fixture
def mock_computer_literacy_relevance_search_input(
    mock_computer_literacy_relevance_search_json_input,
) -> RelevanceSearchInput:
    """Relevance search input for verifying reranking functionality with mocked input data retrieved from cache."""
    return mock_computer_literacy_relevance_search_json_input.as_relevance_search_input()


@pytest.fixture
def mocks_computer_literacy_session_cache_context(
    mocks_computer_literacy_session_cache_backend, mocks_computer_literacy_cache_name, mocks_computer_literacy_directory
):
    """Sets the env context to reused the cache computer literacy data for testing."""
    config_settings.set("SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND", mocks_computer_literacy_session_cache_backend)
    config_settings.set("SCHOLAR_FLUX_SESSION_CACHE_NAME", mocks_computer_literacy_cache_name)
    config_settings.set("SCHOLAR_FLUX_SESSION_CACHE_DIRECTORY", mocks_computer_literacy_directory)
    yield


@pytest.fixture
async def mock_computer_literacy_relevance_search_app_context(
    mocks_computer_literacy_session_cache_context,
) -> AppContext:
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
            "Error configuring the Computer Literacy CachedSessionManager: expected the "
            f"`cache_service.session_manager` to be a `CachedSessionManager` but received type {type(session_manager)}"
        )

    return AppContext(
        search_service=search_service,
        relevance_search_service=relevance_search_service,
        synthesis_service=synthesis_service,
        provider_service=provider_service,
    )


__all__ = [
    "mocks_computer_literacy_directory",
    "mocks_computer_literacy_session_cache_backend",
    "mocks_computer_literacy_cache_name",
    "mocks_computer_literacy_session_cache_context",
    "mock_computer_literacy_relevance_search_json_input",
    "mock_computer_literacy_relevance_search_markdown_input",
    "mock_computer_literacy_relevance_search_input",
    "mock_computer_literacy_relevance_search_output_path",
    "mock_computer_literacy_relevance_search_output",
    "mock_computer_literacy_api_searches",
    "mock_computer_literacy_relevance_search_app_context",
]

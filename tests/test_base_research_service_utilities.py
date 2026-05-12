"""Tests the for utilities used implemented the BaseResearchService for shared use in searches and synthesis."""

import pytest

from scholar_flux_mcp.exceptions import HistoryCacheInitializationException
from scholar_flux_mcp.services.base_research_service import BaseResearchService
from scholar_flux_mcp.services.history_service import HistoryService


def test_base_research_service_initialization():
    """Tests the initialization of the BaseResearchService with and without an assigned HistoryService."""
    history_service = HistoryService(enable=False)
    research_service = BaseResearchService(history_service=history_service)
    assert isinstance(research_service.history_service, HistoryService)

    research_service = BaseResearchService()
    assert research_service._history_service is None

    with pytest.raises(HistoryCacheInitializationException):
        # Attempting to access an uninitialized history service raises an error
        _ = research_service.history_service

    research_service.history_service = history_service
    assert isinstance(research_service.history_service, HistoryService)


@pytest.mark.asyncio
async def test_base_research_service_history_actions_raises_exception_without_service(
    caplog, mock_ai_synthesis_input, mock_ai_synthesis_output
):
    """Verifies that the BaseResearchService reraises without a assigned HistoryService when raise_on_error=True."""
    research_service = BaseResearchService()

    msg = f"Couldn't store the {type(mock_ai_synthesis_output).__name__} within the output history cache"
    with pytest.raises(HistoryCacheInitializationException):
        await research_service.store_history_cache(mock_ai_synthesis_output)
    assert msg in caplog.text

    msg = (
        f"Couldn't retrieve the output associated with the {type(mock_ai_synthesis_input).__name__} within the output "
        "history cache"
    )

    with pytest.raises(HistoryCacheInitializationException):
        await research_service.retrieve_history_cache(mock_ai_synthesis_input)
    assert msg in caplog.text


@pytest.mark.asyncio
async def test_base_research_service_failed_history_actions_gracefully_continue_when_enabled(
    caplog, mock_ai_synthesis_input, mock_ai_synthesis_output
):
    """Verifies that the BaseResearchService gracefully continues when `raise_on_error=False`."""
    research_service = BaseResearchService()

    msg = f"Couldn't store the {type(mock_ai_synthesis_output).__name__} within the output history cache"
    await research_service.store_history_cache(mock_ai_synthesis_output, raise_on_error=False)
    assert msg in caplog.text
    assert "Skipping output storage..." in caplog.text

    msg = (
        f"Couldn't retrieve the output associated with the {type(mock_ai_synthesis_input).__name__} within the output "
        "history cache"
    )

    retrieved = await research_service.retrieve_history_cache(mock_ai_synthesis_input, raise_on_error=False)
    assert retrieved is None
    assert msg in caplog.text
    assert "Skipping output retrieval..." in caplog.text


@pytest.mark.parametrize(
    "obj,expected_type,allow_missing",
    [(0, str, False), ("1", int, False), ({}, list, True), ([], dict, True), (None, int, False)],
)
def test_service_dependency_validation(obj, expected_type, allow_missing):
    """Verifies that a type error is raised when validating an object with an unexpected type."""
    with pytest.raises(TypeError, match=f"expected a {expected_type.__name__}, but received {type(obj).__name__}"):
        _ = BaseResearchService._validate_service_dependency(obj, expected_type, allow_missing=allow_missing)

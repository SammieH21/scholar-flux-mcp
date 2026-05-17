import importlib
import re
from time import sleep
from unittest.mock import patch

import pytest
import sqlalchemy
import sqlmodel

from scholar_flux_mcp.exceptions import HistoryCacheInitializationException, SQLModelImportError
from scholar_flux_mcp.models import SynthesisOutput
from scholar_flux_mcp.models.history import HISTORY_TABLES
from scholar_flux_mcp.server.io.history import OutputHistoryFormatter
from scholar_flux_mcp.services import HistoryService
from scholar_flux_mcp.utils.helpers import os_env_context
from tests.testing_utilities import raise_error


@pytest.fixture
def mock_overwritten_ai_synthesis_output(mock_ai_synthesis_output) -> SynthesisOutput:
    """Helper for testing minor variations in output when stored via HistoryService."""
    synthesis_input = mock_ai_synthesis_output.synthesis_input.model_copy(update={"year_from": 2020})
    synthesis_output = mock_ai_synthesis_output.model_copy(update={"synthesis_input": synthesis_input})
    return synthesis_output


def test_history_service_session_initialization():
    """Verifies that the URL default is a valid sqlite URL."""
    assert HistoryService.is_available()
    history_service = HistoryService(verify_connection=True)
    history_service.verify_url_string(history_service.config["url"])

    history_table_names: set[str] = {
        table_name
        for model in HISTORY_TABLES
        if (table_name := getattr(model, "__tablename__", None)) and not table_name.startswith("base")
    }

    assert len(history_table_names.difference(sqlmodel.SQLModel.metadata.tables.keys())) == 0


async def test_history_service_roundtrip_synthesis_storage(
    mock_synthesis_output,
    tmp_path,
    cleanup,
):
    """Verifies that the `HistoryService` correctly stores SynthesisOutputs."""
    url = f"sqlite:///{tmp_path}/test-db.sqlite"
    history_service = HistoryService(url=url)
    await history_service.store(mock_synthesis_output)
    assert mock_synthesis_output == await history_service.retrieve(mock_synthesis_output.synthesis_input)


async def test_history_service_roundtrip_synthesis_storage_with_rejected_evidence_items(
    mock_synthesis_output_with_rejected_evidence_output,
    tmp_path,
    cleanup,
):
    """Verifies that the `HistoryService` correctly stores SynthesisOutputs with rejected evidence items."""
    url = f"sqlite:///{tmp_path}/test-db-two.sqlite"
    history_service = HistoryService(url=url)
    await history_service.store(mock_synthesis_output_with_rejected_evidence_output)
    assert mock_synthesis_output_with_rejected_evidence_output == await history_service.retrieve(
        mock_synthesis_output_with_rejected_evidence_output.synthesis_input
    )


async def test_history_service_storage(
    mock_ai_synthesis_output,
    mock_overwritten_ai_synthesis_output,
    mock_computer_literacy_relevance_search_output,
    tmp_path,
    cleanup,
):
    """Verifies that the `HistoryService` correctly stores RelevanceSearch and SynthesisOutputs."""
    history_service = HistoryService()

    synthesis_input = mock_ai_synthesis_output.synthesis_input
    cl_relevance_search_input = mock_computer_literacy_relevance_search_output.relevance_search_input

    assert await history_service.retrieve(synthesis_input) is None
    assert await history_service.retrieve(cl_relevance_search_input) is None

    await history_service.store(mock_ai_synthesis_output)
    await history_service.store(mock_overwritten_ai_synthesis_output)  # raises from a uniqueness integrity error
    await history_service.store(mock_computer_literacy_relevance_search_output)

    retrieved_synthesis_output = await history_service.retrieve(synthesis_input)
    assert retrieved_synthesis_output == mock_ai_synthesis_output

    retrieved_overwritten_synthesis_output = await history_service.retrieve(
        mock_overwritten_ai_synthesis_output.synthesis_input
    )
    assert retrieved_overwritten_synthesis_output == mock_overwritten_ai_synthesis_output

    search_input = mock_ai_synthesis_output.search_output.search_input
    retrieved_search_output = await history_service.retrieve(search_input)
    assert retrieved_search_output == mock_ai_synthesis_output.search_output

    relevance_search_input = mock_ai_synthesis_output.relevance_search_output.relevance_search_input
    retrieved_relevance_search_output = await history_service.retrieve(relevance_search_input)
    assert retrieved_relevance_search_output == mock_ai_synthesis_output.relevance_search_output

    retrieved_cl_relevance_search_output = await history_service.retrieve(cl_relevance_search_input)
    assert retrieved_cl_relevance_search_output == mock_computer_literacy_relevance_search_output

    cl_search_input = mock_computer_literacy_relevance_search_output.search_output.search_input
    retrieved_search_output = await history_service.retrieve(cl_search_input)
    assert retrieved_search_output == mock_computer_literacy_relevance_search_output.search_output


async def test_history_ttl_cache_expiration(mock_ai_synthesis_output, tmp_path, cleanup):
    """Verifies that TTL works as intended to filter records by age."""
    tmp_uri = "sqlite:///" + str(tmp_path / "testdb_ttl.sqlite")
    history_service = HistoryService(url=tmp_uri, raise_on_error=True, enable=True)
    await history_service.store(mock_ai_synthesis_output)
    synthesis_input = mock_ai_synthesis_output.synthesis_input
    search_input = mock_ai_synthesis_output.search_output.search_input  # type: ignore
    relevance_search_input = mock_ai_synthesis_output.relevance_search_output.relevance_search_input  # type: ignore

    assert await history_service.retrieve(synthesis_input)
    sleep(0.3)
    # Nothing older than a 100th of a second
    assert not await history_service.retrieve(synthesis_input, ttl=0.01)
    assert not await history_service.retrieve(relevance_search_input, ttl=0.01)
    assert not await history_service.retrieve(search_input, ttl=0.01)

    # The same records should be more recent than 100 seconds
    assert await history_service.retrieve(relevance_search_input, ttl=101)
    assert await history_service.retrieve(search_input, ttl=101)


async def test_history_service_cache_deletion(
    mock_ai_synthesis_output, mock_computer_literacy_relevance_search_output, tmp_path, cleanup
):
    """Verifies that the `HistoryService` correctly uses the `clear` method for cache deletion."""
    history_service = HistoryService(url="sqlite:///:memory:", raise_on_error=True)

    synthesis_input = mock_ai_synthesis_output.synthesis_input
    relevance_search_input = mock_ai_synthesis_output.relevance_search_output.relevance_search_input
    search_input = mock_ai_synthesis_output.search_output.search_input
    cl_relevance_search_input = mock_computer_literacy_relevance_search_output.relevance_search_input
    cl_search_input = mock_computer_literacy_relevance_search_output.search_output.search_input

    await history_service.store(mock_ai_synthesis_output)
    await history_service.store(mock_computer_literacy_relevance_search_output)

    await history_service.clear()

    assert await history_service.retrieve(synthesis_input) is None
    assert await history_service.retrieve(relevance_search_input) is None
    assert await history_service.retrieve(search_input) is None

    assert await history_service.retrieve(cl_relevance_search_input) is None
    assert await history_service.retrieve(cl_search_input) is None


def test_history_service_with_custom_path(tmp_path, cleanup):
    """Verifies that the `HistoryService` can establish a connection with a custom sqlite URI using a tmp filepath."""
    history_service = HistoryService()
    tmp_uri = "sqlite:///" + str(tmp_path / "testdb.sqlite")
    history_service = HistoryService(url=tmp_uri, verify_connection=True)
    assert history_service.config["url"] == tmp_uri
    assert history_service.is_available()


def test_history_service_with_custom_path_via_env(monkeypatch, tmp_path, cleanup):
    """Verifies that the `HistoryService` can establish a connection with a custom sqlite URI from the environment."""
    history_service = HistoryService()
    tmp_uri = "sqlite:///" + str(tmp_path / "env_testdb.sqlite")
    monkeypatch.setenv("SCHOLAR_FLUX_MCP_HISTORY_URL", tmp_uri)
    history_service = HistoryService(url=tmp_uri, verify_connection=True)
    assert history_service.config["url"] == tmp_uri
    assert history_service.is_available()


def test_history_research_tool_formatter(mock_computer_literacy_relevance_search_output, mock_ai_synthesis_output):
    """Verifies that the formatter correctly generates a summary from SynthesisOutput."""
    summary = OutputHistoryFormatter.format_recent_history_markdown(
        [
            mock_ai_synthesis_output,
        ]
    )
    assert isinstance(summary, str) and "1. Research Synthesis" in summary

    relevance_searches = [
        mock_ai_synthesis_output.relevance_search_output,
        mock_computer_literacy_relevance_search_output,
    ]
    summary = OutputHistoryFormatter.format_recent_history_markdown(relevance_searches)
    assert isinstance(summary, str) and "1. Relevance Search" in summary and "2. Relevance Search" in summary

    relevance_searches = [
        mock_ai_synthesis_output.relevance_search_output,
        mock_computer_literacy_relevance_search_output,
    ]
    summary = OutputHistoryFormatter.format_recent_history_markdown(
        [search.search_output for search in relevance_searches]
    )
    assert isinstance(summary, str) and "1. Record Search" in summary and "2. Record Search" in summary


@pytest.mark.parametrize("initialize,initialization_status", ([True, True], [False, False], [None, True]))
def test_enable_history_service_cache_initialization(initialize, initialization_status):
    """Tests the initialization of the `HistoryService` given when setting `enable` to `True`, `False`, or `None`."""
    with os_env_context("SCHOLAR_FLUX_MCP_ENABLE_HISTORY", "None"):
        history_service = HistoryService(enable=initialize)
        assert history_service.enabled is initialization_status
        assert isinstance(history_service._engine, sqlalchemy.Engine) == initialization_status


@pytest.mark.parametrize("initialize,initialization_status", ([True, True], [False, False], [None, True]))
def test_enable_history_service_cache_initialization_via_env(initialize, initialization_status, monkeypatch):
    """Tests `HistoryService` initialization when setting `SCHOLAR_FLUX_MCP_ENABLE_HISTORY`."""
    with os_env_context("SCHOLAR_FLUX_MCP_ENABLE_HISTORY", str(initialize)):
        history_service = HistoryService(enable=None)
        assert history_service.enabled is initialization_status
        assert isinstance(history_service._engine, sqlalchemy.Engine) == initialization_status


async def test_history_service_returns_valid_list_of_records(mock_ai_synthesis_output, monkeypatch, tmp_path, cleanup):
    """Verifies that the `HistoryService` can successfully retrieve a list of `SearchRecords."""
    history_service = HistoryService()
    tmp_uri = "sqlite:///" + str(tmp_path / "env_testdb.sqlite")
    monkeypatch.setenv("SCHOLAR_FLUX_MCP_HISTORY_URL", tmp_uri)
    history_service = HistoryService(url=tmp_uri, verify_connection=True)

    total_records = len(mock_ai_synthesis_output.indexed_records)
    await history_service.store(mock_ai_synthesis_output)

    record_history = await history_service.retrieve_record_history(max_history=total_records)
    retrieved_records = len(record_history)

    assert retrieved_records == total_records

    first_record = mock_ai_synthesis_output.indexed_records[0]
    record_history = await history_service.retrieve_record_history(max_history=1, record_hash=first_record.record_hash)
    assert record_history
    assert len(record_history) == 1


async def test_history_record_searches_with_fuzzy_similarity(mock_ai_synthesis_output, tmp_path, cleanup):
    """Verifies that the `HistoryService` can successfully use fuzzy search similarity to filter and reorder records."""
    history_service = HistoryService()
    tmp_uri = "sqlite:///" + str(tmp_path / "record_env_testdb.sqlite")
    history_service = HistoryService(url=tmp_uri, verify_connection=True)

    assert mock_ai_synthesis_output.indexed_records
    await history_service.store(mock_ai_synthesis_output)

    synthesis_records = mock_ai_synthesis_output.search_output.records

    all_retrieved = await history_service.retrieve_record_history(topic="random topic", similarity_threshold=None)
    assert len(all_retrieved) == len(synthesis_records)

    for record in synthesis_records:
        topic = record.title
        assert record.title
        retrieved = await history_service.retrieve_record_history(topic=topic, max_history=1, similarity_threshold=0.9)

        assert len(retrieved) == 1 and record == retrieved[0]


async def test_history_calculate_fuzzy_similarity(mock_ai_synthesis_output, tmp_path, cleanup):
    """Verifies that the `HistoryService` can successfully calculate history item-topic similarity."""
    history_service = HistoryService()
    tmp_uri = "sqlite:///" + str(tmp_path / "calculate_record_similarity_testdb.sqlite")
    history_service = HistoryService(url=tmp_uri, verify_connection=True)

    assert mock_ai_synthesis_output.indexed_records
    await history_service.store(mock_ai_synthesis_output)

    synthesis_records = mock_ai_synthesis_output.search_output.records
    record = synthesis_records[0]
    record_two = synthesis_records[1]

    record_topic_similarity = history_service.calculate_fuzzy_topic_similarity(record, topic=record.topic)
    assert record_topic_similarity.score == 1.0

    record_topic_similarity = history_service.calculate_fuzzy_topic_similarity(record_two, topic=record.topic)
    assert record_topic_similarity.score < 0.9  # Records would be identified as dupes otherwise


async def test_history_url_selection_with_persistence(tmp_path, cleanup, monkeypatch, caplog):
    """Verifies that DB URL respects the `SCHOLAR_FLUX_MCP_PERSIST_HISTORY."""
    pytest.importorskip("scholar_flux", reason="The history URL selection need scholar-flux to continue")

    monkeypatch.setattr(HistoryService, "DEFAULT_PERSIST_HISTORY", True)

    # With cache persistence
    dir_selection = "scholar_flux.package_metadata.directories.PackageDirectorySettings.get_default_writable_directory"
    monkeypatch.setattr(dir_selection, lambda *args, **kwargs: tmp_path)
    cache_partial_url = f"sqlite:///{tmp_path}"
    assert cache_partial_url in HistoryService.get_default_url()

    # Without cache persistence
    nonpersistence_url = HistoryService.get_default_url(persist_cache=False)
    assert nonpersistence_url == "sqlite:///:memory:"

    ## Simulating filesystem nonavailability
    err = "Directly raised exception"
    monkeypatch.setattr(dir_selection, raise_error(RuntimeError, err))
    nonpersistence_url_fallback = HistoryService.get_default_url(persist_cache=True)

    assert nonpersistence_url_fallback == "sqlite:///:memory:"
    assert err in caplog.text

    caplog.clear()

    ## Simulating ScholarFlux not being available
    monkeypatch.setattr("scholar_flux_mcp.services.history_service.package_directory_settings", None)
    nonpersistence_url_fallback_two = HistoryService.get_default_url(persist_cache=True)

    assert nonpersistence_url_fallback_two == "sqlite:///:memory:"
    assert (
        "The ScholarFlux base package is not installed. Restart the ScholarFluxMCP server after installing"
    ) in caplog.text


async def test_history_output_searches_with_fuzzy_similarity(
    mock_ai_synthesis_output,
    mock_synthesis_output,
    mock_computer_literacy_relevance_search_output,
    monkeypatch,
    tmp_path,
    cleanup,
):
    """Verifies that the `HistoryService` can successfully use fuzzy search similarity to filter and reorder outputs."""
    history_service = HistoryService()
    tmp_uri = "sqlite:///" + str(tmp_path / "output_env_testdb.sqlite")
    monkeypatch.setenv("SCHOLAR_FLUX_MCP_HISTORY_URL", tmp_uri)
    history_service = HistoryService(url=tmp_uri, verify_connection=True)

    assert mock_ai_synthesis_output.indexed_records
    await history_service.store(mock_ai_synthesis_output)
    await history_service.store(mock_synthesis_output)
    await history_service.store(mock_computer_literacy_relevance_search_output)

    all_retrieved = await history_service.retrieve_recent_history(
        topic="random topic", similarity_threshold=None, final_output_only=True
    )
    assert len(all_retrieved) == 3

    for output in (mock_ai_synthesis_output, mock_synthesis_output, mock_computer_literacy_relevance_search_output):
        topic = output.topic
        assert topic
        retrieved = await history_service.retrieve_recent_history(topic=topic, max_history=1, similarity_threshold=0.9)

        assert len(retrieved) == 1 and topic == retrieved[0].topic


async def test_history_service_methods_with_uninitialized_db(mock_computer_literacy_relevance_search_output, caplog):
    """Verifies that core methods correctly handle HistoryCacheInitializationException errors when encountered."""
    history_service = HistoryService(enable=False)
    err = r"The current `HistoryService` instance has not yet fully initialized the output history cache\."

    with pytest.raises(HistoryCacheInitializationException, match=err):
        history_service.engine
    assert re.search(err, caplog.text) is not None
    caplog.clear()

    with pytest.raises(HistoryCacheInitializationException, match=err):
        history_service.session
    assert re.search(err, caplog.text) is not None
    caplog.clear()

    with pytest.raises(HistoryCacheInitializationException, match=err):
        history_service.verify_connection()
    assert re.search(err, caplog.text) is not None
    caplog.clear()

    with pytest.raises(HistoryCacheInitializationException, match=err):
        await history_service.store(mock_computer_literacy_relevance_search_output)
    assert re.search(err, caplog.text) is not None
    caplog.clear()

    with pytest.raises(HistoryCacheInitializationException, match=err):
        await history_service.retrieve(mock_computer_literacy_relevance_search_output.relevance_search_input)
    assert re.search(err, caplog.text) is not None
    caplog.clear()


async def test_history_service_deps_missing():
    """Verifies the behavior of the history service when `sqlmodel` and `history_models` are missing."""
    import scholar_flux_mcp.models.history
    import scholar_flux_mcp.models.type_aliases
    import scholar_flux_mcp.package_metadata
    import scholar_flux_mcp.package_metadata.dependencies
    import scholar_flux_mcp.server.io.history
    import scholar_flux_mcp.server.main
    import scholar_flux_mcp.services.history_service

    try:
        with patch.dict("sys.modules", {"sqlmodel": None}):
            importlib.reload(scholar_flux_mcp.package_metadata.dependencies)
            importlib.reload(scholar_flux_mcp.package_metadata)
            importlib.reload(scholar_flux_mcp.server.io.history)
            importlib.reload(scholar_flux_mcp.server.main)
            importlib.reload(scholar_flux_mcp.models.type_aliases)
            importlib.reload(scholar_flux_mcp.services.history_service)

            history_service = scholar_flux_mcp.services.history_service.HistoryService(
                enable=True, raise_on_error=False
            )

            with pytest.raises(SQLModelImportError):
                history_service.initialize_database()

    finally:
        # reimport core packages
        importlib.reload(scholar_flux_mcp.package_metadata.dependencies)
        importlib.reload(scholar_flux_mcp.package_metadata)
        importlib.reload(scholar_flux_mcp.server.io.history)
        importlib.reload(scholar_flux_mcp.server.main)
        importlib.reload(scholar_flux_mcp.models.type_aliases)
        importlib.reload(scholar_flux_mcp.services.history_service)

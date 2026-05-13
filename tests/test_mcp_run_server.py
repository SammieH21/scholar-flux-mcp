"""Verifies that the MCP server and its tools are usable directly via the `FastMCP` client and `asyncio`."""

from __future__ import annotations

import contextlib
import importlib
import json
import re

import pytest
from fastmcp.client import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import FastMCPTransport
from mcp.types import TextContent

from scholar_flux_mcp.models import APIProviders, ProviderMetadataDescriptions, ResearchTopic
from scholar_flux_mcp.server.io.health_check import HealthCheckToolInput
from scholar_flux_mcp.server.io.history import (
    ClearHistoryToolInput,
    GetHistoryToolInput,
    RecentHistoryToolInput,
    RecordHistoryToolInput,
)
from scholar_flux_mcp.server.io.providers import ListProvidersInput
from scholar_flux_mcp.server.io.relevance_search import RelevanceSearchFormatter
from scholar_flux_mcp.server.io.search import SearchFormatter
from scholar_flux_mcp.server.io.synthesis import SynthesisFormatter
from scholar_flux_mcp.services import CacheService, HistoryService
from tests.testing_utilities import raise_error

SESSION_CACHE_BACKENDS = ("sqlite", "filesystem", "memory", "redis", "mongodb", "inmemory", "gridfs")
RESPONSE_CACHE_STORAGE_DEVICES = ("sqlite", "sql", "mongodb", "mongo", "null", "memory", "inmemory", "duckdb", "redis")


@pytest.fixture
def reload_mcp_module():
    """Fixture for reloading the ScholarFluxMCP server.

    Helpful for loading updated env variables for later testing.

    """
    import scholar_flux_mcp.server.main

    importlib.reload(scholar_flux_mcp.server.main)
    yield
    importlib.reload(scholar_flux_mcp.server.main)


@pytest.fixture
async def main_mcp_client():
    """Fixture for launching the ScholarFluxMCP server to later verify its functionality with pytest."""
    from scholar_flux_mcp.server.main import mcp

    async with Client(transport=mcp) as mcp_client:
        yield mcp_client


@pytest.fixture
def mock_lifespan(mock_ai_synthesis_app_context):
    """Helper for mocking the synthesis context."""

    @contextlib.asynccontextmanager
    async def _mock_lifespan(server):
        yield mock_ai_synthesis_app_context

    return _mock_lifespan


@pytest.fixture
async def mock_mcp_client_with_history(
    mock_ai_synthesis_app_context,
    mock_ai_synthesis_output,
    mock_computer_literacy_relevance_search_output,
    reload_mcp_module,
    mock_lifespan,
    monkeypatch,
):
    from scholar_flux_mcp.server.main import mcp

    mock_ai_search_output = mock_ai_synthesis_output.search_output
    assert mock_ai_search_output

    history_service = HistoryService(enable=True)
    await history_service.store(mock_ai_search_output)
    await history_service.store(mock_computer_literacy_relevance_search_output)
    await history_service.store(mock_ai_synthesis_output)

    monkeypatch.setattr(mock_ai_synthesis_app_context, "history_service", history_service)

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)
    return mcp


@pytest.fixture
async def mock_mcp_client_with_pydantic_ai_test_mocks(
    mock_ai_synthesis_app_context,
    patch_ollama_available,
    reload_mcp_module,
    mock_lifespan,
    monkeypatch,
    mock_pydantic_ai_agent,
    mock_embedding_model,
):
    """Mocks the AI learning synthesis app context to a test embedding model and synthesis agent for testing."""
    from scholar_flux_mcp.server.main import mcp

    history_service = HistoryService(enable=True)

    mock_ai_synthesis_app_context.synthesis_service.record_topic_similarity_embedder.get_or_create_embedder()
    mock_ai_synthesis_app_context.synthesis_service.synthesis_agent.get_or_create_agent()
    monkeypatch.setattr(mock_ai_synthesis_app_context, "history_service", history_service)

    with (
        mock_ai_synthesis_app_context.synthesis_service.record_topic_similarity_embedder.embedder.override(
            model=mock_embedding_model
        ),
        mock_ai_synthesis_app_context.synthesis_service.synthesis_agent.agent.override(model=mock_pydantic_ai_agent),
    ):
        monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)
        yield mcp


def extract_json_text(result: CallToolResult) -> dict | list | CallToolResult | None:
    """Helper method for extracting JSON TextContent from tool output."""
    with contextlib.suppress(ValueError, TypeError, AttributeError):
        json_content = result.content or []
        content_text = (
            [json.loads(getattr(text, "text", "")) for text in json_content if isinstance(text, TextContent)]
            if json_content
            else None
        )

        return content_text[0] if content_text and len(content_text or []) == 1 else content_text
    return result  # return the original result if extraction fails


async def test_list_tools(main_mcp_client: Client[FastMCPTransport]):
    """Verifies that `list_tools` correctly indicates the range of core tools that are available for MCP integration."""
    list_tools = await main_mcp_client.list_tools()
    tool_names = [tool.name for tool in list_tools]

    assert "scholar_flux_record_search" in tool_names
    assert "scholar_flux_synthesize_research_summary" in tool_names
    assert "scholar_flux_list_providers" in tool_names


async def test_list_providers(main_mcp_client: Client[FastMCPTransport]):
    """Verifies that `list_providers` correctly indicates the range of possible providers."""
    json_params = {"params": ListProvidersInput(response_format="json")}
    markdown_params = {"params": ListProvidersInput(response_format="markdown")}

    result = await main_mcp_client.call_tool(name="scholar_flux_list_providers", arguments=json_params)

    json_result = extract_json_text(result)
    assert result and isinstance(json_result, dict)

    json_provider_dict = json_result.get("providers") or {}
    supported_providers = len(list(APIProviders))
    assert supported_providers <= len(json_provider_dict)
    assert supported_providers == len([provider for provider in json_provider_dict.values() if provider["available"]])

    result_md = await main_mcp_client.call_tool(name="scholar_flux_list_providers", arguments=markdown_params)
    md_text = getattr(result_md.content[0], "text", "")

    missing_provider_display_names = {
        display_name
        for provider in APIProviders
        if (display_name := ProviderMetadataDescriptions(provider).value.display_name) not in md_text
    }
    assert not missing_provider_display_names


async def test_list_providers_gracefully_handles_unexpected_exceptions(
    mock_mcp_client_with_history,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    monkeypatch,
):
    """Verifies that the `scholar_flux_list_providers` tool formats and transmits messages on unexpected errors."""
    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)

    err = "Directly raised exception"
    monkeypatch.setattr(
        mock_ai_synthesis_app_context.provider_service,
        "get_providers",
        raise_error(RuntimeError, err),
    )

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        list_providers_tool_markdown_output = await mcp_client.call_tool(
            name="scholar_flux_list_providers",
            arguments={"params": ListProvidersInput(response_format="markdown")},
        )
    list_providers_markdown_str = list_providers_tool_markdown_output.content[0].text
    assert err in list_providers_markdown_str


async def test_search_tool_outputs_json(
    mock_ai_search_json_input, monkeypatch, mock_ai_synthesis_app_context, reload_mcp_module, mock_lifespan
):
    """Verifies that the `record_search` MCP tool correctly returns json data with the required structure."""
    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)
    async with Client(transport=mcp) as mcp_client:
        result = await mcp_client.call_tool(
            name="scholar_flux_record_search", arguments={"params": mock_ai_search_json_input}
        )

    assert result and result.content
    json_text = result.content[0].text
    json_data = json.loads(json_text)
    assert json_data and "records" in json_data and "search_input" in json_data

    # At least one record from cache will the full text. All should have `full_text` and `abstract` keys.
    assert json_data["records"] and any(
        record["full_text"] is not None and record["abstract"] is not None for record in json_data["records"]
    )


async def test_search_tool_outputs_markdown(
    mock_ai_search_markdown_input, monkeypatch, mock_ai_synthesis_app_context, reload_mcp_module, mock_lifespan
):
    """Verifies that the `record_search` MCP tool correctly returns markdown data with the required content."""
    providers = {
        ProviderMetadataDescriptions(provider).value.display_name
        for provider in mock_ai_search_markdown_input.providers
    }

    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)
    async with Client(transport=mcp) as mcp_client:
        result = await mcp_client.call_tool(
            name="scholar_flux_record_search", arguments={"params": mock_ai_search_markdown_input}
        )

    assert result and result.content

    search_markdown = result.content[0].text

    assert search_markdown.startswith("## Search Results")
    assert re.search(r"\*\*Found [1-9][0-9]* records\*\* across \d+ queries", search_markdown) is not None
    assert all(re.search(rf"### {provider} \(\d+ records\)", search_markdown) is not None for provider in providers)


async def test_relevance_search(
    mock_ai_relevance_search_markdown_input,
    monkeypatch,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    mock_embedding_model,
):
    """Verifies that the `scholar_flux_record_relevance_search` MCP tool correctly returns markdown data."""
    from scholar_flux_mcp.server.main import mcp

    mock_ai_synthesis_app_context.synthesis_service.record_topic_similarity_embedder.get_or_create_embedder()
    with mock_ai_synthesis_app_context.synthesis_service.record_topic_similarity_embedder.embedder.override(
        model=mock_embedding_model
    ):
        monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)
        async with Client(transport=mcp) as mcp_client:
            result = await mcp_client.call_tool(
                name="scholar_flux_record_relevance_search",
                arguments={"params": mock_ai_relevance_search_markdown_input},
            )
    assert result and result.content

    relevance_search_markdown = result.content[0].text

    assert relevance_search_markdown.startswith("## Search Results")
    assert re.search("Found [1-9][0-9]* records", relevance_search_markdown) is not None
    assert re.search(r"\*\*Filtered Record Count\*\*: [0-9]+", relevance_search_markdown) is not None
    assert re.search(r"\*\*Sources\*\*:", relevance_search_markdown) is not None
    assert re.search(r"Records by Relevance \(Descending Order\):", relevance_search_markdown) is not None


async def test_synthesis(
    mock_ai_synthesis_markdown_input,
    monkeypatch,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    mock_embedding_model,
    mock_pydantic_ai_agent,
):
    """Verifies that the `synthesize_research_summary` MCP tool correctly returns markdown data."""
    from scholar_flux_mcp.server.main import mcp

    mock_ai_synthesis_app_context.synthesis_service.record_topic_similarity_embedder.get_or_create_embedder()
    mock_ai_synthesis_app_context.synthesis_service.synthesis_agent.get_or_create_agent()
    with (
        mock_ai_synthesis_app_context.synthesis_service.record_topic_similarity_embedder.embedder.override(
            model=mock_embedding_model
        ),
        mock_ai_synthesis_app_context.synthesis_service.synthesis_agent.agent.override(model=mock_pydantic_ai_agent),
    ):
        monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)
        async with Client(transport=mcp) as mcp_client:
            result = await mcp_client.call_tool(
                name="scholar_flux_synthesize_research_summary", arguments={"params": mock_ai_synthesis_markdown_input}
            )
    assert result and result.content
    synthesis_markdown = result.content[0].text

    assert re.search(rf"> {mock_ai_synthesis_markdown_input.question}", synthesis_markdown)

    # A List of categories to further narrow down the synthesis topic
    assert re.search(r"\*\*Categories\*\*: \[[ 'a-zA-Z0-9]+\]", synthesis_markdown)

    assert re.search(r"\*\*Queries\*\*:", synthesis_markdown)
    assert all(re.search(rf"- {query}\n", synthesis_markdown) for query in mock_ai_synthesis_markdown_input.queries)

    # Either a header or synthesis not available
    assert re.search(r"## Synthesis(\n|not available)", synthesis_markdown) is not None

    additional_fields = ["Records Analyzed", "Confidence Score", "Evidence Grounding"]

    for field in additional_fields:
        assert re.search(rf"\*\*{field}\*\*: \d+", synthesis_markdown) is not None


async def test_search_raises_on_unexpected_error(
    mock_ai_search_json_input, monkeypatch, mock_ai_synthesis_app_context, reload_mcp_module, mock_lifespan
):
    """Verifies that the `scholar_flux_record_search` MCP tool correctly handles relevance search errors."""
    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)
    monkeypatch.setattr(
        mock_ai_synthesis_app_context.search_service, "search", raise_error(RuntimeError, "Directly raised exception")
    )

    async with Client(transport=mcp) as mcp_client:
        result = await mcp_client.call_tool(
            name="scholar_flux_record_search", arguments={"params": mock_ai_search_json_input}
        )

    assert result and result.content
    json_text = result.content[0].text

    err = "Directly raised exception"
    msg = f"The `scholar_flux_record_search` tool failed to retrieve and process valid academic records. {err}"
    assert msg in json_text


async def test_relevance_search_on_unexpected_error(
    mock_ai_relevance_search_markdown_input,
    monkeypatch,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    mock_embedding_model,
):
    """Verifies that the `scholar_flux_record_relevance_search` MCP tool correctly handles relevance search errors."""
    from scholar_flux_mcp.server.main import mcp

    mock_ai_synthesis_app_context.synthesis_service.record_topic_similarity_embedder.get_or_create_embedder()

    with mock_ai_synthesis_app_context.synthesis_service.record_topic_similarity_embedder.embedder.override(
        model=mock_embedding_model
    ):
        monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)
        monkeypatch.setattr(
            mock_ai_synthesis_app_context.relevance_search_service,
            "relevance_search",
            raise_error(RuntimeError, "Directly raised exception"),
        )
        async with Client(transport=mcp) as mcp_client:
            result = await mcp_client.call_tool(
                name="scholar_flux_record_relevance_search",
                arguments={"params": mock_ai_relevance_search_markdown_input},
            )

    assert result and result.content
    json_text = result.content[0].text

    err = "Directly raised exception"
    msg = f"The `scholar_flux_record_relevance_search` tool did not return a valid result. {err}"
    assert msg in json_text


async def test_synthesis_on_unexpected_error(
    mock_ai_synthesis_markdown_input,
    monkeypatch,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    mock_embedding_model,
    mock_pydantic_ai_agent,
):
    """Verifies that the `synthesize_research_summary` MCP tool correctly handles unexpected issues during synthesis."""
    from scholar_flux_mcp.server.main import mcp

    mock_ai_synthesis_app_context.synthesis_service.record_topic_similarity_embedder.get_or_create_embedder()
    mock_ai_synthesis_app_context.synthesis_service.synthesis_agent.get_or_create_agent()
    with (
        mock_ai_synthesis_app_context.synthesis_service.record_topic_similarity_embedder.embedder.override(
            model=mock_embedding_model
        ),
        mock_ai_synthesis_app_context.synthesis_service.synthesis_agent.agent.override(model=mock_pydantic_ai_agent),
    ):
        monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)
        monkeypatch.setattr(
            mock_ai_synthesis_app_context.relevance_search_service,
            "relevance_search",
            raise_error(RuntimeError, "Directly raised exception"),
        )
        async with Client(transport=mcp) as mcp_client:
            result = await mcp_client.call_tool(
                name="scholar_flux_synthesize_research_summary", arguments={"params": mock_ai_synthesis_markdown_input}
            )
    assert result and result.content
    json_text = result.content[0].text

    err = "Directly raised exception"
    msg = f"The `scholar_flux_synthesize_research_summary` tool did not return a valid result. {err}"
    assert msg in json_text


async def test_history_output_retrieval(
    mock_mcp_client_with_history,
    mock_ai_synthesis_output,
    mock_computer_literacy_relevance_search_output,
):
    """Verifies that the `HistoryService` correctly enables the retrieval of previously stored output history."""

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        recent_history_tool_output = await mcp_client.call_tool(
            name="scholar_flux_list_recent_history",
            arguments={"params": RecentHistoryToolInput(response_format="json")},
        )
    assert recent_history_tool_output and recent_history_tool_output.content
    recent_history_json_str = recent_history_tool_output.content[0].text
    recent_history_output = json.loads(recent_history_json_str)
    assert "recent_history" in recent_history_output
    recent_history_markdown = recent_history_output["recent_history"]

    input_hash_set = {search_input["input_hash"] for search_input in recent_history_markdown}

    assert mock_ai_synthesis_output.search_output.search_input.input_hash in input_hash_set
    assert mock_computer_literacy_relevance_search_output.relevance_search_input.input_hash in input_hash_set
    assert mock_ai_synthesis_output.synthesis_input.input_hash in input_hash_set


async def test_record_history_json_output_retrieval(
    mock_mcp_client_with_history,
    mock_ai_synthesis_output,
):
    """Verifies that the `HistoryService` correctly enables the retrieval of previously stored JSON record history."""

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        record_history_tool_output = await mcp_client.call_tool(
            name="scholar_flux_list_record_history",
            arguments={"params": RecordHistoryToolInput(response_format="json", max_records=25)},
        )
    assert record_history_tool_output and record_history_tool_output.content

    record_history_json_str = record_history_tool_output.content[0].text
    record_history_output = json.loads(record_history_json_str)
    assert "record_history" in record_history_output
    record_history_json = record_history_output["record_history"]

    record_hash_set = {record["record_hash"] for record in record_history_json}
    synthesis_record_hash_set = {record.record_hash for record in mock_ai_synthesis_output.search_output.records}

    # Verifies that all json record hashes exist in the original record list
    assert not record_hash_set.difference(synthesis_record_hash_set)


async def test_record_history_markdown_output_retrieval(
    mock_mcp_client_with_history,
    mock_ai_synthesis_output,
):
    """Verifies that the `HistoryService` correctly enables the retrieval of stored markdown record history."""

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        record_history_tool_output = await mcp_client.call_tool(
            name="scholar_flux_list_record_history",
            arguments={"params": RecordHistoryToolInput(response_format="markdown")},
        )
    assert record_history_tool_output and record_history_tool_output.content
    record_history_markdown_str = record_history_tool_output.content[0].text

    assert all(
        f"{i}. **{record.title}**" in record_history_markdown_str
        for i, record in enumerate(mock_ai_synthesis_output.search_output.records, start=1)
    )


async def test_get_history_search_output(
    mock_mcp_client_with_history,
    mock_ai_synthesis_output,
):
    """Verifies that search output history can be retrieved by input hash via `scholar_flux_get_history_output`."""
    mock_ai_search_output = mock_ai_synthesis_output.search_output
    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        last_associated_history_output = await mcp_client.call_tool(
            name="scholar_flux_get_history_output",
            arguments={
                "params": GetHistoryToolInput(
                    input_hash=mock_ai_search_output.search_input.input_hash, response_format="json"
                )
            },
        )

    assert last_associated_history_output
    search_output_json_str = last_associated_history_output.content[0].text
    search_output_fmt = SearchFormatter.format(mock_ai_search_output, "json")
    assert search_output_json_str == search_output_fmt


async def test_get_history_relevance_search_output(
    mock_mcp_client_with_history,
    mock_computer_literacy_relevance_search_output,
):
    """Verifies that relevance search outputs can be retrieved by input hash via `scholar_flux_get_history_output`."""
    relevance_search_input = mock_computer_literacy_relevance_search_output.relevance_search_input
    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        last_associated_history_output = await mcp_client.call_tool(
            name="scholar_flux_get_history_output",
            arguments={
                "params": GetHistoryToolInput(input_hash=relevance_search_input.input_hash, response_format="json")
            },
        )

    assert last_associated_history_output
    relevance_search_output_json_str = last_associated_history_output.content[0].text
    relevance_search_output_fmt = RelevanceSearchFormatter.format(
        mock_computer_literacy_relevance_search_output, "json"
    )
    assert relevance_search_output_json_str == relevance_search_output_fmt


async def test_get_history_synthesis_output(
    mock_mcp_client_with_history,
    mock_ai_synthesis_output,
):
    """Verifies that synthesis output history can be retrieved by input hash via `scholar_flux_get_history_output`."""

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        last_associated_history_output = await mcp_client.call_tool(
            name="scholar_flux_get_history_output",
            arguments={
                "params": GetHistoryToolInput(
                    input_hash=mock_ai_synthesis_output.synthesis_input.input_hash, response_format="json"
                )
            },
        )

    assert last_associated_history_output
    synthesis_output_json_str = last_associated_history_output.content[0].text
    synthesis_output_fmt = SynthesisFormatter.format(mock_ai_synthesis_output, "json")
    assert synthesis_output_json_str == synthesis_output_fmt


async def test_get_nonexistent_history_output(
    mock_mcp_client_with_history,
):
    """Verifies that `scholar_flux_get_history_output` gracefully indicates when an input hash doesn't exist."""

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        last_associated_history_output = await mcp_client.call_tool(
            name="scholar_flux_get_history_output",
            arguments={"params": GetHistoryToolInput(input_hash="a-non-existent-hash", response_format="json")},
        )

    assert last_associated_history_output
    synthesis_output_json_str = last_associated_history_output.content[0].text
    assert "A valid history output item could not be found given the current parameters." in synthesis_output_json_str


async def test_history_output_clear(
    mock_mcp_client_with_history,
    mock_ai_synthesis_output,
):
    """Verifies that the history cache is cleared when the `scholar_flux_clear_history` tool is called."""

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        clear_history_tool_result = await mcp_client.call_tool(
            name="scholar_flux_clear_history",
            arguments={"params": ClearHistoryToolInput()},
        )

    cleared_history_tool_text = clear_history_tool_result.content[0].text

    assert "The output history cache was successfully cleared." in cleared_history_tool_text

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        recent_history_tool_output = await mcp_client.call_tool(
            name="scholar_flux_list_recent_history",
            arguments={"params": RecentHistoryToolInput(response_format="json")},
        )

    assert recent_history_tool_output and recent_history_tool_output.content
    cleared_history_output_json = json.loads(recent_history_tool_output.content[0].text)
    assert cleared_history_output_json["recent_history"] == "No history data is available."


async def test_record_history_with_fuzzy_similarity(
    mock_mcp_client_with_history,
    mock_ai_synthesis_output,
):
    """Verifies that the mcp tool successfully uses fuzzy similarity to filter and reorder records."""
    assert mock_ai_synthesis_output.search_output.records  # sanity check

    first_topic = mock_ai_synthesis_output.search_output.records[0].topic

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        record_history_tool_output = await mcp_client.call_tool(
            name="scholar_flux_list_record_history",
            arguments={
                "params": RecordHistoryToolInput(response_format="json", topic=first_topic, similarity_threshold=0.0)
            },
        )
    assert record_history_tool_output and record_history_tool_output.content
    record_history_json_str = record_history_tool_output.content[0].text
    record_history_output = json.loads(record_history_json_str)
    assert "record_history" in record_history_output
    record_history_list = record_history_output["record_history"]

    assert len(mock_ai_synthesis_output.search_output.records) <= len(record_history_list)
    first_title, first_abstract = record_history_list[0]["title"], record_history_list[0]["abstract"]
    assert first_title and first_title in first_topic
    assert first_abstract and first_abstract in first_topic


async def test_recent_history_output_searches_with_fuzzy_similarity(
    mock_mcp_client_with_history,
    mock_ai_synthesis_output,
):
    """Verifies that the `scholar_flux_list_recent_history` tool can use similarity to filter and reorder outputs."""
    first_topic = mock_ai_synthesis_output.topic

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        recent_history_tool_output = await mcp_client.call_tool(
            name="scholar_flux_list_recent_history",
            arguments={"params": RecentHistoryToolInput(response_format="json", topic=first_topic)},
        )

    assert recent_history_tool_output and recent_history_tool_output.content
    recent_history_json_str = recent_history_tool_output.content[0].text
    recent_history_output = json.loads(recent_history_json_str)
    assert "recent_history" in recent_history_output
    recent_history_list = recent_history_output["recent_history"]

    assert ResearchTopic.create(recent_history_list[0]).topic == first_topic
    assert len(recent_history_list[0]) > 1  # should be reranked, not filtered (no similarity_threshold)


async def test_health_check_gracefully_handles_unexpected_exceptions(
    mock_mcp_client_with_history,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    monkeypatch,
):
    """Verifies that the `scholar_flux_health_check` tool formats and transmits messages on unexpected errors."""
    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)

    err = "Directly raised exception"
    monkeypatch.setattr(mock_ai_synthesis_app_context.cache_service, "check_health", raise_error(RuntimeError, err))

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        recent_health_check_tool_markdown_output = await mcp_client.call_tool(
            name="scholar_flux_health_check",
            arguments={"params": HealthCheckToolInput(response_format="markdown")},
        )
    recent_health_check_markdown_str = recent_health_check_tool_markdown_output.content[0].text
    assert err in recent_health_check_markdown_str


async def test_unexpected_health_status_from_bad_history_service(
    mock_mcp_client_with_history,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    monkeypatch,
):
    """Verifies that the history service handles unexpected errors during initialization to survive
    `.check_health()`."""
    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)

    history_service = HistoryService(url="sqlite:///:memory:", enable=True)
    err = "Directly raised exception"
    monkeypatch.setattr(history_service, "verify_connection", raise_error(RuntimeError, err))
    monkeypatch.setattr(mock_ai_synthesis_app_context, "history_service", history_service)

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        recent_health_check_tool_markdown_output = await mcp_client.call_tool(
            name="scholar_flux_health_check",
            arguments={"params": HealthCheckToolInput(response_format="markdown")},
        )

    assert recent_health_check_tool_markdown_output and recent_health_check_tool_markdown_output.content
    recent_health_check_markdown_str = recent_health_check_tool_markdown_output.content[0].text
    assert re.search(r"\*\*Status:\*\* [Uu]nhealthy", recent_health_check_markdown_str) is not None
    assert re.search(rf"\*\*Error:\*\* {err}", recent_health_check_markdown_str) is not None


async def test_unexpected_health_status_from_bad_cache(
    mock_mcp_client_with_history,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    monkeypatch,
):
    """Verifies that the cache service handles unexpected errors during initialization to survive `.check_health()`."""
    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)
    cache_service = CacheService()
    err = "Directly raised exception"
    monkeypatch.setattr(
        "scholar_flux_mcp.services.cache_service.CachedSessionManager.__init__", raise_error(RuntimeError, err)
    )
    monkeypatch.setattr(mock_ai_synthesis_app_context, "cache_service", cache_service)

    await cache_service.initialize()

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        recent_health_check_tool_markdown_output = await mcp_client.call_tool(
            name="scholar_flux_health_check",
            arguments={"params": HealthCheckToolInput(response_format="markdown")},
        )

    assert recent_health_check_tool_markdown_output and recent_health_check_tool_markdown_output.content
    recent_health_check_markdown_str = recent_health_check_tool_markdown_output.content[0].text
    assert re.search(r"\*\*Status:\*\* [Uu]nhealthy", recent_health_check_markdown_str) is not None
    assert (
        re.search(rf"\*\*Error:\*\* Session manager initialization failed: {err}", recent_health_check_markdown_str)
        is not None
    )


async def test_list_recent_history_gracefully_handles_unexpected_exceptions(
    mock_mcp_client_with_history,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    monkeypatch,
):
    """Verifies that the `scholar_flux_list_recent_history` tool formats and transmits messages on unexpected errors."""
    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)

    err = "Directly raised exception"
    monkeypatch.setattr(
        mock_ai_synthesis_app_context.history_service,
        "retrieve_recent_history",
        raise_error(RuntimeError, err),
    )

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        recent_history_tool_markdown_output = await mcp_client.call_tool(
            name="scholar_flux_list_recent_history",
            arguments={"params": RecentHistoryToolInput(response_format="markdown")},
        )
    recent_history_markdown_str = recent_history_tool_markdown_output.content[0].text
    assert err in recent_history_markdown_str


async def test_list_record_history_gracefully_handles_unexpected_exceptions(
    mock_mcp_client_with_history,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    monkeypatch,
):
    """Verifies that the `scholar_flux_list_record_history` tool formats and transmits messages on unexpected errors."""
    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)

    err = "Directly raised exception"
    monkeypatch.setattr(
        mock_ai_synthesis_app_context.history_service,
        "retrieve_record_history",
        raise_error(RuntimeError, err),
    )

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        record_history_tool_markdown_output = await mcp_client.call_tool(
            name="scholar_flux_list_record_history",
            arguments={"params": RecordHistoryToolInput(response_format="markdown")},
        )
    record_history_markdown_str = record_history_tool_markdown_output.content[0].text
    assert err in record_history_markdown_str


async def test_get_history_output_gracefully_handles_unexpected_exceptions(
    mock_mcp_client_with_history,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    monkeypatch,
):
    """Verifies that the `scholar_flux_get_history_output` tool formats and transmits messages on unexpected errors."""
    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)

    err = "Directly raised exception"
    monkeypatch.setattr(
        mock_ai_synthesis_app_context.history_service,
        "retrieve_recent_history",
        raise_error(RuntimeError, err),
    )

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        history_output_tool_markdown_output = await mcp_client.call_tool(
            name="scholar_flux_get_history_output",
            arguments={"params": GetHistoryToolInput(response_format="markdown", input_hash="0123456789101112")},
        )
    history_output_markdown_str = history_output_tool_markdown_output.content[0].text
    assert err in history_output_markdown_str


async def test_clear_history_gracefully_handles_unexpected_exceptions(
    mock_mcp_client_with_history,
    mock_ai_synthesis_app_context,
    reload_mcp_module,
    mock_lifespan,
    monkeypatch,
):
    """Verifies that the `scholar_flux_clear_history` tool formats and transmits messages on unexpected errors."""
    from scholar_flux_mcp.server.main import mcp

    monkeypatch.setattr(mcp._mcp_server, "lifespan", mock_lifespan)

    err = "Directly raised exception"
    monkeypatch.setattr(
        mock_ai_synthesis_app_context.history_service,
        "clear",
        raise_error(RuntimeError, err),
    )

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        clear_history_tool_markdown_output = await mcp_client.call_tool(
            name="scholar_flux_clear_history",
            arguments={"params": ClearHistoryToolInput()},
        )
    clear_history_markdown_str = clear_history_tool_markdown_output.content[0].text
    assert err in clear_history_markdown_str


async def test_health_check_status_healthy_on_startup(
    mock_mcp_client_with_pydantic_ai_test_mocks,
):
    """Verifies that the `HealthStatus` of the package correctly reflects as `healthy`.

    For healthy status, the ScholarFlux MCP server requires that an enabled service (cache/history) is either enabled
    and accessible or disabled altogether. For example, enabling a redis server on a fictional port -2345678 should show
    `unhealthy` whereas a disabled cache server should instead show `disabled` and treated differently from `unhealthy`.

    """

    async with Client(transport=mock_mcp_client_with_pydantic_ai_test_mocks) as mcp_client:
        recent_health_check_tool_output = await mcp_client.call_tool(
            name="scholar_flux_health_check",
            arguments={"params": HealthCheckToolInput(response_format="json")},
        )
    assert recent_health_check_tool_output and recent_health_check_tool_output.content
    recent_health_check_json_str = recent_health_check_tool_output.content[0].text
    recent_health_check_output = json.loads(recent_health_check_json_str)
    assert "status" in recent_health_check_output and recent_health_check_output["status"] == "healthy"

    cache_result = recent_health_check_output["services"]["cache"]
    assert cache_result and isinstance(cache_result, dict)
    cache_details = cache_result["details"]

    assert (
        cache_details.get("initialized") is True
        and cache_details.get("namespace") is not None
        and cache_details.get("user_agent") is not None
    )
    assert cache_details.get("session_cache_backend") in SESSION_CACHE_BACKENDS
    assert cache_details.get("response_cache_storage") in RESPONSE_CACHE_STORAGE_DEVICES

    history_result = recent_health_check_output["services"]["history"]
    assert history_result and isinstance(history_result, dict)
    history_details = history_result["details"]

    assert history_details.get("initialized") is True and history_details.get("url") is not None


async def test_health_check_status_markdown_format(
    mock_mcp_client_with_history,
):
    """Verifies that the `HealthStatus` of the package is correctly displayed in markdown format."""

    async with Client(transport=mock_mcp_client_with_history) as mcp_client:
        recent_health_check_tool_markdown_output = await mcp_client.call_tool(
            name="scholar_flux_health_check",
            arguments={"params": HealthCheckToolInput(response_format="markdown")},
        )

    assert recent_health_check_tool_markdown_output and recent_health_check_tool_markdown_output.content
    recent_health_check_markdown_str = recent_health_check_tool_markdown_output.content[0].text
    assert re.search(r"\*\*Status:\*\* [Hh]ealthy", recent_health_check_markdown_str) is not None

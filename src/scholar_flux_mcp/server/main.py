"""ScholarFlux MCP Server main entry point.

This module initializes the FastMCP server with all tools and manages the application lifecycle including cache
initialization.

"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from scholar_flux_mcp.exceptions.import_exceptions import CoreDependencyImportError, MCPImportError
from scholar_flux_mcp.exceptions.mcp_server_exceptions import MCPServerInitializationException

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    # Note: FastMCP is the current name until the release of `>=mcp 2.x.x`. MCPServer renaming expected 2026 Q1
    from mcp.server.fastmcp import Context, FastMCP
    from mcp.server.session import ServerSession
    from mcp.types import ToolAnnotations
else:
    try:
        from mcp.server.fastmcp import Context, FastMCP
        from mcp.server.session import ServerSession
        from mcp.types import ToolAnnotations
    except ImportError:
        Context = FastMCP = ServerSession = None
        ToolAnnotations = dict

from scholar_flux_mcp.models import (
    RelevanceSearchOutput,
    SearchOutput,
    SynthesisOutput,
)
from scholar_flux_mcp.server.app_context import AppContext
from scholar_flux_mcp.server.io.base import BaseFormatter
from scholar_flux_mcp.server.io.health_check import HealthCheckFormatter, HealthCheckToolInput
from scholar_flux_mcp.server.io.history import (
    ClearHistoryToolInput,
    GetHistoryToolInput,
    OutputHistoryFormatter,
    RecentHistoryToolInput,
    RecordHistoryFormatter,
    RecordHistoryToolInput,
)

# Import tool input models
from scholar_flux_mcp.server.io.providers import (
    ListProvidersInput,
    ProvidersFormatter,
)
from scholar_flux_mcp.server.io.relevance_search import RelevanceSearchFormatter, RelevanceSearchToolInput
from scholar_flux_mcp.server.io.search import SearchFormatter, SearchToolInput
from scholar_flux_mcp.server.io.synthesis import SynthesisFormatter, SynthesisToolInput
from scholar_flux_mcp.server.transport import MCPTransports, TransportSettings, TransportType
from scholar_flux_mcp.services import (
    CacheService,
    GroundingService,
    HistoryService,
    ProviderService,
    RelevanceSearchService,
    SearchService,
    SynthesisService,
)

logger = logging.getLogger(__name__)

# =========================================================================
# MCP SERVER LOGIC
# =========================================================================


def create_server(transport: TransportType | TransportSettings | MCPTransports | None = None) -> FastMCP:
    """Create and configure the ScholarFlux MCP server.

    Args:
        transport (TransportType | TransportSettings | MCPTransports | None):
            Transport settings that can be defined directly from the `MCPTransports` enum, the `TransportSettings`,
            or as a `sse`, `stdio`, or `streamable-http` string.

        Note: For consistency, environment variables are evaluated on import.

    Returns:
        FastMCP: Configured MCP server instance.

    """
    if FastMCP is None or Context is None:
        raise MCPImportError("The MCP Server cannot be activated. The python `mcp` module is not installed.")

    transport_settings = MCPTransports.get_transport_setting(transport) or MCPTransports.get_default()

    # Initializes cache, history and provider services
    cache_service = CacheService()

    # History service: enabled by default but can be disabled via environment
    history_service = HistoryService()

    # Provider service: used to retrieve information about the APIs supported by ScholarFlux
    provider_service = ProviderService()

    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
        """Manage application lifecycle.

        Initializes services on startup and cleans up on shutdown.
        Services are made available to tools through the lifespan state.

        Args:
            server (FastMCP): The `FastMCP` server instance

        Yields:
            AppContext:
                The services used by ScholarFlux MCP for record retrieval, synthesis, caching, and other functionality.

        """
        logger.info("ScholarFlux MCP Server starting...")

        try:
            # Initialize cache service
            await cache_service.initialize()
            logger.info("Cache service initialized")

            # Initialize dependent services
            search_service = SearchService(cache_service, history_service=history_service)
            relevance_search_service = RelevanceSearchService(search_service, history_service=history_service)
            grounding_service = GroundingService()
            synthesis_service = SynthesisService(
                relevance_search_service, grounding_service, history_service=history_service
            )

            logger.info("All services initialized")

            # Yield state to tools
            yield AppContext(
                search_service=search_service,
                relevance_search_service=relevance_search_service,
                synthesis_service=synthesis_service,
                provider_service=provider_service,
                history_service=history_service,
            )

        except CoreDependencyImportError:
            raise
        except Exception as e:
            # Log and raise if an irrecoverable error occurs
            logger.error(f"Failed to initialize the MCPServer: {e}")
            raise MCPServerInitializationException(
                f"The MCPServer failed to initialize due to an unexpected error: {e}"
            ) from e

        finally:
            # Cleanup
            logger.info("ScholarFlux MCP Server shutting down...")

    # Create FastMCP server with lifespan
    server = FastMCP("ScholarFlux MCP", lifespan=app_lifespan, **transport_settings.options)

    # Register tools for the MCP Server. Service access is deferred until runtime through the lifespan context
    _register_tools(server)

    return server


def _register_tools(server: FastMCP) -> None:
    """Registers tools used by the MCP Server to interact with the ScholarFlux base package.

    Each MCP tool and its assigned schema is registered immediately and access the required
    `SearchService`, `SynthesisService`, and `ProviderService` at runtime through the
    `ctx.request_context.lifespan_context` after lifespan initialization.

    Args:
        server (FastMCP): The FastMCP server instance.

    """

    @server.tool(
        name="scholar_flux_record_search",
        annotations=ToolAnnotations(
            title="Multi-Provider Academic Record Search",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def scholar_flux_record_search(
        params: SearchToolInput,
        ctx: Context[ServerSession, AppContext],
    ) -> str:
        """Search for academic literature across multiple scholarly databases.

        This tool queries academic databases (PubMed, PLOS, OpenAlex, Crossref, etc.)
        and returns normalized results with consistent field names across all providers.

        Use this tool when you need to:
        - Find research on a specific topic
        - Gather literature for a systematic review
        - Identify recent publications in a field
        - Search for academic records by keywords, authors, or concepts

        Args:
            params (SearchToolInput): Search parameters including the queries, providers, and record filters.
            ctx (Context[ServerSession, AppContext]): MCP context with access to lifespan state.

        Returns:
            Search results as markdown summary or JSON data.

        """
        logger.info(f"Search requested: {params.query}")

        try:
            # Get search service from lifespan context
            lifespan_ctx = ctx.request_context.lifespan_context
            search_service = lifespan_ctx.search_service

            search_input = params.as_search_input()

            result = await search_service.search(
                search_input,
                from_history_cache=params.from_history_cache,
                store_history_cache=params.store_history_cache,
            )

            # Format output
            return SearchFormatter.format(
                result, response_format=params.response_format, max_records=params.max_records
            )

        except Exception as e:
            msg = f"The `scholar_flux_record_search` tool failed to retrieve and process valid academic records. {e}"
            return SearchFormatter.format_error_message(e, msg, response_format=params.response_format)

    @server.tool(
        name="scholar_flux_record_relevance_search",
        annotations=ToolAnnotations(
            title="Multi-Provider Academic Record Relevance Search",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def scholar_flux_record_relevance_search(
        params: RelevanceSearchToolInput,
        ctx: Context[ServerSession, AppContext],
    ) -> str:
        """Retrieve research findings using embeddings to sort records via topic similarity scoring.

        This tool searches academic databases for papers related to your research question, then uses an embedding model
        to rerank the findings by record-topic embedding cosine similarity given the question, listed categories, and
        the query set used for record retrieval.

        Use this tool when you need to:

        - Understand the current state of research on a topic
        - Identify studies that most similar to the topic of interest.
        - Filter out unrelated studies as a preprocessing step for literature review or meta analyses

        Args:
            params (RelevanceSearchToolInput):
                Relevance search parameters, including the question, categories, and queries.
            ctx (Context[ServerSession, AppContext]): MCP context with access to lifespan state.

        Returns:
            str: Synthesized findings as a markdown report or JSON.

        """
        try:
            # Get services from lifespan context
            lifespan_ctx = ctx.request_context.lifespan_context
            relevance_search_service = lifespan_ctx.relevance_search_service

            relevance_search_input = params.as_relevance_search_input()

            # Runs the relevance search pipeline: search -> dedup -> embed ranking -> output if uncached
            result = await relevance_search_service.relevance_search(
                relevance_search_input,
                from_history_cache=params.from_history_cache,
                store_history_cache=params.store_history_cache,
            )

            return RelevanceSearchFormatter.format(
                result, response_format=params.response_format, max_records=params.max_records
            )

        except Exception as e:
            msg = f"The `scholar_flux_record_relevance_search` tool did not return a valid result. {e}"
            return RelevanceSearchFormatter.format_error_message(e, msg, response_format=params.response_format)

    @server.tool(
        name="scholar_flux_synthesize_research_summary",
        annotations=ToolAnnotations(
            title="Synthesize Research Summary",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def scholar_flux_synthesize_research_summary(
        params: SynthesisToolInput,
        ctx: Context[ServerSession, AppContext],
    ) -> str:
        """Synthesize research findings using AI-powered analysis.

        This tool searches academic databases for papers related to your research
        question, then uses an AI model to synthesize the findings into a coherent,
        evidence-based response with citations.

        Use this tool when you need to:
        - Get an evidence-based answer to a general research question
        - Understand the current state of research on a topic
        - Identify key findings and trends in scientific literature
        - Generate a literature synthesis with proper citations

        Args:
            params (SynthesisToolInput): Synthesis parameters including question and categories.
            ctx (Context[ServerSession, AppContext]): MCP context with access to lifespan state.

        Returns:
            str: Synthesized findings as a markdown report or JSON.

        """
        logger.info(f"Synthesis requested: {params.question[:50]}...")

        try:
            # Get services from lifespan context
            lifespan_ctx = ctx.request_context.lifespan_context
            synthesis_service = lifespan_ctx.synthesis_service

            synthesis_input = params.as_synthesis_input()

            # Runs the full pipeline: search -> dedup -> embed ranking -> synthesizing -> ground -> output if uncached
            result = await synthesis_service.synthesize(
                synthesis_input,
                from_history_cache=params.from_history_cache,
                store_history_cache=params.store_history_cache,
            )

            # Formats the result as raw JSON or markdown format
            return SynthesisFormatter.format(result, response_format=params.response_format)

        except Exception as e:
            msg = f"The `scholar_flux_synthesize_research_summary` tool did not return a valid result. {e}"
            return SynthesisFormatter.format_error_message(e, msg, response_format=params.response_format)

    @server.tool(
        name="scholar_flux_list_recent_history",
        annotations=ToolAnnotations(
            title="List the most recently recorded history outputs",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def scholar_flux_list_recent_history(
        params: RecentHistoryToolInput,
        ctx: Context[ServerSession, AppContext],
    ) -> str:
        """List all recent outputs within a particular time period.

        Args:
            params (RecentHistoryToolInput): Recent history input parameters.
            ctx (Context[ServerSession, AppContext]): MCP context.

        Returns:
            str: A list of all recent outputs within the specified time period

        """
        lifespan_ctx = ctx.request_context.lifespan_context
        tool_history_service = lifespan_ctx.history_service

        try:
            async with tool_history_service.initialize_session() as session:
                recent_history_list = await tool_history_service.retrieve_recent_history(
                    session=session,
                    research_tool_type=params.research_tool_type,
                    ttl=params.ttl,
                    max_history=params.max_history,
                    successful_only=params.successful_only,
                    topic=params.topic,
                    similarity_threshold=params.similarity_threshold,
                )
                return OutputHistoryFormatter.format(recent_history_list, response_format=params.response_format)

        except Exception as e:
            msg = f"The `scholar_flux_list_recent_history` tool failed to list the recently stored output history. {e}"
            return RecordHistoryFormatter.format_error_message(e, msg, response_format=params.response_format)

    @server.tool(
        name="scholar_flux_list_record_history",
        annotations=ToolAnnotations(
            title="List recently stored records across all searches",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def scholar_flux_list_record_history(
        params: RecordHistoryToolInput,
        ctx: Context[ServerSession, AppContext],
    ) -> str:
        """List all recent records within a particular time period.

        Args:
            params (RecordHistoryToolInput): Record history input parameters.
            ctx (Context[ServerSession, AppContext]): MCP context.

        Returns:
            str: A list of all recent records within the specified time period

        """
        lifespan_ctx = ctx.request_context.lifespan_context
        tool_history_service = lifespan_ctx.history_service

        try:
            # Handles filtering via TTL and max records
            record_history_list = await tool_history_service.retrieve_record_history(
                ttl=params.ttl,
                max_history=params.max_records,
                topic=params.topic,
                similarity_threshold=params.similarity_threshold,
            )
            return RecordHistoryFormatter.format(
                record_history_list,
                response_format=params.response_format,
                display_full_text=params.display_full_text,
            )

        except Exception as e:
            msg = f"The `scholar_flux_list_record_history` tool failed to list the stored record history. {e}"
            return RecordHistoryFormatter.format_error_message(e, msg, response_format=params.response_format)

    @server.tool(
        name="scholar_flux_get_history_output",
        annotations=ToolAnnotations(
            title="Retrieves the most recent successfully processed output given the input hash",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def scholar_flux_get_history_output(
        params: GetHistoryToolInput,
        ctx: Context[ServerSession, AppContext],
    ) -> str:
        """List the most recent history output associated with the tool type and input hash.

        Args:
            params (GetHistoryToolInput): History input parameters.
            ctx (Context[ServerSession, AppContext]): MCP context.

        Returns:
            str: The most recent history input associated with the input hash if identified.

        """
        lifespan_ctx = ctx.request_context.lifespan_context
        tool_history_service = lifespan_ctx.history_service

        try:
            async with tool_history_service.initialize_session() as session:
                recent_history_list = await tool_history_service.retrieve_recent_history(
                    session=session,
                    input_hash=params.input_hash,
                    research_tool_type=params.research_tool_type,
                    ttl=params.ttl,
                    max_history=1,
                    successful_only=params.successful_only,
                    raise_on_error=True,
                )

                history_output = recent_history_list[0].to_output() if recent_history_list else None

            match history_output:
                case SynthesisOutput():
                    return SynthesisFormatter.format(history_output, response_format=params.response_format)
                case RelevanceSearchOutput():
                    return RelevanceSearchFormatter.format(history_output, response_format=params.response_format)
                case SearchOutput():
                    return SearchFormatter.format(history_output, response_format=params.response_format)
                case _:
                    return BaseFormatter.format_message(
                        "A valid history output item could not be found given the current parameters.",
                        response_format=params.response_format,
                    )

        except Exception as e:
            msg = (
                "The `scholar_flux_get_history_output` tool failed to retrieve the most recently stored output history "
                f"item: {e}"
            )
            return BaseFormatter.format_error_message(e, msg, response_format=params.response_format)

    @server.tool(
        name="scholar_flux_clear_history",
        annotations=ToolAnnotations(
            title="Clears the history of all previously recorded outputs",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def scholar_flux_clear_history(
        params: ClearHistoryToolInput,
        ctx: Context[ServerSession, AppContext],
    ) -> str:
        """Clears the history of all record search, relevance search, and synthesis outputs.

        Args:
            params (ClearHistoryToolInput): Parameters for clearing the research tool output history.
            ctx (Context[ServerSession, AppContext]): MCP context.

        Returns:
            str: JSON string indicating whether the output history was successfully cleared.

        """
        lifespan_ctx = ctx.request_context.lifespan_context
        tool_history_service = lifespan_ctx.history_service

        try:
            await tool_history_service.clear(raise_on_error=True)
            return "The output history cache was successfully cleared."

        except Exception as e:
            msg = f"The `scholar_flux_clear_history` tool failed to clear the history cache due to an error: {e}"
            return BaseFormatter.format_error_message(e, msg)  # markdown by default

    @server.tool(
        name="scholar_flux_list_providers",
        annotations=ToolAnnotations(
            title="List Academic Database Providers",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def scholar_flux_list_providers(
        params: ListProvidersInput,
        ctx: Context[ServerSession, AppContext],
    ) -> str:
        """List available academic database providers with their capabilities.

        Args:
            params (ListProvidersInput): Format parameters.
            ctx (Context[ServerSession, AppContext]): MCP context.

        Returns:
            str: Provider list as markdown or JSON.

        """
        lifespan_ctx = ctx.request_context.lifespan_context
        tool_provider_service = lifespan_ctx.provider_service

        try:
            live_providers = await tool_provider_service.get_providers() if tool_provider_service else []
            return ProvidersFormatter.format(live_providers, response_format=params.response_format)

        except Exception as e:
            msg = f"The `scholar_flux_list_providers` tool failed to list all supported providers: {e}"
            return ProvidersFormatter.format_error_message(e, msg, response_format=params.response_format)

    @server.tool(
        name="scholar_flux_health_check",
        annotations=ToolAnnotations(
            title="Verify the health status of services used across server tools",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def scholar_flux_health_check(
        params: HealthCheckToolInput,
        ctx: Context[ServerSession, AppContext],
    ) -> str:
        """Indicates the health status of core ScholarFlux MCP server components.

        Args:
            params (HealthCheckToolInput): Health check parameters.
            ctx (Context[ServerSession, AppContext]): MCP context with access to lifespan state.

        Returns:
            str: Health status as JSON or markdown.

        """
        try:
            lifespan_ctx = ctx.request_context.lifespan_context
            mcp_health_check = await lifespan_ctx.check_health()

            return HealthCheckFormatter.format(mcp_health_check, response_format=params.response_format)
        except Exception as e:
            msg = f"The `scholar_flux_health_check` tool failed to determine the health of the MCP server: {e}"
            return HealthCheckFormatter.format_error_message(e, msg, response_format=params.response_format)


# Globally Load options and define the MCP server instance
MCP_TRANSPORT_SETTINGS: TransportSettings = MCPTransports.get_default(verbose=True)
mcp = create_server(transport=MCP_TRANSPORT_SETTINGS)


def main() -> None:
    """Run the MCP server."""
    if FastMCP is None or Context is None:
        raise MCPImportError("The MCP Server cannot be activated. The python `mcp` module is not installed.")

    logger.info(f"Starting ScholarFlux MCP Server (transport: '{MCP_TRANSPORT_SETTINGS.name}')")
    mcp.run(transport=MCP_TRANSPORT_SETTINGS.name)  # default stdio


if __name__ == "__main__":
    main()

__all__ = ["main", "mcp"]

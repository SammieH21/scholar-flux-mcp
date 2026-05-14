"""Synthesis service for ScholarFlux MCP server.

Uses PydanticAI to synthesize research findings from normalized academic records. Single responsibility: AI-powered
synthesis.

"""

from __future__ import annotations

import logging
from typing import Any

from scholar_flux_mcp.agents import (
    RecordTopicSimilarityEmbedder,
    SynthesisAgent,
    SynthesisAgentOutput,
    SynthesisContext,
)
from scholar_flux_mcp.exceptions import (
    AgentException,
    EvidenceGroundingException,
    PydanticAIImportError,
    RecordDeduplicationException,
)
from scholar_flux_mcp.models import (
    EvidenceGroundingInput,
    RelevanceSearchInput,
    SynthesisInput,
    SynthesisOutput,
)
from scholar_flux_mcp.services.base_research_service import BaseResearchService
from scholar_flux_mcp.services.grounding_service import GroundingService
from scholar_flux_mcp.services.history_service import HistoryService
from scholar_flux_mcp.services.relevance_search_service import RelevanceSearchService
from scholar_flux_mcp.services.search_service import SearchService

logger = logging.getLogger(__name__)


class SynthesisService(BaseResearchService):
    """Service for performing research literature synthesis using PydanticAI.

    Orchestrates the synthesis workflow:
    1. Search for relevant records via SearchService
    2. Filter and rank by relevance
    3. Use PydanticAI agent to synthesize findings
    4. Structure output with citations

    """

    def __init__(
        self,
        relevance_search_service: RelevanceSearchService,
        grounding_service: GroundingService | None = None,
        *,
        synthesis_agent: SynthesisAgent | None = None,
        **kwargs: Any,
    ):
        """Initialize synthesis service.

        Args:
            relevance_search_service (RelevanceSearchService):
                The service used for record retrieval and reranking via the configured embedding model.
            grounding_service (GroundingService): The service used for verifying AI-generated synthesis output.
            synthesis_agent (SynthesisAgent):
                The `PydanticAI` agent used to generate the research synthesis given the provided input parameters.
            **kwargs: Additional keyword arguments used to initialize the BaseResearchService.

        """
        self.relevance_search_service = relevance_search_service
        self.grounding_service = grounding_service or GroundingService()
        self.synthesis_agent = synthesis_agent or SynthesisAgent()

        super().__init__(**kwargs)

    @classmethod
    def create(
        cls,
        search_service: SearchService,
        grounding_service: GroundingService | None = None,
        *,
        record_topic_similarity_embedder: RecordTopicSimilarityEmbedder | None = None,
        synthesis_agent: SynthesisAgent | None = None,
        history_service: HistoryService | None = None,
    ) -> SynthesisService:
        """Factory method for creating a new SynthesisService from its core components.

        Args:
            search_service (SearchService):
                A `SearchService` for requesting academic records from APIs.
            grounding_service (GroundingService | None):
                The grounding service used to verify AI-generated synthesis output.
            record_topic_similarity_embedder (RecordTopicSimilarityEmbedder | None):
                The record-topic similarity embedder used to rerank and filter records by topic similarity.
            synthesis_agent (SynthesisAgent | None):
                The `PydanticAI` agent used to generate the research synthesis given the provided input parameters.

        """
        relevance_search_service = RelevanceSearchService(
            search_service=search_service,
            record_topic_similarity_embedder=record_topic_similarity_embedder,
            history_service=history_service,
        )
        return cls(
            relevance_search_service,
            grounding_service=grounding_service,
            synthesis_agent=synthesis_agent,
            history_service=history_service,
        )

    @property
    def relevance_search_service(self) -> RelevanceSearchService:
        """Returns the relevance_search service used to ground LLM responses in facts."""
        if not self._relevance_search_service:
            raise ValueError("A relevance_search service has not yet been assigned.")
        return self._relevance_search_service

    @relevance_search_service.setter
    def relevance_search_service(self, relevance_search_service: RelevanceSearchService) -> None:
        """Sets the RelevanceSearchService with validation.

        Used on initialization to verify the structure of the service.

        """
        self._validate_service_dependency(relevance_search_service, RelevanceSearchService, allow_missing=False)
        self._relevance_search_service = relevance_search_service

    @property
    def search_service(self) -> SearchService:
        """Returns the search service that is used to retrieve data from Academic Databases."""
        return self.relevance_search_service.search_service

    @property
    def record_topic_similarity_embedder(self) -> RecordTopicSimilarityEmbedder:
        """Returns the `RecordTopicSimilarityEmbedder` used to embed and calculate the record-topic similarity."""
        return self.relevance_search_service.record_topic_similarity_embedder

    @property
    def grounding_service(self) -> GroundingService:
        """Returns the grounding service used to ground LLM responses in facts."""
        if not self._grounding_service:
            raise ValueError("A grounding service has not yet been assigned.")
        return self._grounding_service

    @grounding_service.setter
    def grounding_service(self, grounding_service: GroundingService) -> None:
        """Sets the GroundingService with validation.

        Used on initialization to verify the structure of the service.

        """
        self._validate_service_dependency(grounding_service, GroundingService, allow_missing=False)
        self._grounding_service = grounding_service

    @property
    def synthesis_agent(self) -> SynthesisAgent:
        """Returns the agent that synthesizes academic records into a research summary."""
        if not self._synthesis_agent:
            raise ValueError("A valid research synthesis agent has not yet been created.")
        return self._synthesis_agent

    @synthesis_agent.setter
    def synthesis_agent(self, agent: SynthesisAgent) -> None:
        """Validates that the received value is a `SynthesisAgent` before assignment."""
        self._validate_service_dependency(agent, SynthesisAgent, allow_missing=False)
        self._synthesis_agent = agent

    async def synthesize(
        self,
        params: SynthesisInput,
        from_history_cache: bool = False,
        store_history_cache: bool = False,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> SynthesisOutput:
        """Synthesizes research literature findings from the question and parameters provided in the `SynthesisInput`.

        Args:
            params (SynthesisInput):
                The validated parameters used to customize the research synthesis. Includes the question, queries,
                categories, and basic `SearchInput` options.
            from_history_cache (bool):
                Indicates whether the history cache should be used if available.
            store_history_cache (bool):
                Indicates whether the output should be stored within the history cache when retrieved.
            force_refresh (bool):
                Whether cached results should be refreshed while enabling record search and relevance_search cache
                retrieval.
            **kwargs: Additional keyword parameters passed to `MultiCoordinator.search_pages()` via the `SearchService`.

        Returns:
            Structured synthesis with evidence and citations.

        """
        try:
            self._validate_service_dependency(params, SynthesisInput, allow_missing=False)

            ### Cache Retrieval (If available) ###
            cached_synthesis_output = (
                await self.retrieve_history_cache(params, raise_on_error=False)
                if from_history_cache and not force_refresh and self.history_enabled
                else None
            )

            if cached_synthesis_output is not None:
                return cached_synthesis_output

            ### Retrieval and embedding phase ###
            relevance_search_input = RelevanceSearchInput.from_synthesis_params(params)
            relevance_search_output = await self.relevance_search_service.relevance_search(
                relevance_search_input, from_history_cache=from_history_cache, store_history_cache=False, **kwargs
            )

            ### Pydantic AI Agent Synthesis Generation ###
            # Create the `SynthesisContext`
            synthesis_records = relevance_search_output.indexed_records[: params.max_records]
            logger.info(f"Using {len(synthesis_records)} records for synthesis")
            synthesis_context = SynthesisContext(
                question=params.question,
                categories=params.categories,
                records=synthesis_records,
            )

            # Run the research synthesis with the `SynthesisAgent`
            agent_output = await self.synthesis_agent(synthesis_context)
            grounding_input = EvidenceGroundingInput(
                evidence_items=agent_output.evidence_summaries, source_records=synthesis_records
            )

            ### Grounding Phase ###

            # Convert the `SynthesisAgentOutput` to the post-processed `SynthesisOutput`
            grounding_output = self.grounding_service.ground_evidence(grounding_input)

            # Return each component of the synthesis gen. Non-core fields are not including during model serialization.
            synthesis_output = SynthesisOutput(
                synthesis_input=params,
                agent_output=agent_output,
                grounding_output=grounding_output,
                model_name=self.synthesis_agent.model_name,
                relevance_search_output=relevance_search_output,
            )

            # Store the results within the history cache when not already cached.
            if store_history_cache and self.history_enabled:
                await self.store_history_cache(synthesis_output, raise_on_error=False)

            return synthesis_output

        ### Error Handling ###
        except (RuntimeError, PydanticAIImportError):
            # Re-raises uncaught RuntimeErrors for inspection
            raise
        except (RecordDeduplicationException, EvidenceGroundingException) as e:
            # Represents internal errors that are unlikely to occur but should be logged and raised if they do occur
            task = "evidence grounding" if isinstance(e, EvidenceGroundingException) else "record deduplication"
            logger.error(f"Synthesis failed due to an internal error preventing {task}: {e}")
            raise
        except AgentException as e:
            logger.error(f"Synthesis failed due to an error during SynthesisAgent execution: {e}")
            return self._empty_synthesis(params, str(e))
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Return partial result on error
            return self._empty_synthesis(params, str(e))

    def _empty_synthesis(self, params: SynthesisInput, reason: str) -> SynthesisOutput:
        """Creates an empty synthesis result that indicates when an error has occurred.

        Args:
            params (SynthesisInput): The original synthesis parameters.
            reason (str): Indicates the reason for why the synthesis failed.

        Returns:
            SynthesisOutput: The output of the research synthesis with error information.

        """
        agent_output = SynthesisAgentOutput(
            synthesis=f"Unable to synthesize findings: {reason}",
            confidence_score=0.0,
            limitations=[reason],
            suggested_queries=[
                f"Try searching: {params.question}",
                "Consider broadening search categories",
                "Check if providers are accessible",
            ],
        )

        return SynthesisOutput(
            synthesis_input=params,
            agent_output=agent_output,
        )


__all__ = ["SynthesisService"]

"""Synthesis Agent for ScholarFlux MCP server.

Uses PydanticAI to synthesize research findings from normalized academic records. Single responsibility: AI-powered
synthesis.

"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import fields, is_dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from scholar_flux_mcp.agents.models import AgentABC, PydanticAIModelFactory
from scholar_flux_mcp.models import ServiceHealth, SynthesisAgentOutput, SynthesisContext
from scholar_flux_mcp.utils import SearchRecordPreprocessingUtils

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.models import Model
else:
    try:
        from pydantic_ai import Agent
        from pydantic_ai.exceptions import UserError
        from pydantic_ai.models import Model
    except ImportError:
        Agent = None
        Model = None
        UserError = Exception
from scholar_flux_mcp.exceptions import (
    AgentInitializationException,
    AgentUninitializedException,
    InvalidAgentParameterException,
    PydanticAIImportError,
)

logger = logging.getLogger(__name__)


class SynthesisAgent(AgentABC):
    """Helper class that encapsulates the logic used to create and interface with LLMs."""

    SYNTHESIS_INSTRUCTIONS: str = dedent(
        """
        You are an expert academic research synthesizer specializing in academic literature.
        Your task is to analyze academic records and synthesize their findings into a coherent,
        evidence-based response to research questions.

        ## Guidelines

        1. **Evidence-Based**: Only make claims supported by the provided records.
        2. **Citation by Index**: Reference academic records by their index number (record_index) ONLY
        3. **Referenced Text Extraction**: When creating an evidence summary, extract 1-3 sentences of verbatim supporting
           evidence from the record and include them in the `referenced_text` field for verification.
        4. **No Hallucination**: Do NOT invent, modify, or fabricate record titles, DOIs, or metadata
        5. **Nuance**: Acknowledge conflicting findings and limitations
        6. **Clarity**: Use clear, accessible language while maintaining scientific accuracy

        ## Output Requirements

        - Provide a comprehensive synthesis addressing the research question
        - List 3-7 key findings as bullet points
        - For evidence_summaries, use record_index (0-based integer) to reference records
          Example: {"record_index": 0, "referenced_text": "CBT shows 60% efficacy", "finding": "CBT demonstrates...", "relevance_score": 0.9}
        - Annotate statements in the synthesis summary with the record index in brackets when feasible
          Example: Most research agrees on X [21; 32; 7]. Mo also establishes a causal relationship with Y [34].
        - Assign a confidence score (0.0-1.0) based on evidence quality
        - Note any limitations in the available evidence
        - Suggest follow-up queries for deeper investigation

    """
    )

    def __init__(self, agent: Agent[SynthesisContext, SynthesisAgentOutput] | None = None):
        """Initializes a SynthesisAgent with either a predefined or a lazily created PydanticAI agent."""
        self.agent = agent

    def get_or_create_agent(self, provider: str | Model | None = None) -> Agent[SynthesisContext, SynthesisAgentOutput]:
        """Create PydanticAI agent for synthesis."""
        with contextlib.suppress(AgentUninitializedException):
            return self.agent
        try:
            agent = self._create_agent(provider)
            self.agent = agent

            return agent
        except (AgentInitializationException, InvalidAgentParameterException, PydanticAIImportError):
            raise
        except Exception as e:
            msg = f"Failed to create a PydanticAI Agent for research synthesis: {e}"
            raise AgentInitializationException(msg) from e

    @classmethod
    def _create_agent(cls, provider: str | Model | None = None) -> Agent[SynthesisContext, SynthesisAgentOutput]:
        """Create PydanticAI agent for synthesis."""
        if Agent is None or Model is None:
            raise PydanticAIImportError()

        try:
            model = provider if isinstance(provider, Model) else PydanticAIModelFactory.create(provider)

            return Agent(
                model,
                deps_type=SynthesisContext,
                output_type=SynthesisAgentOutput,
                instructions=cls.SYNTHESIS_INSTRUCTIONS,
            )
        except UserError as e:
            raise AgentInitializationException(
                f"PydanticAI encountered an error on LLM configuration initialization: {e}"
            ) from e

    @classmethod
    def create_prompt(cls, context: SynthesisContext, *args: Any, **kwargs: Any) -> str:
        """Generates the prompt to be used for the synthesis of a research summary from available evidence.

        Args:
            context (SynthesisContext): A synthesis context dependency or similarly duck-typed dataclass.
            *args: Additional positional arguments to pass to `SearchRecordPreprocessingUtils.build_record_context`
            *kwargs: Additional keyword arguments to pass to `SearchRecordPreprocessingUtils.build_record_context`

        Returns:
            The generated prompt containing the record context required to answer the research question.

        """
        try:
            if not isinstance(context, SynthesisContext) and not (
                is_dataclass(context) and all(hasattr(context, field.name) for field in fields(SynthesisContext))
            ):
                raise RuntimeError(f"Expected a valid SynthesisContext, but received type {type(context)}")

            record_contexts = SearchRecordPreprocessingUtils.build_record_context(context.records, *args, **kwargs)
            record_summaries = "\n".join(record_contexts) if record_contexts else "No records available."

            prompt = dedent(
                f"""
                IMPORTANT: Reference records by their index number ONLY.
                Do NOT invent, modify, or hallucinate record metadata.
                Use record_index (0-based) in your evidence_summaries.

                ## Research Question
                {context.question}

                ## Research Categories
                {', '.join(c.value.name for c in context.categories) if context.categories else "N/A"}

                ## Available records ({len(context.records)} total)
                {record_summaries}

                ---

                Please synthesize the findings from these records to answer the research question.
                Focus on evidence quality and note any conflicting findings.

            """
            )
        except Exception as e:
            msg = f"Failed to create the prompt from the available synthesis context: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e
        return prompt

    async def synthesize_records(self, context: SynthesisContext) -> SynthesisAgentOutput:
        """Synthesize records retrieved across several academic databases.

        Args:
            context (SynthesisContext): Synthesis parameters including question, queries, and categories.

        Returns:
            Structured synthesis with evidence and citations.

        """
        prompt = self.create_prompt(context)
        agent = self.get_or_create_agent()

        try:
            logger.info("Running synthesis...")
            token_count = SearchRecordPreprocessingUtils.estimate_token_count(prompt)
            logger.info(f"Est. token count {token_count}.\nModel: {self.model_name}")
            result = await agent.run(prompt, deps=context)
            output = result.output
            # Validate output has required attributes
            if not isinstance(output, SynthesisAgentOutput):
                raise RuntimeError(
                    "The agent did not successfully synthesize and return the expected SynthesisAgentOutput."
                )

            if not output.evidence_summaries:
                logger.warning("The 'evidence_summaries' field from the synthesized output is empty.")

            logger.info(f"Synthesis complete. Evidence items: {len(output.evidence_summaries)}")
        except PydanticAIImportError as e:
            logger.error(e.message)
            raise

        except Exception as e:
            msg = f"Agent execution failed: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

        return output

    async def __call__(self, context: SynthesisContext) -> SynthesisAgentOutput:
        """Convenience method for synthesizing records retrieved across several academic databases.

        This method calls SynthesisAgent.synthesize_records under the hood to return AI synthesized output.

        Args:
            context (SynthesisContext): Synthesis parameters including question, queries, and categories.

        Returns:
            Structured synthesis with evidence and citations.

        """
        return await self.synthesize_records(context)

    async def get_stats(self) -> dict[str, Any]:
        """Show the current configuration of the SynthesisAgent.

        Returns:
            dict[str, Any]: Agent initialization status and model name.

        """

        try:
            agent = self.get_or_create_agent()
            return {
                "initialized": True,
                "model_name": self.get_model_name(agent),
            }

        except Exception as e:
            return {"initialized": False, "model_name": None, "error": str(e)}

    async def check_health(self) -> ServiceHealth:
        """Helper used to check the health status of the SynthesisAgent."""
        details = await self.get_stats()
        error = details.get("error")
        health_status = "healthy" if not error else "unhealthy"

        return ServiceHealth(
            name="SynthesisAgent",
            status=health_status,
            details=details,
            error=error,
        )


__all__ = ["SynthesisAgent", "SynthesisAgentOutput", "SynthesisContext"]

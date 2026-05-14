"""Defines the research synthesis tool input model and the output formatter for ScholarFlux MCP server."""

import json
from typing import Any

from pydantic import Field

from scholar_flux_mcp.models import (
    APIProviders,
    BaseSynthesisParams,
    GroundedEvidenceItem,
    ResearchCategory,
    ResearchQueryList,
    ResponseFormat,
    SynthesisInput,
    SynthesisOutput,
    cached_property,
)
from scholar_flux_mcp.server.io.base import BaseFormatter, BaseResearchToolInput
from scholar_flux_mcp.server.io.search import SearchFormatter


class SynthesisToolInput(BaseResearchToolInput, BaseSynthesisParams):
    """Input model for the `scholar_flux_synthesize_research_summary` tool.

    Synthesize findings from research literature using AI-powered analysis across multiple domains.

    """

    queries: ResearchQueryList = Field(default_factory=list)
    categories: list[str] = Field(
        default=["general"],
        description=(
            "Research categories to focus synthesis on. Options: "
            "meta_analysis, theoretical, empirical, qualitative, quantitative, depression, anxiety, ptsd, stress, "
            "bipolar, schizophrenia, substance_use, ocd, adhd, general_wellbeing, intervention, epidemiology, "
            "mental_health, mathematics, computation, general, other. Default: general"
        ),
    )

    def as_synthesis_input(self) -> SynthesisInput:
        """Factory method that converts and processes the search tool input into a `SynthesisInput`."""
        provider_list = APIProviders.as_provider_list(self.providers) or list(SynthesisInput.DEFAULT_PROVIDERS)
        category_list = ResearchCategory.as_category_list(self.categories) or [ResearchCategory.OTHER]

        return SynthesisInput(
            question=self.question,
            queries=self.queries,
            categories=category_list,
            max_records=self.max_records,
            year_from=self.year_from,
            year_to=self.year_to,
            open_access_only=self.open_access_only,
            providers=provider_list,
            similarity_threshold=self.similarity_threshold,
        )

    @cached_property
    def input_hash(self) -> str:
        """Creates a `SynthesisInput` hash for the current tool input model to enable efficient identification."""
        return self.as_synthesis_input().input_hash


class SynthesisFormatter(BaseFormatter):
    """A research synthesis formatter designed to create an MCP research synthesis markdown summary from the output."""

    @classmethod
    def format(
        cls,
        output: SynthesisOutput,
        response_format: ResponseFormat | str = ResponseFormat.MARKDOWN,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Formats the synthesis results either as a markdown summary or JSON response."""
        format = ResponseFormat(response_format)
        return (
            cls.format_synthesis_json(output, *args, **kwargs)
            if format == ResponseFormat.JSON
            else cls.format_synthesis_markdown(output, *args, **kwargs)
        )

    @classmethod
    def format_synthesis_header(cls, result: SynthesisOutput) -> str:
        """Formats the summary section of the synthesized output."""
        categories = ResearchCategory.as_category_names(result.categories_analyzed, display_name=True)
        query_list = "\n".join(f"- {q}" for q in result.queries)

        summary = cls.format_multiline_string(
            f"""\
            # Research Synthesis

            ## Research Question
            > {result.question}

            **Categories**: {categories}

            **Queries**:
            {query_list}


            **Records Analyzed**: {result.records_analyzed}

            **Confidence Score**: {result.confidence_score:.0%}

            """
        )

        if grounding_stats := result.grounding_stats:
            grounded = grounding_stats.references_grounded
            rejected = grounding_stats.references_rejected
            grounding_error = grounding_stats.error
            summary += f"**Evidence Grounding**: {grounded} verified records, {rejected} rejected"
            if grounding_error:
                summary += (
                    "\n\n**Warning**: An internal error occurred during grounding, preventing the verification of all "
                    "available evidence. Verify the results to ensure references and AI generated statements are sound.\n"
                )
        if result.record_topic_similarity_output is None:
            summary += (
                "\n\n**Note**: Embedding-based record relevance reranking was unavailable. Records were not filtered by "
                "topic similarity."
            )

        return summary

    @classmethod
    def format_evidence(cls, evidence: GroundedEvidenceItem, i: int) -> str:
        """Prepares and formats the records that are used as evidence for the synthesized report."""
        year = f" ({evidence.year})" if evidence.year else ""
        doi = f" DOI: `{evidence.doi}`" if evidence.doi else ""
        url = f" [[Link]]({evidence.url})" if evidence.url else ""
        referenced_text = evidence.referenced_text or ""
        referenced_text_fmt = f"\n**Referenced Text**: {referenced_text}\n" if referenced_text else "\n\n"

        reference = f"**{evidence.title}**{year}{doi}{url}"

        # Add authors if available
        if authors := SearchFormatter.format_author_list(evidence.record):
            reference += f"\n\n**Authors**: {authors}"

        return cls.format_multiline_string(
            f"""
            {i}. {reference}

            **Index**: [{evidence.record_index}]

            **Relevance Score**: {evidence.relevance_score:.0%}

            **Summary**: {evidence.finding}

            **Abstract**: {evidence.abstract}
            {referenced_text_fmt}
            **Source**: {evidence.display_name}

            """
        )

    @classmethod
    def format_synthesis_body(cls, result: SynthesisOutput) -> str:
        """Formats the body of the synthesis markdown report."""
        # Build remaining sections
        sections = []
        sections.append("\n\n---\n\n## Synthesis\n\n" + result.synthesis + "\n")

        if result.key_findings:
            sections.append("\n## Key Findings\n")
            sections.extend(f"- {finding}" for finding in result.key_findings)
            sections.append("")

        if result.grounded_evidence:
            sections.append("\n## Supporting Evidence\n")
            sections.extend(cls.format_evidence(ev, i) for i, ev in enumerate(result.grounded_evidence, 1))

        if result.limitations:
            sections.append("\n## Limitations\n")
            sections.extend(f"- {limitation}" for limitation in result.limitations)
            sections.append("")

        if result.suggested_queries:
            sections.append("\n## Suggested Follow-up\n")
            sections.extend(f"- {query}" for query in result.suggested_queries)
            sections.append("")

        return "\n".join(sections)

    @classmethod
    def format_synthesis_markdown(cls, result: SynthesisOutput) -> str:
        """Format synthesis result as a markdown report."""
        summary = cls.format_synthesis_header(result)
        body = cls.format_synthesis_body(result)
        llm = result.model_name
        embedding_model = result.embedding_model_name

        footer = cls.format_multiline_string(
            f"""
            ---
            *This synthesis was generated using ScholarFlux MCP with PydanticAI.
            LLM: {llm} | Embedding Model: {embedding_model}
            Always verify findings with primary sources.*
            """
        )

        return f"{summary}\n\n{body}\n\n{footer}"

    @classmethod
    def format_synthesis_json(cls, output: SynthesisOutput, *args: Any, **kwargs: Any) -> str:
        """Uses a valid `SynthesisOutput` results to create a serialized json string."""
        model_json = output.model_dump(*args, **kwargs)
        return json.dumps(model_json, indent=2, default=str)


__all__ = [
    "SynthesisToolInput",
    "SynthesisFormatter",
]

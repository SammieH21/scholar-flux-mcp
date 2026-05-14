"""Relevance Search tool for ScholarFlux MCP server.

Provides embedded record searches ranked by relevance to the original query and question via PydanticAI.

"""

from collections import Counter
from typing import Any

from pydantic import Field

from scholar_flux_mcp.models import (
    APIProviders,
    BaseRelevanceSearchParams,
    RelevanceSearchInput,
    RelevanceSearchOutput,
    ResearchCategory,
    ResearchQueryList,
    ResponseFormat,
    SearchOutput,
    cached_property,
)
from scholar_flux_mcp.server.io.base import BaseFormatter, BaseResearchToolInput
from scholar_flux_mcp.server.io.search import SearchFormatter
from scholar_flux_mcp.utils.helpers import coerce_int


class RelevanceSearchToolInput(BaseResearchToolInput, BaseRelevanceSearchParams):
    """Input model for the `scholar_flux_relevance_search` tool.

    Synthesize findings from research literature using AI-powered analysis across multiple domains.

    """

    queries: ResearchQueryList = Field(default_factory=list)  # Explicit override (although identical to base params)

    categories: list[str] = Field(
        default=["general"],
        description=(
            "Research categories to focus relevance search on. Options: meta_analysis, theoretical, empirical, "
            "qualitative, quantitative, depression, anxiety, ptsd, stress, bipolar, schizophrenia, substance_use, ocd, "
            "adhd, general_wellbeing, intervention, epidemiology, mental_health, mathematics, computation, general, "
            "other. Default: general"
        ),
    )

    def as_relevance_search_input(self) -> RelevanceSearchInput:
        """Formats and processes the instance into a `RelevanceSearchInput` usable for later retrieval and reranking."""
        provider_list = APIProviders.as_provider_list(self.providers)
        category_list = ResearchCategory.as_category_list(self.categories) or [ResearchCategory.OTHER]

        return RelevanceSearchInput(
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
        """Creates a `RelevanceSearchInput` hash for the current tool input model to enable efficient identification."""
        return self.as_relevance_search_input().input_hash


class RelevanceSearchFormatter(BaseFormatter):
    """Helper class for constructing a markdown summary for relevance searches."""

    DISPLAY_RECORDS: int = 25
    DISPLAY_FIELD_TRUNCATION_LENGTH: int | None = 400

    @classmethod
    def format_relevance_search_markdown(
        cls, output: RelevanceSearchOutput, *, by_provider: bool = False, max_records: int | None = None
    ) -> str:
        """Uses a valid `RelevanceSearchOutput` results to create a markdown summary."""
        summary = (
            SearchFormatter.format_record_search_markdown(output, max_records=max_records)
            if by_provider
            else cls.format_record_search_markdown(output, max_records=max_records)
        )
        summary += cls.format_footer(output)
        return summary

    @classmethod
    def format_relevance_search_json(cls, output: RelevanceSearchOutput, *args: Any, **kwargs: Any) -> str:
        """Uses a valid `RelevanceSearchOutput` results to create a serialized json string."""
        exclude_fields = {
            "relevance_search_input",
            "record_topic_similarity_output",
            "search_output",
        }
        kwargs.setdefault("exclude", exclude_fields)

        return SearchFormatter.format_record_search_json(output, *args, **kwargs)

    @classmethod
    def format(
        cls,
        output: RelevanceSearchOutput,
        response_format: ResponseFormat | str = ResponseFormat.MARKDOWN,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Formats the search results either as a markdown summary or JSON response."""
        format = ResponseFormat(response_format)
        max_records = kwargs.pop("max_records", None)
        return (
            cls.format_relevance_search_json(output, *args, **kwargs)  # JSON format surfaces all records automatically
            if format == ResponseFormat.JSON
            else cls.format_relevance_search_markdown(output, *args, max_records=max_records, **kwargs)
        )

    @classmethod
    def format_record_search_markdown(
        cls,
        output: SearchOutput | RelevanceSearchOutput,
        *,
        max_records: int | None = None,
        field_truncation_length: int | None = None,
    ) -> str:
        """Format search results as markdown summary.

        Args:
            output (SearchOutput): A SearchOutput instance originating from the execution of the SearchService.
            max_records (int | None): The total number of records to show per provider
            field_truncation_length (int | None): The maximum character length that a field can have before truncation

        Returns:
            str: The SearchOutput formatted as a markdown string.

        """
        max_records = coerce_int(max_records) or cls.DISPLAY_RECORDS
        field_truncation_length = field_truncation_length or cls.DISPLAY_FIELD_TRUNCATION_LENGTH

        lines: list[str] = []
        indent: int = 3

        output_header = cls.format_header(output)

        lines.append(output_header)
        lines.append("### Records by Relevance (Descending Order):\n")

        for i, record in enumerate(output.records[:max_records], 1):  # Show first N records
            formatted_record_string = cls.format_record(
                record, i, field_truncation_length=field_truncation_length, display_record_source=True, indent=indent
            )
            lines.append(formatted_record_string + "\n")

        if len(output.records) > max_records:
            pad = " " * indent
            lines.append(f"{pad}*... and {len(output.records) - max_records} more records*\n")

        return "\n".join(lines)

    @classmethod
    def format_header(cls, output: SearchOutput | RelevanceSearchOutput) -> str:
        """Overrides the output header from the SearchFormatter to additionally include per-provider record totals."""
        header = SearchFormatter.format_header(output)

        # Record Count by provider
        record_count_by_provider = Counter(record.provider_name for record in output.records)
        records_by_provider = "\n".join(
            f"- **{cls.get_display_name(provider)}**: ({record_count} records)"
            for provider, record_count in record_count_by_provider.items()
        )
        similarity_threshold = (
            f"{output.similarity_threshold:.2f}"
            if isinstance(output, RelevanceSearchOutput) and output.similarity_threshold is not None
            else "N/A"
        )

        relevance_search_header = cls.format_multiline_string(
            f"""\
            {header}
            ### Record Totals:


            **Record-Topic Similarity Threshold**: {similarity_threshold}

            **Filtered Record Count**: {len(output.records)}

            **Sources**:
            {records_by_provider}
            """
        )

        return relevance_search_header

    @classmethod
    def format_footer(cls, output: RelevanceSearchOutput) -> str:
        """Generates a footer indicating the source of the embeddings."""
        if not output.record_topic_similarity_output:
            return (
                "A valid set of embeddings could not be generated. Check your embedding configuration for potential "
                "issues."
            )

        embedding_model = output.record_topic_similarity_output.model_name
        footer = cls.format_multiline_string(
            f"""
            ---
            *Record-topic similarity scores were generated using ScholarFlux MCP with PydanticAI.
            Embedding Model: {embedding_model}*
            """
        )

        return footer

    # Aliases for API compatibility and easy accessibility
    format_record = SearchFormatter.format_record
    get_display_name = SearchFormatter.get_display_name


__all__ = ["RelevanceSearchToolInput", "RelevanceSearchFormatter"]

"""History tool for ScholarFlux MCP server.

Formats and presents tool output history for browsing recent research operations.

"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from scholar_flux_mcp.models import (
    ProviderMetadataDescriptions,
    RelevanceSearchOutput,
    ResearchCategory,
    ResponseFormat,
    SearchOutput,
    SearchRecord,
    SynthesisOutput,
)
from scholar_flux_mcp.models.type_aliases import ResearchHistoryOutput, ResearchToolType
from scholar_flux_mcp.server.io.base import BaseFormatter, BaseToolInput
from scholar_flux_mcp.server.io.search import SearchFormatter
from scholar_flux_mcp.utils.helpers import coerce_int, parse_iso_timestamp, truncate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scholar_flux_mcp.models.history import (
        RelevanceSearchExecution,
        SearchExecution,
        SearchRecordHistory,
        SynthesisExecution,
    )
else:
    try:
        from scholar_flux_mcp.models.history import (
            RelevanceSearchExecution,
            SearchExecution,
            SearchRecordHistory,
            SynthesisExecution,
        )
    except (ImportError, ModuleNotFoundError):
        SearchRecordHistory = SearchExecution = RelevanceSearchExecution = SynthesisExecution = None


class GetHistoryToolInput(BaseToolInput):
    """Input model for the `scholar_flux_get_history_output` tool."""

    model_config = ConfigDict(extra="forbid")

    input_hash: str = Field(description="The unique hash associated with the input.")
    research_tool_type: ResearchToolType = Field(
        default="all",
        description=(
            "The research tool to view the history output for. Options: search/record_search, relevance/relevance_search"
            ", synthesis/research_synthesis, all."
        ),
    )
    ttl: int | float = Field(
        default=-1,
        description=(
            "TTL (Maximum recency of cached outputs) measured in seconds. If the storage time of a previous output "
            "exceeds the TTL, it is excluded from the output. Set this value to `-1` to the most recent output "
            "regardless of recency."
        ),
    )
    successful_only: bool = Field(
        default=True, description="Indicates whether the last output should be shown independent of success or failure."
    )


class RecordHistoryToolInput(BaseToolInput):
    """Input model for the `scholar_flux_list_record_history` tool."""

    model_config = ConfigDict(extra="forbid")

    ttl: int | float = Field(
        default=-1,
        description=(
            "TTL (Maximum recency of cached outputs) measured in seconds. If the storage time of a previous output "
            "exceeds the TTL, its corresponding records are excluded from the output. Set this value to `-1` to show "
            "all outputs regardless of recency."
        ),
    )
    max_records: int | None = Field(
        default=None,
        description="The maximum number of records to return within the record history sequence.",
    )
    topic: str | None = Field(
        default=None,
        description=(
            "An Optional topic used to filter and sort cached records via fuzzy similarity. When enabled, the "
            "fuzzy similarity record search uses the title, author, doi, and abstract fields to select and reorder "
            "previously cached records by relevance in descending order."
        ),
    )
    similarity_threshold: int | float | None = Field(
        default=None,
        description=(
            "The similarity threshold used to filter via fuzzy similarity search. This field is used only when a "
            "topic has been specified to filter records by fuzzy text similarity."
        ),
        ge=0.0,
        le=1.0,
    )


class RecentHistoryToolInput(BaseToolInput):
    """Input model for the `scholar_flux_list_recent_history` tool."""

    model_config = ConfigDict(extra="forbid")

    research_tool_type: ResearchToolType = Field(
        default="all",
        description=(
            "The research tool to view the history output for. Options: search/record_search, relevance/relevance_search"
            ", synthesis/research_synthesis, all."
        ),
    )
    ttl: int | float = Field(
        default=-1,
        description=(
            "TTL (Maximum recency of cached outputs) measured in seconds. If the storage time of a previous output "
            "exceeds the TTL, it is excluded from the output. Set this value to `-1` to show all outputs regardless of "
            "recency."
        ),
    )
    max_history: int | None = Field(
        default=None,
        description="The maximum number of history output items to return in the final sequence of relations.",
    )
    successful_only: bool = Field(
        default=True, description="Indicates whether outputs should be shown independent of success or failure."
    )
    topic: str | None = Field(
        default=None,
        description=(
            "An Optional topic used to filter and sort cached record search, relevance search, and research synthesis "
            "outputs by fuzzy similarity. When enabled, the query, question, and category are compared to the "
            "user-specified topic to determine output relevance. Outputted records are ordered by topic similarity in "
            "descending order when they exceed the threshold if specified."
        ),
    )
    similarity_threshold: int | float | None = Field(
        default=None,
        description=(
            "The similarity threshold used to filter outputs via fuzzy similarity search. This field is used only when "
            "a topic has been specified to filter previously stored outputs via fuzzy text similarity."
        ),
        ge=0.0,
        le=1.0,
    )


class ClearHistoryToolInput(BaseModel):
    """Input model for the `scholar_flux_clear_history` tool."""

    model_config = ConfigDict(extra="forbid")

    # No-Op parameters: This tool simply clears the history stored within its respective SQL database.


class BaseSummary(BaseModel):
    """Base class for the creation of summaries from history items."""

    input_hash: str
    tool_type: str

    status: Literal["Success", "Partial Success", "Failed"]
    queries: list[str]
    providers: list[str]
    open_access_only: bool
    max_records: int
    total_records: int
    year_from: int | None = None
    year_to: int | None = None

    stored_at: str | None = None

    @property
    def stored_at_dt(self) -> str:
        """Property that formats the history storage date in ISO datetime format."""
        return self._format_datetime(self.stored_at) if self.stored_at else "N/A"

    @property
    def year_range(self) -> str:
        """Property that formats the `year_from` and `year_to` as a proper date range."""
        return self._format_year_range(self.year_from, self.year_to)

    @property
    def provider_display_names(self) -> list[str]:
        """Formats the name of each API provider into its human readable representation."""
        return ProviderMetadataDescriptions.as_provider_names(self.providers, display_name=True)

    @classmethod
    def _format_datetime(cls, stored_at: str, format: str = "%Y-%m-%d at %H:%M:%S") -> str:
        """Helper for formatting the `stored_at` time of the output."""
        parsed = parse_iso_timestamp(stored_at)
        return parsed.strftime(format) if parsed else "N/A"

    @classmethod
    def _format_year_range(cls, year_from: int | None, year_to: int | None) -> str:
        """Returns a formatted year range string, or empty string if no range is set."""
        if year_from and year_to:
            return f"{year_from}–{year_to}"
        elif year_from:
            return f"{year_from}–present"
        elif year_to:
            return f"{year_to} and earlier"
        return ""


class SearchSummary(BaseSummary):
    """Summary class for formatting and displaying a concise representation of a record search."""

    pages_retrieved: int
    pages_successful: int

    tool_type: str = "Record Search"

    @property
    def displayed_records(self) -> int:
        """Calculates the total number of displayed records based on the `max_records` input."""
        return min(self.max_records, self.total_records)

    @classmethod
    def from_search_output(cls, history_item: SearchExecution | SearchOutput, stored_at: str | None = None) -> Self:
        """Creates a new SearchSummary from the current history item."""
        stored_at = history_item.stored_at if isinstance(history_item, SearchExecution) else stored_at
        search_output = (
            history_item if isinstance(history_item, SearchOutput) else history_item.search_output.to_search_output()
        )
        search_input = search_output.search_input

        pagination = search_output.pagination
        any_successful = search_output.pagination.pages_successful > 0
        status: Literal["Success", "Partial Success", "Failed"] = (
            "Success" if search_output.successful else ("Partial Success" if any_successful else "Failed")
        )
        return cls(
            status=status,
            input_hash=search_input.input_hash,
            queries=search_input.queries,
            providers=ProviderMetadataDescriptions.as_provider_names(search_input.providers),
            max_records=search_input.max_records,
            pages_retrieved=pagination.pages_retrieved,
            pages_successful=pagination.pages_successful,
            total_records=pagination.total_records or len(search_output.records),
            year_from=search_output.search_input.year_from,
            year_to=search_output.search_input.year_to,
            open_access_only=search_input.open_access_only,
            stored_at=stored_at,
        )


class RelevanceSearchSummary(BaseSummary):
    """Summary class for formatting and displaying a concise representation of a relevance search."""

    question: str
    categories: list[str]
    records_analyzed: int
    similarity_threshold: float | None
    embedding_model_name: str

    tool_type: str = "Relevance Search"

    @property
    def category_display_names(self) -> list[str]:
        """Parses and formats each category name as title case for later display."""
        return ResearchCategory.as_category_names(self.categories, display_name=True)

    @classmethod
    def from_relevance_search_output(
        cls,
        history_item: RelevanceSearchExecution | RelevanceSearchOutput,
        stored_at: str | None = None,
    ) -> Self:
        """Creates a new RelevanceSearchSummary from the current history item."""
        stored_at = history_item.stored_at if isinstance(history_item, RelevanceSearchExecution) else stored_at

        relevance_search_output = (
            history_item
            if isinstance(history_item, RelevanceSearchOutput)
            else history_item.to_relevance_search_output()
        )
        relevance_search_input = relevance_search_output.relevance_search_input
        any_successful = relevance_search_output.pagination.pages_successful > 0
        status: Literal["Success", "Partial Success", "Failed"] = (
            "Success" if relevance_search_output.successful else ("Partial Success" if any_successful else "Failed")
        )
        pagination = relevance_search_output.pagination if relevance_search_output.pagination else None
        total_records = (
            pagination.total_records
            if pagination and pagination.total_records
            else len(relevance_search_output.records)
        )

        return cls(
            status=status,
            input_hash=relevance_search_input.input_hash,
            question=relevance_search_input.question,
            queries=relevance_search_input.queries,
            providers=relevance_search_output.providers,
            categories=ResearchCategory.as_category_names(relevance_search_input.categories),
            max_records=relevance_search_input.max_records,
            records_analyzed=relevance_search_output.records_analyzed,
            total_records=total_records,
            year_from=relevance_search_output.relevance_search_input.year_from,
            year_to=relevance_search_output.relevance_search_input.year_to,
            open_access_only=relevance_search_input.open_access_only,
            similarity_threshold=relevance_search_input.similarity_threshold,
            embedding_model_name=relevance_search_output.embedding_model_name,
            stored_at=stored_at,
        )


class SynthesisSummary(RelevanceSearchSummary):
    """Summary class for formatting and displaying a concise representation of a research synthesis."""

    references_grounded: int
    references_rejected: int
    synthesis: str
    confidence_score: float
    key_findings: list[str]
    embedding_model_name: str = "N/A"
    model_name: str = "N/A"

    tool_type: str = "Research Synthesis"

    @classmethod
    def from_synthesis_output(
        cls,
        history_item: SynthesisExecution | SynthesisOutput,
        stored_at: str | None = None,
    ) -> Self:
        """Creates a new SynthesisSummary from the current history item."""
        stored_at = history_item.stored_at if isinstance(history_item, SynthesisExecution) else stored_at

        synthesis_output = (
            history_item if isinstance(history_item, SynthesisOutput) else history_item.to_synthesis_output()
        )
        synthesis_input = synthesis_output.synthesis_input
        status: Literal["Success", "Partial Success", "Failed"] = "Success" if synthesis_output.successful else "Failed"

        total_core_records = (
            (search_output.pagination.total_records or len(search_output.records))
            if (search_output := synthesis_output.search_output)
            else None
        )
        total_records = total_core_records or len(synthesis_output.indexed_records)

        return cls(
            status=status,
            input_hash=synthesis_input.input_hash,
            question=synthesis_input.question,
            queries=synthesis_input.queries,
            providers=ProviderMetadataDescriptions.as_provider_names(synthesis_output.synthesis_input.providers),
            categories=ResearchCategory.as_category_names(synthesis_input.categories),
            synthesis=synthesis_output.synthesis,
            key_findings=synthesis_output.key_findings,
            confidence_score=synthesis_output.confidence_score,
            references_grounded=synthesis_output.grounding_stats.references_grounded,
            references_rejected=synthesis_output.grounding_stats.references_rejected,
            max_records=synthesis_input.max_records,
            records_analyzed=synthesis_output.records_analyzed,
            total_records=total_records,
            similarity_threshold=synthesis_input.similarity_threshold,
            year_from=synthesis_output.synthesis_input.year_from,
            year_to=synthesis_output.synthesis_input.year_to,
            open_access_only=synthesis_input.open_access_only,
            embedding_model_name=synthesis_output.embedding_model_name,
            model_name=synthesis_output.model_name,
            stored_at=stored_at,
        )


class RecordHistoryFormatter(BaseFormatter):
    """Formats raw records and record history objects from output history retrieval into concise, readable summaries."""

    @classmethod
    def format(
        cls,
        output: Sequence[SearchRecord | SearchRecordHistory],
        response_format: ResponseFormat | str = ResponseFormat.MARKDOWN,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Formats the output history either as a markdown summary or JSON response."""
        format = ResponseFormat(response_format)
        return (
            cls.format_record_history_json(output, *args, **kwargs)
            if format == ResponseFormat.JSON
            else cls.format_record_history_markdown(output)
        )

    @classmethod
    def format_record_history_json(
        cls, record_history: Sequence[SearchRecord | SearchRecordHistory], indent: int = 2
    ) -> str:
        """Formats the recently recorded tool output history by storage date in descending order."""
        formatted_records = (
            [cls.convert_record(record).model_dump() for record in record_history] if record_history else []
        )
        return json.dumps({"record_history": formatted_records}, indent=indent)

    @classmethod
    def format_record_history_markdown(cls, record_history: Sequence[SearchRecord | SearchRecordHistory]) -> str:
        """Formats the recently recorded tool output history by storage date in descending order."""
        if not record_history:
            return "No history data is available."

        formatted_records = [
            SearchFormatter.format_record(cls.convert_record(record), i)
            for i, record in enumerate(record_history, start=1)
        ]

        return "\n\n---\n\n".join(formatted_records)

    @classmethod
    def convert_record(cls, record: SearchRecord | SearchRecordHistory) -> SearchRecord:
        """Dispatches summary model creation to the appropriate method based on relation type."""
        if isinstance(record, SearchRecord):
            return record
        if isinstance(record, SearchRecordHistory):
            return record.to_search_record()
        raise TypeError(f"Unknown history output type: {type(record).__name__}")


class OutputHistoryFormatter(BaseFormatter):
    """Formats raw relation objects from output history retrieval into concise, readable summaries.

    Note:
        The `_format_synthesis`, `_format_relevance_search`, and `_format_search` methods each call
        `to_synthesis_output()`, `to_relevance_search_output()`, and `to_search_output()`, respectively, and may require
        ORM relationship traversal to fully load all relevant components of the output history. Ensure that the
        respective session is still active or that all nested history components have been eagerly loaded before use.

    """

    SYNTHESIS_TRUNCATION_LENGTH: int | None = 150

    @classmethod
    def format(
        cls,
        output: Sequence[ResearchHistoryOutput],
        response_format: ResponseFormat | str = ResponseFormat.MARKDOWN,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Formats the output history either as a markdown summary or JSON response."""
        format = ResponseFormat(response_format)
        return (
            cls.format_recent_history_json(output, *args, **kwargs)
            if format == ResponseFormat.JSON
            else cls.format_recent_history_markdown(output)
        )

    @classmethod
    def format_recent_history_json(cls, recent_history: Sequence[ResearchHistoryOutput], indent: int = 2) -> str:
        """Formats the recently recorded tool output history by storage date in descending order."""
        history_summary = (
            [cls._format_history_summary(history_item).model_dump() for history_item in recent_history]
            if recent_history
            else "No history data is available."
        )
        return json.dumps({"recent_history": history_summary}, indent=indent)

    @classmethod
    def format_recent_history_markdown(cls, recent_history: Sequence[ResearchHistoryOutput]) -> str:
        """Formats the recently recorded tool output history by storage date in descending order."""
        if not recent_history:
            return "No history data is available."

        formatted_items: list[str] = []
        for i, history_item in enumerate(recent_history, start=1):
            history_markdown = cls._format_history_item(history_item, i)
            formatted_items.append(history_markdown)

        return "\n\n---\n\n".join(formatted_items)

    @classmethod
    def _format_history_summary(cls, history_item: ResearchHistoryOutput) -> BaseSummary:
        """Dispatches summary model creation to the appropriate method based on relation type."""
        if isinstance(history_item, SearchExecution | SearchOutput):
            return SearchSummary.from_search_output(history_item)
        elif isinstance(history_item, RelevanceSearchExecution | RelevanceSearchOutput):
            return RelevanceSearchSummary.from_relevance_search_output(history_item)
        elif isinstance(history_item, SynthesisExecution | SynthesisOutput):
            return SynthesisSummary.from_synthesis_output(history_item)
        raise TypeError(f"Unknown history output type: {type(history_item).__name__}")

    @classmethod
    def _format_history_item(cls, history_item: ResearchHistoryOutput, i: int) -> str:
        """Dispatches formatting to the appropriate method based on relation type."""
        if isinstance(history_item, SynthesisExecution | SynthesisOutput):
            return cls._format_synthesis_summary(history_item, i)
        elif isinstance(history_item, RelevanceSearchExecution | RelevanceSearchOutput):
            return cls._format_relevance_search_summary(history_item, i)
        elif isinstance(history_item, SearchExecution | SearchOutput):
            return cls._format_search_summary(history_item, i)
        raise TypeError(f"{i}. Unknown history output type: {type(history_item).__name__}")

    @classmethod
    def _format_search_summary(cls, history_item: SearchExecution | SearchOutput | SearchSummary, i: int) -> str:
        """Generates a concise markdown summary from a `SearchExecution` model retrieved from history."""
        summary = (
            history_item if isinstance(history_item, SearchSummary) else SearchSummary.from_search_output(history_item)
        )
        queries = ", ".join(summary.queries) if summary.queries else "N/A"
        providers = ", ".join(summary.provider_display_names) if summary.providers else "N/A"

        timestamp_fmt = f" ({stored_at_dt})" if (stored_at_dt := summary.stored_at_dt) else ""

        formatted_summary = cls.format_multiline_string(
            f"""\
            **{i}. {summary.tool_type}**{timestamp_fmt}
            - **Input Hash:** `{summary.input_hash}`
            - **Queries:** {queries}
            - **Providers:** {providers}
            - **Maximum Record Limit:** {summary.max_records}
            - **Records:** {summary.displayed_records} returned, ({summary.total_records} retrieved)
            - **Pages:** {summary.pages_successful}/{summary.pages_retrieved} successful
            """
        )
        if summary.year_range:
            formatted_summary += f"- **Year range:** {summary.year_range}\n"
        if summary.open_access_only:
            formatted_summary += "- **Open access only:** Yes\n"
        formatted_summary += f"- **Status:** {summary.status}"
        return formatted_summary

    @classmethod
    def _format_relevance_search_summary(
        cls, history_item: RelevanceSearchExecution | RelevanceSearchOutput | RelevanceSearchSummary, i: int
    ) -> str:
        """Generates a concise markdown summary from a `RelevanceSearchExecution` model retrieved from history."""
        summary = (
            history_item
            if isinstance(history_item, RelevanceSearchSummary)
            else RelevanceSearchSummary.from_relevance_search_output(history_item)
        )
        queries = ", ".join(summary.queries) if summary.queries else "N/A"
        providers = ", ".join(summary.provider_display_names) if summary.providers else "N/A"
        categories = ", ".join(summary.category_display_names) if summary.categories else "None"
        records_analyzed = summary.records_analyzed
        year_range = summary.year_range

        timestamp_fmt = f" ({stored_at_dt})" if (stored_at_dt := summary.stored_at_dt) else ""

        formatted_summary = cls.format_multiline_string(
            f"""\
            **{i}. {summary.tool_type}**{timestamp_fmt}
            - **Input Hash:** `{summary.input_hash}`
            - **Question:** {summary.question}
            - **Categories:** {categories}
            - **Queries:** {queries}
            - **Providers:** {providers}
            - **Max Analyzable Records:** {summary.max_records}
            - **Records:** {records_analyzed} ranked ({summary.total_records} retrieved)
            - **Similarity threshold:** {summary.similarity_threshold}
            - **Embedding Model:** {summary.embedding_model_name}
            """
        )
        if year_range:
            formatted_summary += f"- **Year range:** {year_range}\n"
        formatted_summary += f"- **Status:** {summary.status}"
        return formatted_summary

    @classmethod
    def _format_synthesis_summary(
        cls,
        history_item: SynthesisExecution | SynthesisOutput | SynthesisSummary,
        i: int,
        *,
        synthesis_text_truncation_length: int | None = None,
    ) -> str:
        """Generates a concise markdown summary from a `SynthesisExecution` model retrieved from history."""
        summary = (
            history_item
            if isinstance(history_item, SynthesisSummary)
            else SynthesisSummary.from_synthesis_output(history_item)
        )
        queries = ", ".join(summary.queries) if summary.queries else "N/A"
        providers = ", ".join(summary.provider_display_names) if summary.providers else "N/A"
        categories = ", ".join(summary.category_display_names) if summary.categories else "None"
        year_range = summary.year_range

        grounded = summary.references_grounded
        rejected = summary.references_rejected

        # If None, the full synthesis is displayed
        truncation_length = coerce_int(synthesis_text_truncation_length or cls.SYNTHESIS_TRUNCATION_LENGTH)
        synthesis = truncate(summary.synthesis, truncation_length)

        timestamp_fmt = f" ({stored_at_dt})" if (stored_at_dt := summary.stored_at_dt) else ""

        formatted_summary = cls.format_multiline_string(
            f"""\
            **{i}. {summary.tool_type}**{timestamp_fmt}
            - **Input Hash:** `{summary.input_hash}`
            - **Question:** {summary.question}
            - **Categories:** {categories}
            - **Queries:** {queries}
            - **Providers:** {providers}
            - **Synthesis:** {synthesis}
            - **Confidence:** {summary.confidence_score:.2f}
            - **Max Analyzable Records:** {summary.max_records}
            - **Records:** {summary.records_analyzed} analyzed ({summary.total_records} retrieved)
            - **Similarity threshold:** {summary.similarity_threshold}
            - **Grounding:** {grounded} verified, {rejected} rejected
            - **LLM:** {summary.model_name}
            - **Embedding Model:** {summary.embedding_model_name}
            """
        )
        if year_range:
            formatted_summary += f"- **Year range:** {year_range}\n"
        if summary.key_findings:
            formatted_summary += f"- **Total key findings:** {len(summary.key_findings)}\n"

        formatted_summary += f"- **Status:** {summary.status}"
        return formatted_summary


__all__ = [
    "RecentHistoryToolInput",
    "RecordHistoryToolInput",
    "GetHistoryToolInput",
    "OutputHistoryFormatter",
    "ClearHistoryToolInput",
]

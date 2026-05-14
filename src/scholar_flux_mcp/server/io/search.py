"""Defines the record search tool input model and the output formatter for ScholarFlux MCP server."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import Self

from scholar_flux_mcp.models import (
    APIProviders,
    BaseSearchParams,
    IndexedSearchRecord,
    ProviderMetadataDescriptions,
    RelevanceSearchOutput,
    ResponseFormat,
    SearchCoordinatorConfig,
    SearchInput,
    SearchOutput,
    SearchRecord,
    cached_property,
)
from scholar_flux_mcp.server.io.base import BaseFormatter, BaseResearchToolInput
from scholar_flux_mcp.utils.helpers import as_tuple, coerce_int, truncate


class SearchToolInput(BaseResearchToolInput, BaseSearchParams):
    """Input model for the `scholar_flux_record_search` tool.

    Search across multiple academic databases for scholarly records with automatic schema normalization.

    """

    max_records: int = Field(
        default=25,
        description="Maximum records to retrieve per provider (1-200)",
        ge=1,
        le=200,
    )

    _search_coordinator_config: list[SearchCoordinatorConfig] = PrivateAttr(
        default_factory=list,
    )

    @model_validator(mode="before")
    @classmethod
    def prepare_search_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Model validator used to prepare each `ScholarFluxSearchConfig` prior to further field validation."""
        queries = values.get("queries") or ""
        providers = APIProviders.as_provider_list(values.get("providers") or list(cls.DEFAULT_PROVIDERS))

        values["queries"] = queries if isinstance(queries, list) else list(as_tuple(queries))
        values["providers"] = [provider_name.value for provider_name in providers]
        return values

    @model_validator(mode="after")
    def validate_search_config(self) -> Self:
        """Validator for creating and verifying each SearchCoordinator configuration before further field validation."""
        self._search_coordinator_config = [
            SearchCoordinatorConfig(query=q, provider_name=APIProviders(provider.lower()))
            for provider in self.providers
            for q in self.queries
        ]
        return self

    @property
    def query(self) -> str:
        """Return a singular query from a list of queries."""
        return self.queries[0] if self.queries else ""

    @property
    def search_coordinator_config(self) -> list[SearchCoordinatorConfig]:
        """Public property for accessing the created configurations for each API/query."""
        return self._search_coordinator_config

    def as_search_input(self) -> SearchInput:
        """Factory method that converts and processes the search tool input into a `SearchInput`."""
        return SearchInput(
            search_coordinator_config=self.search_coordinator_config,
            max_records=self.max_records,
            pages=self.pages,
            year_from=self.year_from,
            year_to=self.year_to,
            open_access_only=self.open_access_only,
        )

    @cached_property
    def input_hash(self) -> str:
        """Creates a `SearchInput` hash for the current tool input model to enable efficient identification."""
        return self.as_search_input().input_hash


class SearchFormatter(BaseFormatter):
    """A search result formatter designed to create an MCP record search markdown summary before transmission."""

    DISPLAY_RECORDS: int = 25
    DISPLAY_FIELD_TRUNCATION_LENGTH: int | None = 400

    @classmethod
    def get_display_name(cls, name: str) -> str:
        """Helper method for getting a human-readable name for a provider."""
        provider_info = ProviderMetadataDescriptions.get(name)
        return provider_info.value.display_name if provider_info else name.title()

    @classmethod
    def format(
        cls,
        output: SearchOutput | RelevanceSearchOutput,
        response_format: ResponseFormat | str = ResponseFormat.MARKDOWN,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Formats the search results either as a markdown summary or JSON response."""
        format = ResponseFormat(response_format)
        max_records = kwargs.pop("max_records", None)
        return (
            cls.format_record_search_json(output, *args, **kwargs)  # all records shown automatically
            if format == ResponseFormat.JSON
            else cls.format_record_search_markdown(output, *args, max_records=max_records, **kwargs)
        )

    @classmethod
    def format_record_search_markdown(
        cls,
        output: SearchOutput | RelevanceSearchOutput,
        *,
        max_records: int | None = None,
        field_truncation_length: int | None = None,
    ) -> str:
        """Formats search results as markdown summary.

        Args:
            output (SearchOutput): A SearchOutput instance originating from the execution of the SearchService.
            max_records (int | None): The total number of records to show per provider
            field_truncation_length (int | None): The maximum character length that a field can have before truncation

        Returns:
            str: The SearchOutput formatted as a markdown string.

        """
        max_records = coerce_int(max_records) or cls.DISPLAY_RECORDS

        lines: list[str] = []
        indent: int = 3

        output_title = cls.format_header(output)
        lines.append(output_title)

        # Group by provider
        records_by_provider: dict[str, list[SearchRecord]] = defaultdict(list)
        for record in output.records:
            records_by_provider[record.provider_name].append(record)

        # Format each provider's results
        for provider, provider_records in records_by_provider.items():
            display_name = cls.get_display_name(provider)

            lines.append(f"### {display_name} ({len(provider_records)} records)")
            lines.append("")

            for i, record in enumerate(provider_records[:max_records], 1):  # Show first N records
                formatted_record_string = cls.format_record(
                    record, i, field_truncation_length=field_truncation_length, indent=indent
                )
                lines.append(formatted_record_string + "\n")

            if len(provider_records) > max_records:
                pad = " " * indent
                lines.append(f"*{pad}... and {len(provider_records) - max_records} more records*")
                lines.append("")

        return "\n".join(lines)

    @classmethod
    def format_record_search_json(cls, output: SearchOutput | RelevanceSearchOutput, *args: Any, **kwargs: Any) -> str:
        """Uses valid `SearchOutput` or `RelevanceSearchOutput` results to create a serialized json string."""
        model_json = output.model_dump(*args, **kwargs)
        return json.dumps(model_json, indent=2, default=str)

    @classmethod
    def format_header(cls, output: SearchOutput | RelevanceSearchOutput) -> str:
        """Formats the markdown header for the retrieved output."""
        # Format query display
        query_display = ", ".join(output.queries)
        total_records = output.pagination.total_records
        providers_successful = output.pagination.providers_successful

        return cls.format_multiline_string(
            f"""\
            ## Search Results: {query_display}

            **Found {total_records} records** across {providers_successful} queries
            """
        )

    @classmethod
    def format_footer(cls, *args: Any, **kwargs: Any) -> str:
        """No-op class method used to modify the footer of the search markdown summary when overridden."""
        return ""

    @classmethod
    def format_author_list(cls, record: SearchRecord | IndexedSearchRecord) -> str:
        """Formats the author field for display in markdown format."""
        if not record.authors:
            # returned as an empty string for consistency and display compatibility
            return ""
        authors = record.authors if isinstance(record.authors, str) else ", ".join(record.authors[:3])
        if len(record.authors) > 3 if isinstance(record.authors, list) else False:
            authors += " et al."
        return authors

    @classmethod
    def format_record(
        cls,
        record: SearchRecord | IndexedSearchRecord,
        i: int | None = None,
        *,
        display_record_source: bool = False,
        field_truncation_length: int | None = None,
        indent: int = 3,
    ) -> str:
        """Formats a record as a human readable string."""
        record_parts: list[str] = []

        pad = " " * indent if isinstance(indent, int) else 0
        title = record.title or "Untitled"
        year = f" ({record.year})" if record.year else ""
        doi = f" DOI: `{record.doi}`" if record.doi else ""
        url = f" [[Link]]({record.url})" if record.url else ""
        record_source: str | None = record.journal or record.publisher
        record_source_fmt = f" {record_source}".rstrip(" .") + "." if record_source else ""
        keywords: list | str | None = record.keywords or record.subjects
        keywords_fmt = ", ".join(keywords) if isinstance(keywords, list) else keywords
        open_access = ("Yes" if record.open_access else "No") if record.open_access is not None else "Unknown"
        topic_similarity_score = record.topic_similarity_score if isinstance(record, IndexedSearchRecord) else None

        record_parts.append(f"{i}. **{title}**{year}{record_source_fmt}{doi}{url}")

        # Add the authors field if available
        if authors := cls.format_author_list(record):
            record_parts.append(f"{pad}*{authors}*")

        field_truncation_length = field_truncation_length or cls.DISPLAY_FIELD_TRUNCATION_LENGTH
        # Add truncated abstract
        if record.abstract:
            abstract = truncate(record.abstract, field_truncation_length).removeprefix("\n")
            record_parts.append(f"{pad}> {abstract}")

        if display_record_source:
            record_parts.append(f"{pad}Source: {record.display_name}")

        if isinstance(topic_similarity_score, int | float):
            topic_similarity_str = f"{pad}Topic similarity: {topic_similarity_score:.2f}"
            record_parts.append(topic_similarity_str)

        if keywords_fmt:
            record_parts.append(f"{pad}Keywords: {keywords_fmt}")

        if record.citation_count is not None:
            record_parts.append(f"{pad}Citations: {record.citation_count}")
        record_parts.append(f"{pad}Open Access: {open_access}")

        return "\n".join(record_parts)


__all__ = ["SearchToolInput", "SearchFormatter"]

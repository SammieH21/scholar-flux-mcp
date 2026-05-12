"""Defines `SearchRecordPreprocessingUtils` for preparing search records/queries/prompts for LLMs and embedding."""

from __future__ import annotations

import logging
import os
from textwrap import dedent
from typing import TYPE_CHECKING, overload

from pydantic import TypeAdapter
from rapidfuzz import fuzz, process

from scholar_flux_mcp.models.enums import ResearchCategory
from scholar_flux_mcp.models.schemas import IndexedSearchRecordList, SearchRecord, SearchRecordList
from scholar_flux_mcp.utils.helpers import as_tuple, coerce_int

if TYPE_CHECKING:
    from scholar_flux.utils.record_types import NormalizedRecordList


logger = logging.getLogger(__name__)

indexed_record_list_adapter: TypeAdapter[IndexedSearchRecordList] = TypeAdapter(IndexedSearchRecordList)


class SearchRecordPreprocessingUtils:
    """Helper utilities for preparing responses prior to inference with LLM and Embedding models."""

    FILTER_MISSING_FIELDS: set = {"abstract", "title", "doi", "provider_name", "query", "page"}
    DEDUPLICATION_FIELDS: tuple[str, ...] = ("title", "abstract", "authors", "year")
    RECORD_TRUNCATION_LENGTH: int | None = coerce_int(os.getenv("SCHOLAR_FLUX_MCP_RECORD_TRUNCATION_LENGTH", 3000))
    FIELD_TRUNCATION_LENGTH: int | None = coerce_int(os.getenv("SCHOLAR_FLUX_MCP_FIELD_TRUNCATION_LENGTH", 1300))
    TEXT_TOKEN_LIMIT: int | None = coerce_int(os.getenv("SCHOLAR_FLUX_MCP_GROUNDING_TOKEN_LIMIT", 120000))
    RECORD_EMBEDDING_TOKEN_LIMIT: int | None = coerce_int(
        os.getenv("SCHOLAR_FLUX_MCP_RECORD_EMBEDDING_TOKEN_LIMIT", 2000)
    )
    DEFAULT_CHARACTERS_PER_TOKEN: int = 3

    @overload
    @classmethod
    def validate_search_records(cls, records: SearchRecordList) -> SearchRecordList:
        """When a list is received, the list of search records is returned as is after successful validation."""
        ...

    @overload
    @classmethod
    def validate_search_records(cls, records: SearchRecord) -> SearchRecord:
        """When a single search record is received, it is returned as is."""
        ...

    @classmethod
    def validate_search_records(cls, records: SearchRecord | SearchRecordList) -> SearchRecordList | SearchRecord:
        """Validates that `records` is a list containing a valid record set."""
        if not isinstance(records, list | SearchRecord | dict):
            raise TypeError(f"Expected a record or list of records, but received type {type(records)}.")

        return (
            [record if isinstance(record, SearchRecord) else SearchRecord.model_validate(record) for record in records]
            if isinstance(records, list)
            else SearchRecord.model_validate(records)
        )

    @classmethod
    def convert_records(
        cls,
        normalized: NormalizedRecordList,
        max_records: int | None = None,
        *,
        open_access_only: bool = False,
        year_from: int | float | None = None,
        year_to: int | float | None = None,
    ) -> SearchRecordList:
        """Convert normalized dicts to SearchRecord models.

        Args:
            normalized: Raw normalized records from ScholarFlux.
            max_records: Maximum records to keep.

        Returns:
            List of SearchRecord instances.

        """
        # Track records per provider for limiting
        records: list[SearchRecord] = []
        total_records: int = 0

        for record in normalized:
            if cls.FILTER_MISSING_FIELDS and any(not record.get(field) for field in cls.FILTER_MISSING_FIELDS):
                continue

            year = record.get("year")
            if isinstance(year_from, int | float) and not (isinstance(year, int) and year >= year_from):
                continue

            if isinstance(year_to, int | float) and not (isinstance(year, int) and year <= year_to):
                continue

            if open_access_only and record.get("open_access") is False:
                continue

            try:
                # Convert to SearchRecord
                search_record = SearchRecord.model_validate(record)
                records.append(search_record)
                total_records += 1
            except Exception as e:
                logger.debug(f"Failed to convert record: {e}")
                continue

            if max_records is not None and total_records >= max_records:
                break

        return records

    @classmethod
    def deduplicate_records(
        cls, records: SearchRecordList, fields: list[str] | tuple | None = None, similarity_threshold: float = 90
    ) -> SearchRecordList:
        """Helper method for deduplicating records prior to grounding."""
        logger.info("Deduplicating records...")

        validated_records = cls.validate_search_records(records)

        sorted_records = sorted(
            validated_records, key=lambda record: (record.year or -1, record.information_content()), reverse=True
        )

        if not (fields and isinstance(fields, list | tuple)):
            fields = cls.DEDUPLICATION_FIELDS

        deduplicated_records: list[SearchRecord] = []
        seen_identifiers: set[str] = set()
        seen_dois: set[str] = set()
        seen_titles: set[str] = set()

        for record in sorted_records:
            field_values = [
                field_value for field in fields if (field_value := record.get(field)) and isinstance(field_value, str)
            ]
            record_identifier = " ".join(field_values)

            if not record.information_content():
                continue

            if not seen_identifiers or not (
                record.doi in seen_dois
                or record.title in seen_titles
                or process.extractOne(
                    record_identifier, seen_identifiers, scorer=fuzz.ratio, score_cutoff=similarity_threshold
                )
            ):
                deduplicated_records.append(record)
                seen_identifiers.add(record_identifier)

                if record.title:
                    seen_titles.add(record.title)
                if record.doi:
                    seen_dois.add(record.doi)

        if records_removed := len(sorted_records) - len(deduplicated_records):
            logger.info(f"Removed {records_removed} duplicated records...")

        return deduplicated_records

    @classmethod
    def build_record_context(
        cls,
        records: SearchRecordList,
        *,
        fields: list[str] | None = None,
        record_truncation_length: int | None = None,
        token_limit: int | None = None,
    ) -> list | None:
        """Format records as a numbered list for AI prompt.

        Args:
            records (SearchRecordList):
                List of SearchRecord or dict record objects.
            fields (List[str] | None): A list of fields used to build the record context.
            record_truncation_length (int | None):
                The truncation length to use when converting fields into text strings (Default=800 characters).
            token_limit (int | None):
                Indicates the total number of tokens that should be retained before context building halts early.

        Returns:
            list | None: Formatted list of record strings with indexed record list for AI consumption.

        """
        if not isinstance(records, list):
            logger.error("Incorrect type received")
            return None

        lines: list[str] = []

        logger.info(f"Building context for a total of {len(records or [])} records...")

        total_token_count: int = 0
        indexed_records: IndexedSearchRecordList = indexed_record_list_adapter.validate_python(records)

        for record in indexed_records:
            record_string = record.to_string(
                fields=fields, record_truncation_length=record_truncation_length or cls.RECORD_TRUNCATION_LENGTH
            )

            indexed_record_string = f"[{record.index}] {record_string}"
            token_limit = token_limit if token_limit is not None else cls.TEXT_TOKEN_LIMIT

            if isinstance(token_limit, int) and token_limit > 0:
                record_token_count = cls.estimate_token_count(indexed_record_string)

                if total_token_count + record_token_count >= token_limit:
                    logger.warning(
                        "The total number of tokens in the record list exceeds the token limit of "
                        f"{token_limit}. Instead returning a total of {len(lines)} record strings."
                    )

                    break
                total_token_count += record_token_count
            lines.append(indexed_record_string)

        return lines

    @classmethod
    def prepare_embedding_research_topic(
        cls, queries: str | list[str], question: str, categories: str | list[ResearchCategory] | None = None
    ) -> str:
        """Prepares the query embedding string for use with later embedding approaches during synthesis."""

        category_string = ", ".join(
            ", ".join(ResearchCategory(category).value.terms) for category in as_tuple(categories)
        )
        query_string = "; ".join(as_tuple(queries))

        return dedent(
            f"""\
            {question}
            --
            {category_string}
            --
            {query_string}"""
        ).replace("--\n\n--", "--\n")

    @classmethod
    def prepare_embedding_record_string(
        cls,
        record: SearchRecord,
        fields: list[str] | None = None,
        token_limit: int | None = None,
        *,
        characters_per_token: int = DEFAULT_CHARACTERS_PER_TOKEN,
    ) -> str:
        """Helper method for preparing a record for embedding and similarity score calculations."""
        current_token_limit = token_limit if token_limit is not None else cls.RECORD_EMBEDDING_TOKEN_LIMIT
        current_characters_per_token = (
            characters_per_token
            if isinstance(characters_per_token, int) and characters_per_token > 0
            else cls.DEFAULT_CHARACTERS_PER_TOKEN
        )

        # estimate the size of the current record:
        record_string = record.to_string(fields=fields, record_truncation_length=None)
        if not (isinstance(current_token_limit, int) and current_token_limit > 0):
            return record_string

        record_token_count = cls.estimate_token_count(record_string, current_characters_per_token)
        token_count_diff = record_token_count - current_token_limit
        return (
            record_string[: -(token_count_diff * current_characters_per_token)]
            if token_count_diff > 0
            else record_string
        )

    @classmethod
    def estimate_token_count(cls, text: str, characters_per_token: int | None = None) -> int:
        """Helper for generating a ballpark estimate of the number of tokens required to include a record in context."""
        if text and not isinstance(text, str):
            raise TypeError(f"Expected a string to estimate LLM/embedding token count but received type {type(text)}")
        current_characters_per_token = (
            characters_per_token if isinstance(characters_per_token, int) else cls.DEFAULT_CHARACTERS_PER_TOKEN
        )
        return max(1, len(text or "") // current_characters_per_token)


__all__ = ["SearchRecordPreprocessingUtils"]

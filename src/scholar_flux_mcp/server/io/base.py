"""Defines the `BaseFormatter` as an abstract base class and template for preparing MCP tool responses."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scholar_flux_mcp.models.enums import ResponseFormat, ResponseFormatString
from scholar_flux_mcp.models.schemas import BaseSearchParams, QueryList
from scholar_flux_mcp.utils.helpers import format_multiline_string

logger = logging.getLogger(__name__)


@runtime_checkable
class SupportsModelDump(Protocol):
    """Helper for ascertaining whether a given object supports `.model_dump()` (e.g., BaseModel subclasses)."""

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Method for coercing pydantic models into dictionaries."""
        ...


T = TypeVar("T", bound=SupportsModelDump)


class BaseToolInput(BaseModel):
    """Input model for the core ScholarFlux MCP tools."""

    model_config = ConfigDict(extra="forbid")

    response_format: ResponseFormatString = Field(
        default="markdown",
        description="Output format: 'markdown' for readable list, 'json' for structured data",
    )


class BaseResearchToolInput(BaseToolInput):
    """Shared base for search, relevance search, and synthesis tools."""

    DEFAULT_PROVIDERS: ClassVar[set[str]] = {"pubmed", "plos", "openalex"}

    queries: QueryList = Field(...)  # RelevanceSearch and Synthesis has a slightly different annotation
    providers: list[str] = Field(
        default_factory=lambda: list(BaseResearchToolInput.DEFAULT_PROVIDERS),
        description=(
            "Academic database providers to search. Options: "
            "pubmed, crossref, plos, arxiv, openalex, core, springernature. "
            "Default: pubmed, plos, openalex"
        ),
    )

    from_history_cache: bool = Field(
        default=False,
        description="Indicates whether the tool result should attempt to retrieve a cached result given the input",
    )

    store_history_cache: bool = Field(
        default=True,
        description="Indicates whether the tool result should be stored in cache after successful processing",
    )
    force_refresh: bool = Field(
        default=False,
        description="Forcing the re-execution of the tool while still enabling cache retrieval from previous stages",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_queries(cls, data: dict) -> dict:
        """Validates and infers the `queries` parameter from informational categories when an empty list is received.

        Uses provided queries if available and otherwise derives the field from known ResearchCategory fields.

        Args:
            dict: A dictionary of preliminary fields received prior to pydantic validation.

        Returns:
            dict: An updated model with inferred queries when otherwise missing.

        """
        # When available, return the `queries` parameter as is.
        if not data.get("queries") and "categories" in cls.model_fields:
            categories = data.get("categories") or []
            inferred_queries = BaseSearchParams.infer_queries(categories)
            data["queries"] = inferred_queries
        if queries := data.get("queries"):
            logger.info("Each API will be queried using the following queries: %s", queries)
        return data


class BaseFormatter(ABC, Generic[T]):
    """An abstract base class formatter designed as a template for formatting MCP tool responses before transmission."""

    JSON_INDENT: int = 2

    @classmethod
    def format_json_string(
        cls,
        output: SupportsModelDump | dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Formats MCP tool output models as a serialized JSON response.

        Args:
            output (SupportsModelDump | dict):
                A dictionary or pydantic model or a similar class that `model_dump()` for output serialization.

        Returns:
            str: The serialized model fields formatted as a JSON string.

        """
        if not isinstance(output, SupportsModelDump | dict):
            raise TypeError(
                f"Expected a pydantic model or class that supports `model_dump()`, but instead received {type(output).__name__}"
            )

        model_json = output if isinstance(output, dict) else output.model_dump(*args, **kwargs)
        return json.dumps(model_json, indent=cls.JSON_INDENT, default=str)

    @classmethod
    @abstractmethod
    def format(cls, output: T, response_format: ResponseFormat | str = ResponseFormat.MARKDOWN) -> str:
        """Abstract class method controlling how tool response output is formatted."""
        ...

    @classmethod
    def format_multiline_string(cls, txt: str) -> str:
        """Helper function for preparing multiline strings with dedent and line stripping."""
        return format_multiline_string(txt)  # delegates behavior to the helper, can be overridden on a formatter basis

    @classmethod
    def format_error_message(
        cls,
        error: BaseException,
        message: str | None = None,
        *,
        response_format: ResponseFormatString = ResponseFormat.MARKDOWN,
        verbose: bool = True,
    ) -> str:
        """Formats the error message as a JSON or markdown formatted string based on the specified response format."""
        err_type = type(error).__name__
        msg = message or str(error)

        if verbose:
            logger.error(f"{err_type}: {msg}")

        return (
            cls.format_json_string({"error": err_type, "message": msg})
            if response_format == ResponseFormat.JSON
            else f"{err_type}: {msg}"
        )

    @classmethod
    def format_message(cls, message: str, *, response_format: ResponseFormatString = ResponseFormat.MARKDOWN) -> str:
        """Formats a basic message for API compatibility with the selected response format (JSON or markdown."""
        return cls.format_json_string({"message": message}) if response_format == ResponseFormat.JSON else str(message)


__all__ = ["BaseToolInput", "BaseResearchToolInput", "SupportsModelDump", "BaseFormatter"]

"""Module defining core aliases used for ScholarFlux MCP type-checking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

from typing_extensions import TypeAliasType

from scholar_flux_mcp.models.schemas import (
    RelevanceSearchInput,
    RelevanceSearchOutput,
    SearchInput,
    SearchOutput,
    SearchRecord,
    SynthesisInput,
    SynthesisOutput,
)
from scholar_flux_mcp.server.io import (
    RelevanceSearchToolInput,
    SearchToolInput,
    SynthesisToolInput,
)

ResearchToolInput = TypeAliasType(
    "ResearchToolInput",
    SynthesisInput
    | SynthesisToolInput
    | RelevanceSearchInput
    | RelevanceSearchToolInput
    | SearchInput
    | SearchToolInput,
)

if TYPE_CHECKING:
    from scholar_flux_mcp.models.history import (
        ResearchHistoryExecution,
        SearchRecordElement,
        SearchRecordHistory,
    )
else:
    try:
        from scholar_flux_mcp.models.history import (
            ResearchHistoryExecution,
            SearchRecordElement,
            SearchRecordHistory,
        )
    except ImportError:
        ResearchHistoryExecution = SearchRecordElement = SearchRecordHistory = Any


ResearchToolOutput = TypeAliasType(
    "ResearchToolOutput",
    (SearchOutput | RelevanceSearchOutput | SynthesisOutput),
)

ResearchHistoryOutput = TypeAliasType(
    "ResearchHistoryOutput",
    ResearchToolOutput | ResearchHistoryExecution,
)

ResearchToolType = TypeAliasType(
    "ResearchToolType",
    Literal["search", "record_search", "relevance", "relevance_search", "synthesis", "research_synthesis", "all"],
)
RESEARCH_TOOL_TYPES: tuple[str, ...] = get_args(ResearchToolType.__value__)

SupportsTopicSimilarity = TypeVar(
    "SupportsTopicSimilarity",
    bound=ResearchToolInput | ResearchHistoryOutput | SearchRecord | SearchRecordHistory,
)

__all__ = [
    "ResearchToolInput",
    "ResearchToolOutput",
    "ResearchHistoryOutput",
    "ResearchToolType",
    "RESEARCH_TOOL_TYPES",
    "SupportsTopicSimilarity",
]

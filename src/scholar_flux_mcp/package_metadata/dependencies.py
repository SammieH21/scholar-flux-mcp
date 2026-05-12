"""Defines dependencies needed by ScholarFluxMCP and the means to identify dependencies that are installed at runtime.

Classes:
    - ServiceDependency: Dataclass used to identify a core dependency needed by at least one service
    - ScholarFluxMCPDependencies: Enum identifying required and installed dependencies used within ScholarFluxMCP

"""

import importlib.util
import logging
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Literal

from typing_extensions import TypeAliasType

from scholar_flux_mcp.exceptions.import_exceptions import (
    CoreDependencyImportError,
    MCPImportError,
    PydanticAIImportError,
    RapidFuzzImportError,
    ScholarFluxImportError,
    SQLModelImportError,
)

logger = logging.getLogger()


CoreServices = TypeAliasType(
    "CoreServices",
    Literal[
        "SearchService",
        "RelevanceSearchService",
        "SynthesisService",
        "HistoryService",
        "MCP Server",
        "GroundingService",
    ],
)


@dataclass
class ServiceDependency:
    """Helper recording core dependencies used within the `ScholarFlux` MCP server."""

    name: str
    exception: type[CoreDependencyImportError]
    required_by: list[CoreServices]


class ScholarFluxMCPDependencies(Enum):
    """Records core `ScholarFluxMCP` dependencies used throughout the MCP server."""

    SCHOLAR_FLUX = ServiceDependency(
        "scholar_flux",
        ScholarFluxImportError,
        required_by=["SearchService", "RelevanceSearchService", "SynthesisService"],
    )
    PYDANTIC_AI = ServiceDependency("pydantic_ai", PydanticAIImportError, required_by=["SynthesisService"])
    RAPIDFUZZ = ServiceDependency(
        "rapidfuzz", RapidFuzzImportError, required_by=["GroundingService", "RelevanceSearchService", "HistoryService"]
    )
    SQLMODEL = ServiceDependency("sqlmodel", SQLModelImportError, required_by=["HistoryService"])
    MCP = ServiceDependency("mcp", MCPImportError, required_by=["MCP Server"])

    @classmethod
    @cache
    def _check_dependency(cls, dependency_name: str) -> bool:
        """Check if a dependency (of ScholarFluxMCP) is available using importlib.

        Args:
            dependency_name (str): The name of the dependency to check.

        Returns:
            bool: True if the dependency is available, False otherwise.
        """
        return importlib.util.find_spec(dependency_name) is not None

    @classmethod
    def identify_missing_dependencies(cls, verbose: bool = False) -> list[ServiceDependency]:
        """Helper for identifying all missing dependencies that are not currently installed."""
        missing = [dependency.value for dependency in cls if not cls._check_dependency(dependency.value.name)]

        if missing and verbose:
            package_names = "; ".join(dep.name for dep in missing)
            logger.warning(f"The following package dependencies are missing {package_names}")
        return missing


__all__ = ["CoreServices", "ScholarFluxMCPDependencies", "ServiceDependency"]

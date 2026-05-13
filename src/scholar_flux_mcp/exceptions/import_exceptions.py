"""Defines exceptions related to missing ScholarFlux MCP dependencies."""

from typing import Any


class CoreDependencyImportError(Exception):
    """Base exception for Dependencies that are required but missing."""

    DEPENDENCY_NAME: str | None
    DEPENDENCY_TYPE: str

    def __init_subclass__(
        cls, dependency_name: str | None = None, dependency_type: str = "core", **kwargs: Any
    ) -> None:
        """Enables the declaration of the dependency name that corresponds to the current import error."""
        cls.DEPENDENCY_NAME = dependency_name
        cls.DEPENDENCY_TYPE = dependency_type or ""
        super().__init_subclass__(**kwargs)

    def __init__(self, message: str | None = None) -> None:
        """Initializes the `CoreDependencyImportError` for import-specific error handling when missing dependencies."""
        self.message = message if message else self._generate_message()
        super().__init__(self.message)

    @classmethod
    def _generate_message(cls) -> str:
        """Defines the core error message used for dependency-related errors unless overridden by subclasses."""
        dependency_type = f"{cls.DEPENDENCY_TYPE} dependency".strip().capitalize()
        return (
            f"{dependency_type}: {cls.DEPENDENCY_NAME} is not installed. Restart the ScholarFluxMCP "
            f"server after installing the '{cls.DEPENDENCY_NAME}' package to use this feature."
            if cls.DEPENDENCY_NAME is not None
            else f"{cls.DEPENDENCY_TYPE} dependency is not installed."  # Fallback if the package name isn't known
        )


class ScholarFluxImportError(CoreDependencyImportError, dependency_name="scholar-flux"):
    """Import error raised when the ScholarFlux module is not available."""

    @classmethod
    def _generate_message(cls) -> str:
        """Defines a scholar-flux specific missing dependency error message."""
        return (
            "The ScholarFlux base package is not installed. Restart the ScholarFluxMCP server after installing "
            "`ScholarFlux` via `pip install scholar-flux` to use core MCP functionality."
        )


class SQLModelImportError(CoreDependencyImportError, dependency_name="sqlmodel"):
    """Import error raised when the SQLModel module is not available."""

    pass


class MCPImportError(CoreDependencyImportError, dependency_name="mcp[cli]"):
    """Import error raised when the FastMCP is not available."""

    pass


class RapidFuzzImportError(CoreDependencyImportError, dependency_name="rapidfuzz"):
    """Import error raised when the RapidFuzz dependency is not available."""

    pass


class PydanticAIImportError(CoreDependencyImportError, dependency_name="pydantic_ai"):
    """Import error raised when the PydanticAI is not available."""

    @classmethod
    def _generate_message(cls) -> str:
        """Defines a pydantic_ai specific missing dependency error message."""
        return (
            "PydanticAI is not installed. Restart the ScholarFluxMCP server after installing `pydantic_ai` via "
            "`pip install pydantic_ai` to use the synthesis functionality."
        )


class PydanticAIProviderExtraImportError(PydanticAIImportError, dependency_name="pydantic_ai"):
    """Import error raised when a specific PydanticAI provider extra is not available."""


__all__ = [
    "CoreDependencyImportError",
    "ScholarFluxImportError",
    "SQLModelImportError",
    "MCPImportError",
    "PydanticAIImportError",
    "PydanticAIProviderExtraImportError",
    "RapidFuzzImportError",
]

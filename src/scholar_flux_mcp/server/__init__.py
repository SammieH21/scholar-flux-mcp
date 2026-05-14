"""ScholarFlux MCP Server package.

Provides the main server entry point.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scholar_flux_mcp.utils.lazy_loader import lazy_import_attr

if TYPE_CHECKING:
    from scholar_flux_mcp.server.main import create_server, mcp  # noqa: TCH004

_lazy_imports = {("scholar_flux_mcp.server.main", "create_server"), ("scholar_flux_mcp.server.main", "mcp")}


def __getattr__(name: str) -> Any:
    """Enables the lazy retrieval of the MCP server initialization components within `scholar_flux_mcp.server.main`.

    Lazy imports are not loaded until they are explicitly needed by a package resource or by a user. When `mcp[cli]` is
    not already installed, an MCPImportError will be raised only when functionality from the module is explicitly
    accessed.

    """
    return lazy_import_attr(name, _lazy_imports, __name__)


_load_mcp = __getattr__


__all__ = ["create_server", "mcp"]


def __dir__() -> list[str]:
    """Implements a basic `dir` method for the `scholar_flux_mcp.server` directory.

    Helpful for providing a list of available modules and objects for IDE autocompletion, including
    locally defined symbols and lazily imported MCP components.

    """
    return list(globals().keys()) + [object_name for (_, object_name) in _lazy_imports]

"""MCP server component tests used to verify lazy loading behavior."""

import importlib
from unittest.mock import patch

import pytest

import scholar_flux_mcp
import scholar_flux_mcp.server as server
from scholar_flux_mcp.exceptions import MCPImportError


def test_lazy_import_mcp_component_utils():
    """Testing whether the `mcp` and `create_server` MCP components can be accessed by lazy importing."""
    mcp = server.mcp
    assert mcp is not None
    # Optionally, check type or module
    assert mcp.__module__.endswith("server")  # whether imported from the mcp or scholar_flux_mcp namespace
    assert "mcp" in dir(server)


def test_lazy_import_caching():
    """Tests the `server` module to determine whether server component loads access the same object even if aliased."""
    from scholar_flux_mcp import create_server as create_server3
    from scholar_flux_mcp.server import create_server
    from scholar_flux_mcp.server import create_server as create_server2

    assert create_server is create_server2
    assert create_server is create_server3


def test_scholar_flux_mcp_component_imports():
    """Verifies that attempting to import components from the `server.main` module defers to `server._load_mcp`"""
    assert scholar_flux_mcp._load_mcp("mcp") is scholar_flux_mcp.mcp
    assert scholar_flux_mcp._load_mcp("create_server") is scholar_flux_mcp.create_server


def test_mcp_package_raises_if_nonexistent_on_import():
    """Verifies that attempting to initialize server components without the `mcp[cli]` package raises an MCPImportError.

    For this reason, the `server.main` submodule is lazy loaded, effectively deferring loading until actually needed if
    the MCP server isn't required initially.

    """

    with patch.dict("sys.modules", {"mcp.server.session": None}):
        # Shouldn't raise an error
        importlib.reload(server)

        # MCP component functionality defined here - will raise
        with pytest.raises(MCPImportError):
            importlib.reload(server.main)


def test_nonexistent_import():
    """Tests the behavior of the dynamically retrieved lazy imports when attempting to load a non-existent module.

    Validates whether attempting to load a non-existent `non_existent_mcp_component` module will raise the expected
    AttributeError.

    """
    with pytest.raises(ImportError):
        from scholar_flux.utils import non_existent_mcp_component  # noqa: F401

    with pytest.raises(AttributeError):
        _ = server.non_existent_mcp_component

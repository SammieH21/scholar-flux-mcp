"""Smoke tests for the initialization of the scholar-flux-mcp package."""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from tests.testing_utilities import raise_error


def test_package_initialization():
    """Verifies that the scholar-flux-mcp package can be imported without issue."""
    import scholar_flux_mcp

    # Exact version depends on install method vs local development
    assert isinstance(scholar_flux_mcp.__version__, str) and scholar_flux_mcp.__version__


def test_package_app_initialization():
    """Verifies that CLI calls `main()`, effectively starting the current package without error (requires FastMCP)."""
    from scholar_flux_mcp.server.main import mcp

    mcp.run = MagicMock()  # type: ignore

    from scholar_flux_mcp.__main__ import main

    main()
    mcp.run.assert_called_once()


def test_package_version_from_development(monkeypatch):
    """Verifies that the scholar-flux-mcp package still imports when a version cannot be inferred via importlib."""
    import scholar_flux_mcp

    try:
        with monkeypatch.context() as m:
            m.setattr("importlib.metadata.version", raise_error(ImportError, "Package not found"))
            importlib.reload(scholar_flux_mcp.package_metadata)
            importlib.reload(scholar_flux_mcp)
            assert "local" in scholar_flux_mcp.__version__
    finally:
        importlib.reload(scholar_flux_mcp.package_metadata)
        importlib.reload(scholar_flux_mcp)

    # Exact version depends on install method vs local development
    assert isinstance(scholar_flux_mcp.__version__, str) and scholar_flux_mcp.__version__


def test_package_app_mcp_fails_without_fastmcp():
    """Verifies that `main()` raises the expected error when fast MCP is not installed."""
    import scholar_flux_mcp.exceptions
    import scholar_flux_mcp.server.main

    # Fallback in case the run unexpectedly continues
    scholar_flux_mcp.server.main.run = MagicMock()  # type: ignore

    try:
        with patch.dict("sys.modules", {"mcp.server.session": None}):
            with pytest.raises(scholar_flux_mcp.exceptions.MCPImportError):
                importlib.reload(scholar_flux_mcp.server.main)
                scholar_flux_mcp.server.main.main()

    finally:
        importlib.reload(scholar_flux_mcp.server.main)

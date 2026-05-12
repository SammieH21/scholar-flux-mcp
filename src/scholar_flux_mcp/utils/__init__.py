"""Main entrypoint for core utilities used throughout the ScholarFluxMCP server."""

from typing import TYPE_CHECKING

from scholar_flux_mcp.utils.logging import masker, setup_mcp_logging
from scholar_flux_mcp.utils.preprocessing_utils import SearchRecordPreprocessingUtils

if TYPE_CHECKING:
    # Defining a single import location across scholar_flux_mcp for the singleton SF config
    from scholar_flux.utils import config_settings
else:
    try:
        from scholar_flux.utils import config_settings
    except ImportError:
        config_settings = None

__all__ = ["masker", "setup_mcp_logging", "SearchRecordPreprocessingUtils", "config_settings"]

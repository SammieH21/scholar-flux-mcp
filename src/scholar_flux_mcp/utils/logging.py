"""ScholarFlux MCP Server secure logging setup.

This module builds off of core scholar-flux logging and masking utilities to initialize a general sensitive data masker
and a logger that uses the masker to avoid exposing sensitive data such as passwords, tokens, and API keys.

"""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scholar_flux.security import MaskingFilter, SensitiveDataMasker
    from scholar_flux.utils import resolve_log_level, setup_logging
else:
    try:
        from scholar_flux.security import MaskingFilter, SensitiveDataMasker
        from scholar_flux.utils import resolve_log_level, setup_logging
    except ImportError:
        SensitiveDataMasker = MaskingFilter = setup_logging = resolve_log_level = None


def setup_mcp_logging(logger: logging.Logger | None = None, masker: SensitiveDataMasker | None = None) -> None:
    """Simple utility method for creating a masked logger for ScholarFluxMCP."""
    mcp_logger = logging.getLogger(__name__) if logger is None else logger

    # Configure logging before imports as a fallback for debugging in case ScholarFlux isn't installed at the time
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if masker is not None and MaskingFilter is not None and setup_logging is not None and resolve_log_level is not None:
        env_log_level = (
            resolve_log_level(os.getenv("SCHOLAR_FLUX_MCP_LOG_LEVEL")) if resolve_log_level is not None else None
        )
        log_level = env_log_level or logging.INFO
        masking_filter = MaskingFilter(masker)
        setup_logging(mcp_logger, log_level=log_level, propagate_logs=False, logging_filter=masking_filter)


masker = SensitiveDataMasker() if SensitiveDataMasker is not None else None

__all__ = ["setup_mcp_logging", "masker"]

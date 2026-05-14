"""ScholarFlux MCP Server secure logging setup.

This module builds off of core scholar-flux logging and masking utilities to initialize a general sensitive data masker
and a logger that uses the masker to avoid exposing sensitive data such as passwords, tokens, and API keys.

"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scholar_flux.security import MaskingFilter, SensitiveDataMasker
    from scholar_flux.utils import resolve_log_level, resolve_log_stream, setup_logging
else:
    try:
        from scholar_flux.security import MaskingFilter, SensitiveDataMasker
        from scholar_flux.utils import resolve_log_level, resolve_log_stream, setup_logging
    except ImportError:
        SensitiveDataMasker = MaskingFilter = setup_logging = resolve_log_level = resolve_log_stream = None


from scholar_flux_mcp.utils.config import config_settings
from scholar_flux_mcp.utils.helpers import coerce_bool, try_none, with_fallback


def setup_mcp_logging(
    logger: logging.Logger | None = None,
    *,
    masker: SensitiveDataMasker | None = None,
    log_directory: str | None = None,
    log_file: str | None = None,
    log_level: int | None = None,
    propagate_logs: bool | None = None,
    **kwargs: Any,
) -> None:
    """Simple utility method for creating a masked logger for ScholarFluxMCP."""
    mcp_logger = logging.getLogger(__name__) if logger is None else logger

    # Configure logging before imports as a fallback for debugging in case ScholarFlux isn't installed at the time
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if (
        MaskingFilter is not None
        and setup_logging is not None
        and resolve_log_level is not None
        and resolve_log_stream is not None
        and config_settings is not None
    ):
        # Note: Much of the following functionality uses `try_none` to ignore "None" and empty strings from env vars
        env_log_level = with_fallback(
            try_none(config_settings.get("SCHOLAR_FLUX_MCP_LOG_LEVEL")), config_settings.get("SCHOLAR_FLUX_LOG_LEVEL")
        )
        resolved_log_level = resolve_log_level(log_level if try_none(log_level) is not None else env_log_level)

        env_log_stream = try_none(config_settings.get("SCHOLAR_FLUX_MCP_LOG_STREAM")) or config_settings.get(
            "SCHOLAR_FLUX_LOG_STREAM"
        )
        # Defer logging stream selection to scholar-flux when `stream` is not provided
        if (
            stream := resolve_log_stream(with_fallback(try_none(kwargs.pop("stream", None)), env_log_stream))
        ) is not None:
            kwargs["stream"] = stream

        env_log_file = try_none(config_settings.get("SCHOLAR_FLUX_MCP_LOG_FILE")) or try_none(
            config_settings.get("SCHOLAR_FLUX_LOG_FILE")
        )
        log_file = log_file or env_log_file  # don't log by default if not set.
        # also generally use the same directory for base scholar-flux unless explicitly overridden
        log_directory = (
            log_directory
            or try_none(config_settings.get("SCHOLAR_FLUX_MCP_LOG_DIRECTORY"))
            or try_none(config_settings.get("SCHOLAR_FLUX_LOG_DIRECTORY"))
        )

        env_propagate_logs = with_fallback(
            coerce_bool(config_settings.get("SCHOLAR_FLUX_MCP_PROPAGATE_LOGS")),
            coerce_bool(config_settings.get("SCHOLAR_FLUX_PROPAGATE_LOGS")),
        )
        propagate_logs = with_fallback(coerce_bool(propagate_logs), env_propagate_logs) or False

        masking_filter = MaskingFilter(masker) if masker else None
        setup_logging(
            mcp_logger,
            log_level=with_fallback(resolved_log_level, logging.WARNING),
            propagate_logs=propagate_logs,  # generally no need to propagate logs unless explicitly set
            logging_filter=masking_filter,
            log_file=log_file,
            log_directory=log_directory,
            **kwargs,
        )

        if try_none(with_fallback(try_none(log_level), env_log_level)) is not None and resolved_log_level is None:
            mcp_logger.warning(f"'{env_log_level}' is not a valid log level. defaulted to log level WARNING instead.")


__all__ = ["setup_mcp_logging"]

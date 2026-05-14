"""ScholarFlux MCP Server initialization utilities.

This module provides initialization functions specifically for the ScholarFlux MCP server, building on top of the base
scholar_flux package's configuration and logging systems.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from scholar_flux_mcp.utils.config import config_settings, update_mcp_config_settings
from scholar_flux_mcp.utils.logging import setup_mcp_logging

if TYPE_CHECKING:
    from scholar_flux import masker
    from scholar_flux.security import SensitiveDataMasker
    from scholar_flux.utils import ConfigLoader
else:
    try:
        from scholar_flux import masker
    except ImportError:
        masker = None
from scholar_flux_mcp.utils.helpers import coerce_bool, with_fallback


def initialize_mcp_package(
    log: bool | None = None,
    *,
    env_path: str | None = None,
    reload_env: bool | None = None,
    reload_os_env: bool | None = None,
    logger: logging.Logger | None = None,
    verbose: bool = False,
    **logging_kwargs: Any,
) -> tuple[ConfigLoader | None, logging.Logger, SensitiveDataMasker | None]:
    """Initializes the ScholarFlux MCP server from MCP-specific components.

    This initializer assumes base scholar_flux package is already initialized and will mainly operate as a no-op
    otherwise.

    Args:
        log (bool | None):
            Whether to enable MCP logging. Defaults to None, but can be set by the `SCHOLAR_FLUX_MCP_ENABLE_LOGGING`
            environment variable.
        env_path (str | None):
            An optional path containing the ScholarFluxMCP configuration to an MCP specific .env file.
        reload_env (bool | None):
            Determines whether environment variables will be loaded/reloaded from the provided `env_path` or a
            current `config_settings.env_path`. Defaults to False, indicating that variables are not reloaded
            from a .env.
        reload_os_env (bool | None):
            Determines whether environment variables will be loaded/reloaded from the Operating System's global
            environment.
        logger (logging.Logger | None):
            Logger used for the ScholarFluxMCP server
        verbose (bool):
            Indicates whether or not to log changed configuration variable names.
        **logging_kwargs: Dictionary defining log configuration settings.

    Returns:
        Tuple[scholar_flux.ConfigLoader | None, logging.Logger, scholar_flux.security.SensitiveDataMasker | None]:
            A tuple containing the config settings, an MCP-specific logger, and a sensitive data masker
            Note that when scholar-flux is not available, the masker and logger will be returned as `None` and a
            warning will be logged. Otherwise, the configuration and logger are returned normally.

    """
    # Update MCP configuration settings
    if config_settings is not None:
        update_mcp_config_settings(
            config_settings,
            env_path=env_path,
            reload_env=reload_env,
            reload_os_env=reload_os_env,
            verbose=verbose,
        )

    # Initialize MCP-specific logger
    mcp_logger = logger if logger is not None else logging.getLogger("scholar_flux_mcp")

    env_enable_logging = (
        with_fallback(
            config_settings.get("SCHOLAR_FLUX_MCP_ENABLE_LOGGING"),
            config_settings.get("SCHOLAR_FLUX_ENABLE_LOGGING"),
        )
        if config_settings is not None
        else None
    )

    # Uses overloads to constrain the value to bool, retrieving the first boolean
    enable_logging = with_fallback(coerce_bool(with_fallback(log, env_enable_logging)), True)

    # Set up MCP logging: the enhanced setup_mcp_logging handles configuration when logging is enabled
    if enable_logging:
        # Pass through any logging_kwargs and let it handle the configuration.
        setup_mcp_logging(
            mcp_logger,
            masker=masker,
            **logging_kwargs,
        )
    else:
        mcp_logger.handlers = []
        mcp_logger.addHandler(logging.NullHandler())

    if masker is None or config_settings is None:
        # Note - the error isn't actually raised here. instead a message is generated.
        mcp_logger.warning(
            "The `scholar-flux` package is not available. Some functionality may not be operational. "
            "(e.g., record search, relevance search, research synthesis). Restart the "
            "ScholarFluxMCP server after installing `ScholarFlux` via `pip install scholar-flux` to use core MCP "
            "functionality."
        )

    return config_settings, mcp_logger, masker


__all__ = ["initialize_mcp_package", "masker"]

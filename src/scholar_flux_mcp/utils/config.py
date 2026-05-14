"""Retrieves and configures the base ScholarFlux package for use with the MCP server."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Defining a single import location across scholar_flux_mcp for the singleton SF config
    from scholar_flux.utils import ConfigLoader, config_settings
else:
    try:
        from scholar_flux.utils import config_settings
    except ImportError:
        config_settings = None

logger = logging.getLogger(__name__)

DEFAULT_MCP_CONFIG_SETTINGS: dict[str, Any] = {
    "SCHOLAR_FLUX_MCP_ENABLE_LOGGING": os.getenv("SCHOLAR_FLUX_MCP_ENABLE_LOGGING"),
    "SCHOLAR_FLUX_MCP_LOG_LEVEL": os.getenv("SCHOLAR_FLUX_MCP_LOG_LEVEL"),
    "SCHOLAR_FLUX_MCP_LOG_STREAM": os.getenv("SCHOLAR_FLUX_MCP_LOG_STREAM"),
    "SCHOLAR_FLUX_MCP_LOG_FILE": os.getenv("SCHOLAR_FLUX_MCP_LOG_FILE"),
    "SCHOLAR_FLUX_MCP_LOG_DIRECTORY": os.getenv("SCHOLAR_FLUX_MCP_LOG_DIRECTORY"),
    "SCHOLAR_FLUX_MCP_PROPAGATE_LOGS": os.getenv("SCHOLAR_FLUX_MCP_PROPAGATE_LOGS"),
}


def update_mcp_config_settings(
    config_settings: ConfigLoader,
    *,
    env_path: str | None,
    reload_env: bool | None = None,
    reload_os_env: bool | None = None,
    verbose: bool = False,
) -> None:
    """Updates the config_settings to add core env variables for ScholarFluxMCP.

    Args:
        config_settings (ConfigLoader):
            A scholar_flux.utils.config_loader.ConfigLoader instance containing the core ScholarFlux configuration.
        env_path (str | None):
            An optional path containing the ScholarFluxMCP configuration to an MCP specific .env file.
        reload_env (bool | None):
            Determines whether environment variables will be loaded/reloaded from the provided `env_path` or a
            current `config_settings.env_path`. Defaults to False, indicating that variables are not reloaded from a
            .env.
        reload_os_env (bool | None):
            Determines whether environment variables will be loaded/reloaded from the Operating System's global
            environment.
        verbose (bool):
            Indicates whether or not to log changed configuration variable names.

    """
    if config_settings is None:
        return

    # Adds the config settings to the ConfigLoader, overwritten by reload_os_env if available
    unset_env_settings = {
        env: value for env, value in DEFAULT_MCP_CONFIG_SETTINGS.items() if env not in config_settings.config
    }
    config_settings.update_config(unset_env_settings)  # add each to the `config_settings` if not already present
    config_settings.load_config(
        env_path=env_path,
        reload_env=reload_env if reload_env is not None else bool(env_path),  # if an env path is provided, update
        reload_os_env=bool(reload_os_env),  # load newly provided environment variables specific to MCP
        verbose=verbose,
    )


__all__ = ["config_settings"]

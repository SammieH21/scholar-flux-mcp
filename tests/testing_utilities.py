# /tests/testing_utilities
"""Helper modules inspired from scholar_flux and used to test ScholarFlux-MCP functionality."""

import os
from collections.abc import Callable
from typing import Optional


def enable_debugging():
    """Helper function that defines the environment variables needed to enable logging by default in ScholarFlux."""
    # Logs
    os.environ["SCHOLAR_FLUX_ENABLE_LOGGING"] = "true"
    os.environ["SCHOLAR_FLUX_PROPAGATE_LOGS"] = "true"
    os.environ["SCHOLAR_FLUX_LOG_LEVEL"] = "DEBUG"
    os.environ["SCHOLAR_FLUX_MCP_LOG_LEVEL"] = "DEBUG"


def prepare_env():
    """Helper function that temporarily configures env variables needed to enable consistent logging and testing."""
    enable_debugging()

    # Ensure we're using test configuration
    os.environ["SCHOLAR_FLUX_MCP_ENABLE_HISTORY"] = "false"

    disable_env_list = [
        "SCHOLAR_FLUX_DEFAULT_MAILTO",
        "SCHOLAR_FLUX_DEFAULT_USER_AGENT",
        "SCHOLAR_FLUX_DEFAULT_PROVIDER",
        "SCHOLAR_FLUX_CACHE_DIRECTORY",
        "SCHOLAR_FLUX_SESSION_CACHE_NAME",
        "SCHOLAR_FLUX_SESSION_CACHE_DIRECTORY",
        "SCHOLAR_FLUX_DEFAULT_LOG_DIRECTORY",
        "CORE_API_KEY",
        "PUBMED_API_KEY",
        "CROSSREF_API_KEY",
        "SPRINGER_NATURE_API_KEY",
    ]

    for env_var in disable_env_list:
        os.environ.pop(env_var, None)


def raise_error(exception_type: type[BaseException], message: Optional[str] = None) -> Callable:
    """Helper method for manually raising an error message."""
    return lambda *args, **kwargs: (_ for _ in ()).throw(exception_type(message) if message else exception_type())


__all__ = ["enable_debugging", "prepare_env", "raise_error"]

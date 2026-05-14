"""Tests for the initialization of ScholarFluxMCP.

This test suite set covers importing, configuration, and logging.
The ScholarFlux MCP server shares the basic initialization structure as the base package with a few minor differences.
As a result, much of this test suite minus MCP-specific integration was retrieved and subsequently adapted from the
base package test suite.

What's covered:
    - smoke tests for package initialization
    - logger initialization and verification
    - environment configuration testing and fallbacks

"""

import importlib
import logging
from sys import stderr, stdin, stdout
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr
from scholar_flux.utils.logger import log_level_context

from scholar_flux_mcp.utils.config import DEFAULT_MCP_CONFIG_SETTINGS
from scholar_flux_mcp.utils.initializer import config_settings, initialize_mcp_package
from scholar_flux_mcp.utils.logging import resolve_log_level, resolve_log_stream, setup_mcp_logging
from tests.testing_utilities import raise_error


def test_package_initialization():
    """Verifies that the scholar-flux-mcp package can be imported without issue."""
    import scholar_flux_mcp

    # Exact version depends on install method vs local development
    assert isinstance(scholar_flux_mcp.__version__, str) and scholar_flux_mcp.__version__


def test_package_app_main():
    """Verifies that the `__main__` entrypoint calls `main()` from the server module during package initialization."""
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
        with (
            patch.dict("sys.modules", {"mcp.server.session": None}),
            pytest.raises(scholar_flux_mcp.exceptions.MCPImportError),
        ):
            importlib.reload(scholar_flux_mcp.server.main)

    finally:
        importlib.reload(scholar_flux_mcp.server.main)


################# Log/Config Init (Inherited from ScholarFlux) ####################


def test_initialization_env_path_fallback(restore_config, recwarn, caplog):
    """Verifies that initialization records a warning and uses defaults when an env_path is of an incorrect type."""
    env_path = 10021
    updated_config_settings, _, _ = initialize_mcp_package(env_path=env_path)  # type: ignore
    assert updated_config_settings
    assert restore_config == updated_config_settings.config
    msg = (
        f"The variable, `env_path` must be a string or path, but received a variable of {type(env_path)}. "
        "Attempting to load environment settings from default .env locations instead..."
    )
    assert msg in caplog.text
    warning_message = str(recwarn[0].message)
    assert msg in warning_message


def test_logging_setup_with_directory(tmp_path, cleanup, caplog):
    """Tests whether a log file can be successfully set up in a temp directory."""
    log_file = "application.log"
    logger = logging.getLogger("test_logger")

    setup_mcp_logging(logger, log_file=log_file, log_directory=tmp_path, log_level=logging.INFO)
    assert logger.level == logging.INFO


@pytest.mark.parametrize(
    "env_log_level,should_warn,resolved,expected",
    (
        (logging.DEBUG, False, logging.DEBUG, logging.DEBUG),
        (0, False, 0, 0),
        (45, False, 45, 45),
        ("", False, None, logging.WARNING),
        ([1, 2, 3], True, None, logging.WARNING),
        ("None", False, None, logging.WARNING),
        ("Unknown Log Level", True, None, logging.WARNING),
        ("DEBUG", False, logging.DEBUG, logging.DEBUG),
        ("info", False, logging.INFO, logging.INFO),
    ),
)
def test_log_level_resolved_from_env(
    env_log_level, should_warn, resolved, expected, restore_config, monkeypatch, caplog
):
    """Tests whether `initialize_mcp_package` gracefully handles log level resolution from the env when possible."""
    assert resolve_log_level(env_log_level) == resolved  # Either resolves correctly or returns None

    logger = logging.getLogger("test_logger")
    with monkeypatch.context() as m:
        config_settings.config.pop("SCHOLAR_FLUX_LOG_LEVEL", None)
        m.delenv("SCHOLAR_FLUX_LOG_LEVEL", raising=False)

        config_settings.set("SCHOLAR_FLUX_MCP_LOG_LEVEL", str(env_log_level))
        m.setenv("SCHOLAR_FLUX_MCP_LOG_LEVEL", str(env_log_level))

        _ = initialize_mcp_package(logger=logger, propagate_logs=True)
        assert logger.level == expected
        warned = f"'{env_log_level}' is not a valid log level. defaulted to log level WARNING instead." in caplog.text
        assert warned ^ (not should_warn)


@pytest.mark.parametrize(
    "stream,expected",
    (
        (stdout, stdout),  # The default
        (stderr, stderr),  # Allow for a little grace
        (None, stderr),  # The default
        ("STDERR", stderr),
        ("STDOUT", stdout),  # STDOUT should be resolvable despite caps
        ("STD Out", stdout),  # Allow for a little grace
        ("stdout", stdout),
        (True, stderr),
        ("Incorrect stream value", stderr),  # default to stderr when invalid
        (stdin, stderr),  # Not quite the expected value for a logger
        (False, False),  # Turns off logging
        ("False", False),  # Also turns off logging after resolving "False" to `False`
        ("FALSE", False),  # Case insensitive
    ),
)
def test_log_stream_resolution(stream, expected, restore_config, caplog):
    """Tests whether `resolve_log_stream` gracefully handles stream resolution when possible."""
    assert config_settings
    assert resolve_log_stream(stream) == expected

    logger = logging.getLogger("test_logger")
    config_settings.set("SCHOLAR_FLUX_LOG_STREAM", stream)
    initialize_mcp_package(logger=logger)
    logger_stream = getattr(logger.handlers[0], "stream", None)
    assert (expected is False and isinstance(logger.handlers[0], logging.NullHandler)) or logger_stream == expected


def test_initializer_logger_creation_without_modification(caplog):
    """Tests whether the initializer can create a new logger without modifying the original "scholar_flux_mcp" logger.

    The `scholar_flux_mcp` logger is a package level logger that can be retrieved using
    `logger = logging.getLogger("scholar_flux_mcp")` and is set at the level of `DEBUG` at the beginning of the test suite.

    The `initialize_mcp_package` function is used to set up logging and masking based on the config set with environment
    variables and optional direct overrides. `new-logger` should not modify the original `scholar_flux_mcp` logger.

    """

    logger = logging.getLogger("scholar_flux_mcp")
    new_logger = logging.getLogger("new-logger")

    # setting up a new logger with the log level - ERROR (this shouldn't modify the original `scholar_flux_mcp` logger)
    _ = initialize_mcp_package(logger=new_logger, log_level=logging.WARNING, propagate_logs=True)

    assert new_logger.level == logging.WARNING
    assert logger.level == logging.DEBUG

    message = "This message should show in the logs"
    new_message = "This message shouldn't show in the logs"

    # tests ran with the `scholar_flux_mcp` logger should be displayed under the DEBUG logging level as usual
    logger.debug(message)

    # the new logger only logs at the `WARNING` level, so this shouldn't show in caplog
    new_logger.debug(new_message)

    assert new_message not in caplog.text
    assert message in caplog.text


def test_logging_context_with_new_logger(caplog):
    """Tests whether the log level for package level loggers can be successfully overridden via `log_level_context`."""
    test_logger = logging.getLogger("context-logger-testing")
    initialize_mcp_package(log=False, logger=test_logger, log_level=logging.WARNING, propagate_logs=True)

    message = "this message should show in the log"
    with log_level_context(logging.DEBUG, test_logger):
        test_logger.debug(message)
        assert message in caplog.text

    # critical log levels only
    message = "this message should NOT show in the log"
    with log_level_context(log_level=logging.CRITICAL, logger=test_logger):
        test_logger.error(message)
        assert message not in caplog.text


def test_package_level_logging_context(caplog):
    """Verifies that the `log_level_context` modifies the log level for the package logger as intended when set."""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    message = "Does this show in the logs?"
    with log_level_context(logging.CRITICAL, logger):
        logger.warning(message)
        assert message not in caplog.text
        with log_level_context(logging.DEBUG, logger):
            logger.info(message)
            assert message in caplog.text
        caplog.clear()
        with log_level_context(logging.DEBUG, allow_lower_level=False, logger=logger):
            logger.info(message)
            assert message not in caplog.text


def test_initializer_without_logging(caplog):
    """Tests whether the initializer correctly ensures that logging does not occur with log = False on setup."""
    test_logger = logging.getLogger("null-logger-testing")
    initialize_mcp_package(log=False, logger=test_logger, log_level=logging.DEBUG)
    message = "this message shouldn't show in the log"
    test_logger.debug(message)
    assert message not in caplog.text


def test_initializer_with_env(restore_config, cleanup, tmp_path, monkeypatch, caplog):
    """Tests whether the initializer can effectively use `.env` files to load config/logger environment variables."""
    test_logger = logging.getLogger("env-logger-testing")
    env_path = tmp_path / ".env"

    provider = "arXiv"
    crossref_api_key_env_var = "CROSSREF_API_KEY"
    mocked_crossref_api_key = "COMPLETELY_FAKE_API_KEY1234"

    with open(env_path, "w") as f:
        f.writelines("SCHOLAR_FLUX_MCP_LOG_LEVEL=ERROR\n")
        f.writelines(f"SCHOLAR_FLUX_DEFAULT_PROVIDER={provider}")

    with monkeypatch.context() as m:
        m.setenv(crossref_api_key_env_var, mocked_crossref_api_key)

        result_config_settings, _, _ = initialize_mcp_package(
            env_path=env_path, reload_os_env=True, logger=test_logger, propagate_logs=True
        )
        assert result_config_settings

        assert test_logger.level == logging.ERROR
        assert config_settings.get("SCHOLAR_FLUX_DEFAULT_PROVIDER") == provider
        assert config_settings is not None
        assert SecretStr(mocked_crossref_api_key) == result_config_settings.get(crossref_api_key_env_var)
        assert mocked_crossref_api_key not in caplog.text


def test_config_with_missing_env(cleanup, tmp_path, caplog):
    """Tests that failed attempts to load configurations will pass gracefully and log when the env file is missing."""
    test_logger = logging.getLogger("env-logger-testing")
    env_path = tmp_path / ".env"  # defines the path to a non-existent .env

    result_config_settings, _, _ = initialize_mcp_package(env_path=env_path, logger=test_logger)
    assert result_config_settings

    # All settings found in the result config should match the original configuration settings stored via ScholarFlux
    assert all(
        value == config_settings.get(key)
        for key, value in result_config_settings.config.items()
        if key in DEFAULT_MCP_CONFIG_SETTINGS
    )


def test_initializer_propagation(caplog):
    """Tests whether the initializer correctly specifies log message propagation consistently via `propagate_logs`."""
    test_logger = logging.getLogger("test-logger-propagation")
    initialize_mcp_package(logger=test_logger, log_level=logging.DEBUG, propagate_logs=True)
    assert test_logger.propagate is True
    message = "test log message propagation"
    test_logger.debug(message)
    assert message in caplog.text
    caplog.clear()
    initialize_mcp_package(logger=test_logger, propagate_logs=False)
    assert test_logger.propagate is False
    assert message not in caplog.text

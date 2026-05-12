"""Utility fixtures for cleaning up the environment before and after tests."""

import os

import pytest

from scholar_flux_mcp.utils import config_settings


@pytest.fixture(scope="function")
def restore_config():
    """Restores the package configuration settings and environment after the conclusion of each test when used."""
    # If the scholar-flux base package is not installed, this is a no-op
    if config_settings is not None:
        config = config_settings.config.copy()
        yield config
        config_settings.config = config
    else:
        yield


@pytest.fixture(scope="function")
def cleanup(tmp_path):
    """A helper utility that cleans up temporary files and directories created with `tmp_path` after each test."""
    yield
    # Remove all files and directories inside tmp_path
    for root, dirs, files in os.walk(tmp_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


__all__ = ["restore_config", "cleanup"]

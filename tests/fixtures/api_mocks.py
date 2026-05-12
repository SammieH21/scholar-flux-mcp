"""Defines fixtures used to test research synthesis functionality."""

import uuid

import pytest

from scholar_flux_mcp.agents.models import (
    ModelProviders,
)


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Fixture used to mock API keys for later tests."""
    google_gemini_key = str(uuid.uuid4())
    with monkeypatch.context() as m:
        m.setenv("OLLAMA_API_KEY", str(uuid.uuid4()))
        m.setenv("ANTHROPIC_API_KEY", str(uuid.uuid4()))
        m.setenv("GOOGLE_API_KEY", google_gemini_key)
        m.setenv("GEMINI_API_KEY", google_gemini_key)
        m.setenv("OPENAI_API_KEY", str(uuid.uuid4()))
        m.setenv("HUGGINGFACE_API_KEY", str(uuid.uuid4()))
        yield


@pytest.fixture
def mock_ollama_base_url(monkeypatch):
    """Fixture used to temporarily set the Ollama base URL for mock synthesis testing."""
    with monkeypatch.context() as m:
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_BASE_URL", ModelProviders.OLLAMA.value.default_base_url)
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_CLOUD_BASE_URL", ModelProviders.OLLAMA_CLOUD.value.default_base_url)
        yield


@pytest.fixture
def patch_ollama_available(monkeypatch):
    """Temporarily patches the PydanticAIProviderInfo._check_ollama_available to mock URL availability."""
    from scholar_flux_mcp.agents.models import PydanticAIModelFactory

    if not PydanticAIModelFactory._check_ollama_available():
        monkeypatch.setattr(
            PydanticAIModelFactory, "create", lambda *args, **kwargs: PydanticAIModelFactory._create_ollama_model()
        )
    yield


__all__ = ["mock_api_keys", "mock_ollama_base_url", "patch_ollama_available"]

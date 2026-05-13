"""Defines fixtures used to test research synthesis functionality."""

import uuid

import pytest

from scholar_flux_mcp.agents.models import (
    EmbeddingModelProviders,
    ModelProviders,
    PydanticAIEmbeddingModelFactory,
    PydanticAIModelFactory,
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
    """Temporarily patches the PydanticAIModelFactory and PydanticAIEmbeddingModelFactory to show ollama as available.

    Models for all other providers are checked normally.

    """
    check_llm_available = PydanticAIModelFactory._check_provider_available
    check_embedder_available = PydanticAIEmbeddingModelFactory._check_provider_available

    def mock_check_llm_provider_available(provider: str | ModelProviders) -> bool:
        """Mocks the underlying Ollama LLM as available for use."""
        model_provider = ModelProviders.get(provider)
        if model_provider is ModelProviders.OLLAMA:
            return True
        # For other providers, call the original method
        return check_llm_available(provider)

    def mock_check_embedding_provider_available(provider: str | EmbeddingModelProviders) -> bool:
        """Mocks the underlying Ollama embedding model as available for use."""
        model_provider = EmbeddingModelProviders.get(provider)
        if model_provider is EmbeddingModelProviders.OLLAMA:
            return True
        return check_embedder_available(provider)

    embedder_factory = "scholar_flux_mcp.agents.record_topic_similarity_embedder.PydanticAIEmbeddingModelFactory"
    llm_model_factory = "scholar_flux_mcp.agents.synthesis_agent.PydanticAIModelFactory"

    if not PydanticAIModelFactory._check_ollama_endpoint_available():
        monkeypatch.setattr(PydanticAIModelFactory, "_check_provider_available", mock_check_llm_provider_available)
        monkeypatch.setattr(f"{llm_model_factory}._check_provider_available", mock_check_llm_provider_available)

    if not PydanticAIEmbeddingModelFactory._check_ollama_endpoint_available():
        monkeypatch.setattr(
            PydanticAIEmbeddingModelFactory, "_check_provider_available", mock_check_embedding_provider_available
        )
        monkeypatch.setattr(f"{embedder_factory}._check_provider_available", mock_check_embedding_provider_available)
    yield


__all__ = ["mock_api_keys", "mock_ollama_base_url", "patch_ollama_available"]

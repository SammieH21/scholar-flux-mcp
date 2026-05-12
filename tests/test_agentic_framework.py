"""Tests the functionality agentic helpers used in research synthesis."""

import importlib
import logging
import uuid
from unittest.mock import MagicMock, patch

import pytest

from scholar_flux_mcp.agents import PydanticAIEmbeddingModelFactory, PydanticAIModelFactory
from scholar_flux_mcp.agents.models import (
    AnthropicModel,
    GoogleModel,
    ModelProviders,
    OpenAIChatModel,
    OpenAIEmbeddingModel,
)
from scholar_flux_mcp.agents.synthesis_agent import SynthesisAgent
from scholar_flux_mcp.exceptions import (
    AgentInitializationException,
    AgentUnavailableException,
    InvalidAgentParameterException,
)


@pytest.fixture(autouse=True)
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


@pytest.fixture(autouse=True)
def mock_base_url(monkeypatch):
    """Fixture used to temporarily set base URLs for later testing."""
    with monkeypatch.context() as m:
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_BASE_URL", ModelProviders.OLLAMA.value.default_base_url)
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_CLOUD_BASE_URL", ModelProviders.OLLAMA_CLOUD.value.default_base_url)
        m.setenv("SCHOLAR_FLUX_MCP_OPENAI_ENDPOINT", "https://api.openai.com/v1")
        yield


@pytest.fixture
def patch_check_provider_available(monkeypatch):
    """Temporarily patches the PydanticAIProviderInfo._check_provider_available to mock URL availability."""
    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(lambda url: mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("scholar_flux_mcp.agents.models.urllib.request.urlopen", mock_response)
    yield


@pytest.mark.parametrize(
    "model_name,base_url", (("my-new-model", "http://localhost:11434"), ("another-new-model", "http://ollama:11435"))
)
def test_model_factory_create_ollama_model(model_name, base_url, monkeypatch):
    """Verifies that the _create_ollama_model factory method works as intended."""

    with monkeypatch.context() as m:
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_MODEL", model_name)
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_BASE_URL", base_url)

        model = PydanticAIModelFactory._create_ollama_model()
        assert model.model_name == model_name
        assert model.base_url.removesuffix("/") == f"{base_url}/v1"


def test_model_factory_ollama_default(monkeypatch, patch_check_provider_available, caplog):
    """Verifies that the _create_ollama_model factory method works as intended."""

    model_name = "llama3.3:8b"
    base_url = "http://localhost:11437"
    with monkeypatch.context() as m, caplog.at_level(logging.INFO):
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_MODEL", model_name)
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_BASE_URL", base_url)

        assert PydanticAIModelFactory._check_ollama_available()
        model = PydanticAIModelFactory.create()
        assert isinstance(model, OpenAIChatModel)
        assert model.base_url == f"{base_url}/v1/"
        assert "Using Ollama for research synthesis" in caplog.text


def test_embedding_model_factory_ollama_default(monkeypatch, patch_check_provider_available, caplog):
    """Verifies that the _create_ollama_model factory method works as intended."""

    model_name = "embeddinggemma:latest"
    base_url = "http://localhost:11437"
    with monkeypatch.context() as m, caplog.at_level(logging.INFO):
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_EMBEDDING_MODEL", model_name)
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_EMBEDDING_BASE_URL", base_url)

        assert PydanticAIEmbeddingModelFactory._check_ollama_available()
        embedder = PydanticAIEmbeddingModelFactory.create()
        assert isinstance(embedder.model, OpenAIEmbeddingModel)
        assert embedder.model.base_url == f"{base_url}/v1/"
        assert "Using Ollama for embedding" in caplog.text


def test_model_factory_ollama_cloud_default(monkeypatch, caplog, mock_api_keys):
    """Verifies that the PydanticAIModelFactory uses Ollama Cloud if possible when local ollama is not available."""
    with monkeypatch.context() as m, caplog.at_level(logging.INFO):
        m.setattr(PydanticAIModelFactory, "_check_ollama_available", lambda: False)
        model = PydanticAIModelFactory.create()
        assert model.base_url.removesuffix("/") == "https://ollama.com/v1"


def test_model_factory_anthropic_default(monkeypatch, caplog, mock_api_keys):
    """Verifies that that the model factory defaults to Anthropic when neither Ollama and Ollama cloud are available."""
    with monkeypatch.context() as m, caplog.at_level(logging.INFO):
        m.setattr(PydanticAIModelFactory, "_check_ollama_available", lambda: False)
        m.delenv("OLLAMA_API_KEY")

        model = PydanticAIModelFactory.create()
        assert isinstance(model, AnthropicModel)
        assert "Using Anthropic Claude for research synthesis" in caplog.text


def test_model_factory_gemini_default(monkeypatch, caplog, mock_api_keys):
    """Verifies that that the model factory defaults to Google neither Ollama and Anthropic are available."""
    with monkeypatch.context() as m, caplog.at_level(logging.INFO):
        m.setattr(PydanticAIModelFactory, "_check_ollama_available", lambda: False)
        m.delenv("OLLAMA_API_KEY")
        m.delenv("ANTHROPIC_API_KEY")

        # assert not PydanticAIModelFactory._check_ollama_available()
        model = PydanticAIModelFactory.create()
        assert isinstance(model, GoogleModel)
        assert "Using Google Gen-AI for research synthesis" in caplog.text


def test_model_factory_openai_default(monkeypatch, caplog, mock_api_keys):
    """Verifies that that the model factory defaults to OpenAI as a fallback."""
    with monkeypatch.context() as m, caplog.at_level(logging.INFO):
        m.setattr(PydanticAIModelFactory, "_check_ollama_available", lambda: False)
        m.delenv("OLLAMA_API_KEY")
        m.delenv("ANTHROPIC_API_KEY")
        m.delenv("GOOGLE_API_KEY")
        m.delenv("GEMINI_API_KEY")

        # assert not PydanticAIModelFactory._check_ollama_available()
        model = PydanticAIModelFactory.create()
        assert isinstance(model, OpenAIChatModel)
        assert "Using OpenAI for research synthesis" in caplog.text


def test_model_factory_raises_on_unavailable_models(monkeypatch, caplog, mock_api_keys):
    """Verifies that the `PydanticAIModelFactory` raises an AgentUnavailableException when no model is available."""
    with monkeypatch.context() as m, caplog.at_level(logging.INFO):
        m.setattr(PydanticAIModelFactory, "_check_ollama_available", lambda: False)
        m.delenv("OLLAMA_API_KEY")
        m.delenv("ANTHROPIC_API_KEY")
        m.delenv("GOOGLE_API_KEY")
        m.delenv("GEMINI_API_KEY")
        m.delenv("OPENAI_API_KEY")

        # assert not PydanticAIModelFactory._check_ollama_available()
        with pytest.raises(AgentUnavailableException) as excinfo:
            _ = PydanticAIModelFactory.create()
        assert "No suitable model matches the provided default: [Any]" in str(excinfo.value)


def test_embedding_model_factory_raises_on_unavailable_models(monkeypatch, caplog, mock_api_keys):
    """Verifies that the embedding model factory raises an EmbedderUnavailableException when no model is available."""
    with monkeypatch.context() as m, caplog.at_level(logging.INFO):
        m.setattr(PydanticAIEmbeddingModelFactory, "_check_ollama_available", lambda: False)
        m.delenv("OLLAMA_API_KEY")
        m.delenv("GOOGLE_API_KEY")
        m.delenv("GEMINI_API_KEY")
        m.delenv("OPENAI_API_KEY")

        # assert not PydanticAIModelFactory._check_ollama_available()
        with pytest.raises(Exception) as excinfo:
            _ = PydanticAIEmbeddingModelFactory.create()
        assert "No suitable embedding model matches the provided default: [Any]" in str(excinfo.value)


@pytest.mark.parametrize("url", ({"url": "incorrect"}, "localhost:11434", 23, [1, 2, 3]))
def test_model_factory_check_ollama_available_detects_bad_urls(url, monkeypatch):
    """Verifies that `_check_ollama_available` is able to correctly handle edge cases, returning False for bad URLs."""
    with monkeypatch.context() as m:
        m.setenv("SCHOLAR_FLUX_MCP_OLLAMA_BASE_URL", str(url))
        assert PydanticAIModelFactory._check_ollama_available() is False


@pytest.mark.parametrize(
    "model_name,provider_name,endpoint",
    (
        ("gpt-4o", "OpenAI", None),
        ("lfm2.5-thinking:1.2b-bf16", "", "http://localhost:11434/v1"),
        ("gpt-5", "Azure", "https://an-example-resource.openai.azure.com"),
    ),
)
def test_model_factory_create_openai_model(model_name, provider_name, endpoint, monkeypatch, mock_api_keys):
    """Verifies that the _create_openai_model factory method works as intended."""

    with monkeypatch.context() as m:
        m.setenv("SCHOLAR_FLUX_MCP_OPENAI_MODEL", str(model_name))
        m.setenv("SCHOLAR_FLUX_MCP_OPENAI_PROVIDER", provider_name)
        m.setenv(
            "SCHOLAR_FLUX_MCP_OPENAI_ENDPOINT", str(endpoint) if endpoint or provider_name.lower() != "azure" else ""
        )

        if provider_name.lower() == "azure":
            m.setenv("AZURE_OPENAI_ENDPOINT", endpoint)

        model = PydanticAIModelFactory._create_openai_model()
        assert model.model_name == model_name
        assert (not endpoint) ^ (model.base_url.removesuffix("/") == endpoint)


@pytest.mark.parametrize(
    "provider_name,expected",
    (
        # Verifying variations on the model provider name
        ("ollama", ModelProviders.OLLAMA),
        ("anthropic", ModelProviders.ANTHROPIC),
        ("google", ModelProviders.GOOGLE),
        ("openai", ModelProviders.OPENAI),
        ("Ollama", ModelProviders.OLLAMA),
        ("Anthropic", ModelProviders.ANTHROPIC),
        ("Google", ModelProviders.GOOGLE),
        ("OpenAI", ModelProviders.OPENAI),
        # Verifying that configs resolve
        (ModelProviders.OLLAMA.value, ModelProviders.OLLAMA),
        (ModelProviders.ANTHROPIC.value, ModelProviders.ANTHROPIC),
        (ModelProviders.GOOGLE.value, ModelProviders.GOOGLE),
        (ModelProviders.OPENAI.value, ModelProviders.OPENAI),
        # Verifying that enum providers resolve back to themselves
        (ModelProviders.OLLAMA, ModelProviders.OLLAMA),
        (ModelProviders.ANTHROPIC, ModelProviders.ANTHROPIC),
        (ModelProviders.GOOGLE, ModelProviders.GOOGLE),
        (ModelProviders.OPENAI, ModelProviders.OPENAI),
    ),
)
def test_models_providers_get_resolution(provider_name, expected):
    """Verifies that `ModelProviders.get` resolves providers with name variations, configs, and enum categories."""
    assert ModelProviders.get(provider_name) is expected


@pytest.mark.parametrize("non_provider_name", ("UnknownProvider", 23, list))
def test_models_providers_get_with_invalid_names_returns_none(non_provider_name):
    """Verifies that unknown strings and invalid types return None when used with `ModelProviders.get."""
    assert ModelProviders.get(non_provider_name) is None


def test_synthesis_agent_raises_on_unavailable_models(monkeypatch, caplog, mock_api_keys):
    """Verifies that the SynthesisAgent raises a AgentInitializationException when all models are unavailable."""
    with monkeypatch.context() as m, caplog.at_level(logging.INFO):
        m.setattr(PydanticAIModelFactory, "_check_ollama_available", lambda: False)
        m.delenv("OLLAMA_API_KEY")
        m.delenv("ANTHROPIC_API_KEY")
        m.delenv("GOOGLE_API_KEY")
        m.delenv("GEMINI_API_KEY")
        m.delenv("OPENAI_API_KEY")

        synthesis_agent = SynthesisAgent()
        with pytest.raises(AgentInitializationException) as excinfo:
            _ = synthesis_agent.get_or_create_agent()
        err = "Failed to create a PydanticAI Agent for research synthesis: No suitable model matches the provided default: [Any]"
        assert err in str(excinfo.value)


def test_synthesis_agent_raises_initialization_exception_for_user_errors(monkeypatch, caplog, mock_api_keys):
    """Verifies that the SynthesisAgent will propagate errors such as missing API keys on specific model selections."""
    with monkeypatch.context() as m, caplog.at_level(logging.INFO):
        m.delenv("ANTHROPIC_API_KEY")  # will raise if an API key doesn't exist when using Claude

        synthesis_agent = SynthesisAgent()
        with pytest.raises(AgentInitializationException) as init_excinfo:
            _ = synthesis_agent.get_or_create_agent("anthropic")
        err = "PydanticAI encountered an error on LLM configuration initialization"
        assert err in str(init_excinfo.value)

    invalid_value = "Not an agent"
    with pytest.raises(InvalidAgentParameterException) as param_excinfo:
        synthesis_agent.agent = "Not an agent"  # type: ignore

    assert f"The SynthesisAgent expected a PydanticAI Agent instance, but received type {type(invalid_value)}." in str(
        param_excinfo.value
    )


def test_pydantic_ai_missing():
    """Verifies the behavior of the `scholar_flux_mcp.agents.models` module when pydantic_ai is missing."""
    import scholar_flux_mcp.agents.models

    try:
        with patch.dict("sys.modules", {"pydantic_ai": None}):
            importlib.reload(scholar_flux_mcp.agents.models)
            importlib.reload(scholar_flux_mcp.agents.synthesis_agent)
            from scholar_flux_mcp.agents.models import (
                Agent,
                AnthropicModel,
                AnthropicModelSettings,
                Embedder,
                EmbeddingResult,
                GoogleEmbeddingSettings,
                GoogleModel,
                GoogleModelSettings,
                Model,
                ModelSettings,
                OllamaProvider,
                OpenAIChatModel,
                OpenAIChatModelSettings,
                OpenAIEmbeddingSettings,
                OpenAIProvider,
                PydanticAIImportError,
                PydanticAIModelFactory,
            )
            from scholar_flux_mcp.agents.synthesis_agent import SynthesisAgent

            assert Agent is None
            assert Model is None
            assert Embedder is None
            assert EmbeddingResult is None
            assert OpenAIChatModel is None
            assert AnthropicModel is None
            assert GoogleModel is None
            assert OllamaProvider is None
            assert OpenAIProvider is None
            assert ModelSettings is dict
            assert OpenAIChatModelSettings is dict
            assert OpenAIEmbeddingSettings is dict
            assert AnthropicModelSettings is dict
            assert GoogleModelSettings is dict
            assert GoogleEmbeddingSettings is dict

            err = "PydanticAI is not installed. Restart the ScholarFluxMCP server after installing `pydantic_ai`"
            with pytest.raises(PydanticAIImportError) as model_factory_excinfo:
                _ = PydanticAIModelFactory.create("ollama")
            assert err in str(model_factory_excinfo.value)

            synthesis_agent = SynthesisAgent()
            with pytest.raises(PydanticAIImportError) as agent_creation_excinfo:
                synthesis_agent.get_or_create_agent("anthropic")
            assert err in str(agent_creation_excinfo.value)

    finally:
        importlib.reload(scholar_flux_mcp.agents.models)
        importlib.reload(scholar_flux_mcp.agents.synthesis_agent)

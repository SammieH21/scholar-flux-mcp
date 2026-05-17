"""Defines the default Agent and Embedding frameworks for ScholarFluxMCP as subclassable components.

Uses PydanticAI to define the base classes and factory utilities that aid in the synthesis of research findings from
normalized academic papers.

The PydanticAIModelFactory and PydanticAIEmbeddingModelFactory classes are used to simplify the process of instantiating
a model subclass for later synthesis using environment variables. If not explicitly specified, model configurations for
agents and embeddings are selected based on priority.

Fallback Chain:
    - LLM models:       Ollama (local) → Ollama (Cloud) → Anthropic (cloud) → Google (cloud) → OpenAI (cloud)
    - Embedding models: Ollama (local) → Google → OpenAI


Public API:
  - ModelProviders: Enum indicating the full range of pre-defined LLM providers available for use.
  - EmbeddingModelProviders: Enum indicating the full range of pre-defined embedding model providers available for use.
  - PydanticAIModelFactory: Helper class used for the (stateless) creation of large language model configurations.
  - AgentABC: An abstract base class defining the structure for all future agents created for ScholarFluxMCP.
  - AgentDepsType: A type variable representing the input dependencies for future agents.
  - AgentOutputType: A type variable representing the output structure of agent responses.
  - PydanticAIEmbeddingModelFactory: Helper class used for the (stateless) creation of Embedding configurations.
  - EmbedderABC: Abstract base class defining the structure for all embedding subclasses created for ScholarFluxMCP.

"""

from __future__ import annotations

import contextlib
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel

from scholar_flux_mcp.exceptions import (
    AgentUnavailableException,
    AgentUninitializedException,
    EmbedderUnavailableException,
    EmbedderUninitializedException,
    InvalidAgentParameterException,
    InvalidEmbedderParameterException,
    PydanticAIImportError,
    PydanticAIProviderExtraImportError,
)
from scholar_flux_mcp.models.core import computed_property

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.embeddings import Embedder, EmbeddingResult, EmbeddingSettings
    from pydantic_ai.embeddings.google import GoogleEmbeddingModel, GoogleEmbeddingSettings
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel, OpenAIEmbeddingSettings
    from pydantic_ai.models import Model
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings
else:
    try:
        from pydantic_ai import Agent
        from pydantic_ai.embeddings import Embedder, EmbeddingResult, EmbeddingSettings
        from pydantic_ai.models import Model
        from pydantic_ai.settings import ModelSettings
    except ImportError:
        Agent = None
        Embedder = None
        EmbeddingResult = None
        Model = None
        # ModelSettings are dicts: Prevent issues with loading when PydanticAI/Google is not already installed
        ModelSettings = dict
        EmbeddingSettings = dict

    try:
        from pydantic_ai.embeddings.google import GoogleEmbeddingModel, GoogleEmbeddingSettings
        from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    except ImportError:
        GoogleModel = None
        GoogleEmbeddingModel = None
        GoogleModelSettings = dict
        GoogleEmbeddingSettings = dict
    try:
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    except ImportError:
        AnthropicModel = None
        AnthropicModelSettings = dict
    try:
        # Shared Ollama and OpenAI settings:
        from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel, OpenAIEmbeddingSettings
        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    except ImportError:
        OpenAIChatModel = None
        OpenAIEmbeddingModel = None
        OpenAIChatModelSettings = dict
        OpenAIEmbeddingSettings = dict
    try:
        # attempt import separately from shared config for Ollama and OpenAI
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError:
        OpenAIProvider = None
    try:
        from pydantic_ai.providers.ollama import OllamaProvider
    except ImportError:
        OllamaProvider = None

from scholar_flux_mcp.utils.helpers import coerce_numeric

logger = logging.getLogger(__name__)


AgentDepsType = TypeVar("AgentDepsType")
AgentOutputType = TypeVar("AgentOutputType", bound=BaseModel)
ModelSettingsType = TypeVar("ModelSettingsType", bound=ModelSettings)


@dataclass
class BaseProviderSettings:
    """A model-providers dataclass that defines the model settings associated with a specific provider.

    Attributes:
        name (str):
            The name of the current provider.
        default_base_url (str):
            The URL for the current provider. Defaults to an empty string.
        _url_env_var (str | None):
            Indicates the name of provider URL environment variable that should override the default.
        _provider_env_var (str | None):
            Environment variable indicating the name of provider that should override the default (OpenAI-specific).
        _api_key_env_var (str | list[str] | None):
            Indicates the API key environment variable for determining provider availability.
        _pydantic_ai_extra (str | None):
            Indicates the PydanticAI extra that is required for the current model provider.
        _dependency_available (bool):
            Indicates whether the current dependency is available.

    """

    name: str = field(kw_only=True)
    default_base_url: str = field(default="")
    _url_env_var: str | None = field(default=None, repr=False)
    _provider_env_var: str | None = field(default=None, repr=False)
    _api_key_env_var: str | list[str] | None = field(default=None, repr=False)
    _pydantic_ai_extra: str | None = field(default=None, repr=False)
    _dependency_available: bool = field(default=True, repr=False)  # defined at runtime by checking imports

    @computed_property
    def base_url(self) -> str:
        """Returns the base URL for the current provider if applicable to the provider."""
        base_url: str | None = os.getenv(self._url_env_var) if self._url_env_var else None
        return base_url or self.default_base_url

    @computed_property
    def provider(self) -> str:
        """Returns the name of the current provider."""
        custom_provider = os.getenv(self._provider_env_var) if self._provider_env_var else None
        provider = custom_provider or self.name
        return provider.lower()

    @property
    def dependency_available(self) -> bool:
        """Indicates whether the extra dependency needed for the PydanticAI provider is available."""
        if not self._dependency_available:
            logger.debug(f"The PydanticAI extra dependency for the provider, `{self.name}`, is not installed.")
        return self._dependency_available

    def validate_dependency(self) -> None:
        """Verifies that the PydanticAI extra dependency for the current provider is available."""
        if not self._dependency_available:
            extra = f"pydantic-ai-slim['{self._pydantic_ai_extra}'] " if self._pydantic_ai_extra else ""
            installation_command = f"via `pip install {extra.rstrip()}` " if extra else ""
            raise PydanticAIProviderExtraImportError(
                f"The PydanticAI extra {extra}for `{self.__class__.__name__}` is not installed. Restart the "
                f"ScholarFluxMCP server after installing the missing extra {installation_command}to use the provider."
            )

    @property
    def api_key_available(self) -> bool:
        """Indicates whether the API key for the current provider is available."""
        api_key_env_vars = [self._api_key_env_var] if isinstance(self._api_key_env_var, str) else self._api_key_env_var
        return bool(api_key_env_vars and any(bool(os.getenv(env_var)) for env_var in api_key_env_vars))


@dataclass
class ModelProviderSettings(BaseProviderSettings):
    """A model-providers dataclass that defines the model settings associated with a specific provider.

    Attributes:
        default_model (str): The default LLM for a provider that is selected when initializing with default settings.
        model_settings (ModelSettings | None): An optional dictionary of settings for the current LLM.
        timeout (float | None): The timeout in seconds for requests to the current model provider.
        _model_env_var (str | None): Indicates the name of LLM that should override the provider default model.

    """

    default_model: str = field(kw_only=True)
    model_settings: ModelSettings | None = field(default=None, repr=False)
    timeout: float | None = field(default=None)
    _model_env_var: str | None = field(default=None, repr=False)

    @computed_property
    def model(self) -> str:
        """Returns the model for the current provider."""
        model: str | None = os.getenv(self._model_env_var) if self._model_env_var else None
        return model or self.default_model


@dataclass
class EmbeddingModelProviderSettings(BaseProviderSettings):
    """A model-providers dataclass that defines the model settings associated with a specific provider.

    Attributes:
        default_model (str): The default embedder for a provider that is selected when initializing with defaults.
        model_settings (EmbeddingSettings | None): An optional dictionary of settings for the current embedding model.
        _embedding_model_env_var (str | None): Indicates the name of embedder that should override the provider default.

    """

    default_model: str = field(kw_only=True)
    model_settings: EmbeddingSettings | None = field(default=None, repr=False)
    _embedding_model_env_var: str | None = field(default=None, repr=False)

    @computed_property
    def model(self) -> str:
        """Returns the embedding model for the current provider."""
        embedding_model: str | None = (
            os.getenv(self._embedding_model_env_var) if self._embedding_model_env_var else None
        )
        return embedding_model or self.default_model


class ModelProviders(Enum):
    """A `ModelProviders` enum that defines all default providers and their associated settings."""

    OLLAMA = ModelProviderSettings(
        name="ollama",
        default_model="glm-4.7-flash:latest",
        default_base_url="http://localhost:11434",
        timeout=300,
        _model_env_var="SCHOLAR_FLUX_MCP_OLLAMA_MODEL",
        _url_env_var="SCHOLAR_FLUX_MCP_OLLAMA_BASE_URL",
        _api_key_env_var="OLLAMA_API_KEY",
        _pydantic_ai_extra="openai",
        _dependency_available=OllamaProvider is not None and OpenAIChatModel is not None,
    )

    OLLAMA_CLOUD = ModelProviderSettings(
        name="ollama_cloud",
        default_model="devstral-2:123b-cloud",
        default_base_url="https://ollama.com/v1",
        _model_env_var="SCHOLAR_FLUX_MCP_OLLAMA_CLOUD_MODEL",
        _url_env_var="SCHOLAR_FLUX_MCP_OLLAMA_CLOUD_BASE_URL",
        _api_key_env_var="OLLAMA_API_KEY",
        _pydantic_ai_extra="openai",
        _dependency_available=OllamaProvider is not None and OpenAIChatModel is not None,
    )

    ANTHROPIC = ModelProviderSettings(
        name="anthropic",
        default_model="claude-haiku-4-5",
        model_settings=AnthropicModelSettings(
            anthropic_cache_instructions="1h",
            anthropic_cache_messages="1h",
            anthropic_cache_tool_definitions="1h",
        ),
        _model_env_var="SCHOLAR_FLUX_MCP_ANTHROPIC_MODEL",
        _api_key_env_var="ANTHROPIC_API_KEY",
        _pydantic_ai_extra="anthropic",
        _dependency_available=AnthropicModel is not None,
    )
    GOOGLE = ModelProviderSettings(
        name="google",
        default_model="gemini-2.5-flash",
        _model_env_var="SCHOLAR_FLUX_MCP_GOOGLE_MODEL",
        _api_key_env_var=["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        _pydantic_ai_extra="google",
        _dependency_available=GoogleModel is not None,
    )
    OPENAI = ModelProviderSettings(
        name="openai",
        default_model="gpt-5",
        model_settings=OpenAIChatModelSettings(
            # Enable extended 24h cache retention for repeated synthesis contexts
            openai_prompt_cache_retention="24h",
            # Use 'flex' tier for 50% discount on non-real-time batch processing
            # Switch to 'auto' or 'default' if real-time responses are required
            openai_service_tier="auto",
        ),
        _model_env_var="SCHOLAR_FLUX_MCP_OPENAI_MODEL",
        _url_env_var="SCHOLAR_FLUX_MCP_OPENAI_ENDPOINT",
        _provider_env_var="SCHOLAR_FLUX_MCP_OPENAI_PROVIDER",
        _api_key_env_var="OPENAI_API_KEY",
        _pydantic_ai_extra="openai",
        _dependency_available=OpenAIChatModel is not None and OpenAIProvider is not None,
    )

    @classmethod
    def get(cls, provider: str | ModelProviderSettings | ModelProviders) -> ModelProviders | None:
        """Attempts to retrieve `ModelProviderSettings` metadata for the current provider.

        Returns:
            ModelProviders: The provider entry when the provider is registered within the `ModelProviders` enum
            None: When the provider cannot be found or an invalid type is passed.

        """
        if isinstance(provider, cls | ModelProviderSettings):
            provider = provider.name

        try:
            return cls[provider.upper()] if isinstance(provider, str) else None

        except (KeyError, TypeError, ValueError):
            return None


class EmbeddingModelProviders(Enum):
    """An `EmbeddingModelProviders` enum that defines all default embedding models by API model provider."""

    OLLAMA = EmbeddingModelProviderSettings(
        name="ollama",
        default_model="embeddinggemma:latest",
        default_base_url="http://localhost:11434",
        _embedding_model_env_var="SCHOLAR_FLUX_MCP_OLLAMA_EMBEDDING_MODEL",
        _url_env_var="SCHOLAR_FLUX_MCP_OLLAMA_EMBEDDING_BASE_URL",
        _api_key_env_var="OLLAMA_API_KEY",
        _pydantic_ai_extra="openai",
        _dependency_available=OllamaProvider is not None and OpenAIEmbeddingModel is not None,
    )

    GOOGLE = EmbeddingModelProviderSettings(
        name="google",
        default_model="gemini-embedding-001",
        _embedding_model_env_var="SCHOLAR_FLUX_MCP_GOOGLE_EMBEDDING_MODEL",
        _api_key_env_var=["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        _pydantic_ai_extra="google",
        _dependency_available=GoogleEmbeddingModel is not None,
    )
    OPENAI = EmbeddingModelProviderSettings(
        name="openai",
        default_model="text-embedding-3-small",
        _embedding_model_env_var="SCHOLAR_FLUX_MCP_OPENAI_EMBEDDING_MODEL",
        _url_env_var="SCHOLAR_FLUX_MCP_OPENAI_EMBEDDING_ENDPOINT",
        _provider_env_var="SCHOLAR_FLUX_MCP_OPENAI_EMBEDDING_PROVIDER",
        _api_key_env_var="OPENAI_API_KEY",
        _pydantic_ai_extra="openai",
        _dependency_available=OpenAIProvider is not None and OpenAIEmbeddingModel is not None,
    )

    @classmethod
    def get(
        cls, provider: str | EmbeddingModelProviderSettings | EmbeddingModelProviders
    ) -> EmbeddingModelProviders | None:
        """Attempts to retrieve `EmbeddingModelProviderSettings` metadata for the current provider.

        Returns:
            EmbeddingModelProviders:
                The embedding model provider entry for the matching embedding model provider.
                None: When the provider of the embedding model cannot be found or an invalid type is passed.

        """
        if isinstance(provider, cls | EmbeddingModelProviderSettings):
            provider = provider.name

        try:
            return cls[provider.upper()] if isinstance(provider, str) else None

        except (KeyError, TypeError, ValueError):
            return None


class PydanticAIProviderInfo:
    """Helper class containing core methods that define a provider's availability and possible configurations."""

    @classmethod
    def _check_endpoint_available(cls, url: str) -> bool:
        """Helper method that verifies whether a URL is accessible by sending a request."""
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False


class PydanticAIModelFactory:
    """Helper class for the selection and creation of basic PydanticAI model configurations for Agent creation."""

    # Default model to use for agentic workflows with PydanticAI
    DEFAULT_MODEL_PROVIDER: str | None = os.getenv("SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER") or None
    # Connection defaults
    DEFAULT_REQUEST_TIMEOUT: float = coerce_numeric(os.getenv("SCHOLAR_FLUX_MCP_REQUEST_TIMEOUT")) or 120

    @classmethod
    def _check_ollama_endpoint_available(cls, base_url: str | None = None) -> bool:
        """Check if Ollama server is running."""
        endpoint = f"{base_url or ModelProviders.OLLAMA.value.base_url}/api/tags"
        return PydanticAIProviderInfo._check_endpoint_available(endpoint)

    @classmethod
    def _check_provider_available(cls, provider: str | ModelProviders) -> bool:
        """Check whether a provider is available and whether an API key (if needed) can be found from the OS env."""
        model_provider = ModelProviders.get(provider)
        if model_provider is ModelProviders.OLLAMA:
            ollama_available = cls._check_ollama_endpoint_available()
            return ollama_available and model_provider.value.dependency_available
        if model_provider is ModelProviders.OLLAMA_CLOUD:
            return ModelProviders.OLLAMA_CLOUD.value.api_key_available and model_provider.value.dependency_available
        if model_provider is ModelProviders.ANTHROPIC:
            return ModelProviders.ANTHROPIC.value.api_key_available and model_provider.value.dependency_available
        if model_provider is ModelProviders.GOOGLE:
            return ModelProviders.GOOGLE.value.api_key_available and model_provider.value.dependency_available
        if model_provider is ModelProviders.OPENAI:
            requires_api_key = not model_provider.value.base_url and model_provider.value.provider.lower() == "openai"
            api_key_missing = requires_api_key and not model_provider.value.api_key_available
            return not api_key_missing and model_provider.value.dependency_available
        return False

    @classmethod
    def create(cls, provider: str | None = None) -> OpenAIChatModel | AnthropicModel | GoogleModel:
        """Configures a PydanticAI model based on the user-specified model default for research synthesis.

        Returns:
            OpenAIChatModel | AnthropicModel | GoogleModel:
                A model configuration that uses either Anthropic, Google, or an OpenAI-compatible endpoint such as
                Ollama.

        Note:
            This factory method checks for model availability in the following order of priority:

            Ollama (local) → Ollama (Cloud) → Anthropic (cloud) → Google (cloud) → OpenAI (cloud).

            If a default is not specified and an API key is required for a provider but unavailable, this method will
            automatically use the following provider if available. If no providers are available, an
            `AgentUnavailableException` is raised instead.

        """
        if Agent is None or Model is None or ModelSettings is None:
            raise PydanticAIImportError()
        provider_default: str | None = (
            provider or os.getenv("SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER") or cls.DEFAULT_MODEL_PROVIDER or None
        )

        # Check explicit preference
        provider_settings = ModelProviders.get(provider_default) if provider_default else None

        if (
            provider_default is None and cls._check_provider_available(ModelProviders.OLLAMA)
        ) or provider_settings is ModelProviders.OLLAMA:
            logger.info("Using Ollama for research synthesis")
            return cls._create_ollama_model()
        if (
            provider_default is None and cls._check_provider_available(ModelProviders.OLLAMA_CLOUD)
        ) or provider_settings is ModelProviders.OLLAMA_CLOUD:
            logger.info("Using Ollama Cloud for research synthesis")
            return cls._create_ollama_cloud_model()
        if (
            provider_default is None and cls._check_provider_available(ModelProviders.ANTHROPIC)
        ) or provider_settings is ModelProviders.ANTHROPIC:
            logger.info("Using Anthropic Claude for research synthesis")
            return cls._create_anthropic_model()

        if (
            provider_default is None and cls._check_provider_available(ModelProviders.GOOGLE)
        ) or provider_settings is ModelProviders.GOOGLE:
            logger.info("Using Google Gen-AI for research synthesis")
            return cls._create_google_model()

        if (
            provider_default is None and cls._check_provider_available(ModelProviders.OPENAI)
        ) or provider_settings is ModelProviders.OPENAI:
            logger.info("Using OpenAI for research synthesis")
            try:
                return cls._create_openai_model()
            # PydanticAI may raise an uncaught OpenAIError when an API key doesn't exist.
            except Exception as e:
                # if OpenAI was explicitly requested
                if provider_settings is ModelProviders.OPENAI:
                    logger.exception(e)
                    raise
                logger.warning(e, exc_info=True)

        unsuccessful_default = provider_default if provider_default is not None else "[Any]"
        raise AgentUnavailableException(
            f"No suitable model matches the provided default: {unsuccessful_default}. "
            "Install and run Ollama (https://ollama.com) for local inference and set `SCHOLAR_FLUX_MCP_OLLAMA_MODEL` "
            "or set one of ANTHROPIC_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY for cloud providers."
        )

    @classmethod
    def _create_ollama_cloud_model(
        cls, settings: OpenAIChatModelSettings | ModelSettings | None = None, timeout: float | None = None
    ) -> OpenAIChatModel:
        """Creates an Ollama configuration via the OpenAIChatModel compatible endpoint."""
        ollama_cloud_defaults = ModelProviders.OLLAMA_CLOUD.value
        ollama_cloud_defaults.validate_dependency()
        model_settings = cls._add_timeout(
            settings or ollama_cloud_defaults.model_settings or OpenAIChatModelSettings(),
            timeout=timeout or ollama_cloud_defaults.timeout,
        )

        return OpenAIChatModel(
            model_name=ollama_cloud_defaults.model,
            provider=OllamaProvider(base_url=ollama_cloud_defaults.base_url),
            settings=model_settings,
        )

    @classmethod
    def _create_ollama_model(
        cls, settings: OpenAIChatModelSettings | ModelSettings | None = None, timeout: float | None = None
    ) -> OpenAIChatModel:
        """Creates an Ollama configuration via the OpenAIChatModel compatible endpoint."""
        ollama_defaults = ModelProviders.OLLAMA.value
        ollama_defaults.validate_dependency()
        model_name: str = ollama_defaults.model
        base_url: str = ollama_defaults.base_url
        model_settings = cls._add_timeout(
            settings or ollama_defaults.model_settings or OpenAIChatModelSettings(),
            timeout=timeout or ollama_defaults.timeout,
        )

        return OpenAIChatModel(
            model_name=model_name,
            provider=OllamaProvider(base_url=f"{base_url.removesuffix('/v1')}/v1"),
            settings=model_settings,
        )

    @classmethod
    def _create_anthropic_model(
        cls, settings: AnthropicModelSettings | ModelSettings | None = None, timeout: float | None = None
    ) -> AnthropicModel:
        """Creates an AnthropicModel configuration."""
        anthropic_defaults = ModelProviders.ANTHROPIC.value
        anthropic_defaults.validate_dependency()
        model_name: str = anthropic_defaults.model
        model_settings = cls._add_timeout(
            settings or anthropic_defaults.model_settings or AnthropicModelSettings(),
            timeout=timeout or anthropic_defaults.timeout,
        )

        return AnthropicModel(
            model_name=model_name,
            settings=model_settings,
        )

    @classmethod
    def _create_google_model(
        cls, settings: GoogleModelSettings | ModelSettings | None = None, timeout: float | None = None
    ) -> GoogleModel:
        """Creates a Google Model configuration."""
        google_defaults = ModelProviders.GOOGLE.value
        google_defaults.validate_dependency()

        model_settings = cls._add_timeout(
            settings or google_defaults.model_settings or GoogleModelSettings(),
            timeout=timeout or google_defaults.timeout,
        )

        return GoogleModel(
            model_name=google_defaults.model,
            settings=model_settings,
        )

    @classmethod
    def _create_openai_model(
        cls, settings: OpenAIChatModelSettings | ModelSettings | None = None, timeout: float | None = None
    ) -> OpenAIChatModel:
        """Create an OpenAIChatModel configuration."""
        openai_defaults = ModelProviders.OPENAI.value
        openai_defaults.validate_dependency()
        provider_name: str = openai_defaults.provider
        base_url: str | None = openai_defaults.base_url
        model_settings = cls._add_timeout(
            settings or openai_defaults.model_settings or OpenAIChatModelSettings(),
            timeout=timeout or openai_defaults.timeout,
        )

        return OpenAIChatModel(
            model_name=openai_defaults.model,
            provider=OpenAIProvider(base_url=base_url) if base_url else provider_name,  # type: ignore
            settings=model_settings,
        )

    @classmethod
    def _default_timeout(cls) -> float:
        """Returns the default timeout that PydanticAI should wait before retrying the request."""
        return coerce_numeric(os.getenv("SCHOLAR_FLUX_MCP_REQUEST_TIMEOUT")) or cls.DEFAULT_REQUEST_TIMEOUT

    @classmethod
    def _add_timeout(
        cls,
        settings: ModelSettingsType,
        timeout: float | None = None,
    ) -> ModelSettingsType:
        """Adds a timeout to the current `ModelSettings` type when possible.

        Otherwise returns the value as is.

        """
        timeout_settings = coerce_numeric(timeout) or cls._default_timeout()
        settings["timeout"] = timeout_settings
        return settings


class PydanticAIEmbeddingModelFactory:
    """Helper class for the selection and creation of basic PydanticAI embedding models configurations."""

    # Default model to use for agentic workflows with PydanticAI
    DEFAULT_MODEL_PROVIDER: str | None = os.getenv("SCHOLAR_FLUX_MCP_DEFAULT_EMBEDDING_MODEL_PROVIDER") or None

    @classmethod
    def _check_ollama_endpoint_available(cls, base_url: str | None = None) -> bool:
        """Check if Ollama server is running."""
        endpoint = f"{base_url or EmbeddingModelProviders.OLLAMA.value.base_url}/api/tags"
        return PydanticAIProviderInfo._check_endpoint_available(endpoint)

    @classmethod
    def _check_provider_available(cls, provider: str | EmbeddingModelProviders) -> bool:
        """Check if an embedding provider is available and whether an API key (if needed) can be found from the env."""
        model_provider = EmbeddingModelProviders.get(provider)
        if model_provider is EmbeddingModelProviders.OLLAMA:
            ollama_available = cls._check_ollama_endpoint_available()
            return ollama_available and model_provider.value.dependency_available
        if model_provider is EmbeddingModelProviders.GOOGLE:
            return model_provider.value.dependency_available and model_provider.value.api_key_available
        if model_provider is EmbeddingModelProviders.OPENAI:
            # OpenAI services that is known to require an api key
            requires_api_key = not model_provider.value.base_url and model_provider.value.provider.lower() == "openai"
            api_key_missing = requires_api_key and not model_provider.value.api_key_available
            return not api_key_missing and model_provider.value.dependency_available
        return False

    @classmethod
    def create(cls, provider: str | None = None) -> Embedder:
        """Configures a PydanticAI embedding model based on the user-specified model default for research embeddings.

        Note:
            This factory method checks for embedding model availability in the following order of priority:

            Ollama (local) → Google (cloud) → OpenAI (cloud).

            If a default is not specified and an API key is required for a provider but unavailable, this method will
            automatically use the following provider if available. If no providers are available, an
            EmbedderUnavailableException is raised instead.

        Returns:
            Embedder:
                An embedding model configuration that can be directly called and used with `embed_topic`

        """
        if Embedder is None or EmbeddingSettings is None:
            raise PydanticAIImportError()
        provider_default: str | None = (
            provider
            or os.getenv("SCHOLAR_FLUX_MCP_DEFAULT_EMBEDDING_MODEL_PROVIDER")
            or cls.DEFAULT_MODEL_PROVIDER
            or None
        )

        # Check explicit preference
        provider_settings = EmbeddingModelProviders.get(provider_default) if provider_default else None

        if (
            provider_default is None and cls._check_provider_available(EmbeddingModelProviders.OLLAMA)
        ) or provider_settings is EmbeddingModelProviders.OLLAMA:
            logger.info("Using Ollama for embeddings")
            return cls._create_ollama_model()

        if (
            provider_default is None and cls._check_provider_available(EmbeddingModelProviders.GOOGLE)
        ) or provider_settings is EmbeddingModelProviders.GOOGLE:
            logger.info("Using Google Gen-AI for embeddings")
            return cls._create_google_model()

        if (
            provider_default is None and cls._check_provider_available(EmbeddingModelProviders.OPENAI)
        ) or provider_settings is EmbeddingModelProviders.OPENAI:
            logger.info("Using OpenAI for embeddings")
            try:
                return cls._create_openai_model()
            # PydanticAI may raise an uncaught OpenAIError when an API key doesn't exist.
            except Exception as e:
                # if OpenAI was explicitly requested
                if provider_settings is EmbeddingModelProviders.OPENAI:
                    logger.exception(e)
                    raise
                logger.warning(e, exc_info=True)

        unsuccessful_default = provider_default if provider_default is not None else "[Any]"
        raise EmbedderUnavailableException(
            f"No suitable embedding model matches the provided default: {unsuccessful_default}. "
            "Install and run Ollama (https://ollama.com) for local embeddings and set "
            "`SCHOLAR_FLUX_MCP_OLLAMA_EMBEDDING_MODEL`, or set GOOGLE_API_KEY or OPENAI_API_KEY for cloud embeddings."
        )

    @classmethod
    def _create_ollama_model(cls, settings: OpenAIEmbeddingSettings | EmbeddingSettings | None = None) -> Embedder:
        """Create a local Ollama embedding model configuration."""
        ollama_defaults = EmbeddingModelProviders.OLLAMA.value
        ollama_defaults.validate_dependency()
        model_name: str = ollama_defaults.model
        base_url: str | None = ollama_defaults.base_url

        model = OpenAIEmbeddingModel(
            model_name=model_name,
            provider=OllamaProvider(base_url=f"{base_url}/v1"),
            settings=settings or ollama_defaults.model_settings or OpenAIEmbeddingSettings(),
        )
        return Embedder(model)

    @classmethod
    def _create_google_model(cls, settings: GoogleEmbeddingSettings | EmbeddingSettings | None = None) -> Embedder:
        """Create a Google embedding model configuration."""
        google_defaults = EmbeddingModelProviders.GOOGLE.value
        google_defaults.validate_dependency()
        model_name: str = google_defaults.model

        model = GoogleEmbeddingModel(
            model_name=model_name, settings=settings or google_defaults.model_settings or GoogleEmbeddingSettings()
        )
        return Embedder(model)

    @classmethod
    def _create_openai_model(cls, settings: OpenAIEmbeddingSettings | EmbeddingSettings | None = None) -> Embedder:
        """Create an OpenAI embedding model configuration."""
        openai_defaults = EmbeddingModelProviders.OPENAI.value
        openai_defaults.validate_dependency()
        base_url: str | None = openai_defaults.base_url
        provider_name: str = openai_defaults.provider

        model = OpenAIEmbeddingModel(
            model_name=openai_defaults.model,
            provider=OpenAIProvider(base_url=base_url) if base_url else provider_name,  # type: ignore
            settings=settings or openai_defaults.model_settings or OpenAIEmbeddingSettings(),
        )
        return Embedder(model)


class AgentABC(ABC, Generic[AgentDepsType, AgentOutputType]):
    """Abstract base class that defines the basic methods used by ScholarFluxMCP to execute Agentic workflows."""

    def __init__(self, agent: Agent[AgentDepsType, AgentOutputType] | None = None, *args: Any, **kwargs: Any) -> None:
        """Defines the initialization of the Agents."""
        self.agent = agent

    @property
    def agent(self) -> Agent[AgentDepsType, AgentOutputType]:
        """Returns the agent that synthesizes academic records into a research summary."""
        if not self._agent:
            class_name = self.__class__.__name__
            raise AgentUninitializedException(f"A PydanticAI Agent has not yet been created by the {class_name}.")
        return self._agent

    @agent.setter
    def agent(self, agent: Agent[AgentDepsType, AgentOutputType] | None = None) -> None:
        """Sets the agent with class validation, raising a `TypeError` if the agent input has the wrong type."""
        if agent is not None and not isinstance(agent, Agent):
            class_name = self.__class__.__name__
            raise InvalidAgentParameterException(
                f"The {class_name} expected a PydanticAI Agent instance, but received type {type(agent)}."
            )
        self._agent = agent

    @classmethod
    @abstractmethod
    def create_prompt(cls, *args: Any, **kwargs: Any) -> str:
        """Generates the prompt for the current PydanticAI agent executor."""
        ...

    def get_or_create_agent(self, *args: Any, **kwargs: Any) -> Agent[AgentDepsType, AgentOutputType]:
        """Optionally override this method to lazily create a new PydanticAI agent if one does not already exist."""
        return self.agent if self._agent else self._create_agent(*args, **kwargs)

    @classmethod
    def _create_agent(cls, *args: Any, **kwargs: Any) -> Agent[AgentDepsType, AgentOutputType]:
        """Factory class method for generating a PydanticAI agent."""
        raise NotImplementedError

    @abstractmethod
    async def __call__(self, context: AgentDepsType, *args: Any, **kwargs: Any) -> AgentOutputType:
        """Convenience method for initializing and running an agent.

        Args:
            context (AgentDepsType): The input dependencies that are passed to the PydanticAI agent when run.

        Returns:
            AgentOutputType: The synthesized output of the current AI model.

        """
        ...

    @classmethod
    def get_model_name(cls, agent: Agent[AgentDepsType, AgentOutputType]) -> str:
        """Helper for extracting the model name of the current LLM."""
        return getattr(agent.model, "model_name", str(agent.model))

    @property
    def model_name(self) -> str:
        """Returns the human-readable LLM model name and version in use."""
        with contextlib.suppress(AgentUninitializedException):
            return self.get_model_name(self.agent)
        return "<Uninitialized>"


class EmbedderABC(ABC):
    """Abstract embedder base class wrapper that defines the basic methods used to embed records and queries."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        """Helper for creating a basic PydanticAI Embedder for calculating query/document similarity scores."""
        self.embedder = embedder

    @property
    def embedder(self) -> Embedder:
        """Returns the embedder for calculating query/document similarity scores."""
        if not self._embedder:
            raise EmbedderUninitializedException("A PydanticAI embedder has not yet been created.")
        return self._embedder

    @embedder.setter
    def embedder(self, embedder: Embedder | None = None) -> None:
        """Sets the embedder with class validation, raising a `TypeError` if the embedding input has the wrong type."""
        if embedder is not None and not isinstance(embedder, Embedder):
            class_name = self.__class__.__name__
            raise InvalidEmbedderParameterException(
                f"The {class_name} expected a PydanticAI Embedder instance, but received type {type(embedder)}."
            )
        self._embedder = embedder

    def get_or_create_embedder(self, *args: Any, **kwargs: Any) -> Embedder:
        """Helper for retrieving an existing embedder if available or creating a new embedder if it doesn't exist."""
        return self.embedder if self._embedder else self._create_embedder(*args, **kwargs)

    @classmethod
    def _create_embedder(cls, *args: Any, **kwargs: Any) -> Embedder:
        """Factory class method for creating a new embedder.

        To be overridden by subclasses.

        """
        raise NotImplementedError

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> EmbeddingResult:
        """Convenience method for initializing an embedding model and using it to embed documents and queries.

        Args:
            *args: Arbitrary Positional arguments (designed for subclasses to implement)
            *kwargs: Arbitrary keyword arguments (designed for subclasses to implement)

        Returns:
            EmbeddingResult: A PydanticAI EmbeddingResult class.

        """
        ...

    @classmethod
    def get_model_name(cls, embedder: Embedder) -> str:
        """Helper for retrieving the name of the model used by the current agent."""
        return getattr(embedder.model, "model_name", str(embedder.model))

    @property
    def model_name(self) -> str:
        """Returns the name of the embedding model currently being used."""
        with contextlib.suppress(EmbedderUninitializedException):
            return self.get_model_name(self.embedder)
        return "<Uninitialized>"


__all__ = [
    "AgentDepsType",
    "AgentOutputType",
    "AgentABC",
    "EmbedderABC",
    "EmbeddingModelProviders",
    "ModelProviders",
    "PydanticAIModelFactory",
    "PydanticAIEmbeddingModelFactory",
]

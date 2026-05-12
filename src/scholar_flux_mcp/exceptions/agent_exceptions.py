"""Defines exceptions used for handling edge cases in agent/embedding creation and execution."""


class AgentException(Exception):
    """Exception raised when errors arise during agent creation or execution."""

    pass


class EmbedderException(Exception):
    """Exception raised upon encountering errors during embedder creation or execution."""

    pass


class AgentUninitializedException(AgentException):
    """Exception raised when attempting to use an agent that hasn't been initialized."""

    pass


class AgentInitializationException(AgentException):
    """Exception raised when large language model configuration is unsuccessful."""

    pass


class InvalidAgentParameterException(AgentException):
    """Exception raised when attempting to assign an invalid value to an agent."""

    pass


class AgentUnavailableException(AgentException):
    """Exception raised when no agent is available for selection via the `PydanticAIModelFactory`."""

    pass


class AgentResearchSynthesisFailedException(AgentException):
    """Exception raised when `synthesize_records` fails to generate a research synthesis during execution."""

    pass


class EmbedderUninitializedException(EmbedderException):
    """Exception raised when attempting to use an embedder that hasn't been created."""

    pass


class EmbedderInitializationException(EmbedderException):
    """Exception raised when embedding model configuration is unsuccessful."""

    pass


class InvalidEmbedderParameterException(EmbedderException):
    """Exception raised when attempting to assign an invalid value to an embedding model."""

    pass


class EmbedderUnavailableException(AgentException):
    """Exception raised when no embedding model is available for selection via the `PydanticAIEmbeddingModelFactory`."""

    pass


class DocumentEmbeddingFailedException(EmbedderException):
    """Exception raised when `embedder.embed` or `embedder.embed_documents` encounters an error."""

    pass


__all__ = [
    "AgentException",
    "EmbedderException",
    "AgentUnavailableException",
    "AgentUninitializedException",
    "EmbedderUninitializedException",
    "AgentInitializationException",
    "EmbedderInitializationException",
    "EmbedderUnavailableException",
    "InvalidAgentParameterException",
    "AgentResearchSynthesisFailedException",
    "InvalidEmbedderParameterException",
    "DocumentEmbeddingFailedException",
]

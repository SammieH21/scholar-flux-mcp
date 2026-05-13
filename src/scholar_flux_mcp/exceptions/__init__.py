"""Imports custom exceptions for core dependencies that would prevent ScholarFluxMCP from loading when missing."""

from scholar_flux_mcp.exceptions.agent_exceptions import (
    AgentException,
    AgentInitializationException,
    AgentResearchSynthesisFailedException,
    AgentUnavailableException,
    AgentUninitializedException,
    DocumentEmbeddingFailedException,
    EmbedderException,
    EmbedderInitializationException,
    EmbedderUnavailableException,
    EmbedderUninitializedException,
    InvalidAgentParameterException,
    InvalidEmbedderParameterException,
)
from scholar_flux_mcp.exceptions.history_exceptions import (
    HistoryCacheConnectionFailed,
    HistoryCacheDeletionException,
    HistoryCacheException,
    HistoryCacheInitializationException,
    HistoryCacheParameterValidationException,
    HistoryCacheRetrievalException,
    HistoryCacheStorageException,
    HistoryCacheVerificationException,
)
from scholar_flux_mcp.exceptions.import_exceptions import (
    CoreDependencyImportError,
    MCPImportError,
    PydanticAIImportError,
    PydanticAIProviderExtraImportError,
    RapidFuzzImportError,
    ScholarFluxImportError,
    SQLModelImportError,
)
from scholar_flux_mcp.exceptions.mcp_server_exceptions import (
    MCPServerException,
    MCPServerInitializationException,
)
from scholar_flux_mcp.exceptions.synthesis_exceptions import (
    EvidenceGroundingException,
    EvidenceGroundingParameterException,
    RecordDeduplicationException,
)

__all__ = [
    "MCPServerException",
    "MCPServerInitializationException",
    "CoreDependencyImportError",
    "MCPImportError",
    "ScholarFluxImportError",
    "SQLModelImportError",
    "PydanticAIImportError",
    "PydanticAIProviderExtraImportError",
    "RapidFuzzImportError",
    "AgentException",
    "AgentUnavailableException",
    "EmbedderException",
    "AgentUninitializedException",
    "EmbedderUnavailableException",
    "EmbedderUninitializedException",
    "AgentInitializationException",
    "EmbedderInitializationException",
    "InvalidAgentParameterException",
    "AgentResearchSynthesisFailedException",
    "InvalidEmbedderParameterException",
    "DocumentEmbeddingFailedException",
    "RecordDeduplicationException",
    "EvidenceGroundingException",
    "EvidenceGroundingParameterException",
    "HistoryCacheException",
    "HistoryCacheInitializationException",
    "HistoryCacheConnectionFailed",
    "HistoryCacheRetrievalException",
    "HistoryCacheStorageException",
    "HistoryCacheDeletionException",
    "HistoryCacheVerificationException",
    "HistoryCacheParameterValidationException",
]

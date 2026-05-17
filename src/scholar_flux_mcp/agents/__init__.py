"""ScholarFluxMCP Module defining Agents for AI-driven research summary synthesis."""

from scholar_flux_mcp.agents.models import (
    AgentABC,
    EmbeddingModelProviders,
    ModelProviders,
    PydanticAIEmbeddingModelFactory,
    PydanticAIModelFactory,
)
from scholar_flux_mcp.agents.record_topic_similarity_embedder import RecordTopicSimilarityEmbedder
from scholar_flux_mcp.agents.synthesis_agent import SynthesisAgent, SynthesisAgentOutput, SynthesisContext

__all__ = [
    "AgentABC",
    "EmbeddingModelProviders",
    "ModelProviders",
    "PydanticAIModelFactory",
    "PydanticAIEmbeddingModelFactory",
    "RecordTopicSimilarityEmbedder",
    "SynthesisContext",
    "SynthesisAgent",
    "SynthesisAgentOutput",
]

"""ScholarFluxMCP Module defining Agents for AI-driven research summary synthesis."""

from scholar_flux_mcp.agents.models import AgentABC, PydanticAIEmbeddingModelFactory, PydanticAIModelFactory
from scholar_flux_mcp.agents.record_topic_similarity_embedder import RecordTopicSimilarityEmbedder
from scholar_flux_mcp.agents.synthesis_agent import SynthesisAgent, SynthesisAgentOutput, SynthesisContext

__all__ = [
    "AgentABC",
    "PydanticAIModelFactory",
    "PydanticAIEmbeddingModelFactory",
    "RecordTopicSimilarityEmbedder",
    "SynthesisContext",
    "SynthesisAgent",
    "SynthesisAgentOutput",
]

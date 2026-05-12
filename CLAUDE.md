# CLAUDE.md

Quick-reference context for AI coding assistants working on ScholarFlux MCP.

### Last updated 5/12/2026 (**v0.1.0**)

> For complete, authoritative information, consult:
> - [README.md](README.md) — overview, features, quickstart, architecture
> - [ScholarFlux (base package)](https://github.com/SammieH21/scholar-flux) — underlying academic API orchestration library
> - [PydanticAI docs](https://ai.pydantic.dev/) — AI agent framework used for synthesis and embeddings
> - [MCP Protocol](https://modelcontextprotocol.io/) — Model Context Protocol specification

## Project Overview

ScholarFlux MCP is an **MCP server** for academic research — search, synthesis, and citation verification as composable tools. Built on [ScholarFlux](https://github.com/SammieH21/scholar-flux) for multi-provider search orchestration and [PydanticAI](https://ai.pydantic.dev/) for LLM/embedding agents, it adds deduplication, embedding-based reranking, AI-powered synthesis with structured citation injection, post-generation citation grounding, and output history on top of ScholarFlux's search infrastructure.

**Core pipeline:** Search (7+ providers) → Normalize → Deduplicate → Embed & Rank → Synthesize (PydanticAI) → Ground Citations → Report

## Development Commands

```bash
# Setup
poetry install --all-extras --with dev,testing

# Testing
poetry run pytest -rsx -vv --cov=scholar_flux_mcp --cov-report=term-missing

# Linting
poetry run mypy src tests
poetry run ruff check src tests
poetry run docstr-coverage src

# Formatting
poetry run ruff format src tests
poetry run ruff check src tests --fix

# MCP Inspector (debug tools interactively)
make mcp

# Docker
cd docker && docker compose up -d                              # Basic (MCP + MongoDB)
docker compose --profile with-redis up -d                      # With Redis caching
docker compose --profile with-ollama up -d                     # With local Ollama
docker compose --profile with-redis --profile with-ollama up -d  # Full stack
```

## Available MCP Tools

| Tool | Purpose |
|------|---------|
| `scholar_flux_health_check` | Verify health status of cache and history services |
| `scholar_flux_record_search` | Multi-provider academic record search |
| `scholar_flux_record_relevance_search` | Search + embedding-based similarity reranking |
| `scholar_flux_synthesize_research_summary` | Full pipeline: search → rank → synthesize → ground |
| `scholar_flux_list_providers` | List available academic database providers |
| `scholar_flux_list_recent_history` | Browse recent research tool outputs |
| `scholar_flux_list_record_history` | Browse recently stored records across all searches |
| `scholar_flux_get_history_output` | Retrieve a specific output by input hash |
| `scholar_flux_clear_history` | Clear all stored output history |

Tool input models and output formatters live in `server/io/`. Tool registrations live in `server/main.py`.

## Service Dependency Chain

The three core research tools are layered — each builds on the previous:

```
SearchService  ←  RelevanceSearchService  ←  SynthesisService
(search)           (search + dedup + rank)     (rank + synthesize + ground)
```

After synthesis, the **GroundingService** validates every bracketed `[N]` citation in the AI-generated text against source records by index. Citations referencing out-of-bounds or invalid indices are counted as `references_rejected`; valid ones as `references_grounded`. This is reported in `SynthesisOutput.grounding_stats`.

## Architecture Quick Reference

```
FastMCP Server (server/main.py)
├── create_server()           — Server factory with lifespan management
├── app_lifespan()            — Async context manager initializing all services
├── _register_tools()         — Registers 9 MCP tools with deferred service access
└── AppContext                — Dataclass container for lazily initialized services
     │
     ├── Tool I/O (server/io/)     — Pydantic input validation + markdown/JSON formatting
     │   ├── base.py               — BaseFormatter, BaseToolInput, BaseResearchToolInput (search, relevance search, synthesis)
     │   ├── health_check.py       — HealthCheckToolInput, HealthCheckFormatter
     │   ├── search.py             — SearchToolInput, SearchFormatter
     │   ├── relevance_search.py   — RelevanceSearchToolInput, RelevanceSearchFormatter
     │   ├── synthesis.py          — SynthesisToolInput, SynthesisFormatter
     │   ├── history.py            — RecentHistoryToolInput, RecordHistoryToolInput, GetHistoryToolInput, OutputHistoryFormatter, RecordHistoryFormatter
     │   └── providers.py          — ListProvidersInput
     │
     ├── Services (services/)       — Single-responsibility service layer
     │   ├── BaseResearchService    — Base class with history cache integration + dependency validation
     │   ├── SearchService          — Wraps ScholarFlux MultiSearchCoordinator (extends BaseResearchService)
     │   ├── RelevanceSearchService — Search + dedup + embedding ranking (extends BaseResearchService)
     │   ├── SynthesisService       — Orchestrates: relevance search → agent → grounding (extends BaseResearchService)
     │   ├── GroundingService       — Validates AI citations against source records
     │   ├── HistoryService         — SQLModel-based relational storage for tool output history
     │   ├── CacheService           — Two-tier caching (HTTP session + data response)
     │   └── ProviderService        — Provider metadata and rate limit info
     │
     ├── Agents (agents/)           — PydanticAI-based AI components
     │   ├── SynthesisAgent         — LLM agent for evidence-based research synthesis
     │   ├── RecordTopicSimilarityEmbedder   — Embedding model for record-topic scoring
     │   ├── PydanticAIModelFactory          — LLM model creation with provider cascade
     │   ├── PydanticAIEmbeddingModelFactory — Embedding model creation with cascade
     │   ├── AgentABC                        — Abstract base for all agents
     │   └── EmbedderABC                     — Abstract base for all embedders
     │
     ├── Models (models/)           — Pydantic schemas and validation
     │   ├── core.py                — Package-level annotations, helpers, validators, JSONDataModel base class
     │   ├── enums.py               — Enums defining package-level constants for internal logic (e.g., research subjects, metadata, providers)
     │   ├── schemas.py             — Defines the core schema used for internal processing, search, relevance search, synthesis (~2000 lines)
     │   ├── type_aliases.py        — Aliases for MCP tool I/O and logical model groupings (e.g., ResearchToolInput, ResearchToolType)
     │   └── history.py             — SQLModel classes for relational database storage and persistence
     │
     ├── Exceptions (exceptions/)        — Custom exception hierarchy
     │   ├── agent_exceptions.py         — Agent/Embedder exceptions
     │   ├── import_exceptions.py        — MCPImportError, ScholarFluxImportError, PydanticAIImportError, SQLModelImportError, RapidFuzzImportError
     │   ├── history_exceptions.py       — HistoryCache exceptions (init, retrieval, storage, deletion)
     │   ├── mcp_server_exceptions.py    — MCPServerException, MCPServerInitializationException
     │   └── synthesis_exceptions.py     — Grounding/dedup exceptions
     │
     ├── Utils (utils/)                  — Shared utilities
     │   ├── preprocessing_utils.py      — Record conversion, dedup (rapidfuzz), context building
     │   ├── helpers.py                  — Type coercion, timestamps, truncation
     │   ├── similarity_utils.py         — Cosine similarity calculation
     │   ├── fuzzy_text_similarity.py    — Utilities for fuzzy string matching with `rapidfuzz`
     │   └── logging.py             — Logging setup with sensitive data masking
     │
     └── Transport (server/transport.py) — MCP transport configuration
         ├── StdioTransport              — Default (no additional args)
         ├── ServerSentEventTransport    — SSE with host/port
         └── StreamableHTTPTransport     — HTTP with stateless/json options
```

## Error Handling

- **MCP tools** catch exceptions and return the error as a JSON or Markdown formatted string to the client
- **Lifespan** validates services on assignment and raises an MCPServerException on initialization for irrecoverable misconfiguration
- **SynthesisService** returns `_empty_synthesis` with the error reason on agent failure when `raise_on_error=False`,
- **RelevanceSearchService** falls back to unranked records if embedding fails (configurable via `raise_on_error`)
- **BaseResearchService** catches `HistoryCacheException` in cache helpers, assignment validation, logs warnings, continues without history
- **Import guards**: `PydanticAIImportError`, `RapidFuzzImportError`, `MCPImportError`, `ScholarFluxImportError`, `SQLModelImportError`


## Core Environment Variables

```bash
# Data directory
SCHOLAR_FLUX_HOME=~/.scholar_flux

# Logging
SCHOLAR_FLUX_LOG_LEVEL=WARNING
SCHOLAR_FLUX_MCP_LOG_LEVEL=INFO

# Cache backends
SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND=redis      # mongodb, sqlite, redis, memory, etc.
SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE=redis      # mongodb, sql/sqlite, redis, duckdb, memory, null

# Model providers (force specific provider or let cascade auto-detect)
SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER=               # ollama, ollama_cloud, anthropic, google, openai
SCHOLAR_FLUX_MCP_DEFAULT_EMBEDDING_MODEL_PROVIDER=     # ollama, google, openai

# API keys for academic databases (optional, improves rate limits)
PUBMED_API_KEY=
SPRINGER_NATURE_API_KEY=
CORE_API_KEY=

# API keys for LLM/embedding providers (at least one required for synthesis)
OLLAMA_API_KEY=             # Ollama Cloud
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=             # or GEMINI_API_KEY
OPENAI_API_KEY=

# Preprocessing tuning
SCHOLAR_FLUX_MCP_RECORD_TRUNCATION_LENGTH=3000         # Max chars per record in LLM context
SCHOLAR_FLUX_MCP_FIELD_TRUNCATION_LENGTH=1300           # Max chars per field (abstract, etc.)
SCHOLAR_FLUX_MCP_GROUNDING_TOKEN_LIMIT=120000           # Token budget for grounding context
SCHOLAR_FLUX_MCP_RECORD_EMBEDDING_TOKEN_LIMIT=2000      # Token budget per record for embedding

# Transport
SCHOLAR_FLUX_MCP_TRANSPORT=stdio                       # stdio, sse, streamable-http
SCHOLAR_FLUX_MCP_HOST=127.0.0.1
SCHOLAR_FLUX_MCP_PORT=8000
```

## Code Standards

- **Type hints**: Required on all functions. Verified with `mypy` strict mode
- **Docstrings**: Required, Google style, coverage checked via `docstr-coverage`
- **Line length**: 120 characters max
- **Testing**: `pytest-asyncio` for async tests, fixtures in `tests/fixtures/` and `tests/conftest.py`
- **Formatting**: `ruff`
- **Python**: 3.10+ required

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

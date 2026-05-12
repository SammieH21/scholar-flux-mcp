# ScholarFlux MCP

[![CI](https://github.com/SammieH21/scholar-flux-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/SammieH21/scholar-flux-mcp/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SammieH21/scholar-flux-mcp/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/SammieH21/scholar-flux-mcp/actions/workflows/github-code-scanning/codeql)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-blue)](https://modelcontextprotocol.io)
[![Beta](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/SammieH21/scholar-flux-mcp)
[![mypy: Type Checked](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


An **MCP server** for academic research — search, synthesis, and citation verification as composable tools. Query 7+ academic databases concurrently, rank results by embedding similarity to your research question, generate
evidence-based syntheses, and verify every bracketed citation and referenced text against retrieved source records. Use the full pipeline or any layer independently.
Built on [ScholarFlux](https://github.com/SammieH21/scholar-flux), [PydanticAI](https://github.com/pydantic/pydantic-ai), and [FastMCP](https://gofastmcp.com).

At its core, ScholarFlux MCP is built around a single principle: every citation should be verifiable against the sources it claims to reference. In the words of W. Edwards Deming, "In God we trust; all others must bring data."


> **Research Use Notice**
> This is a **research tool** for academic literature synthesis and discovery. It is **not intended for clinical diagnosis, treatment decisions, or medical advice**. The AI-powered synthesis feature assists researchers in reviewing literature but requires human oversight and professional interpretation. Always consult qualified healthcare professionals for medical or mental health concerns. This tool should be used by researchers, clinicians, and students as part of evidence-based research workflows, not as a substitute for professional medical judgment.

## Features

ScholarFlux MCP connects [ScholarFlux](https://github.com/SammieH21/scholar-flux)'s search infrastructure to an AI synthesis and verification pipeline, exposed as MCP tools:

### From ScholarFlux (inherited)

- **Multi-provider search**: Query PubMed, PLOS, OpenAlex, Crossref, arXiv, CORE, and Springer Nature concurrently with automatic per-provider rate limiting and shared rate limiter coordination
- **Schema normalization**: Provider-specific response formats consolidated into a unified record schema
- **Two-tier caching**: HTTP session cache + processed result cache with pluggable backends (Redis, MongoDB, SQLite, in-memory)

### Added by ScholarFlux MCP

ScholarFlux MCP implements a layered research pipeline on top of ScholarFlux's search infrastructure:

- **Deduplicate → Rank → Filter**: Retrieved records are deduplicated across providers via rapidfuzz fuzzy matching, then scored against your research question using PydanticAI embedding models and filtered by similarity threshold — substantially reducing noise before synthesis
- **Synthesize → Ground → Report**: PydanticAI agents generate evidence-based research summaries with structured context injection and index-based (`[N]`) citation. The GroundingService then validates every citation against source records — checking both index bounds and referenced text similarity via rapidfuzz — reporting verified and rejected counts. Every synthesis includes confidence scores, grounding statistics, explicit limitations, and suggested follow-up queries
- **Zero-config model selection**: The model factory cascades through available providers (Ollama local → Ollama Cloud → Anthropic → Google → OpenAI), adapting to your environment with minimal configuration — for both LLM synthesis and embedding-based reranking
- **Output history**: SQLModel-based relational storage with TTL support and fuzzy topic search for replaying and retrieving previous searches, relevance rankings, and syntheses without re-executing the pipeline
- **MCP interface**: 9 tools registered via FastMCP with transport options (stdio, SSE, streamable HTTP)

Supporting infrastructure:

- **Flexible caching**: Inherits ScholarFlux's cache backends; adds MongoDB or in-memory options with optional Redis for sessions
- **Docker deployment**: Full Docker Compose stack with MongoDB, optional Redis and Ollama, health checks, resource limits, and auto-model-pull on startup

## Why This Exists

Citation hallucination in AI-generated academic content is a documented problem:

| Source | Finding |
|--------|---------|
| Kim et al. (2025) | DOI hallucination exceeded **80%** in lower-income countries across 4 major LLMs |
| Linardon et al. (2025) | GPT-4o fabricated **20–29%** of citations for specialized topics; **45%** of "real" citations contained errors |
| Mugaanyi et al. (2024) | Overall citation accuracy was **33%** in natural sciences, **9%** in humanities |

ScholarFlux handles the process of retrieving, processing, and normalizing records from 7+ providers into downstream data pipelines fit for analysis, data engineering workflows, and literature review. What it doesn't do is synthesize those records into a generated report or verify citations. That's what this project adds: a pipeline that **synthesizes from real sources, then checks its own citations against them**.

Existing tools either detect fabrication after the fact (GPTZero, GhostCite) or prevent it within closed systems (Elicit, Consensus). While both Elicit and Consensus now offer MCP servers for search and synthesis, neither exposes explicit grounding metrics — you get answers with citations, but no verified/rejected counts. ScholarFlux MCP's GroundingService takes a different approach: synthesize, verify, and report what failed — with both verified and rejected counts exposed to the MCP client.

### Example: Synthesis on Citation Hallucination

To illustrate the pipeline, we used ScholarFlux MCP to synthesize research about citation hallucination itself:

**Input:**
```json
{
  "question": "What is the current state of citation hallucination and fabrication in AI-generated academic content, and what verification methods have been proposed to address this problem?",
  "queries": [
    "LLM citation hallucination academic research",
    "AI generated citations verification scholarly",
    "GPT fabricated references detection methods",
    "large language model citation accuracy peer review"
  ],
  "providers": ["arxiv", "openalex", "plos", "crossref"],
  "max_records": 140,
  "pages": 4,
  "year_from": 2023
}
```

**Output (key metrics):**
```
Records Analyzed: 62
Confidence Score: 85%
Evidence Grounding: 7 verified records, 0 rejected
```

The synthesis cited Szeider (2026), who proposed using MCP for citation verification. `0 rejected` here means every citation index pointed to a real retrieved record and the referenced text matched the cited source — the grounding mechanism validates both structural accuracy (did the LLM cite a record that exists in the retrieved set?) and citation accuracy (does the referenced text actually appear in or closely match the cited record via rapidfuzz similarity scoring?).

Full output: [`samples/llm_generated_citations_synthesis.md`](samples/llm_generated_citations_synthesis.md)

## Agentic Synthesis Constraints

The architecture ensures and validates citation accuracy through the implementation of strict architectural constraints:
1. **Index Constraint**: LLMs can only cite by record index (0-N). Values outside of this range are flagged as invalid.
2. **Text Verification**: All cited text must match source records (60%+ fuzzy similarity via `rapidfuzz`)
3. **Structured Output**: Building off of `PydanticAI`, synthesis outputs are constrained to strict structure and type validation. When the LLM fails to produce the response structure, `PydanticAI` automatically retries the request, relaying the issue to the LLM.
4. **Confidence Checks**: On every synthesis generation, the `SynthesisAgent` must relay its confidence in the report that it generates, indicating limitations in the produced report or when analyses require more conclusive evidence.
5. **Post-Generation Checks**: The `GroundingService` validates every index citation and text references to relay when the LLM hallucinates.

This makes hallucinated citations structurally impossible when:
- Records are relevant to the research question
- LLM follows the constrained generation instructions
- Source records contain sufficient content for synthesis.

## Example Output

The synthesis tool produces structured, citation-grounded academic summaries. Here's a truncated example from a query on AI literacy research:

**Input:**
```json
{
  "question": "What are the current, most explored concepts in AI literacy?",
  "queries": ["Artificial Intelligence Literacy", "machine learning literacy"],
  "providers": ["plos", "openalex", "crossref", "springernature"],
  "categories": ["COMPUTATION", "MATHEMATICS"],
  "max_records": 120,
  "pages": 3,
  "year_from": 2023
}
```

**Output (truncated):**

```markdown
# Research Synthesis

## Research Question
> What are the current, most explored concepts in AI literacy?

**Categories**: computation, mathematics
**Queries**:
- Artificial Intelligence Literacy
- machine learning literacy

**Records Analyzed**: 54
**Confidence Score**: 92%
**Evidence Grounding**: 16 verified records, 0 rejected

---

## Synthesis

The research on artificial intelligence literacy has rapidly expanded, particularly
from 2023-2026, with the most explored concepts falling into several interconnected
areas. First, the development and validation of AI literacy frameworks and assessment
instruments represents a dominant research theme, with scholars proposing multidimensional
models encompassing technical competence, ethical reasoning, critical evaluation,
creative application, and adaptive learning [4]. Multiple validated scales have emerged,
including the Artificial Intelligence Literacy Scale [5], the Multidimensional AI
Literacy Competency Scale (MAIL-CS) [17], and the AI Identity Scale [6]...

## Key Findings

- AI literacy is consistently conceptualized as a multidimensional construct
- Framework development and scale validation represent the most mature research area
- Educational interventions across K-8, primary, secondary, and higher education
  demonstrate positive learning outcomes
- A counterintuitive finding shows lower AI literacy correlates with greater AI
  receptivity due to perceived 'magic' of AI

## Supporting Evidence

1. **Artificial intelligence literacy in design intelligence: A review** (2026)
   [Link](https://doi.org/10.36922/dp025520053) via crossref

   **Index**: [4]
   **Relevance Score**: 95%
   **Summary**: AI literacy is multidimensional, encompassing technical competence,
   ethical reasoning, critical evaluation, creative application, and adaptive learning

## Limitations

- Most studies are cross-sectional surveys limiting causal inference
- Many scales are validated in specific cultural contexts limiting generalizability

## Suggested Follow-up

- What are the most effective pedagogical approaches for teaching AI literacy?
- How does generative AI specifically change the landscape of AI literacy education?

---
*This synthesis was generated using ScholarFlux MCP with PydanticAI.
LLM: minimax-m2.5:cloud | Embedding Model: embeddinggemma:latest.
Always verify findings with primary sources.*
```

**Key observations:**
- **Multi-query orchestration**: 2 queries × 4 providers searched with automatic rate limiting via ScholarFlux
- **Embedding-based filtering**: 120 max records → 54 analyzed after similarity ranking
- **Citation grounding**: 16 verified, 0 rejected — transparent fabrication detection
- **AI-generated summaries**: Each evidence item includes both the original abstract and an AI-generated finding summary
- **Model attribution**: Footer shows which LLM and embedding model produced the synthesis

## Quick Start

### Option 1: Local Installation

ScholarFlux MCP is installable via PyPI or from source with Poetry:

```bash
# Install with pip
pip install scholar-flux-mcp

# Or install from source with Poetry
git clone https://github.com/SammieH21/scholar-flux-mcp.git
cd scholar-flux-mcp

# For core package utilities
poetry install

# Run the server
poetry run python -m scholar_flux_mcp
```

### Option 2: Docker

First clone the repo using the following:

```bash
git clone https://github.com/SammieH21/scholar-flux-mcp.git
cd scholar-flux-mcp
```

Afterward, build the docker container:

```bash
cd docker
cp .env.example .env
# Edit .env with your API keys

docker compose up -d
```

Once built, you can start a containerized ScholarFlux MCP server directly with docker:

```bash
# Basic (MCP server only)
docker compose up -d

# With Redis caching
docker compose --profile with-redis up -d

# With local Ollama
docker compose --profile with-ollama up -d

# Full stack
docker compose --profile with-redis --profile with-ollama up -d
```

### Option 3: MCP Inspector

MCP Inspector is a quick and easy way to get started with ScholarFlux MCP and test tools directly on live data. To get started, first ensure that you have ScholarFlux MCP server and [`npm`](https://www.npmjs.com/package/marked) installed. Then run the following:

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run the server
npx @modelcontextprotocol/inspector python -m scholar_flux_mcp

# Or use the Makefile (includes all environment variables)
make mcp
```


## MCP Client Configuration

### Claude Desktop / Claude Code

Add to your MCP configuration file:

**Local installation:**
```json
{
  "mcpServers": {
    "scholar_flux": {
      "command": "python",
      "args": ["-m", "scholar_flux_mcp"]
    }
  }
}
```

**With Docker:**
```json
{
  "mcpServers": {
    "scholar_flux": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "scholar-flux-mcp"]
    }
  }
}
```


## Example Usage

Once configured with an MCP client, you can ask questions like:

> "Search for recent papers on cognitive behavioral therapy for anxiety disorders"

> "Synthesize the research on biomarkers for treatment response in depression"

> "What providers are available and which are best for mental health research?"

## Available Tools

### Health Checks

#### `scholar_flux_health_check`

Verify that the scholar-flux-mcp server is accessible and whether the cache and history services have been successfully initialized.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response_format` | string | "markdown" | "markdown" or "json" |

**Example Output**
```markdown
## Health Check Status: healthy

### Cache
- **Status:** healthy
- **Details:**
  - initialized: True
  - namespace: mcp_cache
  - ttl: None
  - session_cache_backend: redis
  - response_cache_storage: redis
  - user_agent: Sammie Haskin (mailto:***)
  - raise_on_error: False

### History
- **Status:** healthy
- **Details:**
  - initialized: True
  - ttl: None
  - echo: False
  - raise_on_error: True
  - url: sqlite:///./.cache/history.sqlite

```


### Search Tools

#### `scholar_flux_record_search`

Search for academic literature across multiple scholarly databases.

```json
{
  "queries": ["depression treatment CBT efficacy", "anxiety intervention RCT"],
  "providers": ["pubmed", "plos", "openalex"],
  "max_records": 25,
  "year_from": 2020,
  "response_format": "markdown"
}
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `queries` | list[str] | required | Search terms (supports multiple queries) |
| `providers` | list[str] | `["pubmed", "plos", "openalex"]` | Databases to search |
| `max_records` | int | 25 | Results per provider (1-200) |
| `pages` | int | 3 | Result pages per provider (1-10) |
| `year_from` | int | null | Filter by publication year |
| `year_to` | int | null | Filter by publication year |
| `open_access_only` | bool | false | Only open access records |
| `response_format` | string | "markdown" | "markdown" or "json" |
| `from_history_cache` | bool | false | Retrieve from history cache if available |
| `store_history_cache` | bool | true | Store output in history cache |


#### `scholar_flux_record_relevance_search`

Retrieve and rank studies via record-topic embedding cosine similarity.

```json
{
  "question": "Vaccine Flu Prevention Efficacy",
  "queries": ["Flu Vaccine Efficacy", "Influenza Prevention"],
  "categories": ["epidemiology", "intervention", "empirical"],
  "providers": ["pubmed", "plos", "openalex", "crossref"],
  "max_records": 50,
  "pages": 4,
  "year_from": 2000,
  "similarity_threshold": 0.60,
  "response_format": "markdown"
}
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | required | The question or topic. Used to determine record similarity |
| `queries` | list[str] | optional | Search terms for querying academic databases. Defaults to `categories` if not provided |
| `providers` | list[str] | `["pubmed", "plos", "openalex"]` | Databases to search |
| `categories` | list[str] | `["general"]` | Research categories for relevance search |
| `pages` | int | 3  | Pages to retrieve per provider per query |
| `max_records` | int | 25 | Maximum records to retrieve and reindex by topic-similarity |
| `year_from` | int | null | Retain records on or after the publication year |
| `year_to` | int | null | Retain records on or before the publication year |
| `similarity_threshold` | float | 0.0 | Minimum embedding similarity for record inclusion (0.0-1.0) |
| `open_access_only` | bool | false | Only include open access records |
| `from_history_cache` | bool | false | Retrieve from history cache if available |
| `store_history_cache` | bool | true | Store output in history cache |
| `response_format` | string | "markdown" | "markdown" or "json" |
| `force_refresh` | bool | false | Enables history cache retrieval for record searches while re-executing the relevance search stage|

### Synthesis Tools

#### `scholar_flux_synthesize_research_summary`

Synthesize research findings using AI-powered analysis.

```json
{
  "question": "What are the most effective interventions for treatment-resistant depression?",
  "queries": ["treatment resistant depression", "TRD intervention efficacy"],
  "categories": ["depression", "intervention"],
  "providers": ["pubmed", "plos", "openalex", "crossref"],
  "max_records": 100,
  "pages": 3,
  "year_from": 2020,
  "similarity_threshold": 0.5,
  "response_format": "markdown"
}
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | required | Research question to answer |
| `queries` | list[str] | optional | Search terms for querying academic databases. Defaults to `categories` if not provided |
| `providers` | list[str] | `["pubmed", "plos", "openalex"]` | Databases to search |
| `categories` | list[str] | `["general"]` | Research categories for synthesis focus |
| `pages` | int | 3  | Pages to retrieve per provider per query |
| `max_records` | int | 50 | Maximum records to analyze and synthesize (5-200) |
| `year_from` | int | null | Filter by publication year (last 5 years by default) |
| `year_to` | int | null | Filter by publication year |
| `similarity_threshold` | float | 0.5 | Minimum embedding similarity for record inclusion (0.0-1.0) |
| `open_access_only` | bool | false | Only include open access records |
| `from_history_cache` | bool | false | Retrieve from history cache if available |
| `store_history_cache` | bool | true | Store output in history cache |
| `response_format` | string | "markdown" | "markdown" or "json" |
| `force_refresh` | bool | false | Enables history cache retrieval for upstream relevance searches while re-executing the research synthesis stage |

**Research Categories:**

*Methodology:*
- `meta_analysis`, `theoretical`, `empirical`, `qualitative`, `quantitative`

*Mental Health (example domain):*
- `depression`, `anxiety`, `ptsd`, `bipolar`, `schizophrenia`
- `substance_use`, `ocd`, `adhd`, `general_wellbeing`

*Cross-domain:*
- `intervention`, `epidemiology`, `mathematics`, `computation`, `general`, `other`

**Output Structure:**

Each synthesis includes:
- **Header**: Question, categories, queries, records analyzed, confidence score, grounding stats
- **Synthesis**: Narrative with bracketed citation indices `[4]`, `[17]`
- **Key Findings**: 3-8 bullet points summarizing major themes
- **Evidence Items**: Per-record details including:
  - Title, year, DOI, URL, provider
  - Record index (for citation verification)
  - Referenced text and similarity score (for citation verification)
  - Relevance score (embedding similarity)
  - AI-generated finding summary
  - Original abstract
- **Limitations**: Identified gaps in the evidence
- **Suggested Follow-ups**: Recommended queries for deeper investigation
- **Footer**: Model attribution (LLM and embedding model used)



### History Tools

Browse and retrieve previously processed research tool outputs. History is stored using SQLModel with SQLite and supports TTL-based expiration and fuzzy topic filtering — search previous outputs or records by subject, query text, author, or DOI.

#### `scholar_flux_list_recent_history`

List recent research tool outputs.

```json
{
  "research_tool_type": "synthesis",
  "ttl": -1,
  "max_history": 10,
  "response_format": "markdown"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `research_tool_type` | string | `"all"` | Filter by tool type: `search`, `relevance`, `synthesis`, or `all` |
| `ttl` | number | -1 | Max age in seconds (-1 for no limit) |
| `max_history` | int | null | Max items to return |
| `successful_only` | bool | true | Only return successful outputs |
| `response_format` | string | "markdown" | "markdown" or "json" |
| `topic` | string | null | Filter and sort outputs by fuzzy text similarity to this topic. Compares against query, question, and category fields. Results are sorted by similarity in descending order |
| `similarity_threshold` | float | null | Minimum fuzzy similarity score for inclusion (0.0–1.0) when filtering by topic. At 1.0, acts as exact string match and is useful for finding outputs by specific query, categories, and question |


#### `scholar_flux_get_history_output`

Retrieve a specific output by its input hash. Returns the full formatted output (search results, relevance rankings, or synthesis report) for replay without re-executing the pipeline.

```json
{
  "input_hash": "a1b2c3d4e5f6",
  "research_tool_type": "synthesis",
  "response_format": "markdown"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_hash` | string | required | Unique hash identifying the input |
| `research_tool_type` | string | "all" | Filter by tool type |
| `ttl` | number | -1 | Max age in seconds (-1 for no limit) |
| `successful_only` | bool | true | Only return successful outputs |
| `response_format` | string | "markdown" | "markdown" or "json" |

#### `scholar_flux_list_record_history`

List recently stored academic records across all searches. Retrieves individual records from the history database, filterable by TTL, record count, and fuzzy topic similarity.

```json
{
  "ttl": -1,
  "max_records": 25,
  "response_format": "markdown"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ttl` | number | -1 | Max age in seconds (-1 for no limit) |
| `max_records` | int | null | Max records to return |
| `response_format` | string | "markdown" | "markdown" or "json" |
| `topic` | string | null | Filter and sort records by fuzzy text similarity to this topic. Record—topic similarity is calculated based on the title, author, DOI, and abstract. Results ordered by similarity in descending order |
| `similarity_threshold` | float | null | Minimum fuzzy similarity score for inclusion (0.0–1.0) when filtering by topic. At 1.0, filters and sorts via an exact partial string match to the abstract text, author, or DOI. |

#### `scholar_flux_clear_history`

Clear all stored output history. No parameters required.

### Utility Tools

#### `scholar_flux_list_providers`

List available academic database providers with their capabilities.


## Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                     FastMCP Server (main.py)                    │
 │  - Lifespan management (async context manager)                  │
 │  - Tool registration (thin adapters to services)                │
 └─────────────────────────────────────────────────────────────────┘
                                 │
                                 │
                                 ▼
                     ┌──────────────────────────┐
                     │     SynthesisService     │
         ┌────────── │     (Orchestrator)       │ ────────┐
         │           │                          │         │
         │           └──────────────────────────┘         │
         │                       │                        │
         │                       │                        │
         │                       │                        │
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐   ┌──────────────────────┐   ┌─────────────────┐
│                 │   │RelevanceSearchService│   │ SynthesisAgent  │
│GroundingService │   │                      │   │                 │
│                 │   │ - Search + dedup     │   │ - PydanticAI    │
│ - Index + text  │   │ - Embedding ranking  │   │ - Ollama/Claude │
│   validation    │   │ - Similarity filter  │   │ - Structured out│
│ - Fuzz matching │   └──────────────────────┘   └─────────────────┘
└─────────────────┘              │
                                 │
                                 ▼
                         ┌─────────────────┐
                         │  SearchService  │
                         │                 │
                         │ - Coordinator   │
                         │   creation      │
                         │ - Result norm.  │
                         └─────────────────┘
                                 │
                                 ▼
 ┌────────────────────────────────────────────────────────────────┐
 │                    ScholarFlux (Backend)                       │
 │  - Per-provider rate limiting (6s PLOS, 4s arXiv, etc.)        │
 │  - Concurrent threading with shared rate limiters              │
 │  - Schema normalization across 7+ providers                    │
 │  - Two-tier caching (HTTP + processed results)                 │
 └────────────────────────────────────────────────────────────────┘
          │                       │                         │
          ▼                       ▼                         ▼
  ┌────────────────────┐   ┌─────────────────┐   ┌─────────────────┐
  │ CacheService       │   │ HistoryService  │   │ ProviderService │
  │                    │   │                 │   │                 │
  │                    │   │                 │   │ - Provider meta │
  │ - HTTP Session     │   │ - Tool Cache    │   │   Rate limits   │
  │                    │   │                 │   │ - API keys      │
  │ - Response cache   │   │ - Output search │   └─────────────────┘
  │                    │   │                 │
  └────────────────────┘   └─────────────────┘
 ```

### How Search Orchestration Works

ScholarFlux MCP uses [ScholarFlux](https://github.com/SammieH21/scholar-flux) for multi-provider search orchestration. Searches are **concurrent but rate-limited per provider**:

```
MultiSearchCoordinator (ScholarFlux)
├── Thread Pool (per-provider threads)
│   ├── Thread 1: PLOS (shared rate limiter → 6s between requests)
│   ├── Thread 2: arXiv (shared rate limiter → 4s between requests)
│   ├── Thread 3: OpenAlex (shared rate limiter → 1s between requests)
│   └── Thread 4: Crossref (shared rate limiter → 1s between requests)
│
├── Shared Rate Limiter Registry (cross-query coordination)
└── Generator Pipeline (streaming results via concurrent.futures.as_completed)
```

**Key design decisions:**
- **Threading over asyncio**: Simpler for users, better for I/O-bound workloads with rate limits
- **Concurrent execution**: While one provider waits on rate limits, others continue
- **Shared rate limiters**: Multiple queries to the same provider coordinate through a single limiter
- **~3x speedup**: During record retrieval, compared to sequential requests across providers

For details on ScholarFlux's orchestration architecture, see the [ScholarFlux documentation](https://SammieH21.github.io/scholar-flux/).

### Service Responsibilities

| Service | Responsibility | Dependencies |
|---------|---------------|--------------|
| BaseResearchService | Base class for research services with history cache integration | HistoryService (optional) |
| SynthesisService | Full pipeline: relevance search → AI synthesis → citation grounding | RelevanceSearchService, GroundingService, SynthesisAgent |
| RelevanceSearchService | Orchestrates search → dedup → embedding ranking | SearchService, RecordTopicSimilarityEmbedder |
| SearchService | Creates SearchCoordinators, executes multi-provider queries, normalizes results | CacheService |
| CacheService | Two-layer caching: HTTP session cache + data response cache | None (initialized first) |
| ProviderService | Provider metadata, rate limits, API key management | None |
| GroundingService | Validates AI-generated citations via index bounds and referenced text similarity (rapidfuzz) | None |
| HistoryService | SQLModel-based relational storage for tool output history with TTL | None (independent) |
| SynthesisAgent | PydanticAI-based agent for record synthesis with lazy model selection | None (dynamically created) |
| RecordTopicSimilarityEmbedder | Embedding-based record-topic similarity scoring and filtering | None (dynamically created) |

### Data Flow

```
 User Query → MCP Tool → SynthesisService
                               │
                               ▼
                      RelevanceSearchService
                               │
                               ▼
                         SearchService
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
           PubMed           PLOS          OpenAlex ...
        (rate-limited)  (rate-limited)  (rate-limited)
               │               │               │
               └───────────────┼───────────────┘
                               ▼
                      Normalized Records
                      (e.g., 120 retrieved)
                               │
                               ▼
                      Deduplicate (rapidfuzz)
                               │
                               ▼
                  RecordTopicSimilarityEmbedder
                      (similarity ranking)
                               │
                               ▼
                      Filtered Records
                      (e.g., 54 above threshold)
                               │
                               ▼
                       SynthesisAgent
                      (PydanticAI + LLM)
                               │
                               ▼
                      GroundingService
                     (validate citations)
                               │
                               ▼
                       SynthesisOutput
              (16 grounded, 0 rejected)
```

### Design Principles

- **Single Responsibility**: Each service handles one concern
- **Dependency Injection**: Services receive dependencies at initialization
- **Lifespan Management**: Proper async resource management
- **Type Safety**: Full Pydantic validation on all I/O

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SCHOLAR_FLUX_HOME` | Data directory | `~/.scholar_flux` |
| `SCHOLAR_FLUX_LOG_LEVEL` | `ScholarFlux` Logging level | `WARNING` |
| `SCHOLAR_FLUX_MCP_LOG_LEVEL` | MCP server logging level | `INFO` |
| `SCHOLAR_FLUX_REDIS_HOST` | Redis host | `redis` |
| `SCHOLAR_FLUX_MCP_ENABLE_HISTORY` | Enable history service | `true` |
| `SCHOLAR_FLUX_MCP_HISTORY_TTL` | Default TTL for cached outputs (seconds) | - |
| `SCHOLAR_FLUX_MCP_HISTORY_URL` | SQLite/database URL for history storage | - |

### Model Configuration

ScholarFlux MCP supports multiple LLM and embedding providers with automatic fallback. The model factory cascades through available providers until one succeeds:

**LLM Cascade**: Ollama Local → Ollama Cloud → Anthropic → Google → OpenAI

**Embedding Cascade**: Ollama Local → Google → OpenAI

| Variable | Description | Default |
|----------|-------------|---------|
| `SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER` | Force a specific LLM provider | (auto-detect) |
| `SCHOLAR_FLUX_MCP_DEFAULT_EMBEDDING_MODEL_PROVIDER` | Force a specific embedding provider | (auto-detect) |

**Ollama (Local):**
| Variable | Description | Default |
|----------|-------------|---------|
| `SCHOLAR_FLUX_MCP_OLLAMA_BASE_URL` | Local Ollama endpoint | `http://localhost:11434` |
| `SCHOLAR_FLUX_MCP_OLLAMA_MODEL` | Local Ollama LLM | `glm-4.7-flash:latest` |
| `SCHOLAR_FLUX_MCP_OLLAMA_EMBEDDING_MODEL` | Local Ollama embedding model | `embeddinggemma:latest` |

**Ollama Cloud:**
| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_API_KEY` | Ollama Cloud API key | - |
| `SCHOLAR_FLUX_MCP_OLLAMA_CLOUD_MODEL` | Ollama Cloud model | `devstral-2:123b-cloud` |
| `SCHOLAR_FLUX_MCP_OLLAMA_CLOUD_BASE_URL` | Ollama Cloud endpoint | `https://ollama.com/v1` |

**Anthropic:**
| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `SCHOLAR_FLUX_MCP_ANTHROPIC_MODEL` | Anthropic model | `claude-haiku-4-5` |

**Google:**
| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google/Gemini API key | - |
| `SCHOLAR_FLUX_MCP_GOOGLE_MODEL` | Google LLM | `gemini-2.5-flash` |
| `SCHOLAR_FLUX_MCP_GOOGLE_EMBEDDING_MODEL` | Google embedding model | `gemini-embedding-001` |

**OpenAI / OpenAI-Compatible:**
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `SCHOLAR_FLUX_MCP_OPENAI_MODEL` | OpenAI LLM | `gpt-5` |
| `SCHOLAR_FLUX_MCP_OPENAI_EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `SCHOLAR_FLUX_MCP_OPENAI_ENDPOINT` | Custom OpenAI-compatible endpoint | - |
| `SCHOLAR_FLUX_MCP_OPENAI_PROVIDER` | Provider name for OpenAI-compatible APIs | `openai` |

**Note**: Any OpenAI-compatible API (vLLM, Together, Fireworks, local servers) can be used by setting `SCHOLAR_FLUX_MCP_OPENAI_ENDPOINT` and `SCHOLAR_FLUX_MCP_OPENAI_PROVIDER`.

### API Keys

For better rate limits, configure API keys for:
- **PubMed**: [NCBI Account](https://www.ncbi.nlm.nih.gov/account/)
- **Springer Nature**: [Developer Portal](https://dev.springernature.com/)
- **CORE**: [API Dashboard](https://core.ac.uk/services/api)

## Comparison with Existing Tools

The academic AI tool landscape has grown significantly. ScholarFlux MCP occupies a specific niche: **open-source, self-hostable research synthesis with multi-provider orchestration, post-synthesis citation verification, and MCP integration**. Here's how it compares to existing tools as of early 2026.

### Feature Comparison

| Capability | Elicit | Consensus | OpenScholar (Ai2) | Valsci | paper-search-mcp | **ScholarFlux MCP** |
|---|---|---|---|---|---|---|
| **Multi-provider search** | ✅ (3: SS+OA+PM) | ❌ (proprietary index) | ❌ (Semantic Scholar) | ❌ (Semantic Scholar) | ✅ (7, search only) | ✅ † (7, via ScholarFlux) |
| **User selects providers** | ❌ | ❌ | ❌ | ❌ | ⚠️ per-tool | ✅ |
| **Multi-query support** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **AI-powered synthesis** | ✅ | ✅ | ✅ | ⚠️ claim verification | ❌ | ✅ |
| **Post-synthesis grounding** | ⚠️ implicit | ⚠️ implicit | ✅ (inline RAG) | ✅ (bibliometric) | ❌ | ✅ (GroundingService) |
| **Confidence + rejection metrics** | ❌ | ⚠️ consensus meter | ❌ | ✅ (ordinal scale) | ❌ | ✅ |
| **Per-record relevance scores** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Limitations + follow-ups** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Open source** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Local LLM support** | ❌ | ❌ | ⚠️ (own 8B only) | ⚠️ (OpenAI-compat.) | ❌ | ✅ (any Ollama model) |
| **Ollama Cloud models** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (minimax, devstral, etc.) |
| **MCP protocol** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Docker Compose deployment** | N/A (SaaS) | N/A (SaaS) | ❌ | ⚠️ (requires ~1.5TB data) | ❌ | ✅ (turnkey) |
| **Production caching** | N/A (SaaS) | N/A (SaaS) | ❌ | ❌ | ❌ | ✅ † (MongoDB/Redis) |
| **Rate-limited orchestration** | N/A (hosted) | N/A (hosted) | ❌ | ❌ | ❌ | ✅ † |
| **Schema normalization** | N/A (internal) | N/A (internal) | ❌ | ❌ | ❌ | ✅ † |

> **†** Capabilities marked with † are provided by [ScholarFlux](https://github.com/SammieH21/scholar-flux), the underlying search orchestration library. ScholarFlux MCP exposes these to the MCP client but does not implement them.

### Tool Summaries

**[Elicit](https://elicit.com)** — The market leader for hosted academic AI research. Searches 138M+ papers across Semantic Scholar, OpenAlex, and PubMed. Features include Research Agents (Dec 2025), Systematic Review workflows, Reports, and ClinicalTrials.gov integration. Closed-source SaaS with paid tiers. Best for researchers who want a polished, managed experience and don't need provider control or self-hosting.

**[Consensus](https://consensus.app)** — AI search engine over 220M+ peer-reviewed papers with a unique "Consensus Meter" for yes/no research questions, Deep Search for structured reports, and a Scholar Agent for multi-step searches. Now offers an [MCP server](https://consensus.app/home/mcp/) and GPT-5 integration. Closed-source SaaS. Best for quick evidence lookups and gauging scientific agreement on specific claims.

**[OpenScholar](https://github.com/AkariAsai/OpenScholar)** (Ai2/University of Washington) — Open-source RAG-based synthesis model [published in *Nature*, Feb 2026](https://www.nature.com/articles/s41586-025-10072-4). Achieved human-level citation accuracy where GPT-4o fabricated 78–90% of citations. Uses a fine-tuned 8B model against Semantic Scholar. A research artifact with inference code — no Docker deployment, no MCP integration, single datastore. Best for researchers who want to run a dedicated synthesis model and can manage their own infrastructure.

**[Valsci](https://github.com/bricee98/Valsci)** — Open-source, self-hostable batch claim verification tool ([BMC Bioinformatics, 2025](https://doi.org/10.1186/s12859-025-06159-4)). Uses RAG + bibliometric scoring + chain-of-thought to verify scientific claims on an ordinal scale (Contradicted → Highly Supported). Requires ~1.5TB of local Semantic Scholar datasets. Best for large-batch hypothesis validation rather than real-time research synthesis.

**[paper-search-mcp](https://github.com/openags/paper-search-mcp)** — MCP server supporting 7 academic platforms (arXiv, PubMed, bioRxiv, Semantic Scholar, etc.) for search and PDF download. No synthesis, no grounding, no schema normalization. Best as a lightweight search connector when you only need paper discovery via MCP.

**Other academic MCP servers** (arXiv MCP, PubMed MCP, Semantic Scholar MCP, Wiley AI Gateway) — Single-provider MCP wrappers for search and metadata retrieval. No synthesis or cross-provider capabilities.

### What ScholarFlux MCP Adds

ScholarFlux provides reliable multi-provider search with rate limiting, normalization, and caching. The tools above each extend parts of the research workflow. ScholarFlux MCP builds the analysis and verification pipeline that connects search to synthesis:

1. **Deduplication + reranking before synthesis**: Records from multiple providers are deduplicated via rapidfuzz and ranked by embedding similarity to your research question — reducing noise before the LLM sees them
2. **AI synthesis with structured citation injection**: PydanticAI agents generate evidence-based research summaries using index-based citation (`[N]`), constrained to reference only retrieved records
3. **Post-synthesis citation verification**: The GroundingService validates every citation against source records — checking index bounds and using rapidfuzz to verify that referenced text actually matches the cited record — reporting verified counts *and* rejections
4. **Transparency by default**: Confidence scores, grounding statistics, per-record relevance scores, explicit limitations, and suggested follow-up queries in every synthesis — surfaced directly to the MCP client
5. **Zero-config model selection**: The model factory cascades through available providers (Ollama local → Ollama Cloud → Anthropic → Google → OpenAI), adapting to your environment for both LLM synthesis and embedding-based reranking
6. **Output history**: SQLModel-based relational storage lets you replay previous syntheses, relevance searches, and record searches without re-executing the pipeline
7. **Turnkey Docker deployment**: `docker compose up` gives you the full stack — MCP server, MongoDB caching, optional Redis and Ollama with auto-model-pull — with health checks, resource limits, and non-root security
8. **MCP-native**: Integrates directly into Claude Desktop, Claude Code, Neovim, or any MCP-compatible client as part of your existing workflow

### When to Use Each Approach

**Use Elicit or Consensus** when:
- You want a polished, managed SaaS experience with no setup
- You need systematic review workflows with screening criteria (Elicit)
- You want quick consensus checks on specific scientific claims (Consensus)
- Budget for paid tiers is available

**Use OpenScholar** when:
- You want a dedicated, fine-tuned synthesis model optimized for citation accuracy
- You're comfortable setting up ML inference infrastructure
- Single-source (Semantic Scholar) coverage is sufficient

**Use Valsci** when:
- You need to verify hundreds of scientific claims in batch
- You can host ~1.5TB of Semantic Scholar data locally
- Your workflow is claim verification (true/false) rather than open-ended synthesis

**Use ScholarFlux MCP** when:
- You need the **full research lifecycle** — search, synthesis, and grounding — orchestrated as a single pipeline rather than stitched together manually
- You need **grounded synthesis with transparent verification metrics**, including rejection counts that tell you when the LLM fabricated citations
- You need to **control which academic databases** are searched
- You need **multi-query support** to search related concepts together
- You need **zero-config model selection** that adapts to your environment — local Ollama for privacy, Ollama Cloud for minimax/devstral, cloud providers for scale — without manual configuration
- You want a **turnkey Docker stack** without managing terabytes of data
- You want **MCP integration** embedded in your editor or agent workflow
- You're building **production research pipelines** with caching, rate limiting, and error handling


## Related Projects

- [ScholarFlux](https://github.com/SammieH21/scholar-flux) - The underlying academic API orchestration library
- [PydanticAI](https://ai.pydantic.dev/) - AI agent framework used for synthesis
- [Model Context Protocol](https://modelcontextprotocol.io/) - The protocol specification

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

**Note**: This license covers the code only. Data accessed through academic APIs is subject to the respective providers' terms of service - see [NOTICE](NOTICE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up the development environment
- Code style, linting, and testing requirements
- Submitting pull requests

See [GitHub Issues](https://github.com/SammieH21/scholar-flux-mcp/issues) for open tasks.


## Acknowledgments

Thanks to Springer Nature, Crossref, PLOS, PubMed, arXiv, OpenAlex, and CORE for providing public access to their academic databases through their respective APIs.

ScholarFlux MCP is built on top of [ScholarFlux](https://github.com/SammieH21/scholar-flux), [PydanticAI](https://ai.pydantic.dev/), and [FastMCP](https://gofastmcp.com). Special appreciation to the research software engineering community and the open-source contributors who have helped improve ScholarFlux and ScholarFlux MCP through bug reports, feature suggestions, and pull requests.


## Citation

If you use ScholarFlux MCP in your research, please cite it:

```bibtex
@software{scholarfluxmcp,
  author = {Haskin, Sammie},
  title = {ScholarFlux MCP: AI-Powered Academic Research Synthesis and Citation Grounding as MCP Tools},
  year = {2026},
  url = {https://github.com/SammieH21/scholar-flux-mcp},
  version = {0.1.0}
}
```

Citing the tools you use helps support open-source development and aids reproducibility.


## Contact

Questions or suggestions? Open an issue or email scholar.flux@gmail.com.


---

## Project Statistics

- **~19.6k Lines of Code** - ~12.9k LOC source + ~6.7k LOC comprehensive tests
- **93% Test Coverage** - Rigorous testing across core functionality and edge cases
- **Type-Safe Architecture** - Comprehensive type hints throughout the codebase with mypy strict-mode type checking
- **Security-Audited** - Automated CVE scanning via CodeQL and Safety CLI, credential masking
- **Zero Known CVEs** - Continuous security monitoring in CI/CD pipeline

---

**Built with ❤️ for researchers who need verifiable, grounded AI synthesis**


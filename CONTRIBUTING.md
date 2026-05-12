# Contributing to ScholarFlux MCP Server

This MCP server extends [ScholarFlux](https://github.com/SammieH21/scholar-flux) to provide LLMs with academic search and research synthesis capabilities via the [Model Context Protocol](https://modelcontextprotocol.io).

> **⚠️ Project Scope and Responsibility**
> This is a **research tool** for academic literature synthesis, not a clinical or diagnostic tool. Contributors should ensure that all features, documentation, and examples emphasize the need for human oversight and professional interpretation. We do not provide medical advice, and all contributions should maintain this boundary.

## Quick Start

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- [ScholarFlux](https://github.com/SammieH21/scholar-flux) library (installed via Poetry)
- Optional: Redis (for caching), Ollama (for local synthesis)

### Setup

```bash
# Clone the repository
git clone https://github.com/SammieH21/scholar-flux-mcp.git
cd scholar-flux-mcp

# Install with all required and optional dependencies
poetry install --all-extras --with dev,testing

# Verify the installation:
poetry run pytest tests/ -v
```

> **Note:** For detailed setup instructions, see [README.md > Configuration](README.md#configuration).


## Project Structure

```

FastMCP Server (src/scholar_flux_mcp/server/main.py)
├── create_server()           — Server factory with lifespan management
├── app_lifespan()            — Async context manager initializing all services
├── _register_tools()         — Registers 9 MCP tools with deferred service access
└── AppContext                — Dataclass container for lazily initialized services
     │
     ├── Tool I/O (server/io/)     — Pydantic input validation + markdown/JSON formatting
     │   ├── base.py               — BaseFormatter, BaseToolInput BaseResearchToolInput (search, relevance search, synthesis)
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
     │   └── EmbedderABC                     — Abstract base for all embedding models
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
     │   ├── import_exceptions.py        — MCPImportError, ScholarFluxImportError, PydanticAIImportError, SQLModelImportError
     │   ├── history_exceptions.py       — HistoryCache exceptions (init, retrieval, storage, deletion)
     │   ├── mcp_server_exceptions.py    — MCPServerException, MCPServerInitializationException
     │   └── synthesis_exceptions.py     — Grounding/dedup exceptions
     │
     ├── Utils (utils/)                  — Shared utilities
     │   ├── preprocessing_utils.py      — Record conversion, dedup (rapidfuzz), context building
     │   ├── helpers.py                  — Type coercion, timestamps, truncation
     │   ├── similarity_utils.py         — Cosine similarity calculation
     │   ├── fuzzy_text_similarity.py    — Utilities for fuzzy string matching with `rapidfuzz`
     │   └── logging.py                  — Logging setup with sensitive data masking
     │
     └── Transport (server/transport.py) — MCP transport configuration
         ├── StdioTransport              — Default (no additional args)
         ├── ServerSentEventTransport    — SSE with host/port
         └── StreamableHTTPTransport     — HTTP with stateless/json options
tests/                         # Test suite (repo root)
```

**Design Principles:**
- **Single Responsibility**: Each service handles one concern
- **Dependency Injection**: Services receive dependencies at initialization
- **Type Safety**: Full Pydantic validation on all I/O
- **Thin Tools**: MCP tools are adapters between FastMCP and services

## Development

### Environment Variables (Optional)

```bash
# For synthesis with Anthropic (third in priority when `SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER` is not set and neither Ollama nor Ollama Cloud are available)
export ANTHROPIC_API_KEY=your_key
export SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER=anthropic

# For local synthesis with Ollama
export SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER=ollama
export SCHOLAR_FLUX_MCP_OLLAMA_MODEL=glm-4.7-flash:latest

# For enhanced API access
export PUBMED_API_KEY=your_key
export SPRINGER_NATURE_API_KEY=your_key
export CORE_API_KEY=your_key
```

### Enabling Debug Logging

ScholarFlux MCP inherits the logging utility and sensitive masker from the base package.
By default, the log level for the base package and MCP server are set to the `WARNING` log level.
If you need more detailed logs for development from the base package:

```bash
export SCHOLAR_FLUX_ENABLE_LOGGING=TRUE
export SCHOLAR_FLUX_LOG_LEVEL=INFO
export SCHOLAR_FLUX_PROPAGATE_LOGS=TRUE
```

To customize the log level for the MCP server:

```bash
export SCHOLAR_FLUX_MCP_LOG_LEVEL=DEBUG
```   

## Testing & Code Quality

Both ScholarFlux and ScholarFlux MCP use comprehensive testing and linting setups to ensure code quality:

### Running Tests

**Quick test (current Python version):**
```bash
poetry run pytest tests -rsx -vv
```

**Testing with Coverage**

```bash
# Run tests with coverage reporting
poetry run pytest --cov-report=term-missing

# Run with verbose output and coverage
poetry run pytest -rsx -vv --cov=scholar_flux_mcp --cov-report=term-missing

# Run specific test file
poetry run pytest tests/test_synthesis.py -v
```

**Testing Standards:**
- New features require tests
- Mock external dependencies (APIs, databases)
- Test edge cases and error conditions

**Test all Python versions (3.10, 3.11, 3.12, 3.13, optionally 3.14):**
```bash
poetry run tox
```

**With coverage report:**
```bash
poetry run tox -e coverage
```

Coverage reports are generated as both terminal output and XML format in `coverage.xml`.

Before submitting PRs, use `poetry run tox` to verify that the patch works across all supported Python versions (3.10-3.13)

**GitHub Workflow Testing Locally**

For testing GitHub Actions workflows locally, users can use [`act`](https://github.com/nektos/act) if workflow functionality needs to be vetted before implementation.

**Installation:**
```bash
# macOS
brew install act

# Linux/Windows
curl -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

**Important Note for CI Workflow Testing:**

When testing `.github/workflows/ci.yml`, which runs pytest for multiple Python versions in parallel, you may encounter port conflicts with Redis and MongoDB service containers. Since `act` runs all matrix jobs on your local machine (unlike GitHub Actions which uses separate VMs), the services will try to bind to the same ports.

**Solutions:**

1. **Run one Python version at a time** (recommended):
```bash
   act -W .github/workflows/ci.yml --matrix python-version:3.13
```

2. **Run sequentially instead of in parallel**:
   Temporarily modify the workflow to remove the matrix strategy

3. **Use dynamic port mapping**:
   Modify the workflow to use dynamic ports (see GitHub Actions service container documentation)

**Note:** If you run the full matrix workflow with `act` without addressing port conflicts, MongoDB and Redis tests may be skipped for all but one Python version.

For more information on `act`, see the [official documentation](https://github.com/nektos/act).


### Linting

```bash
# Check code quality
poetry run ruff check src tests

# Auto-fix issues
poetry run ruff check src tests --fix

# Type checking
poetry run mypy src tests
```

### Development Shortcuts

A `Makefile` provides quick commands for common tasks during active development:

```bash
make help      # Show all available commands
make install   # Install ScholarFluxMCP with the dependencies for development, testing, and documentation
make test      # Run the test suite with coverage (current Python version only)
make lint      # Check code quality (ruff, docstr-coverage, mypy)
make docs      # Build documentation locally
make shell     # Enter the Poetry virtual environment
make mcp       # Start MCP Inspector to directly test MCP tools
```

## What Can I Contribute?

### Good First Issues

- Documentation improvements
- Additional test coverage
- Bug fixes with failing tests

### New Features

- Additional MCP tools
- New research categories for synthesis
- Caching backend implementations
- History service enhancements

### Bug Fixes

1. Check open issues labeled `bug`
2. Write a failing test that demonstrates the bug
3. Fix the bug
4. Submit PR with test and fix

### Documentation

- README examples
- Tool usage documentation
- Docstring improvements

## Development Guidelines

### Code Style

- **Type hints**: Required for all function signatures
- **Docstrings**: Required for public classes and functions
- **Line length**: 120 characters max (matches scholar-flux)
- **Formatting**: Use `poetry run ruff format`
- **Convention**: Google style for docstrings

## Contribution Workflow

### 1. Branch Naming

```
feature/add-support-for-hugging-face-embeddings
fix/relevance-search-service-embedding-fallback
docs/improve-tool-examples
test/add-synthesis-service-tests
```

**Staging Branch**: `develop` (Staging branch for vetting releases with `testpypi`)
**Production Branch**: `main` (Main branch for major releases via `pypi`)

### 2. Development Process

1. Create a new branch from `main`
2. Write tests first (Test Driven Development encouraged!)
3. Implement your changes
4. Add type hints and docstrings to all new functions
5. Document all public APIs with docstrings following PEP 257
6. Run `poetry run tox -e lint` before committing
7. Ensure all tests pass with `poetry run tox`



### 3. Commit Messages

Follow conventional commits:

```
feat: add Hugging Face model support
fix: resolve synthesis history service bug
docs: add examples for relevance search tools
test: add integration tests for synthesis service
```

### 4. Pull Request Checklist

Before submitting:

- [ ] Tests pass (`make test`)
- [ ] Type Checking (`mypy`) and linting via (`ruff`) pass (`make lint`)
- [ ] New features include tests
- [ ] Docstrings added for public APIs
- [ ] No sensitive data committed
- [ ] Dependencies updated in `pyproject.toml` if needed
- [ ] `poetry lock` run if dependencies changed

### 5. PR Description Template

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation

## Testing
How did you test these changes?
```

## Architecture Notes

### Adding a New MCP Tool

1. Create and add the service for the tool in `services/`
2. Create input model and output formatter in `server/io/`
3. Implement the tool function accepting a Pydantic input model
4. Register in `server/main.py` inside `_register_tools()`

```python
# Tool signature pattern (Pydantic model for input validation + schema generation)
@server.tool(name="scholar_flux_my_tool")
async def scholar_flux_my_tool(
    params: MyToolInput,
    ctx: Context[ServerSession, AppContext],
) -> str:
    # params is already validated by Pydantic
    # ... implementation
    return MyToolFormatter.format(result, response_format=params.response_format) # returns Markdown or JSON formatted output
```

### Adding a New Service

1. Create a service class in `services/`
2. Follow the single-responsibility principle — a tool should ideally perform one task exceptionally rather than several satisfactorily
3. The service should accept dependencies via `__init__`
4. Add the service to the lifespan context in `server/app_context.py`
5. Write tests with mocked dependencies


### Questions?

- Check existing [documentation and tutorials](https://SammieH21.github.io/scholar-flux/) first
- Open a discussion for general questions
- Open an issue for bugs or feature requests
- Email us at scholar.flux@gmail.com for other inquiries

## Resources

- **ScholarFlux Docs (Base)**: [https://SammieH21.github.io/scholar-flux/](https://SammieH21.github.io/scholar-flux/)
- **Security**: See [SECURITY.md](SECURITY.md) for security-related questions
- **Issues**: [GitHub Issues](https://github.com/SammieH21/scholar-flux-mcp/issues)
- **MCP Spec**: [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)
- **AI-Assisted Code Review**: [.github/AI_REVIEW_PROMPTS.md](.github/AI_REVIEW_PROMPTS.md) - Prompts for code/documentation review and test gap analysis
- **Email**: scholar.flux@gmail.com

### Response Times

- **Issues/PRs**: We aim to respond within 3-5 business days
- **Security issues**: Within 72 hours (see [SECURITY.md](SECURITY.md))
- **General inquiries**: Within 1 week


## Code of Conduct

We are committed to providing a welcoming and inclusive environment.

- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on constructive feedback

Find the code of conduct [**here**](CODE_OF_CONDUCT.md)

## Project Status

ScholarFlux MCP is currently in **beta** (v0.1.0). This means:

- APIs may change between versions
- We're actively seeking feedback
- Breaking changes may occur before 1.0
- Security vulnerabilities are addressed promptly
- Contributors have significant impact on project direction

## Recognition

Contributors will be recognized in:
- Release notes for their contributions
- GitHub contributors page
- Project documentation (if desired)

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.
Keep in mind that the overall scope of the license **covers only the code itself**. See [NOTICE](NOTICE) for details.

By contributing, you agree that your contributions will be licensed under the same license.

---

Thank you for helping to improve ScholarFlux MCP!

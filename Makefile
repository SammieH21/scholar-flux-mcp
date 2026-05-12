.PHONY: install test lint format help spell_check docs mcp shell

# MCP Defaults: Edit this configuration as needed based on your own setup
# session cache options: mongodb, sqlite, redis, gridfs, memory, filesystem
# response cache options: mongodb, sql/sqlite, redis, duckdb, memory/inmemory, null
SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND ?= sqlite
SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE ?= sqlite
SCHOLAR_FLUX_LOG_LEVEL ?= WARNING
SCHOLAR_FLUX_MCP_LOG_LEVEL ?= INFO
SCHOLAR_FLUX_MCP_HISTORY_URL ?= sqlite:///./.cache/history.sqlite
SCHOLAR_FLUX_MCP_DEFAULT_EMBEDDING_MODEL_PROVIDER ?= ollama
SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER ?= ollama_cloud
# PydanticAI agent model defaults
SCHOLAR_FLUX_MCP_OLLAMA_BASE_URL ?= http://localhost:11434
SCHOLAR_FLUX_MCP_OLLAMA_MODEL ?= glm-4.7-flash:latest
SCHOLAR_FLUX_MCP_OLLAMA_CLOUD_MODEL ?= minimax-m2.5:cloud
SCHOLAR_FLUX_MCP_OLLAMA_EMBEDDING_MODEL ?= embeddinggemma:latest
SCHOLAR_FLUX_MCP_ANTHROPIC_MODEL ?= claude-haiku-4-5
SCHOLAR_FLUX_MCP_GOOGLE_MODEL ?= gemini-2.5-flash
SCHOLAR_FLUX_MCP_OPENAI_MODEL ?= gpt-5
SCHOLAR_FLUX_MCP_OPENAI_PROVIDER ?= openai
# SCHOLAR_FLUX_MCP_OPENAI_ENDPOINT ?= # add this directly to the makefile below to set a non-default endpoint
SCHOLAR_FLUX_MCP_REQUEST_TIMEOUT ?= 120
# How many input tokens should be passed to agent classes
SCHOLAR_FLUX_MCP_GROUNDING_TOKEN_LIMIT ?= 120000
# The text limit of individual fields
SCHOLAR_FLUX_MCP_RECORD_TRUNCATION_LENGTH ?= 3000

# Designed mainly for linux/Unix (Mac) compatibility. Use [Git Bash](https://gitforwindows.org/) if you encounter any issues using Windows.

# A simple help command to list available targets
help:
	@echo "Available commands:"
	@echo "  install        Installs the ScholarFlux MCP package for development with all extras"
	@echo "  test           Runs tests with pytest within the poetry environment"
	@echo "  lint           Runs linting and type checking tools (e.g., ruff, mypy, docstr-coverage)"
	@echo "  format         Runs Ruff for stylistic code changes and Ruff with --fix for potential linting issues"
	@echo "  docs           Autogenerates Sphinx documentation from in-code docstrings and rst files"
	@echo "  spell_check    Uses cspell to check spelling in python files (docstrings, etc.)"
	@echo "  mcp            Sets up a basic npm mcp inspector session for debugging scholar-flux-mcp configurations"
	@echo "  shell          Activates the project's virtual environment shell"

# Installs dependencies from poetry.lock
install:
	@echo "Installing project dependencies..."
	poetry install --all-extras --with dev,testing
	poetry run mypy --install-types --non-interactive src tests
	poetry run pip install types-requests types-xmltodict types-PyYAML

# Runs tests using `poetry run` to execute commands within the virtual environment
test:
	@echo "Running tests..."
	poetry run pytest  -rsx -vv --cov=scholar_flux_mcp --cov-report=term-missing --cov-report xml

# Runs code quality checks
lint:
	@echo "Running code checks (linting and type checking)..."
	poetry run mypy src tests
	poetry run ruff check src tests
	poetry run docstr-coverage src

# Uses Ruff for stylistic codebase formatting and fixing missing imports, stylistic issues, etc.
format:
	@echo "Formatting code structure..."
	poetry run ruff format src tests
	poetry run ruff check src tests --fix

# Uses cspell to check spelling in python files (docstrings, etc.)
spell_check:
	@echo "Running CSpell spell checker..."
	act -W .github/workflows/spell_checker.yml

# Builds Sphinx documentation
docs:
	poetry run $(MAKE) -C docs clean
	poetry run $(MAKE) -C docs html

# Activates the node package manager MCP inspector package with default session and response cache settings
mcp:
	# Also add API keys for ANTHROPIC, OPENAI, or GEMINI if needed (but try not to hardcode them)
	npx @modelcontextprotocol/inspector \
		-e SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND=${SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND} \
		-e SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE=${SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE} \
		-e SCHOLAR_FLUX_LOG_LEVEL=${SCHOLAR_FLUX_LOG_LEVEL} \
		-e SCHOLAR_FLUX_MCP_LOG_LEVEL=${SCHOLAR_FLUX_MCP_LOG_LEVEL} \
        -e SCHOLAR_FLUX_MCP_HISTORY_URL=${SCHOLAR_FLUX_MCP_HISTORY_URL} \
        -e SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER=${SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER} \
        -e SCHOLAR_FLUX_MCP_DEFAULT_EMBEDDING_MODEL_PROVIDER=${SCHOLAR_FLUX_MCP_DEFAULT_EMBEDDING_MODEL_PROVIDER} \
        -e SCHOLAR_FLUX_MCP_OLLAMA_BASE_URL=${SCHOLAR_FLUX_MCP_OLLAMA_BASE_URL} \
		-e SCHOLAR_FLUX_MCP_OLLAMA_CLOUD_MODEL=${SCHOLAR_FLUX_MCP_OLLAMA_CLOUD_MODEL} \
        -e SCHOLAR_FLUX_MCP_OLLAMA_MODEL=${SCHOLAR_FLUX_MCP_OLLAMA_MODEL} \
        -e SCHOLAR_FLUX_MCP_OLLAMA_EMBEDDING_MODEL=${SCHOLAR_FLUX_MCP_OLLAMA_EMBEDDING_MODEL} \
        -e SCHOLAR_FLUX_MCP_ANTHROPIC_MODEL=${SCHOLAR_FLUX_MCP_ANTHROPIC_MODEL} \
        -e SCHOLAR_FLUX_MCP_GOOGLE_MODEL=${SCHOLAR_FLUX_MCP_GOOGLE_MODEL} \
        -e SCHOLAR_FLUX_MCP_OPENAI_MODEL=${SCHOLAR_FLUX_MCP_OPENAI_MODEL} \
        -e SCHOLAR_FLUX_MCP_OPENAI_PROVIDER=${SCHOLAR_FLUX_MCP_OPENAI_PROVIDER} \
        -e SCHOLAR_FLUX_MCP_REQUEST_TIMEOUT=${SCHOLAR_FLUX_MCP_REQUEST_TIMEOUT} \
		-e SCHOLAR_FLUX_MCP_GROUNDING_TOKEN_LIMIT=${SCHOLAR_FLUX_MCP_GROUNDING_TOKEN_LIMIT} \
		python -m scholar_flux_mcp

# Activates the poetry virtual environment shell
shell:
	poetry shell




# ScholarFluxMCP Server Dockerfile
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Install pip package manager dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
COPY . ./scholar-flux-mcp

# Install the package with all dependencies
RUN cd scholar-flux-mcp && pip install --no-cache-dir .

# ========================================================================================
# Stage 2: Runtime
# This stage starts from slim once more, copying only required files from the build stage.
# ========================================================================================
FROM python:3.11-slim AS runtime

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the virtual environment from the builder stage (ignoring build dependencies in the process)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the ScholarFluxMCP package into the container
COPY --chown=appuser:appuser . ./scholar-flux-mcp/

# Create a ScholarFlux home directory
RUN mkdir -p /app/scholar-flux/{package_cache,logs} && \
    chown -R appuser:appuser /app/scholar-flux

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SCHOLAR_FLUX_HOME=/app/scholar-flux \
    SCHOLAR_FLUX_ENABLE_LOGGING=TRUE \
    SCHOLAR_FLUX_LOG_LEVEL=INFO \
    SCHOLAR_FLUX_MCP_LOG_LEVEL=INFO \
    SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND=sqlite \
    SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE=mongodb

# Switch to non-root user
USER appuser

# Health check (for HTTP transport mode)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import scholar_flux_mcp; print('healthy')" || exit 1

# Default command runs stdio transport
# Override with --transport streamable_http for HTTP mode
ENTRYPOINT ["python", "-m", "scholar_flux_mcp"]
CMD []

# Labels
LABEL org.opencontainers.image.title="ScholarFluxMCP Server" \
      org.opencontainers.image.description="MCP server for academic record retrieval across 7+ providers, featuring embedding-driven record relevance re-ranking and AI-powered research synthesis with citation verification built on ScholarFlux and PydanticAI." \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/SammieH21/scholar-flux-mcp"

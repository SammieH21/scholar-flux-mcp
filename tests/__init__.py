"""ScholarFlux MCP Server Test Suite.

This package contains comprehensive tests for:
- Cache service (MongoDB/Redis backends)
- Search service (multi-provider coordination)
- Synthesis service (PydanticAI integration)
- MCP tools (tool registration and execution)

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=scholar_flux_mcp

Test categories:
- Unit tests: Individual service/component testing
- Integration tests: Cross-service interaction testing
- Tool tests: MCP tool execution testing

"""

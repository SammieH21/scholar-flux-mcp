"""Tests for the MCP transport mechanism used to initialize the ScholarFlux MCP server."""

import pytest

from scholar_flux_mcp.server.transport import (
    MCPTransports,
    ServerSentEventTransport,
    StdioTransport,
    StreamableHTTPTransport,
)


@pytest.mark.parametrize(
    "transport,expected",
    [
        ("sse", MCPTransports.SSE),
        ("stdio", MCPTransports.STDIO),
        ("streamable-http", MCPTransports.STREAMABLE_HTTP),
        ("STREAMABLE HTTP", MCPTransports.STREAMABLE_HTTP),
        ("StdIO", MCPTransports.STDIO),
    ],
)
def test_mcp_transports_enum_selection(transport, expected):
    """Verifies that the MCPTransports enum returns the selected, available setting when requested."""
    setting = MCPTransports(transport)

    assert setting is expected


@pytest.mark.parametrize(
    "transport,expected",
    [
        ("sse", ServerSentEventTransport),
        ("stdio", StdioTransport),
        ("streamable-http", StreamableHTTPTransport),
        ("STREAMABLE HTTP", StreamableHTTPTransport),
        ("StdIO", StdioTransport),
    ],
)
def test_transport_selection(transport, expected):
    """Verifies that the selected transport returns the expected transport setting."""
    setting = MCPTransports.get_transport_setting(transport)
    assert isinstance(setting, expected)


@pytest.mark.parametrize(
    "transport",
    [
        ("ss",),
        ("io",),
        ("http",),
        ("transport HTTP",),
        ("another transport type",),
        ([1, 2, 3],),
        (None,),
    ],
)
def test_transport_selection_with_unknown_setting(transport):
    """Verifies that MCPTransports enum returns None when an unknown setting is identified."""
    with pytest.raises(ValueError):
        setting = MCPTransports(transport)

    mcp_transport_enum = MCPTransports.get(transport)
    assert mcp_transport_enum is None

    setting = MCPTransports.get_transport_setting(transport)  # type: ignore
    assert setting is None


@pytest.mark.parametrize(
    "transport,expected",
    [
        ("sse", ServerSentEventTransport),
        ("stdio", StdioTransport),
        ("streamable-http", StreamableHTTPTransport),
        ("STREAMABLE HTTP", StreamableHTTPTransport),
        ("StdIO", StdioTransport),
        ("Unknown Transport Type", StdioTransport),  # defaults back to stdio when unknown
    ],
)
def test_transport_env_settings_modifies_mcp_transport_protocol(transport, expected, monkeypatch):
    """Verifies that the `SCHOLAR_FLUX_MCP_TRANSPORT` correctly sets the transport type when specified."""
    with monkeypatch.context() as m:
        m.setenv("SCHOLAR_FLUX_MCP_TRANSPORT", transport)
        setting = MCPTransports.get_default()
        assert isinstance(setting, expected)


def test_transport_setting_selection_defaults_stdio_without_assigned_env(monkeypatch):
    """Verifies that the MCP transport setting defaults to STDIO when `SCHOLAR_FLUX_MCP_TRANSPORT` is None."""
    with monkeypatch.context() as m:
        m.delenv("SCHOLAR_FLUX_MCP_TRANSPORT", raising=False)

    setting = MCPTransports.get_default()
    assert isinstance(setting, StdioTransport)

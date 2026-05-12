"""MCP Server Transport definitions and settings for ensuring type safety when initializing the ScholarFlux MCP server.

This module defines the default MCP settings that are used at runtime to ensure that MCP options are defined explicitly
while remaining testable and predictable.

Classes:

    - StdioTransport: Default settings for STDIO transport (No-Op - This transport doesn't require additional arguments)
    - ServerSentEventTransport: Default settings for SSE transport. Defines the host and port for the MCP server
    - StreamableHTTPTransport: Default settings for streamable http transport.
    - MCPTransports: Enum defining each MCP transport type, providing a `get_default` method for type-safe retrieval.

At runtime, the following variables are evaluated:

    - SCHOLAR_FLUX_MCP_TRANSPORT: Defines what transport should be directly used at run time
    - SCHOLAR_FLUX_MCP_HOST: The IP address that should be used to run the MCP server
    - SCHOLAR_FLUX_MCP_PORT: The port that the MCP server should run on

"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from typing import Literal

from scholar_flux_mcp.utils.helpers import coerce_int

CHARACTER_SEPARATORS_PATTERN = re.compile("-| |_")

# Import tool input models

logger = logging.getLogger(__name__)


TransportType = Literal["sse", "streamable-http", "stdio"]

# =========================================================================
# MCP TRANSPORT CONFIGURATION
# =========================================================================


@dataclass
class TransportSettings(ABC):
    """Classes defining the types of transport options available for use."""

    @property
    def options(self) -> dict:
        """Property returning available options as a dict."""
        return asdict(self)

    @property
    @abstractmethod
    def name(self) -> TransportType:
        """Property returning the name of the current transport option."""
        ...


@dataclass
class StdioTransport(TransportSettings):
    """The basic STDIO transport method requires no additional arguments."""

    @property
    def name(self) -> TransportType:
        """Property returning the code of the StdIO transport option."""
        return "stdio"


@dataclass
class ServerSentEventTransport(TransportSettings):
    """Defines basic options used for both `sse` and `streamable-transport`"""

    host: str = os.getenv("SCHOLAR_FLUX_MCP_HOST") or "127.0.0.1"
    port: int = coerce_int(os.getenv("SCHOLAR_FLUX_MCP_PORT")) or 8000

    @property
    def name(self) -> TransportType:
        """Property returning the code of the sse transport option."""
        return "sse"


@dataclass
class StreamableHTTPTransport(ServerSentEventTransport):
    """Defines the default options used for Streamable-HTTP."""

    stateless_http: bool = True
    json_response: bool = True

    @property
    def name(self) -> TransportType:
        """Property returning the code of the `streamable-http` transport option."""
        return "streamable-http"


class MCPTransports(Enum):
    """Defines the available MCP Transport options."""

    STDIO = StdioTransport()
    SSE = ServerSentEventTransport()
    STREAMABLE_HTTP = StreamableHTTPTransport()

    @classmethod
    def get(
        cls, transport_type: MCPTransports | TransportSettings | str | TransportType | None, verbose: bool = True
    ) -> MCPTransports | None:
        """Tries to find the transport from the current set of MCP Transport Types, returning None if not found."""
        try:
            return cls(transport_type)
        except (KeyError, TypeError, ValueError):
            if verbose:
                logger.warning(f"Couldn't find an MCPTransports type matching the value: {transport_type}")
        return None

    @classmethod
    def _missing_(cls, value: object) -> MCPTransports | None:
        """Overrides the behavior of the current enum to retrieve a matching transport when direct matching fails."""
        if not isinstance(value, str):
            return None

        strip_characters = partial(CHARACTER_SEPARATORS_PATTERN.sub, repl="")

        value_fmt = strip_characters(string=value.lower())

        return next(
            (format for format in cls if strip_characters(string=format.name.lower()) == value_fmt),
            None,
        )

    @classmethod
    def get_transport_setting(
        cls, transport_type: MCPTransports | TransportSettings | str | TransportType | None = None
    ) -> TransportSettings | None:
        """Retrieve the current transport setting by resolving against the MCPTransports Enum. Returns None otherwise."""
        return mcp_transport.value if (mcp_transport := cls.get(transport_type)) else None

    @classmethod
    def get_default(cls, verbose: bool = False) -> TransportSettings:
        """Reads the default transport type for the MCP server."""
        transport_type = os.getenv("SCHOLAR_FLUX_MCP_TRANSPORT") or "stdio"
        if transport_settings := cls.get(transport_type, verbose=False):
            return transport_settings.value
        if verbose:
            logger.warning(
                f"The transport default from the environment ({transport_settings}) is not a valid transport type. "
                "Defaulting to `stdio`..."
            )
        return cls.STDIO.value


__all__ = ["TransportType", "TransportSettings", "MCPTransports"]

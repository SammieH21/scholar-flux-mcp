"""Providers tool for ScholarFlux MCP server.

Lists and provides information about available academic database providers.

"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ConfigDict

from scholar_flux_mcp.models import ProviderInfo, ResponseFormat
from scholar_flux_mcp.server.io.base import BaseFormatter, BaseToolInput


class ListProvidersInput(BaseToolInput):
    """Input model for the `scholar_flux_list_providers` tool."""

    model_config = ConfigDict(extra="forbid")


class ProvidersFormatter(BaseFormatter):
    """A provider list formatter designed to create an MCP provider list markdown summary or JSON response."""

    JSON_INDENT: int = 2

    @classmethod
    def format(
        cls,
        output: list[ProviderInfo],
        response_format: ResponseFormat | str = ResponseFormat.MARKDOWN,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Formats the provider list either as a markdown summary or JSON response.

        Args:
            live_providers (list[ProviderInfo]): A list of ProviderInfo instances from the ProviderService.
            response_format (ResponseFormat): Output format, either 'markdown' or 'json'
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            A formatted provider list as a markdown or JSON string

        """
        format = ResponseFormat(response_format)
        return (
            cls.format_providers_json(output, *args, **kwargs)
            if format == ResponseFormat.JSON
            else cls.format_providers_markdown(output, *args, **kwargs)
        )

    @classmethod
    def format_providers_markdown(
        cls,
        live_providers: list[ProviderInfo],
        *,
        max_providers: int | None = None,
    ) -> str:
        """Formats the supported provider list as a markdown report."""
        max_providers = max_providers or len(live_providers)
        display_providers = live_providers[:max_providers]

        lines = [
            "# Available Academic Database Providers",
            "",
            "ScholarFlux supports the following academic databases:",
            "",
        ]

        for info in display_providers:
            lines.append(cls.format_provider(info))
            lines.append("")

        lines.extend(
            [
                "---",
                "",
                "## Usage Tips",
                "",
                "- For **mental health research**, start with `pubmed` and `plos`",
                "- For **preprints and computational methods**, use `arxiv`",
                "- For **bibliometric analysis**, use `openalex`",
                "- Combine multiple providers for comprehensive coverage",
                "",
            ]
        )

        return "\n".join(lines)

    @classmethod
    def format_providers_json(
        cls,
        live_providers: list[ProviderInfo],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Uses valid a list of ProviderInfo instances to create a serialized json string."""
        providers_dict = {
            provider_info.name: provider_info.model_dump(*args, **kwargs) for provider_info in live_providers
        }
        return json.dumps({"providers": providers_dict}, indent=cls.JSON_INDENT, default=str)

    @classmethod
    def format_provider(cls, info: ProviderInfo) -> str:
        """Formats a single provider as a markdown string."""
        display_name = info.display_name or info.name.title()
        desc = info.description or "No description available"
        coverage = info.coverage or "N/A"
        api_key = "Required" if info.requires_api_key else "Not required"
        rate_limit = f"{info.rate_limit}s" if info.rate_limit else "Rate limit info available"
        recommended_for = ", ".join(info.recommended_for) if info.recommended_for else "[Not specified]"

        return cls.format_multiline_string(
            f"""
            ## {display_name}

            **ID**: `{info.name}`
            **URL**: {info.base_url}
            **Description**: {desc}
            **Coverage**: {coverage}
            **Recommended for**: {recommended_for}
            **Rate Limit**: {rate_limit}
            **API Key**: {api_key}
            """
        )


__all__ = ["ListProvidersInput", "ProvidersFormatter"]

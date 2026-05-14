"""Provider service for ScholarFlux MCP server.

Thin wrapper around scholar-flux's provider_registry that provides MCP-specific provider information and validation.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scholar_flux.api.models import ProviderConfig
    from scholar_flux.api.providers import provider_registry
else:
    try:
        from scholar_flux.api.models import ProviderConfig
        from scholar_flux.api.providers import provider_registry
    except ImportError:
        provider_registry = None
        ProviderConfig = None
from scholar_flux_mcp.models import ProviderInfo, ProviderMetadataDescriptions

logger = logging.getLogger(__name__)


class ProviderService:
    """Provider information service delegating to scholar-flux.

    Provides a unified interface for accessing provider metadata, validating provider names, and retrieving display
    information.

    """

    def get_provider_names(self) -> list[str]:
        """Get list of available provider names.

        Returns:
            List of provider name strings (lowercase, normalized).

        """
        return provider_registry.providers if provider_registry is not None else []

    def get_config(self, name: str) -> ProviderConfig | None:
        """Get the ProviderConfig for a specific provider.

        Args:
            name: Provider name (case-insensitive).

        Returns:
            ProviderConfig instance or None if not found.

        """
        return provider_registry.get(name) if provider_registry is not None else None

    def get_display_name(self, name: str) -> str:
        """Get human-readable display name for a provider.

        Args:
            name: Provider name.

        Returns:
            Display name from config or title-cased name if not found.

        """
        return provider_registry.get_display_name(name) or name.title() if provider_registry else name.title()

    def get_all_info(self) -> list[ProviderInfo]:
        """Get detailed information for all providers.

        Returns:
            List of ProviderInfo instances describing:
            - name: Provider identifier
            - display_name: Human-readable name
            - requires_api_key: Whether API key is required
            - rate_limit: Request delay in seconds
            - docs_url: Documentation URL
            - base_url: API base URL

        """
        return [
            info
            for provider_name in self.get_provider_names()
            if (info := self.get_provider_info(provider_name)) is not None
        ]

    def get_provider_info(self, name: str) -> ProviderInfo | None:
        """Get detailed information for a specific provider.

        Args:
            name (str): Provider name.

        Returns:
            ProviderInfo | None: provider metadata if a corresponding configuration was found. None otherwise.

        """
        config = self.get_config(name)
        if not config:
            return None

        return ProviderInfo(
            name=config.provider_name,
            display_name=config.display_name,
            base_url=config.base_url,
            requires_api_key=config.api_key_required,
            api_key_env_var=config.api_key_env_var,
            rate_limit=config.request_delay,
            records_per_page=config.records_per_page,
            docs_url=config.docs_url,
        )

    async def get_providers(self) -> list[ProviderInfo]:
        """Get list of available providers with metadata.

        Returns:
            List of provider information dicts.

        """
        provider_info_list = self.get_all_info()
        return [
            provider_info.annotate(ProviderMetadataDescriptions.get_fields(provider_info.name) or {})
            for provider_info in provider_info_list
        ]


__all__ = ["ProviderService"]

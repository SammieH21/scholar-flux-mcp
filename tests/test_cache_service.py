"""Tests for the CacheService.

This module tests cache initialization and configuration.
Note: Full integration tests require Redis/MongoDB which are mocked here.

"""

from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest
from scholar_flux import CachedSessionManager
from scholar_flux.utils import config_settings

from scholar_flux_mcp.exceptions import ScholarFluxImportError
from scholar_flux_mcp.services.cache_service import CacheService, get_cache_service
from tests.testing_utilities import raise_error


class TestCacheServiceConfiguration:
    """Tests for CacheService configuration and initialization."""

    def test_default_namespace(self):
        """Test default namespace is set."""

        service = CacheService()
        assert service._namespace == "mcp_cache"

    def test_custom_namespace(self):
        """Test custom namespace is used."""

        service = CacheService(namespace="custom_namespace")
        assert service._namespace == "custom_namespace"

    def test_default_ttl_none(self):
        """Test default TTL is None (uses backend defaults)."""
        service = CacheService()
        assert service.ttl is None
        assert service.enabled is False  # not initialized at this point

    def test_custom_ttl(self):
        """Test custom TTL is used."""
        service = CacheService(ttl=3600)
        assert service.ttl == 3600

    def test_default_user_agent(self):
        """Test default user agent is set."""
        service = CacheService()
        assert service._user_agent == "ScholarFlux-MCP/1.0"

    def test_custom_user_agent(self):
        """Test custom user agent is used."""
        service = CacheService(user_agent="Custom/2.0")
        assert service._user_agent == "Custom/2.0"

    def test_default_raise_on_error_false(self):
        """Test default raise_on_error is False."""
        service = CacheService()
        assert service.raise_on_error is False

    def test_custom_raise_on_error(self):
        """Test custom raise_on_error is used."""
        service = CacheService(raise_on_error=True)
        assert service.raise_on_error is True

    def test_not_initialized_before_init(self):
        """Test service is not initialized before calling initialize()."""
        service = CacheService()
        assert service.initialized is False


class TestCacheServiceInitialization:
    """Tests for CacheService initialization."""

    @pytest.mark.asyncio
    async def test_initialize_with_redis_backend(self, restore_config):
        """Test initialize() uses redis backend from environment."""
        config_settings.set("SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE", "redis")
        config_settings.set("SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND", "redis")
        service = CacheService(namespace="test", ttl=7200)
        await service.initialize()
        assert service.enabled is True  # should now be initialized and configured to return a cached session
        assert service._data_cache_manager and "Redis" in service._data_cache_manager.cache_storage.structure()
        assert service._data_cache_manager.cache_storage.ttl == 7200
        assert service._data_cache_manager.cache_storage.namespace == "test"
        assert isinstance(service._session_manager, CachedSessionManager)
        assert service._session_manager and service._session_manager.backend == "redis"
        assert service._session_manager.user_agent == "ScholarFlux-MCP/1.0"
        assert service._session_manager.raise_on_error is False

        assert service.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_with_inmemory_fallback(self, monkeypatch):
        """Test initialize() defaults to inmemory when env not set."""
        mock_data_cache_manager = MagicMock()
        mock_session_manager = MagicMock()

        # Return None to trigger "or 'inmemory'" fallback
        monkeypatch.setattr(config_settings, "get", lambda key: None)

        with (
            patch("scholar_flux_mcp.services.cache_service.DataCacheManager") as mock_dcm,
            patch("scholar_flux_mcp.services.cache_service.CachedSessionManager") as mock_csm,
        ):
            mock_dcm.with_storage.return_value = mock_data_cache_manager
            mock_csm.return_value = mock_session_manager

            service = CacheService()
            await service.initialize()

            # Verify inmemory backend was used as fallback
            mock_dcm.with_storage.assert_called_once_with(
                "inmemory",
                namespace="mcp_cache",
                verify_connection=True,
            )

            assert service.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_graceful_fallback_on_error(self, restore_config, monkeypatch):
        """Test initialize() falls back gracefully when backend fails."""
        config_settings.set("SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE", "redis")

        with patch(
            "scholar_flux.data_storage.data_cache_manager.RedisStorage", side_effect=raise_error(ConnectionError)
        ):
            service = CacheService(raise_on_error=False)
            await service.initialize()

            # Should fall back to inmemory
            assert service._data_cache_manager and service._data_cache_manager
        assert "InMemory" in service._data_cache_manager.cache_storage.structure()  # type: ignore
        #       Session manager should fall back to basic SessionManager
        #       assert isinstance(service._session_manager, SessionManager)
        #       assert isinstance(service._session_manager, CachedSessionManager)
        assert service.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_raises_on_import_error(self, monkeypatch):
        """Test initialize() raises RuntimeError on ImportError."""
        import scholar_flux_mcp.services.cache_service

        monkeypatch.setattr(scholar_flux_mcp.services.cache_service, "DataCacheManager", None)
        with pytest.raises(ScholarFluxImportError):
            service = scholar_flux_mcp.services.cache_service.CacheService()
            await service.initialize()

    @pytest.mark.asyncio
    async def test_initialize_raises_when_raise_on_error_true(
        self,
        restore_config,
    ):
        """Test initialize() raises when raise_on_error is True."""
        config_settings.set("SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE", "redis")

        with patch("scholar_flux.DataCacheManager.with_storage", raise_error(ConnectionError)):
            service = CacheService(raise_on_error=True)

            with pytest.raises(RuntimeError):
                await service.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test initialize() can be called multiple times safely."""
        mock_data_cache_manager = MagicMock()
        mock_session_manager = MagicMock()

        with (
            patch("scholar_flux_mcp.services.cache_service.config_settings") as mock_config,
            patch("scholar_flux_mcp.services.cache_service.DataCacheManager") as mock_dcm,
            patch("scholar_flux_mcp.services.cache_service.CachedSessionManager") as mock_csm,
        ):
            mock_config.get.return_value = "inmemory"
            mock_dcm.with_storage.return_value = mock_data_cache_manager
            mock_csm.return_value = mock_session_manager

            service = CacheService()
            await service.initialize()
            await service.initialize()  # Second call
            await service.initialize()  # Third call

            # Should only create managers once
            mock_dcm.with_storage.assert_called_once()
            mock_csm.assert_called_once()

    @pytest.mark.asyncio
    async def test_properties_raise_before_init(self):
        """Test that accessing properties before init raises error."""
        service = CacheService()

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = service.data_cache_manager

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = service.session_manager

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = service.create_session()

    @pytest.mark.asyncio
    async def test_properties_return_managers_after_init(self):
        """Test that properties return managers after initialization."""
        mock_data_cache_manager = MagicMock()
        mock_session_manager = MagicMock()
        mock_session = MagicMock()
        mock_session_manager.return_value = mock_session

        with (
            patch("scholar_flux_mcp.services.cache_service.config_settings"),
            patch("scholar_flux_mcp.services.cache_service.DataCacheManager") as mock_dcm,
            patch("scholar_flux_mcp.services.cache_service.CachedSessionManager") as mock_csm,
        ):
            mock_dcm.with_storage.return_value = mock_data_cache_manager
            mock_csm.return_value = mock_session_manager

            service = CacheService()
            await service.initialize()

            assert service.data_cache_manager is mock_data_cache_manager
            assert service.session_manager is mock_session_manager
            assert service.create_session() is mock_session


class TestCacheServiceStats:
    """Tests for cache service statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_structure(self, restore_config):
        """Test get_stats returns expected structure."""
        config_settings.set("SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND", "redis")
        config_settings.set("SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE", "redis")

        service = CacheService(namespace="test", ttl=3600)

        stats = await service.get_stats()

        assert stats["initialized"] is False
        assert stats["namespace"] == "test"
        assert stats["ttl"] == 3600
        assert stats["response_cache_storage"] == "redis"
        assert stats["session_cache_backend"] == "redis"
        assert stats["user_agent"] == "ScholarFlux-MCP/1.0"
        assert stats["raise_on_error"] is False

    @pytest.mark.asyncio
    async def test_get_stats_with_defaults(self, monkeypatch):
        """Test get_stats uses defaults when env not set."""
        # Return None to trigger defaults
        monkeypatch.setattr(config_settings, "get", lambda key: None)

        service = CacheService()

        stats = await service.get_stats()

        # Should use defaults
        assert stats["response_cache_storage"] == "inmemory"
        assert stats["session_cache_backend"] == "sqlite"


class TestCacheServiceClear:
    """Tests for cache clearing."""

    @pytest.mark.asyncio
    async def test_clear_cache_returns_true_when_initialized(self):
        """Test clear_cache returns True when cache manager exists."""
        service = CacheService()
        service._data_cache_manager = MagicMock()

        result = await service.clear_cache()
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_cache_clears_from_namespace_when_available(self):
        """Test clear_cache returns True when cache manager exists and a namespace is cleared."""
        service = CacheService()
        service._data_cache_manager = MagicMock()
        service._data_cache_manager.cache_storage.namespace = "valid_namespace"

        result = await service.clear_cache()
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_cache_returns_false_when_not_initialized(self):
        """Test clear_cache returns False when not initialized."""
        service = CacheService()

        result = await service.clear_cache()
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_cache_handles_exceptions(self):
        """Test clear_cache handles exceptions gracefully."""
        service = CacheService()
        service._data_cache_manager = MagicMock()
        # Make the cache manager raise an exception when accessed
        type(service._data_cache_manager.cache_storage).namespace = property(raise_error(RuntimeError, "Test error"))

        result = await service.clear_cache()
        assert result is False


class TestCacheServiceContextManager:
    """Tests for cache service context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes_service(self):
        """Test context manager initializes service on entry."""

        with (
            patch("scholar_flux_mcp.services.cache_service.CacheService.initialize") as mock_init,
            suppress(ScholarFluxImportError),
        ):
            async with get_cache_service() as service:
                mock_init.assert_called_once()
                assert service is not None

    @pytest.mark.asyncio
    async def test_context_manager_with_custom_params(self):
        """Verifies that the context manager accepts custom parameters."""
        with (
            patch.object(CacheService, "__init__", return_value=None) as mock_init,
            patch.object(CacheService, "initialize"),
            suppress(ScholarFluxImportError),
        ):
            async with get_cache_service(
                namespace="custom",
                ttl=7200,
                user_agent="Test/1.0",
                session_expire_after=3600,
                raise_on_error=True,
            ):
                # Verify __init__ was called with correct params
                mock_init.assert_called_once_with(
                    namespace="custom",
                    ttl=7200,
                    user_agent="Test/1.0",
                    session_expire_after=3600,
                    raise_on_error=True,
                )

"""Cache service for ScholarFlux MCP server.

Thin wrapper around DataCacheManager and CachedSessionManager that delegates configuration to scholar-flux while
providing MCP-specific defaults.

"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from requests import Session
from requests_cache import CachedSession

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from scholar_flux import CachedSessionManager, DataCacheManager, SessionManager
else:
    try:
        from scholar_flux import CachedSessionManager, DataCacheManager, SessionManager
        from scholar_flux.utils import config_settings
    except ImportError:
        CachedSessionManager = None
        DataCacheManager = None
        SessionManager = None

from scholar_flux_mcp.exceptions.import_exceptions import ScholarFluxImportError
from scholar_flux_mcp.models.schemas import ServiceHealth
from scholar_flux_mcp.utils import config_settings, masker

logger = logging.getLogger(__name__)


class CacheService:
    """Manages caching infrastructure for ScholarFlux MCP.

    Thin wrapper that delegates cache backend selection and configuration
    to scholar-flux's DataCacheManager and CachedSessionManager. Reads
    configuration from environment variables via scholar-flux's config system.

    Environment variables used (via scholar-flux):
        SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE: Data cache backend (redis, sql, mongodb, inmemory)
        SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND: Session cache backend (redis, sqlite, mongodb, memory)
        SCHOLAR_FLUX_REDIS_HOST: Redis host (if using redis backend)
        SCHOLAR_FLUX_REDIS_PORT: Redis port (if using redis backend)
        SCHOLAR_FLUX_MONGODB_HOST: MongoDB host (if using mongodb backend)
        SCHOLAR_FLUX_MONGODB_PORT: MongoDB port (if using mongodb backend)

    """

    def __init__(
        self,
        namespace: str = "mcp_cache",
        ttl: int | None = None,
        user_agent: str | None = None,
        session_expire_after: int | None = None,
        raise_on_error: bool = False,
    ) -> None:
        """Initialize cache service.

        Args:
            namespace: Cache namespace for isolation across projects/services.
            ttl: Default time-to-live for data cache entries in seconds.
                If None, uses backend defaults (redis/mongodb have TTL support).
            user_agent: User-Agent string for HTTP session manager.
                If None, reads from SCHOLAR_FLUX_DEFAULT_USER_AGENT or uses default.
            session_expire_after: Expiration time for session cache entries in seconds.
                If None, defaults to 86400 (24 hours) via CachedSessionManager.
            raise_on_error: If True, raises exceptions on initialization failure.
                If False, falls back to in-memory/basic session gracefully.

        """
        self._namespace = namespace
        self.ttl = ttl
        self._user_agent = user_agent or os.environ.get("SCHOLAR_FLUX_DEFAULT_USER_AGENT", "ScholarFlux-MCP/1.0")
        self._session_expire_after = session_expire_after
        self.raise_on_error = raise_on_error

        self._data_cache_manager: DataCacheManager | None = None
        self._session_manager: SessionManager | CachedSessionManager | None = None
        self._error: str | None = None

    @property
    def initialized(self) -> bool:
        """Indicates whether the cache service is initialized and available."""
        return self._data_cache_manager is not None and self._session_manager is not None

    async def _initialize_data_cache_manager(self) -> None:
        """Initialize data cache manager using scholar-flux defaults.

        Reads cache backend preferences from environment variables and creates
        appropriate data cache manager. Falls back gracefully to in-memory cache
        if initialization fails and raise_on_error is False.

        Raises:
            RuntimeError: If initialization fails and raise_on_error is True.

        """
        if DataCacheManager is None or config_settings is None:
            raise ScholarFluxImportError()

        try:
            # Get backend from environment (scholar-flux handles defaults)
            backend: Literal["redis", "sql", "sqlalchemy", "mongodb", "pymongo", "inmemory", "memory", "null"] = (
                config_settings.get("SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE") or "inmemory"
            )

            logger.info(f"Initializing data cache with backend: {backend}")

            # Create data cache manager - let scholar-flux handle connection details
            # Storage classes (RedisStorage, MongoDBStorage, etc.) read their own
            # connection parameters from environment variables
            kwargs: dict[str, Any] = {"namespace": self._namespace}
            if self.ttl is not None:
                kwargs["ttl"] = self.ttl

            self._data_cache_manager = DataCacheManager.with_storage(
                backend,
                verify_connection=True,
                **kwargs,
            )

            logger.info(f"Data cache manager initialized (backend: {backend}, namespace: {self._namespace})")

        except Exception as e:
            err = f"Data cache initialization failed: {e}"
            self._error = err

            if self.raise_on_error:
                logger.error(err)
                raise RuntimeError(err) from e

            # Graceful fallback to in-memory cache
            logger.warning(f"{err}. Falling back to in-memory data cache")
            self._data_cache_manager = DataCacheManager.with_storage(
                "inmemory",
                namespace=self._namespace,
            )

    async def _initialize_session_manager(self) -> None:
        """Initialize session manager using scholar-flux defaults.

        Reads session cache backend preferences from environment variables and creates
        appropriate session manager. Falls back gracefully to basic session
        if initialization fails and raise_on_error is False.

        Raises:
            RuntimeError: If initialization fails and raise_on_error is True.

        """
        if CachedSessionManager is None or SessionManager is None or config_settings is None:
            raise ScholarFluxImportError()

        try:
            # Create session manager - backend from environment
            # CachedSessionManager.default_session_backend() reads
            # SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND
            session_backend = config_settings.get("SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND") or "sqlite"
            logger.info(f"Initializing session manager with backend: {session_backend}")

            self._session_manager = CachedSessionManager(
                backend=None,  # Use default_session_backend()
                user_agent=self._user_agent,
                expire_after=self._session_expire_after,
                raise_on_error=self.raise_on_error,
            )
            logger.info(f"Session manager initialized (backend: {session_backend})")

        except Exception as e:
            err = f"Session manager initialization failed: {e}"
            self._error = err

            if self.raise_on_error:
                logger.error(err)
                raise RuntimeError(err) from e

            # Graceful fallback to basic session
            logger.warning(f"{err}. Falling back to basic session manager...")
            self._session_manager = SessionManager(user_agent=self._user_agent)

    async def initialize(self, reinitialize: bool | None = None) -> None:
        """Initialize cache managers using scholar-flux defaults.

        Reads cache backend preferences from environment variables and creates
        appropriate managers. Falls back gracefully to in-memory/basic session
        if initialization fails and raise_on_error is False.

        Args:
            reinitialize (bool | None):
                Optionally reinitializes the cache service when True. If `reinitialize` is None (the default),
                re-initialization will not occur after the first successful call to `CacheService.initialize()`.

        Raises:
            ImportError: If scholar-flux is not installed.
            RuntimeError: If initialization fails and raise_on_error is True.

        """
        if reinitialize is not True and self.initialized:
            return

        self._error = None

        if (
            DataCacheManager is None
            or CachedSessionManager is None
            or SessionManager is None
            or config_settings is None
        ):
            raise ScholarFluxImportError()

        # Resolve user_agent from config if not provided
        if not self._user_agent:
            self._user_agent = config_settings.get("SCHOLAR_FLUX_DEFAULT_USER_AGENT", "ScholarFlux-MCP/1.0")

        await self._initialize_data_cache_manager()
        await self._initialize_session_manager()

        logger.info("Cache service initialized successfully")

    @property
    def data_cache_manager(self) -> DataCacheManager:
        """Get the data cache manager instance.

        Returns:
            DataCacheManager: Initialized cache manager for processed response data.

        Raises:
            RuntimeError: If service not initialized via initialize().

        """
        if not self._data_cache_manager:
            raise RuntimeError("Cache service not initialized. Call initialize() first.")
        return self._data_cache_manager

    @property
    def session_manager(self) -> SessionManager | CachedSessionManager:
        """Get the session manager instance.

        Returns:
            SessionManager | CachedSessionManager: Session factory for creating HTTP sessions.

        Raises:
            RuntimeError: If service not initialized via initialize().

        """
        if not self._session_manager:
            raise RuntimeError("Cache service not initialized. Call initialize() first.")
        return self._session_manager

    @property
    def enabled(self) -> bool:
        """Indicates whether the assigned service's cached session manager is configured to build a cached session."""
        return self._session_manager is not None and isinstance(self.session_manager, CachedSessionManager)

    def create_session(self) -> Session | CachedSession:
        """Create a new HTTP session instance.

        Convenience method that calls the session manager's configure_session() method.
        Each call creates a new session instance (important for thread safety).

        Returns:
            Session | CachedSession: New session instance.

        Raises:
            RuntimeError: If service not initialized via initialize().

        """
        if not self._session_manager:
            raise RuntimeError("Cache service not initialized. Call initialize() first.")
        return self._session_manager()

    async def check_health(self) -> ServiceHealth:
        """Helper used to check the health status of the CacheService."""

        stats = await self.get_stats()
        cache_healthy = self.initialized and self._error is None

        return ServiceHealth(
            name="cache",
            status="healthy" if cache_healthy else ("unhealthy" if self._error else "disabled"),
            details=stats,
            error=self._error,
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get cache service statistics and configuration.

        Returns:
            dict: Cache configuration including backends, namespace, ttl, and status.

        """
        # Read current environment configuration
        session_env = "SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND"
        data_cache_env = "SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE"
        session_backend = (config_settings.get(session_env) or "sqlite") if config_settings else None
        response_cache_backend = (config_settings.get(data_cache_env) or "inmemory") if config_settings else None

        return {
            "initialized": self.initialized,
            "namespace": self._namespace,
            "ttl": self.ttl,
            "session_cache_backend": session_backend,
            "response_cache_storage": response_cache_backend,
            "user_agent": masker.mask_text(self._user_agent) if masker else "***",
            "raise_on_error": self.raise_on_error,
        }

    async def clear_cache(self, namespace: str | None = None) -> bool:
        """Clears cache entries from the response processing cache.

        Args:
            namespace: Optional namespace to clear. If None, clears configured namespace.
                Note: Implementation depends on backend support for clearing.

        Returns:
            bool: True if cache clear was attempted, False if not initialized.

        """
        try:
            if self._data_cache_manager is not None:
                target_namespace = namespace if namespace else self._data_cache_manager.cache_storage.namespace
                logger.info(f"Cache clear requested for namespace: {target_namespace}")
                if target_namespace:
                    with self._data_cache_manager.cache_storage.with_namespace(target_namespace):
                        self._data_cache_manager.cache_storage.delete_all()
                else:
                    self._data_cache_manager.cache_storage.delete_all()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


@asynccontextmanager
async def get_cache_service(
    namespace: str = "mcp_cache",
    ttl: int | None = None,
    user_agent: str | None = None,
    session_expire_after: int | None = None,
    raise_on_error: bool = False,
) -> AsyncGenerator[CacheService, None]:
    """Context manager for cache service lifecycle.

    Initializes cache service on entry and cleans up on exit.

    Args:
        namespace: Cache namespace for isolation.
        ttl: Time-to-live for cache entries in seconds.
        user_agent: User-Agent string for HTTP sessions. If None, resolved from config.
        session_expire_after: Session cache expiration in seconds.
        raise_on_error: Whether to raise exceptions on initialization failure.

    Yields:
        CacheService: Initialized cache service instance.

    Example:
        async with get_cache_service() as cache:
            session = cache.create_session()
            # ... use session

    """
    service = CacheService(
        namespace=namespace,
        ttl=ttl,
        user_agent=user_agent,
        session_expire_after=session_expire_after,
        raise_on_error=raise_on_error,
    )
    try:
        await service.initialize()
        yield service
    finally:
        # Cleanup if needed (currently no explicit cleanup required)
        pass

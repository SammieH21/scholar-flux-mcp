"""Defines exceptions raised due to unexpected errors from the tool output history cache."""


class HistoryCacheException(Exception):
    """Base exception for issues related to output history cache."""

    pass


class HistoryCacheInitializationException(HistoryCacheException):
    """Exception raised when attempting to use an engine while the HistoryService is not yet initialized."""

    pass


class HistoryCacheConnectionFailed(HistoryCacheException):
    """Exception raised when encountering history cache storage connection errors."""

    pass


class HistoryCacheRetrievalException(HistoryCacheException):
    """Exception raised when history cache storage retrieval fails."""

    pass


class HistoryCacheStorageException(HistoryCacheException):
    """Exception raised when updating the history cache storage fails."""

    pass


class HistoryCacheDeletionException(HistoryCacheException):
    """Exception raised when history storage cache deletion fails."""

    pass


class HistoryCacheVerificationException(HistoryCacheException):
    """Exception raised when history storage cache validation fails."""

    pass


class HistoryCacheParameterValidationException(HistoryCacheException):
    """Exception raised when invalid parameters are passed to the history cache."""

    pass


__all__ = [
    "HistoryCacheException",
    "HistoryCacheConnectionFailed",
    "HistoryCacheInitializationException",
    "HistoryCacheRetrievalException",
    "HistoryCacheStorageException",
    "HistoryCacheDeletionException",
    "HistoryCacheVerificationException",
    "HistoryCacheParameterValidationException",
]

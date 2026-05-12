"""Helper module containing metadata relevant to the initialization of scholar-flux-mcp.

This module is currently responsible for the retrieval of the current package version and the enumeration of
current dependencies.

"""

from importlib.metadata import PackageNotFoundError

try:
    from importlib import metadata as _md

    __version__ = _md.version("scholar_flux_mcp")

# If the package cannot be found, assume local development
except (PackageNotFoundError, ImportError):
    __version__ = "0.0.0+local"

from scholar_flux_mcp.package_metadata.dependencies import (
    ScholarFluxMCPDependencies,
    ServiceDependency,
)

# Note: missing dependencies require a server restart to register. This identifies and caches missing deps early:
ScholarFluxMCPDependencies.identify_missing_dependencies(verbose=True)

__all__ = ["__version__", "ScholarFluxMCPDependencies", "ServiceDependency"]

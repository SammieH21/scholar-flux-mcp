"""Defines the health check tool and its output formatter for ScholarFlux MCP server."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from scholar_flux_mcp.models import HealthStatus, ResponseFormat, ResponseFormatString, ServiceHealth
from scholar_flux_mcp.server.io.base import BaseFormatter


class HealthCheckToolInput(BaseModel):
    """Basic input model for the `scholar_flux_health_check` tool."""

    model_config = ConfigDict(extra="forbid")

    response_format: ResponseFormatString = Field(
        default="markdown",
        description="Output format: 'markdown' for readable summary, 'json' for structured data",
    )


class HealthCheckFormatter(BaseFormatter):
    """Formatter for health check responses."""

    @classmethod
    def format(
        cls,
        output: HealthStatus,
        response_format: ResponseFormat | str = ResponseFormat.MARKDOWN,
    ) -> str:
        """Formats the health check response either as a markdown summary or JSON response."""
        format = ResponseFormat(response_format)
        return (
            cls.format_health_check_json(output)
            if format == ResponseFormat.JSON
            else cls.format_health_check_markdown(output)
        )

    @classmethod
    def format_health_check_json(cls, health_status: HealthStatus) -> str:
        """Formats the health check response as a serialized JSON string."""
        return cls.format_json_string(health_status)

    @classmethod
    def format_health_check_markdown(cls, health_status: HealthStatus) -> str:
        """Formats the health check response as markdown."""
        service_health_lines = []
        overall_status = health_status.status or "unknown"
        service_health_lines.append(f"## Health Check Status: {overall_status}")
        service_health_lines.append("")

        for service_health in health_status.services.values():
            service_health_lines.append(cls.format_service_health_check(service_health))
            service_health_lines.append("")

        return "\n".join(service_health_lines)

    @classmethod
    def format_service_health_check(
        cls, service_health: ServiceHealth, service_name: str | None = None, indent: int = 2
    ) -> str:
        """Formats a single service's health check status as a markdown summary string."""
        health_status_lines = []
        status = service_health.status or "unknown"
        error = service_health.error
        name = service_name or service_health.name

        health_status_lines.append(f"### {name.title()}")
        health_status_lines.append(f"- **Status:** {status}")

        if error:
            health_status_lines.append(f"- **Error:** {error}")

        if service_health.details:
            health_status_lines.append("- **Details:**")
            for field, value in service_health.details.items():
                health_status_lines.append(f"{indent * ' '}- {field}: {value}")

        return "\n".join(health_status_lines)


__all__ = ["HealthCheckToolInput", "HealthCheckFormatter"]

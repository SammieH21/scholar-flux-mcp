"""Core model functionality serving as the base for the ScholarFlux MCP server.

This module defines core property decorators, the base JSON data model, and general annotations used throughout the
codebase to ensure data integrity.

"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    SerializationInfo,
    computed_field,
    model_serializer,
)
from pydantic.main import IncEx

from scholar_flux_mcp.utils.helpers import (
    as_tuple,
    format_iso_timestamp,
    parse_iso_timestamp,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# =============================================================================
# ENUMS
# =============================================================================

SelfT = TypeVar("SelfT", bound=object)


def computed_property(func: Callable[[SelfT], Any], **kwargs: Any) -> property:
    """Helper method for defining pydantic computed fields as properties in a type-safe manner compatible with mypy."""
    return computed_field(property(func), **kwargs)


def computed_cached_property(func: Callable[[SelfT], Any], **kwargs: Any) -> cached_property:
    """Helper method for defining pydantic computed fields as cached properties compatible with mypy."""
    return computed_field(cached_property(func), **kwargs)


def validate_iso_timestamp(value: str | datetime) -> datetime:
    """Validates that the input is a datetime object or string ISO timestamp, returning a datetime object if valid."""
    if isinstance(value, datetime):
        return value
    if parsed_value := parse_iso_timestamp(value):
        return parsed_value
    raise ValueError(f"The received object ({value}) could not be parsed as a valid timestamp object.")


def validate_iso_timestamp_string(value: str | datetime) -> str:
    """Validates that the input is a datetime object or string ISO timestamp, returning a formatted string if valid."""
    if isinstance(value, datetime):
        return format_iso_timestamp(value)
    if parse_iso_timestamp(value):
        return value
    raise ValueError(f"The received object ({value}) could not be parsed as a valid timestamp object.")


TimeStamp = Annotated[datetime, BeforeValidator(validate_iso_timestamp)]
TimeStampString = Annotated[str, BeforeValidator(validate_iso_timestamp_string)]

JSON_CORE_FIELDS_ONLY_SERIALIZATION_CONTEXT: ContextVar[bool] = ContextVar(
    "JSON_CORE_FIELDS_ONLY_SERIALIZATION_CONTEXT", default=False
)
JSON_HIDDEN_FIELD_SERIALIZATION_CONTEXT: ContextVar[bool] = ContextVar(
    "JSON_HIDDEN_FIELD_SERIALIZATION_CONTEXT", default=False
)


class JSONDataModel(BaseModel):
    """Basic Pydantic model allowing for the definition of hidden — yet serializable fields during serialization."""

    HIDDEN_FIELDS: ClassVar[set[str]] = set()
    SERIALIZE_CORE_FIELDS_ONLY: ClassVar[bool] = False

    @classmethod
    @contextmanager
    def serialization_context(
        cls, core_fields_only: bool | None = None, serialize_hidden_fields: bool | None = None
    ) -> Iterator[None]:
        """Helper for serializing core fields only when needed, irrespective of `hidden` status."""
        core_only_token: Token[bool] | None = None
        hidden_field_token: Token[bool] | None = None

        try:
            core_only_flag = bool(cls.SERIALIZE_CORE_FIELDS_ONLY if core_fields_only is None else core_fields_only)

            #  Default: hides hidden fields when `serialize_hidden_fields` is not set and `core_fields_only` is True
            hidden_field_flag = bool(core_only_flag if serialize_hidden_fields is None else serialize_hidden_fields)
            core_only_token = JSON_CORE_FIELDS_ONLY_SERIALIZATION_CONTEXT.set(core_only_flag)
            hidden_field_token = JSON_HIDDEN_FIELD_SERIALIZATION_CONTEXT.set(hidden_field_flag)

            yield

        finally:
            if isinstance(core_only_token, Token):
                JSON_CORE_FIELDS_ONLY_SERIALIZATION_CONTEXT.reset(core_only_token)
            if isinstance(hidden_field_token, Token):
                JSON_HIDDEN_FIELD_SERIALIZATION_CONTEXT.reset(hidden_field_token)

    @classmethod
    def _select_model_fields(
        cls, include: IncEx | None = None, exclude: IncEx | None = None
    ) -> tuple[IncEx | None, IncEx | None]:
        """Selects fields for serialization, returning a tuple for `include=` and `exclude=` for later serialization."""
        if JSON_CORE_FIELDS_ONLY_SERIALIZATION_CONTEXT.get():
            core_set = set(cls.model_fields)
            include_set = set(as_tuple(include))
            include = core_set if not include else core_set & include_set
        elif include is None and exclude is None:
            exclude = {} if JSON_HIDDEN_FIELD_SERIALIZATION_CONTEXT.get() else cls.HIDDEN_FIELDS

        return include, exclude

    def model_dump(
        self, *, include: IncEx | None = None, exclude: IncEx | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Uses the current serialization settings to determine whether to serialize core and/or hidden fields."""
        include, exclude = self._select_model_fields(include, exclude)
        return super().model_dump(include=include, exclude=exclude, **kwargs)

    def model_dump_json(self, *, include: IncEx | None = None, exclude: IncEx | None = None, **kwargs: Any) -> str:
        """JSON variant of `model_dump`, using the same logic when serializing the data model into a JSON string."""
        include, exclude = self._select_model_fields(include, exclude)
        return super().model_dump_json(include=include, exclude=exclude, **kwargs)

    @model_serializer(mode="wrap")
    def _context_aware_serializer(self, handler: Any, info: SerializationInfo) -> dict[str, Any]:
        """Serializes select hidden and/or core model fields depending on the current `cls.serialization_context()`.

        This method, by default, uses `_select_model_fields()` to select non-hidden fields while prioritizing explicit
        `include` and `exclude` keywords over default model field selection settings when provided via `model_dump()`.

        Args:
            handler (Any):
                The current serialization handler. Used by pydantic under the hood to handle serialization.
            info (SerializationInfo):
                Contains serialization information guiding the inclusion or exclusion of specific fields as well as
                other serialization settings.

        Returns:
            dict[str, Any]: A dictionary of serialized fields extracted from the current model.

        """
        result = handler(self)

        include_default, exclude_default = self._select_model_fields()
        include = info.include if info.include is not None else include_default
        exclude = info.exclude if info.exclude is not None else exclude_default

        return {
            k: v
            for k, v in result.items()
            if (include is None or k in include) and (exclude is None or k not in exclude)
        }


__all__ = [
    "cached_property",
    "computed_property",
    "computed_cached_property",
    "validate_iso_timestamp",
    "validate_iso_timestamp_string",
    "TimeStamp",
    "TimeStampString",
    "JSONDataModel",
]

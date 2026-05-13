"""This module defines common helpers for ease of later use without requiring additional dependencies.

Several of these originate from the scholar_flux.utils.helpers module but were sourced and/or revised for
Convenience given that the ScholarFlux dependency might not always be installed locally during testing.

Functions:
- try_none
- coerce_int
- coerce_str
- coerce_numeric
- coerce_bool
- generate_iso_timestamp
- parse_iso_timestamp
- format_iso_timestamp
- as_tuple
- truncate
- os_env_context

"""

import os
import re
from collections.abc import Generator, Mapping, MutableSequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TypeVar, overload

T = TypeVar("T", bound=object)


def coerce_int(value: object) -> int | None:
    """Attempts to convert a value to an integer, returning None if the conversion fails.

    Args:
        value (object): The value to attempt to convert into a int.

    Returns:
        Optional[int]: The value converted into an integer if possible, otherwise None

    """
    if isinstance(value, int) or value is None:
        return value

    try:
        return int(value) if isinstance(value, str) else None
    except (ValueError, TypeError):
        return None


def coerce_str(value: object, encoding: str | None = "utf-8") -> str | None:
    """Attempts to convert a value into a string, if possible, returning None if conversion fails.

    Args:
        value (object): The value to attempt to convert into a string.
        encoding (Optional[str]): An optional value used to decode byte strings. Not relevant for data of other types.

    Returns:
        Optional[str]: The value converted into a string if possible, otherwise None

    """
    if isinstance(value, str) or value is None:
        return value

    try:
        return value.decode(encoding or "utf-8") if isinstance(value, bytes) else str(value)
    except (ValueError, TypeError, UnicodeDecodeError):
        return None


def coerce_numeric(value: object) -> float | None:
    """Attempts to convert a value to a float, returning None if the conversion fails.

    Args:
        value (object): The value to attempt to convert into a decimal value.

    Returns:
        float | None: The value converted into a float if possible, otherwise None.

    Note:
        Conversion treats booleans as integers and converts them when observed. To avoid this, use conditional logic.

    """
    if isinstance(value, int | float) or value is None:
        return float(value) if isinstance(value, int) else value

    try:
        return float(value) if isinstance(value, str) else None
    except (ValueError, TypeError):
        return None


@overload
def try_none(value: None) -> None:
    """When `None` is received, `None` is returned as is."""
    ...


@overload
def try_none(value: T) -> None | T:
    """When `T` is received, T is converted into None object when possible."""
    ...


def try_none(
    value: object, none_indicators: tuple[object, ...] = ("none", "unspecified", "unknown", "n/a")
) -> object | None:
    """Converts empty strings, 'none', and empty data containers into None, returning the original value otherwise.

    Args:
        value (object): The value to convert into None when possible
        none_indicators (tuple[Any, ...]): Tuple of values that should be treated as None indicators.

    Returns:
        object | None: The original value if not converted, and None otherwise

    """
    formatted_value = value.strip().lower() if isinstance(value, str) else value
    none_indicators = as_tuple(none_indicators)
    return value if (formatted_value or isinstance(value, int)) and formatted_value not in none_indicators else None


def coerce_bool(
    value: object,
    true_values: tuple[str, ...] = ("T", "true", "yes", "1"),
    false_values: tuple[str, ...] = ("F", "false", "no", "0"),
) -> bool | None:
    """Attempts to convert a value to a boolean value, returning None if the conversion fails.

    Args:
        value (object): The value to attempt to convert into a boolean.
        true_values (tuple[str, ...]): Values to be mapped to True when matched by the input value.
        false_values (tuple[str, ...]): Values to be mapped to False when matched by the input value.

    Returns:
        Optional[bool]: The value converted into a boolean if possible, otherwise None.

    Examples:
        >>> from scholar_flux.utils.helpers import coerce_bool
        >>> coerce_bool("TRUE")
        True
        >>> coerce_bool(1)
        True
        >>> coerce_bool(True, true_values=())
        True
        >>> coerce_bool("maybe", true_values=("Maybe",))
        True
        >>> coerce_bool("NO")
        False
        >>> coerce_bool("0")
        False
        >>> coerce_bool("Unknown?")
        None
        >>> coerce_bool("0", false_values=None)
        None

    """
    if isinstance(value, bool):
        return value

    value_str = coerce_str(try_none(value))

    if value_str is None:
        return None

    if value_str.lower() in {str(value).lower() for value in as_tuple(true_values)}:
        return True
    if value_str.lower() in {str(value).lower() for value in as_tuple(false_values)}:
        return False

    return None


def parse_iso_timestamp(timestamp_str: str) -> datetime | None:
    """Attempts to convert an ISO 8601 timestamp string back to a datetime object.

    Args:
        timestamp_str (str): ISO 8601 formatted timestamp string

    Returns:
        datetime | None: datetime object if parsing succeeds, None otherwise

    """
    if not isinstance(timestamp_str, str):
        return None

    try:
        cleaned = timestamp_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return dt
    except (ValueError, AttributeError, TypeError, OSError):
        return None


def as_tuple(obj: object) -> tuple:
    """Converts objects into tuples when possible and nests objects within a tuple otherwise.

    Args:
        obj (object): The object to nest as a tuple

    Returns:
        tuple: The original object converted into a tuple

    """
    match obj:
        case tuple():
            return obj
        case list():
            return tuple(obj)
        case set():
            return tuple(obj)
        case None:
            return ()
        case _:
            return (obj,)


def generate_datetime() -> datetime:
    """Generates the current UTC datetime for consistent timezone usage across the scholar-flux-mcp codebase.

    Returns:
        datetime: The current datetime in UTC.

    """
    return datetime.now(timezone.utc)


def generate_iso_timestamp() -> str:
    """Generates and formats an ISO 8601 timestamp string in UTC with millisecond precision for round-trip conversion.

    Example usage:
        >>> from scholar_flux.utils import generate_iso_timestamp, parse_iso_timestamp, format_iso_timestamp
        >>> timestamp = generate_iso_timestamp()
        >>> parsed_timestamp = parse_iso_timestamp(timestamp)
        >>> assert parsed_timestamp is not None and format_iso_timestamp(parsed_timestamp) == timestamp

    Returns:
        str: ISO 8601 formatted timestamp (e.g., "2024-03-15T14:30:00.123Z")

    """
    return format_iso_timestamp(generate_datetime())


def format_iso_timestamp(timestamp: datetime) -> str:
    """Formats an iso timestamp string in UTC with millisecond precision.

    Args:
        timestamp (datetime): The datetime object to format.

    Returns:
        str: ISO 8601 formatted timestamp (e.g., "2024-03-15T14:30:00.123+00:00")

    """
    return timestamp.isoformat(timespec="milliseconds")


def truncate(
    value: object,
    max_length: int | None = 40,
    suffix: str = "...",
    show_count: bool = True,
) -> str:
    """Truncates various strings, mappings, and sequences for cleaner representations of objects in CLIs.

    When `max_length` is None, the object is converted into its string representation without truncation.

    Handles:
    - Strings: Truncate with suffix
    - Mappings (dict): Show preview of first N chars with count
    - Sequences (list, tuple): Show preview with count
    - Other objects: Use string representation

    Args:
        value (object): The value to truncate.
        max_length (int | None): Maximum character length before truncation. Truncation si not performed if None.
        suffix (str): String to append when truncated (default: "...").
        show_count (bool): Whether to show item count for collections.

    Returns:
        str: Truncated string representation.

    Examples:
        >>> truncate("A very long string that needs truncation", max_length=20)
        'A very long string...'

        >>> truncate("A very long string that actually doesn't need truncation", max_length=None)
        "A very long string that actually doesn't need truncation"

        >>> truncate({'key1': 'value1', 'key2': 'value2'}, max_length=30)
        "{'key1': 'value1', ...} (2 items)"

        >>> truncate([1, 2, 3, 4, 5], max_length=10)
        '[1, 2, ...] (5 items)'

        >>> truncate({'a': 1}, max_length=50, show_count=False)
        "{'a': 1}"

    """
    # Handle None explicitly
    if value is None:
        return "None"

    if max_length is None:
        return str(value)

    # Handle strings
    if isinstance(value, str):
        if len(value) <= max_length:
            return value
        return value[: max_length - len(suffix)] + suffix

    # Handle mappings (dict, etc.)
    if isinstance(value, Mapping):
        str_repr = str(value)
        if len(str_repr) <= max_length:
            return str_repr

        # Truncate and add count
        truncated = str_repr[: max_length - len(suffix) - 1] + suffix + str_repr[-1]
        if show_count:
            count_suffix = f" ({len(value)} items)" if len(value) != 1 else " (1 item)"
            return truncated + count_suffix
        return truncated

    # Handle sequences (list, tuple, but not strings)
    if isinstance(value, MutableSequence | tuple):
        str_repr = str(value)
        if len(str_repr) <= max_length:
            return str_repr

        # Truncate and add count
        truncated = str_repr[: max_length - len(suffix) - 1] + suffix + str_repr[-1]
        if show_count:
            count_suffix = f" ({len(value)} items)" if len(value) != 1 else " (1 item)"
            return truncated + count_suffix
        return truncated

    # Fallback: convert to string and truncate
    str_repr = str(value)
    if len(str_repr) <= max_length:
        return str_repr
    return str_repr[: max_length - len(suffix)] + suffix


@contextmanager
def os_env_context(env_var: str, value: str | None) -> Generator[None, None, None]:
    """Temporarily overwrites the value of the associated environment variable for the duration of the context.

    Args:
        env_var (str): The name of the OS environment variable to temporarily overwrite.
        value (str | None): A string value to assign. If `None`, the variable is temporarily removed instead.

    Note:
        After the context manager closes, the `env_var` will be assigned its original value if it pre-existed.
        Otherwise the temporary environment variable will be deleted.

    """
    if not isinstance(env_var, str):
        raise TypeError(
            f"The OS environment context manager expected a string environment variable, but received {type(env_var)}"
        )

    if value is not None and not isinstance(value, str):
        raise TypeError(
            "The OS environment context manager expected either a string value to assign the environment variable or "
            f"None to temporarily remove the environment variable, but received {type(value)}"
        )

    current_value = os.environ.get(env_var)

    try:
        if value is not None:
            os.environ[env_var] = value
        else:
            os.environ.pop(env_var, None)
        yield

    finally:
        if current_value is not None:
            os.environ[env_var] = current_value
        else:
            os.environ.pop(env_var, None)


def batch_replace(
    text: str, pattern: re.Pattern | str, replacements: dict[str, str], skip_unmatched: bool = False
) -> str:
    """Performs batch regex replacements on a text string simultaneously.

    Args:
        text (str): The string the patterns to batch replace at once.
        replacements: Dictionary mapping regex patterns to replacement strings

    Returns:
        str: The text containing the simultaneously replaced substrings.

    """
    p = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
    return p.sub(
        lambda match: replacements[m] if (m := match.group()) in replacements or not skip_unmatched else m, text
    )


__all__ = [
    "try_none",
    "coerce_int",
    "coerce_str",
    "coerce_numeric",
    "coerce_bool",
    "generate_datetime",
    "generate_iso_timestamp",
    "parse_iso_timestamp",
    "format_iso_timestamp",
    "as_tuple",
    "truncate",
    "os_env_context",
    "batch_replace",
]

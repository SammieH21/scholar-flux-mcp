"""Tests for the `helpers.py` module.

Tests originate from the Core ScholarFlux package and are adapted for utility.

"""

from datetime import datetime
from time import sleep

import pytest

from scholar_flux_mcp.utils.helpers import (
    as_tuple,
    batch_replace,
    coerce_int,
    coerce_numeric,
    coerce_str,
    generate_datetime,
    generate_iso_timestamp,
    parse_iso_timestamp,
    truncate,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("42.123", 42.123),
        ("atl", None),
        (1, 1.0),
        (False, 0.0),  # edge case
        (3.14, 3.14),
    ],
)
def test_coerce_numeric(value, expected):
    """Tests if coercing numeric strings into floats returns the converted value when possible and None otherwise.

    This function will return a float when the result is valid and None otherwise.

    """
    assert coerce_numeric(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("42", 42),
        ("abc", None),
        (None, None),
        (3.14, None),
    ],
)
def test_coerce_int(value, expected):
    """Tests if coercing integer strings into integers returns the converted value when possible and None otherwise.

    This function will return an integer when the result is valid and None otherwise.

    """
    assert coerce_int(value) == expected


############################## Tests for as_tuple ################################


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((1, 2, 3), (1, 2, 3)),  # tuple returns unchanged
        ([1, 2, 3], (1, 2, 3)),  # list converted to tuple
        ({1, 2, 3}, (1, 2, 3)),  # set converted to tuple (order may vary)
        (None, ()),  # None returns empty tuple
        (42, (42,)),  # scalar wrapped in tuple
        ("string", ("string",)),  # string wrapped in tuple
        ({"a": 1}, ({"a": 1},)),  # dict wrapped in tuple
    ],
)
def test_as_tuple(value, expected):
    """Validates that as_tuple correctly converts or wraps values into tuples."""
    result = as_tuple(value)
    if isinstance(value, set):
        # Sets don't have guaranteed order, so compare as sets
        assert set(result) == set(expected)
    else:
        assert result == expected


def test_as_tuple_preserves_tuple():
    """Verifies that an existing tuple is returned unchanged."""
    original = (1, 2, 3)
    assert as_tuple(original) is original


############################## Tests for coerce_str exception handling ################################


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("42", "42"),
        ("abc", "abc"),
        (None, None),
        (3.14, "3.14"),
        (["abc", "tev"], "['abc', 'tev']"),
        (b"abc", "abc"),
        (sum, "<built-in function sum>"),
    ],
)
def test_coerce_str(value, expected):
    """Tests if coercing types into strings returns the expected string when coercible and None otherwise."""
    assert coerce_str(value) == expected


def test_coerce_str_unicode_decode_error():
    """Validates that coerce_str returns None when bytes cannot be decoded with the specified encoding."""
    # Invalid UTF-8 byte sequence
    invalid_bytes = b"\xff\xfe"
    assert coerce_str(invalid_bytes, encoding="ascii") is None


def test_coerce_str_invalid_encoding():
    """Validates that coerce_str handles invalid encoding gracefully."""
    # Bytes that are valid UTF-8 but not valid ASCII
    utf8_bytes = b"\xc3\xa9"  # é in UTF-8
    assert coerce_str(utf8_bytes, encoding="ascii") is None


############################## Tests for parse_iso_timestamp ################################


@pytest.mark.parametrize(
    "value",
    [
        123,  # integer
        12.34,  # float
        None,  # NoneType
        ["2024-01-01"],  # list
        {"date": "2024-01-01"},  # dict
    ],
)
def test_parse_iso_timestamp_non_string(value):
    """Validates that parse_iso_timestamp returns None for non-string inputs."""
    assert parse_iso_timestamp(value) is None


def test_parse_iso_timestamp_valid():
    """Validates that parse_iso_timestamp correctly parses valid ISO timestamps."""
    result = parse_iso_timestamp("2024-12-18T10:30:00Z")
    assert result is not None
    assert result.year == 2024
    assert result.month == 12
    assert result.day == 18


def test_parse_iso_timestamp_invalid_format():
    """Validates that parse_iso_timestamp returns None for invalid date formats."""
    assert parse_iso_timestamp("not-a-date") is None
    assert parse_iso_timestamp("") is None


def test_generate_iso_timestamp():
    """Verifies that generate_iso_timestamp generates a datetime in the expected form."""
    current_time = generate_datetime()
    sleep(0.1)
    result = generate_iso_timestamp()
    parsed = parse_iso_timestamp(result)
    assert isinstance(current_time, datetime) and isinstance(parsed, datetime)
    assert parsed >= current_time


########################### Tests for Truncation ###############################
def test_truncate_none():
    """Test truncating None returns 'None'."""
    assert truncate(None) == "None"
    assert truncate(None, max_length=10) == "None"


def test_truncate_with_no_max_length():
    """Test truncation without max length returns stringified object."""
    alist = ["z"] * 100
    assert truncate(alist, max_length=None) == str(alist)


def test_truncate_short_string():
    """Test that short strings are not truncated."""
    short_str = "Hello"
    assert truncate(short_str, max_length=40) == "Hello"
    assert truncate(short_str, max_length=10) == "Hello"


def test_truncate_long_string():
    """Test that long strings are truncated with suffix."""
    long_str = "A very long string that needs truncation"
    result = truncate(long_str, max_length=21)
    assert len(result) == 21
    assert result == "A very long string..."
    assert result.endswith("...")


def test_truncate_string_custom_suffix():
    """Test truncating strings with custom suffix."""
    long_str = "A very long string"
    result = truncate(long_str, max_length=15, suffix=">>")
    assert len(result) == 15
    assert result.endswith(">>")


def test_truncate_empty_string():
    """Test truncating empty string."""
    assert truncate("", max_length=40) == ""


def test_truncate_short_dict():
    """Test that short dicts are not truncated."""
    short_dict = {"a": 1, "b": 2}
    result = truncate(short_dict, max_length=50)
    assert result == str(short_dict)


def test_truncate_long_dict_with_count():
    """Test that long dicts are truncated with count."""
    long_dict = {f"key{i}": f"value{i}" for i in range(10)}
    result = truncate(long_dict, max_length=30, show_count=True)

    assert result.endswith("} (10 items)")
    assert "..." in result
    assert result.startswith("{")


def test_truncate_long_dict_without_count():
    """Test that long dicts are truncated without count when show_count=False."""
    long_dict = {f"key{i}": f"value{i}" for i in range(10)}
    result = truncate(long_dict, max_length=30, show_count=False)

    assert " items)" not in result
    assert "..." in result
    assert result.endswith("}")


def test_truncate_single_item_dict():
    """Test that single-item dict shows '(1 item)' not '(1 items)'."""
    single_dict = {"key": "a very long value that will cause truncation to occur"}
    result = truncate(single_dict, max_length=30, show_count=True)

    assert "(1 item)" in result


def test_truncate_empty_dict():
    """Test truncating empty dict."""
    assert truncate({}, max_length=40) == "{}"


def test_truncate_short_list():
    """Test that short lists are not truncated."""
    short_list = [1, 2, 3]
    result = truncate(short_list, max_length=50)
    assert result == str(short_list)


def test_truncate_long_list_with_count():
    """Test that long lists are truncated with count."""
    long_list = [{"id": f"item_{i}", "value": i} for i in range(25)]
    result = truncate(long_list, max_length=40, show_count=True)

    assert result.endswith("] (25 items)")
    assert "..." in result
    assert result.startswith("[")


def test_truncate_long_list_without_count():
    """Test that long lists are truncated without count when show_count=False."""
    long_list = list(range(100))
    result = truncate(long_list, max_length=30, show_count=False)

    assert " items)" not in result
    assert "..." in result
    assert result.endswith("]")


def test_truncate_single_item_list():
    """Test that single-item list shows '(1 item)' not '(1 items)'."""
    single_list = ["a very long string that will definitely cause truncation"]
    result = truncate(single_list, max_length=30, show_count=True)

    assert "(1 item)" in result


def test_truncate_empty_list():
    """Test truncating empty list."""
    assert truncate([], max_length=40) == "[]"


def test_truncate_tuple_with_count():
    """Test that tuples are truncated with count."""
    long_tuple = tuple(range(50))
    result = truncate(long_tuple, max_length=30, show_count=True)

    assert result.endswith(") (50 items)")
    assert "..." in result
    assert result.startswith("(")


def test_truncate_nested_structure():
    """Test truncating nested data structures."""
    nested = {"@xmlns:opensearch": "http://a9.com/-/spec/opensearch/1.1/", "data": [{"id": i} for i in range(10)]}
    result = truncate(nested, max_length=40, show_count=True)

    assert "..." in result
    assert result.startswith("{")
    assert result.endswith("} (2 items)")


def test_truncate_custom_object_short_representation():
    """Test truncating custom objects falls back to string representation."""

    class CustomObj:
        def __str__(self):
            return "A CustomObject"

    obj = CustomObj()
    result = truncate(obj, max_length=25)
    assert result == "A CustomObject"


def test_truncate_custom_object_long_representation():
    """Test truncating custom objects falls back to string representation."""

    class CustomObj:
        def __str__(self):
            return "CustomObject with a very long string representation"

    obj = CustomObj()
    result = truncate(obj, max_length=25)

    assert len(result) == 25
    assert "..." in result


def test_truncate_preserves_brackets():
    """Test that closing brackets are preserved in truncated collections."""
    long_dict = {f"key{i}": "value" for i in range(20)}
    result = truncate(long_dict, max_length=30, show_count=False)
    assert result.endswith("}")

    long_list = list(range(100))
    result = truncate(long_list, max_length=30, show_count=False)
    assert result.endswith("]")


def test_truncate_zero_max_length():
    """Test edge case with very small max_length."""
    result = truncate("hello", max_length=5)
    # Should handle gracefully, likely return just suffix or minimal string
    assert len(result) <= 5


@pytest.mark.parametrize("max_length", [10, 20, 30, 40, 50])
def test_truncate_respects_max_length_constraint(max_length):
    """Test that truncated output respects max_length (excluding count suffix)."""
    long_str = "x" * 100
    result = truncate(long_str, max_length=max_length, show_count=False)
    assert len(result) == max_length


def test_truncate_with_processed_response_metadata():
    """Test realistic use case with API response metadata."""
    metadata = {
        "@xmlns:opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "@xmlns:atom": "http://www.w3.org/2005/Atom",
        "totalResults": 1000,
    }
    result = truncate(metadata, max_length=40, show_count=False)

    assert result.startswith("{")
    assert result.endswith("}")
    assert "..." in result
    assert len(result) == 40


############################## Tests for `batch_replace` ################################


def test_batch_replace_handles_basic_replacements():
    """Verifies that batch_replace can simultaneously handle replacements with several components."""
    txt = "1 fish, 214, ab fish 5?"
    replacements = {"1": "2", "5": "4"}
    expected = "2 fish, 214, ab fish 4?"
    replaced = batch_replace(txt, r"(\d+)", replacements, skip_unmatched=True)
    assert expected == replaced


def test_batch_replace_overlapping_numbers():
    """Verifies that batch_replace can handle replacements with overlapping components."""
    txt = "1 22 333 44 55"
    replacements = {"1": "2", "2": "3", "3": "4", "4": "5", "5": "6", "6": "7", "7": "8"}
    expected = "2 33 444 55 66"
    replaced = batch_replace(txt, r"\d", replacements, skip_unmatched=True)
    assert replaced == expected

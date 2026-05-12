import importlib
from unittest.mock import patch

import pytest

from scholar_flux_mcp.exceptions.import_exceptions import RapidFuzzImportError
from scholar_flux_mcp.utils.fuzzy_text_similarity import (
    BaseTextSimilarity,
    FuzzyRatioSimilarity,
    PartialRatioSimilarity,
)


@pytest.mark.parametrize(
    "sub_text, text, expected",
    [
        ("test", "testing", 1.0),  # partial match
        ("a test string", "this is a test string", 1.0),  # substring present
        ("foo", "bar", 0),  # no match
    ],
)
def test_partial_ratio_calculate(sub_text, text, expected):
    """Verifies that matched substrings are successfully identified when they exist in the text."""
    result = PartialRatioSimilarity.calculate(sub_text, text)
    assert result.score == expected


def test_fuzzy_ratio_same_strings():
    """Verifies that identifical strings result in a similarity score of 100."""
    result = FuzzyRatioSimilarity.calculate("identical", "identical")
    result = FuzzyRatioSimilarity.calculate(":)", ":)")
    assert result.score == 1.0


def test_fuzzy_ratio_different_strings():
    """Verifies that vastly differing strings result in a low fuzzy similarity score."""
    result = FuzzyRatioSimilarity.calculate("hello", "world")
    assert result.score < 20


def test_exceeds_threshold_when_none():
    """Verifies that `exceeds_threshold` is always true when a threshold is not provided."""
    result = PartialRatioSimilarity.calculate("a", "b", threshold=None)
    assert result.exceeds_threshold is True

    result2 = FuzzyRatioSimilarity.calculate("a", "b", threshold=None)
    assert result2.exceeds_threshold is True


def test_exceeds_threshold_below():
    """Verifies that differing strings can be identified as such based on the selected threshold."""
    result = PartialRatioSimilarity.calculate("a", "b", threshold=0.10)
    assert result.exceeds_threshold is False

    result2 = FuzzyRatioSimilarity.calculate("cde", "fgh", threshold=0.10)
    assert result2.exceeds_threshold is False


def test_base_class_abstract_calculate():
    """Verifies that the `calculate()` constructor raises a NotImplementedError when the base class is not overidden."""
    with pytest.raises(NotImplementedError):
        _ = BaseTextSimilarity.calculate()


def test_rapidfuzz_missing():
    """Verifies the behavior of the fuzzy_text_similarity module when `rapidfuzz` is missing."""
    import scholar_flux_mcp.utils.fuzzy_text_similarity

    try:
        with patch.dict("sys.modules", {"rapidfuzz": None}):
            importlib.reload(scholar_flux_mcp.utils.fuzzy_text_similarity)

            from scholar_flux_mcp.utils.fuzzy_text_similarity import (
                BaseTextSimilarity,
                FuzzyRatioSimilarity,
                PartialRatioSimilarity,
                fuzz,
            )

            assert fuzz is None

            with pytest.raises(RapidFuzzImportError):
                assert BaseTextSimilarity.validate_dependency()

            with pytest.raises(RapidFuzzImportError):
                assert FuzzyRatioSimilarity.calculate("a", "b")

            with pytest.raises(RapidFuzzImportError):
                assert PartialRatioSimilarity.calculate("a", "b")

    finally:
        importlib.reload(scholar_flux_mcp.utils.fuzzy_text_similarity)


def test_base_class_abstract_initialization():
    """Verifies that initialization of the `BaseTextSimilarity` class raises a `TypeError` when not overidden."""

    err = "Can't instantiate abstract class BaseTextSimilarity with.*"
    with pytest.raises(TypeError, match=err):
        _ = BaseTextSimilarity(score=10, threshold=0.30)  # type: ignore

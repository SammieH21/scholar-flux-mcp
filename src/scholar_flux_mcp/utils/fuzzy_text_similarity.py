"""Defines data structures for text similarity output, wrapping results for utility and observability."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import Self

if TYPE_CHECKING:
    from rapidfuzz import fuzz
else:
    try:
        from rapidfuzz import fuzz
    except ImportError:
        fuzz = None

from scholar_flux_mcp.exceptions.import_exceptions import RapidFuzzImportError

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class BaseTextSimilarity(ABC):
    """Abstract base class for text similarity implementations."""

    score: float = Field(ge=0.0, le=1.0, kw_only=True)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, kw_only=True)

    @property
    def exceeds_threshold(self) -> bool:
        """Indicates whether the current similarity comparison exceeds the threshold."""
        # If a threshold is not provided, assume it has been met (return True)
        return True if self.threshold is None else self.score > self.threshold

    @classmethod
    @abstractmethod
    def calculate(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Calculates the similarity score between inputs."""
        raise NotImplementedError()

    @classmethod
    def validate_dependency(cls) -> None:
        """Verifies that the rapidfuzz dependency is available for use.

        Can be overridden to support other methods.

        """
        if fuzz is None:
            raise RapidFuzzImportError()


@dataclass
class PartialRatioSimilarity(BaseTextSimilarity):
    """Dataclass containing the result of the partial ratio similarity test."""

    s1: str = Field(alias="sub_text")
    s2: str = Field(alias="text")

    @classmethod
    def calculate(
        cls,
        sub_text: str,
        text: str,
        threshold: float | None = None,
    ) -> Self:
        """Calculates the partial fuzzy similarity between a substring and a text."""
        cls.validate_dependency()
        # scaling partial ratios by 100 to account for the range of expected values
        score = 100 if sub_text in text else fuzz.partial_ratio(sub_text, text)
        return cls(sub_text=sub_text, text=text, score=score / 100.0, threshold=threshold)


@dataclass
class FuzzyRatioSimilarity(BaseTextSimilarity):
    """Dataclass containing the result of the fuzzy ratio similarity test."""

    s1: str
    s2: str

    @classmethod
    def calculate(
        cls,
        s1: str,
        s2: str,
        threshold: float | None = None,
    ) -> Self:
        """Calculates the fuzzy similarity between two strings."""
        cls.validate_dependency()
        score = 100 if s1 == s2 else fuzz.ratio(s1, s2)
        return cls(s1=s1, s2=s2, score=score / 100.0, threshold=threshold)


__all__ = ["PartialRatioSimilarity", "FuzzyRatioSimilarity"]

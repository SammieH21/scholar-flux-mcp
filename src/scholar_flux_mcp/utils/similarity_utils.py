"""A lightweight set of mathematical helpers used later in the calculation of cosine similarity between embeddings."""

import math
from collections.abc import Iterator, Sequence


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Basic dot product used to check for similarity between lists of floats with equal size."""
    return math.fsum(a_i * b_i for a_i, b_i in zip(a, b, strict=True))


def norm(v: Sequence[float]) -> float:
    """Basic vector norm used to later calculate the cosine similarity between lists of floats with equal size."""
    return math.sqrt(math.fsum(math.pow(v_i, 2) for v_i in v))


def cosine_similarity(a: Sequence[float], b: Sequence[float], normalize: bool = True) -> float:
    """Calculates the cosine similarity between two lists/tuples of floats with equal size.

    Args:
        a (Sequence[float]): Vector `a` for computing the cosine similarity between two vectors.
        b (Sequence[float]): Vector `b` for computing the cosine similarity between two vectors.
        normalize (bool):
            Whether to compute and divide by the norm between two vectors. (True by default). Set this to False when
            vectors are pre-normalized

    Returns:
        float: The cosine similarity between vectors `a` and `b`

    """
    if isinstance(a, Iterator):
        a = tuple(a)

    if isinstance(b, Iterator):
        b = tuple(b)

    try:
        if not isinstance(a, Sequence) or not isinstance(b, Sequence):
            raise TypeError(
                f"Expected two lists of floats to compute the cosine similarity but received {type(a)} and {type(b)}"
            )

        if not (a_size := len(a)) == (b_size := len(b)):
            raise ValueError(f"Expected two lists of floats with equal size, but received sizes {a_size} and {b_size}")

        magnitude = norm(a) * norm(b) if normalize else 1.0
        return dot(a, b) / magnitude if magnitude else 0
    except (ValueError, TypeError, ZeroDivisionError) as e:
        raise RuntimeError(
            f"Encountered an error during the computation of cosine similarity between two arrays: {e}"
        ) from e


__all__ = ["dot", "norm", "cosine_similarity"]

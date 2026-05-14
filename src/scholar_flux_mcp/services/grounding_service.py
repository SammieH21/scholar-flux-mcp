"""Grounding service for ensuring AI outputs reference valid source records.

Provides validation and mapping between AI-generated evidence references and the actual SearchRecord objects retrieved
from academic databases.

"""

from __future__ import annotations

import logging

from scholar_flux_mcp.exceptions import EvidenceGroundingException, EvidenceGroundingParameterException
from scholar_flux_mcp.models import (
    AgentEvidenceItem,
    EvidenceGroundingInput,
    EvidenceGroundingOutput,
    EvidenceGroundingStats,
    GroundedEvidenceItem,
    RejectedEvidenceItem,
    SearchRecord,
)
from scholar_flux_mcp.utils.fuzzy_text_similarity import PartialRatioSimilarity

logger = logging.getLogger(__name__)


class GroundingService:
    """Service for grounding AI synthesis outputs to source records.

    Ensures that evidence citations reference only valid records from the original search results, preventing
    hallucinated references.

    """

    DEFAULT_RAISE_ON_ERROR: bool = False
    DEFAULT_CITATION_SIMILARITY_THRESHOLD: float = 0.6

    @classmethod
    def ground_evidence(
        cls, input: EvidenceGroundingInput, similarity_threshold: float | None = None
    ) -> EvidenceGroundingOutput:
        """Map AI-generated evidence references to actual source records.

        Args:
            input (EvidenceGroundingInput):
                Evidence grounding input containing the synthesized evidence items and source records.
            similarity_threshold (float | None):
                The cutoff threshold used to determine whether the current citation originates from the referenced
                record. This value should fall between 0 and 1 and is later scaled into a percentage for compatibility
                with the rapidfuzz scoring implementation.

        Returns:
            EvidenceGroundingOutput: Grounding output with validated evidence items and grounding stats.

        """
        grounded: list[GroundedEvidenceItem] = []
        rejected: list[RejectedEvidenceItem] = []
        seen_indices: set[int] = set()
        stats = EvidenceGroundingStats()

        if not isinstance(input, EvidenceGroundingInput):
            raise EvidenceGroundingParameterException(
                f"The GroundingService expected `EvidenceGroundingInput`, but instead received type {type(input)}"
            )

        threshold = (
            similarity_threshold if similarity_threshold is not None else cls.DEFAULT_CITATION_SIMILARITY_THRESHOLD
        )

        if threshold is not None and not isinstance(threshold, int | float):
            raise EvidenceGroundingParameterException(
                f"The GroundingService expected a valid nonnegative decimal threshold, but instead received type {type(threshold)}"
            )

        if not input.source_records:
            logger.warning("No source records provided for grounding")
            return EvidenceGroundingOutput(grounded_evidence_items=grounded, grounding_stats=stats)

        try:
            for current_item in input.evidence_items:
                idx = current_item.record_index

                # Validate index bounds
                max_idx = len(input.source_records) - 1

                try:
                    cls._validate_index(idx, max_idx, raise_on_error=True)
                except EvidenceGroundingException as e:
                    rejected_item = RejectedEvidenceItem(**current_item.model_dump(), error=str(e))
                    rejected.append(rejected_item)
                    stats.references_rejected += 1
                    continue

                # Skip duplicates (keep first occurrence)
                if idx in seen_indices:
                    logger.debug(f"Skipping duplicate reference to record {idx}")

                    continue

                current_record = input.source_records[idx]

                referenced_text_similarity = (
                    cls.calculate_referenced_text_similarity(
                        current_item, current_record, similarity_threshold=threshold
                    )
                    if current_item.referenced_text
                    else None
                )

                if (
                    referenced_text_similarity is not None
                    and referenced_text_similarity.threshold
                    and not referenced_text_similarity.exceeds_threshold
                ):
                    rejection_reason = (
                        f"The calculated citation similarity score ({referenced_text_similarity.score:.2f}) does not "
                        f"exceed the threshold required for acceptance ({referenced_text_similarity.threshold:.1%})"
                    )
                    rejected_item = RejectedEvidenceItem(
                        **current_item.model_dump(),
                        record=current_record,
                        referenced_text_similarity=referenced_text_similarity.score,
                        error=rejection_reason,
                    )
                    rejected.append(rejected_item)
                    stats.references_rejected += 1
                    continue

                seen_indices.add(idx)

                similarity_score = referenced_text_similarity.score if referenced_text_similarity else None
                # Retrieve the source record and append it to the grounded evidence item
                grounded.append(
                    GroundedEvidenceItem(
                        record=current_record,
                        record_index=current_item.record_index,
                        finding=current_item.finding,
                        referenced_text=current_item.referenced_text,
                        referenced_text_similarity=similarity_score,
                        relevance_score=current_item.relevance_score,
                    )
                )
                stats.references_grounded += 1

            logger.info(f"Grounding complete: {stats.references_grounded} valid, {stats.references_rejected} rejected")
        except Exception as e:
            err = f"An unexpected error occurred during evidence grounding. {e}"
            if cls.DEFAULT_RAISE_ON_ERROR:
                logger.error(err, exc_info=True)
                raise EvidenceGroundingException(err) from e
            logger.warning(f"{err}: Skipping evidence grounding...", exc_info=True)
            grounded = []
            rejected = []
            err_type = e.__class__.__name__
            stats = EvidenceGroundingStats(error=f"{err_type}: {err}")

        return EvidenceGroundingOutput(
            grounded_evidence_items=grounded, rejected_evidence_items=rejected, grounding_stats=stats
        )

    @classmethod
    def calculate_referenced_text_similarity(
        cls, evidence_item: AgentEvidenceItem, record: SearchRecord, similarity_threshold: float | None = None
    ) -> PartialRatioSimilarity | None:
        """Validates whether an existing citation contains the expected quote from the current record."""
        if not evidence_item.referenced_text:
            return None

        if not isinstance(evidence_item, AgentEvidenceItem) or not isinstance(record, SearchRecord):
            return None

        record_string = record.to_string()

        similarity_result = PartialRatioSimilarity.calculate(
            sub_text=evidence_item.referenced_text, text=record_string, threshold=similarity_threshold
        )
        return similarity_result

    @classmethod
    def _validate_index(cls, idx: int, max_idx: int, raise_on_error: bool | None = None) -> bool:
        """Checks whether a record index referenced by a model is within valid bounds.

        Args:
            idx (int): The record index to validate.
            max_idx (int): Maximum valid index (len(records) - 1).
            raise_on_error (bool | None): Whether an error should be raised when encountering an invalid record index.

        Returns:
            bool: True if the indexed reference is valid and False otherwise.

        """
        raise_on_error = raise_on_error if raise_on_error is not None else cls.DEFAULT_RAISE_ON_ERROR
        if not isinstance(idx, int):
            err = (
                "The grounding service expected a valid, non-negative index but instead received type "
                f"'{type(idx).__name__}'"
            )
            if raise_on_error:
                raise EvidenceGroundingException(err)
            logger.warning(err)
            return False

        if idx < 0 or idx > max_idx:
            err = f"Index {idx} out of bounds. Expected an index between [0—{max_idx}]"
            if raise_on_error:
                raise EvidenceGroundingException(err)
            logger.warning(err)
            return False
        return True


__all__ = ["GroundingService"]

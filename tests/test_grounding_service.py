import pytest

from scholar_flux_mcp.exceptions import EvidenceGroundingException, EvidenceGroundingParameterException
from scholar_flux_mcp.models.schemas import (
    AgentEvidenceItem,
    EvidenceGroundingInput,
    IndexedSearchRecordList,
)
from scholar_flux_mcp.services.grounding_service import GroundingService
from tests.testing_utilities import raise_error


@pytest.fixture
def indexed_records(mock_ai_synthesis_output) -> IndexedSearchRecordList:
    """A list of indexed records corresponding to the mocked synthesis output."""
    return mock_ai_synthesis_output.indexed_records


@pytest.fixture
def evidence_items(mock_ai_synthesis_output) -> list[AgentEvidenceItem]:
    """A list of synthesized evidence items corresponding to the synthesis output."""
    return mock_ai_synthesis_output.agent_output.evidence_summaries


@pytest.fixture
def evidence_input_with_ungrounded_citations(indexed_records, evidence_items) -> EvidenceGroundingInput:
    """Input to the `GroundingService` for verifying its core functionality with ungrounded citations."""
    return EvidenceGroundingInput(source_records=indexed_records[:4], evidence_items=evidence_items)


@pytest.fixture
def evidence_input_with_grounded_citations(indexed_records, evidence_items) -> EvidenceGroundingInput:
    """Input to the `GroundingService` for verifying its core functionality grounded citations."""
    return EvidenceGroundingInput(source_records=indexed_records, evidence_items=evidence_items)


@pytest.fixture
def evidence_input_with_referenced_text(evidence_input_with_grounded_citations) -> EvidenceGroundingInput:
    """Input to the `GroundingService` for verifying its core functionality grounded citations and referenced text."""
    source_records_with_references: list[AgentEvidenceItem] = []
    for evidence_item in evidence_input_with_grounded_citations.evidence_items:
        evidence_record = next(
            (
                record
                for record in evidence_input_with_grounded_citations.source_records
                if record.index == evidence_item.record_index
            ),
            None,
        )
        record_string = evidence_record.to_string().split(". ")[0] if evidence_record else None
        source_records_with_references.append(evidence_item.model_copy(update={"referenced_text": record_string}))

    return evidence_input_with_grounded_citations.model_copy(update={"evidence_items": source_records_with_references})


@pytest.fixture
def evidence_input_with_faulty_referenced_texts(evidence_input_with_referenced_text) -> EvidenceGroundingInput:
    """Input to the `GroundingService` for verifying the rejection of evidence items with faulty text references."""
    evidence_items = evidence_input_with_referenced_text.evidence_items

    evidence_items[:2] = [
        item.model_copy(update={"referenced_text": "Bad text reference here..."}) for item in evidence_items[:2]
    ]

    return evidence_input_with_referenced_text.model_copy(update={"evidence_items": evidence_items})


def test_grounding_service_identifies_rejected(evidence_input_with_ungrounded_citations):
    """Verifies that the grounding service appropriately identifies invalid values."""
    evidence_output = GroundingService.ground_evidence(evidence_input_with_ungrounded_citations)
    stats = evidence_output.grounding_stats
    total_source_records = len(evidence_input_with_ungrounded_citations.source_records)

    assert stats.references_rejected >= 1
    # Each evidence item should correspond to only one record
    assert stats.references_rejected + stats.references_grounded == total_source_records


def test_grounding_service_identifies_grounded(evidence_items, evidence_input_with_grounded_citations):
    """Verifies that the grounding service appropriately works and identifies valid values."""
    evidence_output = GroundingService.ground_evidence(evidence_input_with_grounded_citations)
    stats = evidence_output.grounding_stats
    total_evidence_items = len(evidence_items)

    assert stats.references_rejected == 0
    assert stats.references_grounded <= total_evidence_items


def test_grounding_service_operates_with_text_references(evidence_input_with_referenced_text):
    """Verifies that the grounding service appropriately works and identifies valid values."""
    assert any(
        evidence_item
        for evidence_item in evidence_input_with_referenced_text.evidence_items
        if evidence_item.referenced_text
    )
    evidence_output = GroundingService.ground_evidence(evidence_input_with_referenced_text)
    assert not evidence_output.rejected_evidence_items

    stats = evidence_output.grounding_stats
    total_evidence_items = len(evidence_input_with_referenced_text.evidence_items)

    assert stats.references_rejected == 0
    assert stats.references_grounded <= total_evidence_items


def test_grounding_service_operates_with_faulty_text_references(evidence_input_with_faulty_referenced_texts):
    """Verifies that the grounding service correctly identifies evidence items with faulty referenced text."""
    assert any(
        evidence_item
        for evidence_item in evidence_input_with_faulty_referenced_texts.evidence_items
        if evidence_item.referenced_text
    )
    evidence_output = GroundingService.ground_evidence(evidence_input_with_faulty_referenced_texts)
    assert evidence_output.rejected_evidence_items

    stats = evidence_output.grounding_stats
    total_evidence_items = len(evidence_input_with_faulty_referenced_texts.evidence_items)

    faulty_reference_total = len(
        [
            item
            for item in evidence_input_with_faulty_referenced_texts.evidence_items
            if item.referenced_text == "Bad text reference here..."
        ]
    )

    assert faulty_reference_total

    # Only rejections should be those that are due to faulty text
    assert stats.references_rejected == faulty_reference_total

    # The rest should be valid
    assert stats.references_grounded
    assert stats.references_grounded + stats.references_rejected <= total_evidence_items


def test_grounding_service_flags_incorrect_value_types(evidence_input_with_referenced_text, evidence_items):
    """Verifies that the grounding service flags and raises an error for invalid input types."""
    with pytest.raises(EvidenceGroundingParameterException):
        _ = GroundingService.ground_evidence([1, 2, 3])  # type: ignore

    with pytest.raises(EvidenceGroundingParameterException):
        _ = GroundingService.ground_evidence(evidence_items)  # type: ignore

    with pytest.raises(EvidenceGroundingParameterException):
        _ = GroundingService.ground_evidence(
            evidence_input_with_referenced_text,
            similarity_threshold="invalid value",  # type: ignore
        )


def test_grounding_service_fallsback_to_rejection_on_error(
    evidence_items, evidence_input_with_grounded_citations, caplog, monkeypatch
):
    """Verifies that the grounding service falls back to record rejection when validation fails."""
    err = "Directly raised error"

    monkeypatch.setattr(GroundingService, "_validate_index", raise_error(RuntimeError, err))
    evidence_output = GroundingService.ground_evidence(evidence_input_with_grounded_citations)

    stats = evidence_output.grounding_stats
    assert stats.references_grounded == 0
    assert stats.references_rejected <= len(evidence_items)
    assert err in caplog.text


def test_grounding_service_raises_on_validation_error_when_enabled(evidence_input_with_grounded_citations, monkeypatch):
    """Verifies that the grounding service raises an EvidenceGroundingException with `DEFAULT_RAISE_ON_ERROR=True`."""
    err = "Directly raised error"

    monkeypatch.setattr(GroundingService, "_validate_index", raise_error(RuntimeError, err))
    monkeypatch.setattr(GroundingService, "DEFAULT_RAISE_ON_ERROR", True)
    message = rf"An unexpected error occurred during evidence grounding\. {err}.*"

    with pytest.raises(EvidenceGroundingException, match=message):
        _ = GroundingService.ground_evidence(evidence_input_with_grounded_citations)


def test_grounding_service_record_validation_triggers_warnings_without_source_records(caplog):
    """Verifies that the grounding service returns an `EvidenceGroundingOutput when no source records are provided."""
    empty_input = EvidenceGroundingInput(evidence_items=[], source_records=[])
    evidence_output = GroundingService.ground_evidence(empty_input)
    assert not evidence_output.grounded_evidence_items
    assert evidence_output.grounding_stats.references_rejected == 0
    assert evidence_output.grounding_stats.references_grounded == 0

    assert "No source records provided for grounding" in caplog.text


def test_validate_index_behavior(caplog):
    """Verifies that validate index works as a simple helper for checking whether an index is within bounds."""
    assert GroundingService._validate_index(1, 2)  # valid, between [0, 2]
    assert not caplog.text

    assert not GroundingService._validate_index(2, 1)
    assert "Index 2 out of bounds. Expected an index between [0—1]" in caplog.text

    assert not GroundingService._validate_index(-1, 99999)
    assert "Index -1 out of bounds. Expected an index between [0—99999]" in caplog.text

    assert not GroundingService._validate_index("five", 27)  # type: ignore
    assert "The grounding service expected a valid, non-negative index but instead received type 'str'" in caplog.text

"""Tests for SearchRecord preprocessing and transformation for output formatting, embedding searches, and synthesis."""

import pytest
from pydantic import ValidationError

from scholar_flux_mcp.models import (
    ResearchCategory,
    SearchRecord,
)
from scholar_flux_mcp.utils import SearchRecordPreprocessingUtils


def test_record_validation_identifies_record(mock_search_record_list):
    """Verifies that the `SearchRecordPreprocessingUtils` class correctly validates SearchRecords."""
    validated_records = SearchRecordPreprocessingUtils.validate_search_records(mock_search_record_list)
    assert validated_records == mock_search_record_list
    mock_search_record_fields_list = [record.model_dump() for record in mock_search_record_list]

    # should auto-convert valid dictionaries of record fields into a `SearchRecord`
    validated_records2 = SearchRecordPreprocessingUtils.validate_search_records(mock_search_record_fields_list)
    assert mock_search_record_list == validated_records2


def test_record_validation_converts_single_dict_record(mock_search_record_list):
    """Verifies that a single dictionary is correctly converted to a SearchRecord."""
    record_dict = mock_search_record_list[0].model_dump()
    validated_record = SearchRecordPreprocessingUtils.validate_search_records(record_dict)
    assert isinstance(validated_record, SearchRecord)
    assert validated_record.title == mock_search_record_list[0].title


def test_record_validation_rejects_invalid_type():
    """Verifies that non-record types are properly rejected."""
    with pytest.raises(TypeError, match="Expected a record or list of records, but received type <class 'str'>"):
        SearchRecordPreprocessingUtils.validate_search_records("not a record")  # type: ignore


def test_convert_records_filters_missing_required_fields():
    """Verifies that records missing required fields are filtered out."""
    # Create a record missing required fields
    incomplete_record = {
        "provider_name": "pubmed",
        "title": "Test",
        "year": 2023,
    }
    normalized_records = [incomplete_record]

    converted = SearchRecordPreprocessingUtils.convert_records(normalized_records, max_records=10)
    assert len(converted) == 0


def test_convert_records_filters_by_year_range():
    """Verifies that records outside year range are filtered out."""
    records = [
        {
            "provider_name": "pubmed",
            "query": "test",
            "title": "Old Paper",
            "abstract": "A study on BP",
            "doi": "10.4001/979-8-1234-5790-1.ch008",
            "year": 1990,
            "page": 1,
        },
        {
            "provider_name": "pubmed",
            "query": "test",
            "abstract": "A medical paper",
            "title": "Recent Paper",
            "doi": "10.8371/979-8-1234-4710-1.ch009",
            "year": 2023,
            "page": 1,
        },
    ]

    # Filter records from 2000 onwards
    converted = SearchRecordPreprocessingUtils.convert_records(records, max_records=10, year_from=2000)

    assert len(converted) == 1
    assert converted[0].title == "Recent Paper"


def test_convert_records_filters_open_access():
    """Verifies that non-open access records are filtered when open_access_only=True."""
    records = [
        {
            "provider_name": "arxiv",
            "query": "test",
            "title": "Open Access Paper",
            "abstract": "Paper containing scientific research",
            "doi": "10.3108/10321",
            "year": 2023,
            "page": 1,
            "open_access": True,
        },
        {
            "provider_name": "arxiv",
            "query": "test",
            "title": "Scientific Preprint",
            "abstract": "test abstract preprint",
            "doi": "10.20944/preprints202301.3810.v8",
            "year": 2023,
            "page": 1,
            "open_access": False,
        },
    ]

    converted = SearchRecordPreprocessingUtils.convert_records(records, max_records=10, open_access_only=True)

    assert len(converted) == 1
    assert converted[0].title == "Open Access Paper"


def test_convert_records_limits_max_records():
    """Verifies that max_records limit is respected."""
    records = [
        {
            "provider_name": "pubmed",
            "query": "test",
            "title": f"Paper {i}",
            "abstract": f"Abstract {i}",
            "doi": f"10.8371/979-8-1234-4710-1.ch00{i}",
            "year": 2023,
            "page": 1,
        }
        for i in range(10)
    ]

    converted = SearchRecordPreprocessingUtils.convert_records(records, max_records=5)
    assert len(converted) == 5


def test_deduplicate_records_respects_similarity_threshold():
    """Verifies that the fuzzy similarity threshold is respected during deduplication."""
    records = [
        SearchRecord(
            provider_name="pubmed",
            query="test",
            title="Very Similar Title",
            abstract="Similar content",
            year=2023,
            page=1,
        ),
        SearchRecord(
            provider_name="crossref",
            query="test",
            title="Very Similar Title But Different",
            abstract="Slightly different content",
            year=2023,
            page=1,
        ),
    ]

    # Low threshold should remove one
    deduped_low = SearchRecordPreprocessingUtils.deduplicate_records(records, similarity_threshold=50)
    assert len(deduped_low) == 1

    # High threshold should keep both
    deduped_high = SearchRecordPreprocessingUtils.deduplicate_records(records, similarity_threshold=99)
    assert len(deduped_high) == 2


def test_deduplicate_records_prioritizes_higher_information_content():
    """Verifies that records with higher information content are kept."""
    records = [
        SearchRecord(
            provider_name="crossref",
            query="test",
            title="Paper A",
            doi="10.13083/rs.4.rs-1234567/v3",
            abstract="",  # No abstract = low information content
            year=2023,
            page=1,
        ),
        SearchRecord(
            provider_name="crossref",
            query="test",
            title="Paper B",
            doi="10.13083/rs.4.rs-1234567/v3",
            abstract="This is a very detailed abstract with many fields filled.",  # High information content
            year=2023,
            page=1,
        ),
    ]

    deduped = SearchRecordPreprocessingUtils.deduplicate_records(records)
    assert len(deduped) == 1
    assert deduped[0].title == "Paper B"  # Paper B has higher information content


def test_deduplicate_records_empty_list():
    """Verifies that empty record lists are handled correctly."""
    deduped = SearchRecordPreprocessingUtils.deduplicate_records([])
    assert len(deduped) == 0


def test_deduplicate_records_all_invalid():
    """Verifies that lists containing values of invalid types raise a validation error."""
    with pytest.raises(ValidationError):
        _ = SearchRecordPreprocessingUtils.deduplicate_records(["invalid", 123, None, {}], similarity_threshold=90)  # type: ignore


def test_build_record_context_basic(mock_search_record_list):
    """Verifies basic record context building functionality."""
    context = SearchRecordPreprocessingUtils.build_record_context(mock_search_record_list)
    assert context is not None
    assert len(context) > 0


def test_build_record_context_custom_fields(mock_search_record_list):
    """Verifies that custom fields are used when specified."""
    custom_fields = ["title", "year"]
    context = SearchRecordPreprocessingUtils.build_record_context(mock_search_record_list, fields=custom_fields)
    assert context is not None
    for item in context:
        assert "[" in item
        assert "title" in item.lower()
        assert "year" in item.lower()


def test_build_record_context_truncates_long_records(mock_search_record_list):
    """Verifies that long records are truncated appropriately."""
    truncated_context = SearchRecordPreprocessingUtils.build_record_context(
        mock_search_record_list, record_truncation_length=100
    )
    assert truncated_context is not None
    for item in truncated_context:
        # Check that the context is not excessively long
        assert len(item) <= 100 * 3  # Rough token estimate


def test_build_record_context_respects_token_limit(mock_search_record_list, monkeypatch):
    """Verifies that token limit is respected when the token limit is set, truncating the context as a result."""
    context = SearchRecordPreprocessingUtils.build_record_context(mock_search_record_list)
    truncated_context = SearchRecordPreprocessingUtils.build_record_context(mock_search_record_list[:-1])
    assert truncated_context
    # Set a very low token limit
    estimated_tokens = SearchRecordPreprocessingUtils.estimate_token_count(" ".join(truncated_context))
    monkeypatch.setattr(SearchRecordPreprocessingUtils, "TEXT_TOKEN_LIMIT", estimated_tokens)

    truncated_context = SearchRecordPreprocessingUtils.build_record_context(mock_search_record_list)
    assert truncated_context is not None and context is not None

    # Should return fewer records than available
    assert len(truncated_context) < len(context)


def test_prepare_embedding_research_topic_with_categories():
    """Verifies that research topic is prepared correctly with categories."""
    topic = SearchRecordPreprocessingUtils.prepare_embedding_research_topic(
        queries="depression treatment",
        question="What is the best treatment for depression?",
        categories=[ResearchCategory.DEPRESSION, ResearchCategory.INTERVENTION],
    )

    assert "What is the best treatment for depression?" in topic
    assert "depression" in topic.lower()
    assert "intervention" in topic.lower()


def test_prepare_embedding_research_topic_with_multiple_queries():
    """Verifies that multiple queries are properly joined."""
    topic = SearchRecordPreprocessingUtils.prepare_embedding_research_topic(
        queries=["CBT depression", "anxiety treatment"],
        question="Treatments for mental health",
    )

    assert "CBT depression" in topic
    assert "anxiety treatment" in topic
    assert "--" in topic


def test_prepare_embedding_record_string(mock_search_record_list):
    """Verifies that record strings are prepared correctly for embedding."""
    record = mock_search_record_list[0]
    record_string = SearchRecordPreprocessingUtils.prepare_embedding_record_string(record)

    assert isinstance(record_string, str)
    assert len(record_string) > 0
    assert record.title in record_string


def test_prepare_embedding_record_string_respects_token_limit(mock_search_record_list, monkeypatch):
    """Verifies that token limit is respected when preparing record strings."""
    record = mock_search_record_list[0]
    full_record_string = record.to_string()
    tokens = SearchRecordPreprocessingUtils.estimate_token_count(full_record_string)

    monkeypatch.setattr(SearchRecordPreprocessingUtils, "RECORD_EMBEDDING_TOKEN_LIMIT", tokens - 20)
    record_string = SearchRecordPreprocessingUtils.prepare_embedding_record_string(record)
    truncated_tokens = SearchRecordPreprocessingUtils.estimate_token_count(record_string)
    assert truncated_tokens < tokens

    assert isinstance(record_string, str)
    # Should be shorter than full record
    assert len(record_string) < len(full_record_string)


def test_build_record_context_returns_none_for_invalid_input():
    """Verifies that None is returned for invalid input types."""
    context = SearchRecordPreprocessingUtils.build_record_context("not a list")  # type: ignore
    assert context is None


def test_estimate_token_count():
    """Verifies that token count estimation is accurate."""
    text = "This is a test sentence for token estimation."
    token_count = SearchRecordPreprocessingUtils.estimate_token_count(text)

    assert isinstance(token_count, int)
    assert token_count > 0
    # Should be roughly correct based on 3 chars per token
    expected = max(1, len(text) // 3)
    assert token_count == expected


def test_estimate_token_count_empty_string():
    """Verifies that empty strings are handled correctly."""
    token_count = SearchRecordPreprocessingUtils.estimate_token_count("")
    assert token_count == 1  # Returns max(1, len(text) // 3)


def test_estimate_token_count_custom_characters_per_token():
    """Verifies that custom characters per token is respected."""
    text = "This is a test sentence."
    token_count = SearchRecordPreprocessingUtils.estimate_token_count(text, characters_per_token=4)

    expected = max(1, len(text) // 4)
    assert token_count == expected


def test_estimate_token_count_non_string_raises_error():
    """Verifies that non-string input raises a TypeError."""
    with pytest.raises(TypeError, match="Expected a string to estimate LLM/embedding token count"):
        SearchRecordPreprocessingUtils.estimate_token_count(123)  # type: ignore


def test_estimate_token_count_none_value():
    """Verifies that None values are handled correctly."""
    token_count = SearchRecordPreprocessingUtils.estimate_token_count(None)  # type: ignore
    assert token_count == 1

"""Tests for ensuring that embeddings operate as intended with fallbacks where beneficial."""

import pytest
from pydantic_ai.embeddings import Embedder, EmbeddingResult

from scholar_flux_mcp.agents.record_topic_similarity_embedder import (
    RecordTopicSimilarity,
    RecordTopicSimilarityEmbedder,
    SearchRecordEmbedding,
)
from scholar_flux_mcp.exceptions import (
    DocumentEmbeddingFailedException,
    EmbedderUninitializedException,
    InvalidEmbedderParameterException,
)


def test_document_embedder_initialization(mock_embedding_model):
    """Verifies that the RecordTopicSimilarityEmbedder can be loaded as intended."""
    document_embedder = RecordTopicSimilarityEmbedder()
    assert document_embedder._embedder is None
    with pytest.raises(EmbedderUninitializedException) as excinfo:
        _ = document_embedder.embedder
    assert "A PydanticAI embedder has not yet been created." in str(excinfo.value)
    embedder = document_embedder.get_or_create_embedder()
    assert embedder is document_embedder.embedder and isinstance(embedder, Embedder)


def test_document_embedder_invalid_assignment(caplog, mock_embedding_model):
    """Verifies that the RecordTopicSimilarityEmbedder will not accept non-Embedder instances."""
    document_embedder = RecordTopicSimilarityEmbedder()
    invalid_value = 42
    with pytest.raises(InvalidEmbedderParameterException) as excinfo:
        document_embedder.embedder = invalid_value  # type: ignore

    # Should still be None after the error is raised and caught.
    assert (
        f"The RecordTopicSimilarityEmbedder expected a PydanticAI Embedder instance, but received type {type(invalid_value)}."
        in str(excinfo.value)
    )
    assert document_embedder._embedder is None


def test_document_embedder_rankings(mock_indexed_search_record_list):
    """Verifies that the RecordTopicSimilarityEmbedder enables re-sorting and filtering based on selected options."""
    assert mock_indexed_search_record_list
    embedding_similarity_results = [
        RecordTopicSimilarity(
            record_embedding=SearchRecordEmbedding(record=record),
            topic="test topic",
            topic_similarity_score=record.topic_similarity_score,
        )
        for record in mock_indexed_search_record_list
    ]

    reversed_index_scores = [
        record.model_copy(update={"topic_similarity_score": 1 - record.topic_similarity_score + 0.5})
        for record in mock_indexed_search_record_list
    ]
    reversed_embedding_similarity_results = [
        RecordTopicSimilarity(
            record_embedding=SearchRecordEmbedding(record=record),
            topic="test topic",
            topic_similarity_score=record.topic_similarity_score,
        )
        for record in reversed_index_scores
    ]

    filter_sim = RecordTopicSimilarityEmbedder.filter_similarity_scores  # shortcut

    # Basic search to verify that records are filtered as intended. The first two cases verifies sort order when no records are removed
    assert (
        filter_sim(
            embedding_similarity_results,
            max_records=3,
        )
        == mock_indexed_search_record_list
    )
    assert filter_sim(embedding_similarity_results, max_records=None) == mock_indexed_search_record_list
    assert (
        filter_sim(embedding_similarity_results, max_records=2, sort_by_similarity=True)
        == mock_indexed_search_record_list[:2]
    )

    # If max_records==None or max_records==len(reversed_index_scores), don't adjust ordering if sort_by_similarity=False
    assert (
        filter_sim(reversed_embedding_similarity_results, max_records=3, sort_by_similarity=False)
        == reversed_index_scores
    )
    assert (
        filter_sim(reversed_embedding_similarity_results, max_records=None, sort_by_similarity=False)
        == reversed_index_scores
    )

    # Reverse and reindex all index scores after sorting by updated topic similarity score
    reindexed_reversed_index_scores = [
        record.model_copy(update={"index": i}) for i, record in enumerate(reversed_index_scores[::-1])
    ]
    assert (
        filter_sim(reversed_embedding_similarity_results, max_records=3, sort_by_similarity=True)
        == reindexed_reversed_index_scores
    )

    # Retrieve the last two elements of the reversed embedding results without re-sorting by similarity score
    reindexed_reversed_top_two_indexed_scores = [
        record.model_copy(update={"index": i}) for i, record in enumerate(reversed_index_scores[-2:])
    ]
    assert (
        filter_sim(reversed_embedding_similarity_results, max_records=2, sort_by_similarity=False)
        == reindexed_reversed_top_two_indexed_scores
    )


@pytest.mark.asyncio
async def test_document_embedder_embed_text(caplog, mock_embedding_model):
    """Verifies that the basic `RecordTopicSimilarityEmbedder.embed` method correctly embeds text."""
    document_embedder = RecordTopicSimilarityEmbedder()
    document_embedder.get_or_create_embedder()
    doc = "Hello world!!"
    with document_embedder.embedder.override(model=mock_embedding_model):
        embedding_result = await document_embedder.embed(doc)
    assert isinstance(embedding_result, EmbeddingResult)
    assert len(embedding_result.embeddings) == 1 and len(embedding_result.embeddings[0]) >= 8


@pytest.mark.asyncio
async def test_document_embedder_embed_documents(caplog, mock_search_record_list, mock_embedding_model):
    """Verifies that the basic `RecordTopicSimilarityEmbedder.embed` method correctly embeds text."""
    document_embedder = RecordTopicSimilarityEmbedder()
    document_embedder.get_or_create_embedder()
    with document_embedder.embedder.override(model=mock_embedding_model):
        embedding_result = await document_embedder.embed_records(mock_search_record_list)
    assert isinstance(embedding_result, list) and all(
        isinstance(record_embedding, SearchRecordEmbedding) for record_embedding in embedding_result
    )
    assert len(embedding_result) == len(mock_search_record_list)
    embedding_length = list({len(record.embedding) for record in embedding_result})
    assert embedding_length[0] == 8


@pytest.mark.asyncio
async def test_document_embedder_embed_documents_batch_equivalent(
    mock_search_record_list, mock_embedding_model, monkeypatch
):
    """Verifies that the basic `RecordTopicSimilarityEmbedder.embed` method correctly embeds text."""
    document_embedder = RecordTopicSimilarityEmbedder()
    document_embedder.get_or_create_embedder()
    monkeypatch.setattr(RecordTopicSimilarityEmbedder, "DEFAULT_EMBEDDING_BATCH_SIZE", None)

    with document_embedder.embedder.override(model=mock_embedding_model):
        embedding_result = await document_embedder.embed_records(mock_search_record_list, batch_size=None)
        batched_embedding_result = await document_embedder.embed_records(mock_search_record_list, batch_size=2)

        assert embedding_result == batched_embedding_result


@pytest.mark.asyncio
async def test_document_embedder_embed_invalid_value(caplog, mock_embedding_model):
    """Verifies that `.embed()` raises an `DocumentEmbeddingFailedException` when passed non-text values."""
    document_embedder = RecordTopicSimilarityEmbedder()
    document_embedder.get_or_create_embedder()
    invalid_value = [1, 3, 5]
    with (
        document_embedder.embedder.override(model=mock_embedding_model),
        pytest.raises(DocumentEmbeddingFailedException) as excinfo,
    ):
        _ = await document_embedder.embed(invalid_value)  # type: ignore
    err = "Encountered an unexpected error during document embedding"
    assert err in str(excinfo.value)
    assert err in caplog.text

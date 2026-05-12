"""Defines the RecordTopicSimilarityEmbedder for efficient embedding and similarity scoring with PydanticAI."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator, Sequence
from functools import partial
from math import ceil
from operator import itemgetter
from typing import TYPE_CHECKING, Any, overload

from pydantic import TypeAdapter

if TYPE_CHECKING:
    from pydantic_ai.embeddings import Embedder, EmbeddingResult
else:
    try:
        from pydantic_ai.embeddings import Embedder, EmbeddingResult
    except ImportError:
        Embedder = EmbeddingResult = None


from scholar_flux_mcp.agents.models import EmbedderABC, PydanticAIEmbeddingModelFactory
from scholar_flux_mcp.exceptions import (
    DocumentEmbeddingFailedException,
    EmbedderInitializationException,
    EmbedderUninitializedException,
    PydanticAIImportError,
)
from scholar_flux_mcp.models import (
    IndexedSearchRecord,
    RecordTopicSimilarity,
    RecordTopicSimilarityOutput,
    ResearchCategoryList,
    SearchRecord,
    SearchRecordEmbedding,
    SearchRecordList,
    ServiceHealth,
    TopicEmbedding,
)
from scholar_flux_mcp.utils.helpers import as_tuple, coerce_int
from scholar_flux_mcp.utils.preprocessing_utils import SearchRecordPreprocessingUtils
from scholar_flux_mcp.utils.similarity_utils import cosine_similarity

logger = logging.getLogger(__name__)

search_record_adapter: TypeAdapter[SearchRecordList] = TypeAdapter(SearchRecordList)


class RecordTopicSimilarityEmbedder(EmbedderABC):
    """Class responsible for creating and embedding questions and documents via PydanticAI for similarity scoring."""

    DEFAULT_SEARCH_RECORD_EMBEDDING_FIELDS: list[str] = ["title", "year", "journal", "abstract", "keywords"]
    DEFAULT_SEARCH_RECORD_EMBEDDING_SIMILARITY_THRESHOLD: float = 0.0
    DEFAULT_EMBEDDING_BATCH_SIZE: int | None = coerce_int(os.getenv("SCHOLAR_FLUX_MCP_EMBEDDING_BATCH_SIZE"))
    TOPIC_SIMILARITY_RANKING_PRECISION: int | None = 2

    def get_or_create_embedder(self, provider: str | None = None) -> Embedder:
        """Creates a new PydanticAI Embedder for similarity scoring."""
        with contextlib.suppress(EmbedderUninitializedException):
            return self.embedder

        self.embedder = self._create_embedder(provider)

        return self.embedder

    @classmethod
    def _create_embedder(cls, provider: str | None) -> Embedder:
        """Factory method for creating a new embedder using the `PydanticAIEmbeddingModelFactory` when possible."""
        if Embedder is None:
            raise PydanticAIImportError()

        return provider if isinstance(provider, Embedder) else PydanticAIEmbeddingModelFactory.create(provider)

    async def embed_topic(
        self, question: str, queries: str | list[str], categories: ResearchCategoryList | None = None
    ) -> TopicEmbedding:
        """Embeds a question along with a query or multiple queries using a PydanticAI embedding model.

        If an embedding model does not already exist, this method will lazily create the embedding model using the
        `PydanticAIEmbeddingModelFactory` to select a specific embedding model provider (configured with env variables)
        or by selecting the most preferable model when a model isn't explicitly specified.

        Args:
            question (str): The question used to calculate the embedding for the topic of synthesis.
            queries (str | list[str]): A list of queries to be used in the calculation of the topic embedding.
            categories (ResearchCategoryList | None): A list of research categories to include in the topic embedding

         Returns:
             TopicEmbedding: The PydanticAI embedding results for the input query.

        """
        research_topic_string = SearchRecordPreprocessingUtils.prepare_embedding_research_topic(
            queries=queries, question=question, categories=categories
        )
        topic_embedding_result = await self.embed(research_topic_string)
        return TopicEmbedding(
            topic=research_topic_string,
            question=question,
            queries=[queries] if isinstance(queries, str) else queries,
            embedding=topic_embedding_result[0],
            categories=categories or [],
            model_name=self.model_name,
        )

    @overload
    async def embed_records(
        self,
        records: SearchRecord,
        fields: list[str] | None = None,
        token_limit: int | None = None,
        batch_size: int | None = None,
    ) -> SearchRecordEmbedding:
        """When a record and query is passed, a RecordTopicSimilarity result is returned."""
        ...

    @overload
    async def embed_records(
        self,
        records: SearchRecordList,
        fields: list[str] | None = None,
        token_limit: int | None = None,
        batch_size: int | None = None,
    ) -> list[SearchRecordEmbedding]:
        """When a list of records and a query is passed, a list of RecordTopicSimilarity results are returned."""
        ...

    async def embed_records(
        self,
        records: SearchRecord | SearchRecordList,
        fields: list[str] | None = None,
        token_limit: int | None = None,
        batch_size: int | None = None,
    ) -> SearchRecordEmbedding | list[SearchRecordEmbedding]:
        """Embeds a record or multiple records using a PydanticAI embedding model.

        If an embedding model does not already exist, this method will lazily create the embedding model using the
        `PydanticAIEmbeddingModelFactory` to select a specific embedding model provider (configured with env variables)
        or by selecting the most preferable model when a model isn't explicitly specified.

        Args:
            records (SearchRecord | SearchRecordList): A record or list of records to embed
            fields (list[str]): A list of fields to calculate embedding similarity with.
            token_limit (int | None): The total number of tokens that the current record should not exceed.

         Returns:
             SearchRecordEmbedding: The PydanticAI embedding results for the input record or record list.

        """
        embedding_fields = fields or self.DEFAULT_SEARCH_RECORD_EMBEDDING_FIELDS
        search_record_list = search_record_adapter.validate_python(records)
        embedding_record_strings = (
            (SearchRecordPreprocessingUtils.prepare_embedding_record_string(r, embedding_fields, token_limit))
            for r in search_record_list
        )
        record_batch_size = batch_size or self.DEFAULT_EMBEDDING_BATCH_SIZE
        record_embedding_results = await self.embed(embedding_record_strings, batch_size=record_batch_size)

        record_embedding_list = [
            SearchRecordEmbedding(record=record, embedding=embedding, model_name=self.model_name)
            for record, embedding in zip(search_record_list, record_embedding_results.embeddings, strict=True)
        ]
        return record_embedding_list[0] if isinstance(records, SearchRecord) else record_embedding_list

    async def embed(self, text: str | Sequence[str] | Iterator[str], batch_size: int | None = None) -> EmbeddingResult:
        """Helper for embedding a singular document or list of documents."""
        try:
            embedder = self.get_or_create_embedder()

            if isinstance(text, Iterator):
                text = list(text)

            listed_text = [text] if isinstance(text, str) else text

            if listed_text and isinstance(batch_size, int) and len(listed_text) > batch_size:
                total_batches = ceil(len(listed_text) / batch_size)
                batch_iterator = (listed_text[batch_size * i : batch_size * (i + 1)] for i in range(0, total_batches))
                embedded_documents = [
                    await embedder.embed_documents(document_batch) for document_batch in batch_iterator
                ]

                return EmbeddingResult(
                    embeddings=[emb for result in embedded_documents for emb in result.embeddings],
                    inputs=[inp for result in embedded_documents for inp in result.inputs],
                    model_name=embedded_documents[0].model_name,
                    provider_name=embedded_documents[0].provider_name,
                    input_type=embedded_documents[0].input_type,
                )

            return await embedder.embed_documents(listed_text)
        except PydanticAIImportError:
            raise
        except EmbedderInitializationException:
            raise
        except Exception as e:
            err = f"Encountered an unexpected error during document embedding: {e}"
            logger.error(err, exc_info=True)
            raise DocumentEmbeddingFailedException(err) from e

    async def __call__(self, text: str | Sequence[str]) -> EmbeddingResult:
        """Helper method for calling `embed` under the hood for embedding strings."""
        return await self.embed(text)

    async def embedding_similarity(
        self,
        records: SearchRecord | SearchRecordList,
        question: str,
        queries: str | list[str] | None = None,
        categories: ResearchCategoryList | None = None,
        token_limit: int | None = None,
    ) -> RecordTopicSimilarityOutput:
        """Embeds a query and SearchRecord or list of SearchRecords to calculate record similarity."""
        record_list = SearchRecordPreprocessingUtils.validate_search_records(
            [records] if isinstance(records, SearchRecord) else records
        )
        logger.info("Embedding research topic...")
        research_topic_embedding = await self.embed_topic(question, queries=queries or [], categories=categories)
        logger.info("Embedding documents...")
        record_embeddings = await self.embed_records(record_list, token_limit=token_limit)

        logger.info("Calculating record-research topic similarity scores...")
        record_similarity_scores = self.calculate_record_cosine_similarity(record_embeddings, research_topic_embedding)
        logger.info(f"Calculated record-research topic similarity scores for {len(record_similarity_scores)} records.")
        return RecordTopicSimilarityOutput(
            topic_embedding=research_topic_embedding,
            record_similarity_scores=record_similarity_scores,
        )

    @classmethod
    def calculate_record_cosine_similarity(
        cls,
        search_record_embeddings: SearchRecordEmbedding | list[SearchRecordEmbedding],
        research_topic_embedding: TopicEmbedding,
    ) -> list[RecordTopicSimilarity]:
        """Uses the query and record contents in addition to their embeddings to create a new RecordTopicSimilarity list."""

        return [
            RecordTopicSimilarity(
                record_embedding=search_record_embedding,
                topic=research_topic_embedding.topic,
                topic_similarity_score=cosine_similarity(
                    search_record_embedding.embedding, research_topic_embedding.embedding
                ),
            )
            for search_record_embedding in as_tuple(search_record_embeddings)
        ]

    @classmethod
    def indexed_similarity_score_sort_order(
        cls,
        indexed_record_embedding_similarity: tuple[int, RecordTopicSimilarity],
        precision: int | None = None,
    ) -> tuple[float, int]:
        """Sorting method used to sort record embeddings by similarity. Preserves original order otherwise."""
        index, record_embedding_similarity = indexed_record_embedding_similarity
        similarity_score = cls.similarity_score_sort_order(record_embedding_similarity, precision)
        return similarity_score, index

    @classmethod
    def similarity_score_sort_order(
        cls,
        record_embedding_similarity: RecordTopicSimilarity,
        precision: int | None = None,
    ) -> float:
        """Sorting method used to sort record embeddings by similarity. Preserves original order otherwise."""
        precision = precision if isinstance(precision, int) else cls.TOPIC_SIMILARITY_RANKING_PRECISION
        similarity_score = (
            round(record_embedding_similarity.topic_similarity_score, precision)
            if precision is not None
            else record_embedding_similarity.topic_similarity_score
        )
        return -similarity_score

    @classmethod
    def filter_top_n_similarity_scores(
        cls,
        record_similarity_list: Iterator[RecordTopicSimilarity] | Sequence[RecordTopicSimilarity],
        max_records: int,
        *,
        sort_by_similarity: bool = True,
    ) -> Sequence[RecordTopicSimilarity]:
        """Ranks and retains the top `max` embedding similarity scores, returning an ordered list."""
        # precision=99 ensures granularity in similarity score ranking calculations with minimal rounding
        indexed_precision_ordering = partial(cls.indexed_similarity_score_sort_order, precision=99)
        ranked_similarity_list = sorted(enumerate(record_similarity_list), key=indexed_precision_ordering)

        # Retrieves the top n records by index
        ranked_similarity_list = ranked_similarity_list[:max_records]

        # re-sorts by similarity with the default rounding precision when enabled. Otherwise reverts the order.
        ranked_similarity_list.sort(
            key=cls.indexed_similarity_score_sort_order if sort_by_similarity else itemgetter(0)
        )

        # Retrieves only similarity scores after re-sorting
        return [record_embedding_similarity for _, record_embedding_similarity in ranked_similarity_list]

    @classmethod
    def filter_similarity_scores(
        cls,
        record_similarity_list: Iterator[RecordTopicSimilarity] | Sequence[RecordTopicSimilarity],
        max_records: int | None = None,
        *,
        threshold: float | None = 0.5,
        sort_by_similarity: bool = True,
    ) -> Sequence[IndexedSearchRecord]:
        """Filters and sorts records by similarity score, retaining records above the specified scoring threshold."""
        current_similarity_threshold = (
            threshold if threshold is not None else cls.DEFAULT_SEARCH_RECORD_EMBEDDING_SIMILARITY_THRESHOLD
        )
        filtered_similarity_scores: Iterator[RecordTopicSimilarity] | Sequence[RecordTopicSimilarity] = (
            record_embedding_similarity
            for record_embedding_similarity in record_similarity_list
            if record_embedding_similarity.topic_similarity_score >= current_similarity_threshold
        )

        if isinstance(max_records, int) and max_records > 0:
            filtered_similarity_scores = cls.filter_top_n_similarity_scores(
                filtered_similarity_scores, max_records=max_records, sort_by_similarity=sort_by_similarity
            )

        elif sort_by_similarity:
            # Sort order is otherwise preserved during ties and only ranked by similarity score
            filtered_similarity_scores = sorted(filtered_similarity_scores, key=cls.similarity_score_sort_order)

        return [
            IndexedSearchRecord.from_search_record(
                record_embedding_similarity.record,
                index=i,
                topic_similarity_score=record_embedding_similarity.topic_similarity_score,
            )
            for i, record_embedding_similarity in enumerate(filtered_similarity_scores)
        ]

    async def get_stats(self) -> dict[str, Any]:
        """Show the current configuration of the RecordTopicSimilarityEmbedder.

        Returns:
            dict[str, Any]: Embedding model configuration including the embedding initialization status and model name.

        """

        try:
            embedder = self.get_or_create_embedder()
            return {"initialized": True, "model_name": self.get_model_name(embedder)}

        except Exception as e:
            return {"initialized": False, "model_name": None, "error": str(e)}

    async def check_health(self) -> ServiceHealth:
        """Helper used to check the health status of the RecordTopicSimilarityEmbedder."""
        details = await self.get_stats()
        error = details.get("error")
        health_status = "healthy" if not error else "unhealthy"

        return ServiceHealth(
            name="RecordTopicSimilarityEmbedder",
            status=health_status,
            details=details,
            error=error,
        )


__all__ = ["RecordTopicSimilarityEmbedder"]

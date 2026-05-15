"""Module defining the `HistoryService` for storing and retrieving MCP tool outputs from a relational database cache."""

from __future__ import annotations

import asyncio
import itertools
import logging
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, overload
from urllib.parse import urlparse

from sqlalchemy import Engine, create_engine, exc
from sqlalchemy.orm import selectinload

from scholar_flux_mcp.exceptions.history_exceptions import (
    HistoryCacheConnectionFailed,
    HistoryCacheDeletionException,
    HistoryCacheInitializationException,
    HistoryCacheParameterValidationException,
    HistoryCacheRetrievalException,
    HistoryCacheStorageException,
)
from scholar_flux_mcp.exceptions.import_exceptions import RapidFuzzImportError, SQLModelImportError
from scholar_flux_mcp.models import (
    RelevanceSearchInput,
    RelevanceSearchOutput,
    SearchInput,
    SearchOutput,
    SearchRecord,
    ServiceHealth,
    SynthesisInput,
    SynthesisOutput,
)
from scholar_flux_mcp.models.type_aliases import (
    RESEARCH_TOOL_TYPES,
    ResearchHistoryOutput,
    ResearchToolInput,
    ResearchToolOutput,
    ResearchToolType,
    SupportsTopicSimilarity,
)
from scholar_flux_mcp.server.io import (
    RelevanceSearchToolInput,
    SearchToolInput,
    SynthesisToolInput,
)
from scholar_flux_mcp.utils.fuzzy_text_similarity import PartialRatioSimilarity
from scholar_flux_mcp.utils.helpers import coerce_bool, coerce_numeric, coerce_str, try_none
from scholar_flux_mcp.utils.initializer import masker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    import sqlmodel  # if history_models is available, so is sqlmodel

    # also available if sqlalchemy is available
    from sqlalchemy import Engine, create_engine, exc
    from sqlalchemy.orm import selectinload

    import scholar_flux_mcp.models.history as history_models
else:
    try:
        import sqlmodel  # if history_models is available, so is sqlmodel

        # also available if sqlalchemy is available
        from sqlalchemy import Engine, create_engine, exc
        from sqlalchemy.orm import selectinload

        import scholar_flux_mcp.models.history as history_models
    except (ImportError, ModuleNotFoundError):
        history_models = Engine = create_engine = exc = selectinload = None


URI_SCHEMA_PATTERN: re.Pattern = re.compile(r"^[a-zA-Z0-9+]+:///?")


logger = logging.getLogger(__name__)


def _calculate_fuzzy_topic_similarity_worker(
    history_item: SupportsTopicSimilarity,
    topic: str,
    similarity_threshold: int | float | None,
) -> tuple[SupportsTopicSimilarity, PartialRatioSimilarity]:
    """Internal helper used to generate tuples calculating the similarity of the item to the current topic.

    Because multiprocessing methods aren't compatible with lambda or partials (on package reloading)< this worker is
    defined outside of the `HistoryService`, preventing multiprocessing issues in the process.

    Args:
        history_item (ResearchToolInput | ResearchHistoryOutput | SearchRecord | SearchRecordHistory):
            A research tool input, output, or record to calculate fuzzy string similarity to the topic.
        topic (str | None):
            An optional topic string used to filter and sort records by fuzzy similarity. When provided, results
            are filtered to those exceeding `similarity_threshold` and ordered by similarity score descending.
        similarity_threshold (int | float | None):
            The minimum fuzzy similarity score (0.0–1.0) required for a record to be included. Only applied
            when `topic` is provided.

    Returns:
        tuple[SupportsTopicSimilarity, PartialRatioSimilarity]:
            A tuple containing the original history item and a `PartialRatioSimilarity` containing its calculated
            fuzzy similarity to the topic.

    """
    similarity = PartialRatioSimilarity.calculate(
        sub_text=topic, text=history_item.topic, threshold=similarity_threshold
    )
    return (history_item, similarity)


class HistoryService:
    """Implements the sqlmodel methods to interact with MCP history for retrieval, relevance search and synthesis."""

    DEFAULT_CONFIG: dict[str, Any] = {
        "url": lambda: HistoryService.get_default_url(),
        "echo": False,
    }
    DEFAULT_RAISE_ON_ERROR: bool = True
    STORAGE_TYPE: str = "SQL"

    def __init__(
        self,
        url: str | None = None,
        raise_on_error: bool | None = None,
        verify_connection: bool = False,
        ttl: int | float | None = None,
        enable: bool | None = True,
        **sqlmodel_config: Any,
    ) -> None:
        """Initialize the SQLModel storage backend and connect to the server indicated via the `url` parameter.

        This class uses the innate flexibility of SQLModel to support backends such as SQLite, Postgres, DuckDB, etc.

        Args:
            url (Optional[str]):
                Database connection string. This can be provided positionally or as a keyword argument.
            raise_on_error (Optional[bool]):
                Determines whether an error should be raised when encountering unexpected issues when interacting with
                SQLModel. If `None`, the `raise_on_error` attribute defaults to `HistoryService.DEFAULT_RAISE_ON_ERROR`.
            verify_connection (bool):
                If True, verifies the SQL service is available immediately after initialization.
                Raises StorageCacheException if connection fails. Defaults to False.
            ttl (Optional[int]):
                A basic TTL used to determine when data should be retained or omitted from selection during retrieval
            enable (Optional[bool]):
                Indicates whether to initialize the database engine and tables for the history service .
            **sqlmodel_config:
                Additional SQLModel engine/session options passed to sqlalchemy.create_engine Typical parameters include
                the following:
                - url (str): Indicates what server to connect to. Defaults to sqlite in the package directory.

        """
        default_config = self.get_default_config()
        sqlmodel_config["url"] = url or default_config["url"]()  # lazy writable path creation for defaults

        self.config: dict[str, Any] = sqlmodel_config
        self.lock = asyncio.Lock()
        enable_history = enable if enable is not None else coerce_bool(os.getenv("SCHOLAR_FLUX_MCP_ENABLE_HISTORY"))
        enabled = True if enable_history is None and history_models is not None else enable_history

        if enabled:
            try:
                self.initialize_database()
            except SQLModelImportError as e:
                # Raise an error at this point only if `raise_on_error` is not explicitly enabled
                self._handle_exception(e, raise_exception_type=SQLModelImportError if raise_on_error else None)
        else:
            self.engine = None

        self.raise_on_error = raise_on_error if raise_on_error is not None else self.DEFAULT_RAISE_ON_ERROR

        if enabled and verify_connection:
            self.verify_connection()

        ttl = self._validate_ttl(ttl)
        self.ttl = ttl if ttl is not None else self._validate_ttl(os.getenv("SCHOLAR_FLUX_MCP_HISTORY_TTL"))

    @property
    def engine(self) -> Engine:
        """Returns the current engine given the successful database initialization for the HistoryService."""
        if self._engine is None:
            msg = "The current `HistoryService` instance has not yet fully initialized the output history cache."
            logger.error(msg)
            raise HistoryCacheInitializationException(msg)
        return self._engine

    @engine.setter
    def engine(self, engine: Engine | None) -> None:
        """Sets the Engine used by the HistoryService for storage and retrieval."""
        if engine is not None and not isinstance(engine, Engine):
            raise TypeError(f"The HistoryService expected an `engine`, but received {type(engine)}")
        self._engine = engine

    @property
    def enabled(self) -> bool:
        """Indicates whether the history service is initialized and available."""
        if self._engine is None:
            return False

        history_tables = {model.__tablename__ for model in history_models.HISTORY_TABLES}
        missing_tables = history_tables.difference(set(sqlmodel.SQLModel.metadata.tables))
        return not missing_tables

    @property
    def initialized(self) -> bool:
        """Alias indicating whether the history service has been initialized and is available."""
        return self.enabled

    def initialize_database(self) -> None:
        """Initializes the database engine and metadata tables for the HistoryService."""
        if history_models is None or sqlmodel is None:
            raise SQLModelImportError()
        engine = sqlmodel.create_engine(**self.config)
        sqlmodel.SQLModel.metadata.create_all(engine)
        self.engine = engine

    @property
    def session(self) -> sqlmodel.Session:
        """Property enabling the efficient creation of a session from the current `Engine` instance."""
        return sqlmodel.Session(self.engine)

    @asynccontextmanager
    async def initialize_session(self) -> AsyncIterator[sqlmodel.Session]:
        """Initializes a new asynchronous session with the initialized engine of the HistoryService."""

        async with self.lock:
            with self.session as session:
                yield session

    @asynccontextmanager
    async def get_or_create_session(self, session: sqlmodel.Session | None = None) -> AsyncIterator[sqlmodel.Session]:
        """Uses or initializes a new asynchronous session with the initialized engine of the HistoryService."""
        if isinstance(session, sqlmodel.Session):
            if session.is_active:
                yield session
            else:
                with session:
                    yield session

        else:
            async with self.initialize_session() as session:
                yield session

    @classmethod
    def _retrieve_synthesis_history(
        cls,
        session: sqlmodel.Session,
        input: SynthesisInput | SynthesisToolInput | str | None = None,
        *,
        ttl: int | float | None = None,
        successful_only: bool = False,
        final_output_only: bool | None = None,  # No-Op: For input parameter consistency only
    ) -> Iterator[history_models.SynthesisExecution]:
        """Creates a generator matching the provided synthesis input to its outputs ordered from newest to oldest."""
        stmt = (
            sqlmodel.select(history_models.SynthesisExecution)
            .join(history_models.SynthesisInputHistory)
            .join(history_models.SynthesisOutputHistory)
            .options(
                selectinload(history_models.SynthesisExecution.synthesis_input),
            )
        )

        if input is not None and not isinstance(input, SynthesisInput | SynthesisToolInput | str):
            raise TypeError(f"Expected a research synthesis input or input hash, but received {type(input).__name__}")

        if input:
            input_hash = input if isinstance(input, str) else input.input_hash
            stmt = stmt.where(history_models.SynthesisInputHistory.input_hash == input_hash)

        if successful_only:
            stmt = stmt.where(history_models.SynthesisOutputHistory.successful)

        stmt = stmt.order_by(sqlmodel.col(history_models.SynthesisInputHistory.stored_at).desc())

        synthesis_result = session.exec(stmt)

        for synthesis_execution in synthesis_result:
            if ttl is None or not synthesis_execution.synthesis_output.is_expired(ttl):
                yield synthesis_execution

    @classmethod
    def _retrieve_relevance_search_history(
        cls,
        session: sqlmodel.Session,
        input: RelevanceSearchInput | RelevanceSearchToolInput | str | None = None,
        *,
        ttl: int | float | None = None,
        successful_only: bool = False,
        final_output_only: bool | None = False,
    ) -> Iterator[history_models.RelevanceSearchExecution]:
        """Creates a generator matching the provided relevance search input to its outputs from newest to oldest."""
        stmt = (
            sqlmodel.select(history_models.RelevanceSearchExecution)
            .join(history_models.RelevanceSearchInputHistory)
            .join(history_models.RelevanceSearchOutputHistory)
            .options(selectinload(history_models.RelevanceSearchExecution.relevance_search_input))
        )

        if input is not None and not isinstance(input, RelevanceSearchInput | RelevanceSearchToolInput | str):
            raise TypeError(f"Expected a relevance search input or input hash, but received {type(input).__name__}")

        if input:
            input_hash = input if isinstance(input, str) else input.input_hash
            stmt = stmt.where(history_models.RelevanceSearchInputHistory.input_hash == input_hash)

        if successful_only:
            stmt = stmt.where(history_models.RelevanceSearchOutputHistory.successful)

        stmt = stmt.order_by(sqlmodel.col(history_models.RelevanceSearchInputHistory.stored_at).desc())

        relevance_search_result = session.exec(stmt)

        for relevance_search_execution in relevance_search_result:
            unexpired = ttl is None or not relevance_search_execution.relevance_search_output.is_expired(ttl)
            if unexpired and (not final_output_only or relevance_search_execution.synthesis_execution is None):
                yield relevance_search_execution

    @classmethod
    def _retrieve_search_history(
        cls,
        session: sqlmodel.Session,
        input: SearchInput | SearchToolInput | str | None = None,
        *,
        ttl: int | float | None = None,
        successful_only: bool = False,
        final_output_only: bool | None = False,
    ) -> Iterator[history_models.SearchExecution]:
        """Creates a generator matching the provided search input to its outputs from newest to oldest."""
        stmt = (
            sqlmodel.select(history_models.SearchExecution)
            .join(history_models.SearchInputHistory)
            .join(history_models.SearchOutputHistory)
            .options(selectinload(history_models.SearchExecution.search_input))
        )

        if input is not None and not isinstance(input, SearchInput | SearchToolInput | str):
            raise TypeError(f"Expected a search input or input hash, but received {type(input).__name__}")

        if input:
            input_hash = input if isinstance(input, str) else input.input_hash
            stmt = stmt.where(history_models.SearchInputHistory.input_hash == input_hash)

        if successful_only:
            stmt = stmt.where(history_models.SearchOutputHistory.successful)
        stmt = stmt.order_by(sqlmodel.col(history_models.SearchInputHistory.stored_at).desc())

        search_result = session.exec(stmt)

        for search_execution in search_result:
            unexpired = ttl is None or not search_execution.search_output.is_expired(ttl)
            if unexpired and (not final_output_only or search_execution.relevance_search_execution is None):
                yield search_execution

    @overload
    async def retrieve(
        self,
        input: SynthesisInput | SynthesisToolInput,
        *,
        ttl: int | float | None = None,
        successful_only: bool = False,
        raise_on_error: bool | None = None,
    ) -> SynthesisOutput | None:
        """The latest synthesis output is returned when available if a synthesis input is provided."""
        ...

    @overload
    async def retrieve(
        self,
        input: RelevanceSearchInput | RelevanceSearchToolInput,
        raise_on_error: bool | None = None,
        *,
        session: sqlmodel.Session | None = None,
        ttl: int | float | None = None,
        successful_only: bool = False,
        final_output_only: bool | None = False,
    ) -> RelevanceSearchOutput | None:
        """The latest relevance search output is returned when available if a relevance search input is provided."""
        ...

    @overload
    async def retrieve(
        self,
        input: SearchInput | SearchToolInput,
        raise_on_error: bool | None = None,
        *,
        session: sqlmodel.Session | None = None,
        ttl: int | float | None = None,
        successful_only: bool = False,
        final_output_only: bool | None = False,
    ) -> SearchOutput | None:
        """The latest search output is returned when available if a search input is provided."""
        ...

    async def retrieve(
        self,
        input: ResearchToolInput,
        raise_on_error: bool | None = None,
        *,
        session: sqlmodel.Session | None = None,
        ttl: int | float | None = None,
        successful_only: bool = False,
        final_output_only: bool | None = False,
    ) -> ResearchToolOutput | None:
        """Attempts to resolve the input via its associated input hash if stored within history.

        Args:
            input  (SynthesisInput | SynthesisToolInput | RelevanceSearchInput | RelevanceSearchToolInput | SearchInput | SearchToolInput):
                The input that will be looked up within the current relational database via SQLModel.
            raise_on_error (Optional[bool]):
                Determines whether an error should be raised when encountering an exception during history retrieval.
            session (sqlmodel.Session | None = None):
               An optional session used to retrieve previously stored history records. If not provided, a new session is
               initialized instead.
            ttl (int | float | None):
                The TTL (time to live) measured in seconds. When None, the TTL associated with the history service is
                used instead.
            successful_only (bool):
                Indicates whether only successfully processed tool calls should be returned. Tool call success is
                defined by the `successful` property on each output class:

                - SearchOutput: successful if all requests return a valid result
                - RelevanceSearchOutput: successful if all records are successfully embedded.
                - SynthesisOutput: When the synthesis agent returns a valid, complete synthesis.
            final_output_only (bool | None):
                Indicates whether output history should only be retrieved when outputted from its corresponding tool.

        Returns:
            SearchOutput | RelevanceSearchOutput | SynthesisOutput | None:
                A valid research tool output when available and fresh given the TTL. Otherwise None.

        """
        ttl = ttl if ttl is not None else self.ttl
        raise_on_error = raise_on_error if raise_on_error is not None else self.raise_on_error

        try:
            if isinstance(input, SynthesisToolInput | SynthesisInput):
                async with self.get_or_create_session(session) as initialized_session:
                    synthesis_history = next(
                        self._retrieve_synthesis_history(
                            initialized_session,
                            input,
                            ttl=ttl,
                            successful_only=successful_only,
                            final_output_only=final_output_only,
                        ),
                        None,
                    )
                    return (
                        synthesis_history.synthesis_output.to_synthesis_output()
                        if synthesis_history is not None
                        else None
                    )
            elif isinstance(input, RelevanceSearchToolInput | RelevanceSearchInput):
                async with self.get_or_create_session(session) as initialized_session:
                    relevance_search_history = next(
                        self._retrieve_relevance_search_history(
                            initialized_session,
                            input,
                            ttl=ttl,
                            successful_only=successful_only,
                            final_output_only=final_output_only,
                        ),
                        None,
                    )
                    return (
                        relevance_search_history.relevance_search_output.to_relevance_search_output()
                        if relevance_search_history is not None
                        else None
                    )
            elif isinstance(input, SearchToolInput | SearchInput):
                async with self.get_or_create_session(session) as initialized_session:
                    search_history = next(
                        self._retrieve_search_history(
                            initialized_session,
                            input,
                            ttl=ttl,
                            successful_only=successful_only,
                            final_output_only=final_output_only,
                        ),
                        None,
                    )
                    return search_history.search_output.to_search_output() if search_history is not None else None
            else:
                raise TypeError(
                    f"Expected a valid input type but instead received an input of type {type(input).__name__}"
                )
        except HistoryCacheInitializationException:
            raise
        except Exception as e:
            msg = f"An error occurred during search output history retrieval: {str(e)}"
            raise_exception_type = HistoryCacheRetrievalException if raise_on_error else None
            self._handle_exception(e, raise_exception_type=raise_exception_type, message=msg)
        return None

    async def retrieve_recent_history(
        self,
        research_tool_type: ResearchToolType = "all",
        raise_on_error: bool | None = None,
        *,
        session: sqlmodel.Session | None = None,
        input_hash: str | None = None,
        max_history: int | None = None,
        ttl: int | float | None = None,
        successful_only: bool = False,
        final_output_only: bool | None = True,
        topic: str | None = None,
        similarity_threshold: int | float | None = None,
    ) -> Sequence[history_models.ResearchHistoryExecution]:
        """Returns a list of all recorded relational outputs, sorted by creation date and filtered by TTL if provided.

        Args:
            research_tool_type: (Literal["search", "record_search", "relevance", "relevance_search", "research_synthesis", "synthesis", "all"]):
                The tool type to retrieve recent history for.
            raise_on_error (Optional[bool]):
                Determines whether an error should be raised when encountering an exception during history retrieval.
            session (sqlmodel.Session | None):
                An optional session used to retrieve previously stored history records. If not provided, a new session
                is initialized instead.
            input_hash (str | None):
                An optional input hash used to filter results.
            max_history (int| None):
                The total number of history output items to return in the final sequence of relations.
            ttl (int | float | None):
                The TTL (time to live) measured in seconds. When None, all records are returned, regardless of recency
            successful_only (bool):
                Indicates whether only successfully processed tool calls should be returned. Tool call success is
                defined by the `successful` property on each output class.
            final_output_only (bool | None):
                Indicates whether output history should only be retrieved when outputted from its corresponding tool.
            topic (str | None):
                An optional topic string used to filter and sort history items by fuzzy similarity. When provided,
                results are filtered to those exceeding `similarity_threshold` and ordered by similarity score.
            similarity_threshold (int | float | None):
                The minimum fuzzy similarity score (0.0–1.0) required for a history item to be included. Only
                applied when `topic` is provided.

        Returns:
            Sequence[SearchExecution | RelevanceSearchExecution | SynthesisExecution]:
                A list of valid research tool output executions when available and fresh given the TTL. Otherwise None.

        """
        ttl = self._validate_ttl(ttl if ttl is not None else self.ttl)
        raise_on_error = raise_on_error if raise_on_error is not None else self.raise_on_error

        recent_history_list: list[history_models.ResearchHistoryExecution] = []

        try:
            if research_tool_type not in RESEARCH_TOOL_TYPES:
                raise TypeError(f"Expected a valid tool type but instead received: {research_tool_type!r}")
            if input_hash is not None and not isinstance(input_hash, str):
                raise TypeError(f"Expected a valid input hash, but instead received: {type(input_hash).__name__}")
            if max_history is not None and not (isinstance(max_history, int) and max_history >= 0):
                err_class = ValueError if isinstance(max_history, int) else TypeError
                raise err_class(
                    f"Expected `max_history` to be a positive integer, but instead received: '{max_history}'"
                )

            async with self.get_or_create_session(session) as initialized_session:
                if research_tool_type in ("all", "research_synthesis", "synthesis"):
                    recent_history_iterator: Iterator[history_models.ResearchHistoryExecution] = (
                        self._retrieve_synthesis_history(
                            initialized_session,
                            input=input_hash,
                            ttl=ttl,
                            successful_only=successful_only,
                            final_output_only=final_output_only,
                        )
                    )
                    if max_history is not None and not topic:
                        recent_history_iterator = itertools.islice(recent_history_iterator, max_history)
                    recent_history_list += list(recent_history_iterator)

                if research_tool_type in ("all", "relevance", "relevance_search"):
                    recent_history_iterator = self._retrieve_relevance_search_history(
                        initialized_session,
                        ttl=ttl,
                        input=input_hash,
                        successful_only=successful_only,
                        final_output_only=final_output_only,
                    )

                    if max_history is not None and not topic:
                        recent_history_iterator = itertools.islice(recent_history_iterator, max_history)
                    recent_history_list += list(recent_history_iterator)

                if research_tool_type in ("all", "search", "record_search"):
                    recent_history_iterator = self._retrieve_search_history(
                        initialized_session,
                        ttl=ttl,
                        input=input_hash,
                        successful_only=successful_only,
                        final_output_only=final_output_only,
                    )

                    if max_history is not None and not topic:
                        recent_history_iterator = itertools.islice(recent_history_iterator, max_history)
                    recent_history_list += list(recent_history_iterator)

                if topic:
                    recent_history_list = self.filter_fuzzy_similarity(
                        topic=topic, history_items=recent_history_list, similarity_threshold=similarity_threshold
                    )
                else:
                    recent_history_list.sort(key=lambda history_item: history_item.stored_at, reverse=True)

                return recent_history_list[:max_history] if max_history is not None else recent_history_list
        except HistoryCacheInitializationException:
            raise
        except RapidFuzzImportError:
            raise
        except Exception as e:
            msg = f"An error occurred during recent output history retrieval: {str(e)}"
            raise_exception_type = HistoryCacheRetrievalException if raise_on_error else None
            self._handle_exception(e, raise_exception_type=raise_exception_type, message=msg)

        return []

    @classmethod
    def filter_fuzzy_similarity(
        cls,
        history_items: Sequence[SupportsTopicSimilarity] | Iterator[SupportsTopicSimilarity],
        topic: str,
        similarity_threshold: int | float | None,
        sort: bool = True,
        parallel_threshold: int | None = 100,
        max_workers: int = 4,
    ) -> list[SupportsTopicSimilarity]:
        """Calculates the partial similarity between a user-specified topic and an input, stored output, or record.

        This method uses multiprocessing for large history lists to improve performance where beneficial.

        Args:
            history_items (Sequence[SupportsTopicSimilarity] | Iterator[SupportsTopicSimilarity]):
                A sequence or iterator of history items with topic fields. Supported items include
                ResearchToolInput, ResearchHistoryOutput, SearchRecord, SearchRecordHistory instances.
            topic (str):
                The topic string to compare against history items.
            similarity_threshold (int | float | None):
                The minimum fuzzy similarity score (0.0–1.0) required for inclusion.
            sort (bool):
                Whether to sort results by similarity score descending (default: True).
            parallel_threshold (int | None):
                Threshold at which to use multiprocessing for retrieval.
            max_workers (int):
                The maximum number of workers to used to perform a fuzzy similarity search via multiprocessing.

        Returns:
            list[SupportsTopicSimilarity]: Filtered and optionally sorted history items by topic relevance.

        """
        PartialRatioSimilarity.validate_dependency()  # verifies whether the rapidfuzz dependency is available
        # Convert iterator to list for multiprocessing compatibility
        history_list = history_items if isinstance(history_items, list) else list(history_items)

        fuzzy_similarity_func = partial(
            _calculate_fuzzy_topic_similarity_worker,
            topic=topic,
            similarity_threshold=similarity_threshold,
        )
        history_items_by_topic_similarity: (
            Iterator[tuple[SupportsTopicSimilarity, PartialRatioSimilarity]]
            | Sequence[tuple[SupportsTopicSimilarity, PartialRatioSimilarity]]
        )

        # Default to single-core processing when workers is <= 1 or `max_workers` is not an integer
        workers = min(mp.cpu_count(), max_workers) if isinstance(max_workers, int) else 1

        # Use multiprocessing threshold: only parallelize for large lists
        if workers <= 1 or (parallel_threshold is not None and len(history_list) < parallel_threshold):
            history_items_by_topic_similarity = (fuzzy_similarity_func(history_item) for history_item in history_list)

        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Parallel processing with multiprocessing
                history_items_by_topic_similarity = executor.map(fuzzy_similarity_func, history_list)

        history_items_by_topic_similarity = (
            history_item for history_item in history_items_by_topic_similarity if history_item[1].exceeds_threshold
        )

        # Filter out None results and sort if requested
        if sort:
            history_items_by_topic_similarity = sorted(
                history_items_by_topic_similarity, key=lambda item: item[1].score, reverse=True
            )

        return [history_item for history_item, _ in history_items_by_topic_similarity]

    @classmethod
    def _retrieve_records(
        cls,
        session: sqlmodel.Session,
        record_hash: str | None = None,
        ttl: int | float | None = None,
        unique_only: bool = True,
    ) -> Iterator[history_models.SearchRecordHistory]:
        """Extracts a record from the output history by record hash."""
        ttl = cls._validate_ttl(ttl)
        if record_hash is not None and not isinstance(record_hash, str):
            raise TypeError(
                "The history service expected a string record hash to search valid records, but instead received "
                f"type {type(record_hash)}."
            )
        stmt = (
            sqlmodel.select(history_models.SearchRecordHistory)
            .join(history_models.SearchRecordElement)
            .join(history_models.SearchOutputHistory)
        )

        if record_hash:
            stmt = stmt.where(history_models.SearchRecordHistory.record_hash == record_hash)

        stmt = stmt.order_by(sqlmodel.col(history_models.SearchOutputHistory.stored_at).desc())

        record_results = session.exec(stmt)

        seen: set[str] = set()  # Used to ensure unique records are returned unless `unique_only=False`.
        for record in record_results:
            unexpired_records = ttl is None or not record.search_output.is_expired(ttl)
            if unexpired_records and (not unique_only or record.record_hash not in seen):
                seen.add(record.record_hash)
                yield record

    async def retrieve_record_history(
        self,
        raise_on_error: bool | None = None,
        *,
        session: sqlmodel.Session | None = None,
        record_hash: str | None = None,
        max_history: int | None = None,
        ttl: int | float | None = None,
        topic: str | None = None,
        similarity_threshold: int | float | None = None,
    ) -> Sequence[SearchRecord]:
        """Returns a list of all retrieved records matching a particular query or record hash.

        Args:
            raise_on_error (Optional[bool]):
                Determines whether an error should be raised when encountering an exception during history retrieval.
            session (sqlmodel.Session | None):
                An optional session used to retrieve previously stored history records. If not provided, a new session
                is initialized instead.
            record_hash (str | None):
                An optional input record hash used to filter results.
            max_history (int| None):
                The total number of history output items to return in the final sequence of tool execution outputs.
            ttl (int | float | None):
                The TTL (time to live) measured in seconds. When None, all records are returned, regardless of recency.
            topic (str | None):
                An optional topic string used to filter and sort records by fuzzy similarity. When provided, results
                are filtered to those exceeding `similarity_threshold` and ordered by similarity score descending.
            similarity_threshold (int | float | None):
                The minimum fuzzy similarity score (0.0–1.0) required for a record to be included. Only applied
                when `topic` is provided.

        Returns:
            Sequence[SearchRecord]:
                A list of valid records retrieved from the history database.

        """
        ttl = self._validate_ttl(ttl if ttl is not None else self.ttl)
        raise_on_error = raise_on_error if raise_on_error is not None else self.raise_on_error

        try:
            if max_history is not None and not (isinstance(max_history, int) and max_history >= 0):
                err_class = ValueError if isinstance(max_history, int) else TypeError
                raise err_class(
                    f"Expected `max_history` to be a positive integer, but instead received: '{max_history}'"
                )

            async with self.get_or_create_session(session) as initialized_session:
                recent_history_iterator: Iterator[history_models.SearchRecordHistory] = self._retrieve_records(
                    initialized_session,
                    record_hash=record_hash,
                    ttl=ttl,
                )
                if max_history is not None and not topic:
                    recent_history_iterator = itertools.islice(recent_history_iterator, max_history)

                recent_history_tuple = (record.to_search_record() for record in recent_history_iterator)

                if topic:
                    filtered_record_history = self.filter_fuzzy_similarity(
                        recent_history_tuple, topic, similarity_threshold
                    )
                    return filtered_record_history[:max_history] if max_history is not None else filtered_record_history

                return list(recent_history_tuple)

        except HistoryCacheInitializationException:
            raise
        except RapidFuzzImportError:
            raise
        except Exception as e:
            msg = f"An error occurred during recent output history retrieval: {str(e)}"
            raise_exception_type = HistoryCacheRetrievalException if raise_on_error else None
            self._handle_exception(e, raise_exception_type=raise_exception_type, message=msg)

        return []

    @classmethod
    def _store_synthesis_history(
        cls, output: SynthesisOutput | history_models.SynthesisExecution, session: sqlmodel.Session
    ) -> None:
        """Stores the `SynthesisOutput` or constructed `SynthesisExecution` model generated after research synthesis."""
        try:
            synthesis_execution = (
                output
                if isinstance(output, history_models.SynthesisExecution)
                else history_models.SynthesisExecution.from_research_synthesis(output.synthesis_input, output)
            )
            merged = session.merge(synthesis_execution)
            session.commit()
            session.refresh(merged)
        except exc.SQLAlchemyError as e:
            session.rollback()
            msg = f"An error occurred during synthesis output history storage {str(e)}"
            cls._handle_exception(e, raise_exception_type=HistoryCacheStorageException, message=msg)

    @classmethod
    def _store_relevance_search_history(
        cls, output: RelevanceSearchOutput | history_models.RelevanceSearchExecution, session: sqlmodel.Session
    ) -> None:
        """Stores the `RelevanceSearchOutput` or constructed `RelevanceSearchExecution` model."""
        try:
            relevance_search_execution = (
                output
                if isinstance(output, history_models.RelevanceSearchExecution)
                else history_models.RelevanceSearchExecution.from_record_relevance_search(
                    output.relevance_search_input, output
                )
            )
            merged = session.merge(relevance_search_execution)
            session.commit()
            session.refresh(merged)
        except exc.SQLAlchemyError as e:
            session.rollback()
            msg = f"An error occurred during relevance search output history storage {str(e)}"
            cls._handle_exception(e, raise_exception_type=HistoryCacheStorageException, message=msg)

    @classmethod
    def _store_search_history(
        cls, output: SearchOutput | history_models.SearchExecution, session: sqlmodel.Session
    ) -> None:
        """Stores the `SearchOutput` or constructed `SearchExecution` model generated after record search."""
        try:
            search_execution = (
                output
                if isinstance(output, history_models.SearchExecution)
                else history_models.SearchExecution.from_record_search(output.search_input, output)
            )
            merged = session.merge(search_execution)
            session.commit()
            session.refresh(merged)
        except exc.SQLAlchemyError as e:
            session.rollback()
            msg = f"An error occurred during search output history storage {str(e)}"
            cls._handle_exception(e, raise_exception_type=HistoryCacheStorageException, message=msg)

    async def store(
        self,
        output: ResearchHistoryOutput,
        raise_on_error: bool | None = None,
        *,
        session: sqlmodel.Session | None = None,
    ) -> None:
        """Stores a valid output or constructed relation via SQLModel.

        Args:
            output (SearchOutput | SearchExecution | RelevanceSearchOutput | RelevanceSearchExecution | SynthesisOutput | SynthesisExecution):
                A valid output or relation to store within the current relational database via SQLModel.
            raise_on_error (Optional[bool]):
                Determines whether an error should be raised when encountering an exception during history storage.
            session (sqlmodel.Session | None):
                An optional session used to retrieve previously stored history records. If not provided, a new session
                is initialized instead.

        """
        raise_on_error = raise_on_error if raise_on_error is not None else self.raise_on_error
        try:
            if isinstance(output, SynthesisOutput | history_models.SynthesisExecution):
                async with self.get_or_create_session(session) as initialized_session:
                    self._store_synthesis_history(output, initialized_session)
            elif isinstance(output, RelevanceSearchOutput | history_models.RelevanceSearchExecution):
                async with self.get_or_create_session(session) as initialized_session:
                    self._store_relevance_search_history(output, initialized_session)
            elif isinstance(output, SearchOutput | history_models.SearchExecution):
                async with self.get_or_create_session(session) as initialized_session:
                    self._store_search_history(output, initialized_session)
            else:
                raise TypeError(
                    f"Expected a valid output type for history cache storage but instead received type {type(output).__name__}"
                )
        except HistoryCacheInitializationException:
            raise
        except Exception as e:
            msg = f"An error occurred during output history storage: {str(e)}"
            raise_exception_type = HistoryCacheStorageException if raise_on_error else None
            self._handle_exception(e, raise_exception_type=raise_exception_type, message=msg)
        return None

    async def clear(self, raise_on_error: bool | None = None, *, session: sqlmodel.Session | None = None) -> bool:
        """Delete all cached output tables retrieved via ScholarFluxMCP."""
        raise_on_error = raise_on_error if raise_on_error is not None else self.raise_on_error
        try:
            async with self.get_or_create_session(session) as initialized_session:
                for table in history_models.HISTORY_TABLES:
                    initialized_session.exec(sqlmodel.delete(table))
                initialized_session.commit()
            return True
        except Exception as e:
            msg = f"Could not successfully delete all cache output tables for the HistoryService: {e}"
            raise_exception_type = HistoryCacheDeletionException if raise_on_error else None
            self._handle_exception(e, raise_exception_type=raise_exception_type, message=msg)
        return False

    def verify_connection(self) -> None:
        """Verifies that the HistoryService is available for connection with initialized configuration settings."""
        try:
            self.ping(self.engine)
        except HistoryCacheInitializationException:
            raise
        except Exception as e:
            msg = f"Could not initialize a connection for the HistoryService: {e}"
            self._handle_exception(e, raise_exception_type=HistoryCacheConnectionFailed, message=msg)

    @classmethod
    def verify_url_string(cls, url: str) -> None:
        """Helper method for verifying that the current URI has a valid SQLModel resource identifier."""
        if not isinstance(url, str):
            raise HistoryCacheParameterValidationException(
                f"Expected a valid SQLModel URI, but received type {type(url)}"
            )

        url_case = url.lower()
        if (
            URI_SCHEMA_PATTERN.search(url_case) is None
            or (url_case.startswith("duckdb:") and not url_case.startswith("duckdb:///"))
            or (url_case.startswith("sqlite:") and not url_case.startswith("sqlite:///"))
        ):
            raise HistoryCacheParameterValidationException(
                "Only URIs with valid SQL protocols are supported (e.g., postgres://, sqlite:///, duckdb:///, etc.). "
                f"Received: '{url}'"
            )

        result = urlparse(url)

        # If the path is non-empty, then remove special characters after the scheme
        if path := coerce_str(result.path):
            path = path.strip(":/ ")
        if not path:
            raise HistoryCacheParameterValidationException(
                f"Expected a path after the protocol in the SQLModel URI. Only the scheme was received: {url}"
            )

    @classmethod
    def create_default_url(cls) -> str:
        """Creates a default URL that persists a SQLite history database within memory."""
        return "sqlite:///:memory:"

    @classmethod
    def get_default_url(cls) -> str:
        """Retrieves the SQLModel URL from the environment configuration, falling back to the default when invalid.

        Returns:
            str:
                The validated URL from the environment configuration if valid. Otherwise the default URL generated via
                `cls.create_default_url()`.

        Note: This method first attempts to validate the URL string from the environment variable,
        `SCHOLAR_FLUX_MCP_HISTORY_URL`, using the `cls.verify_url_string` class method. When validation fails, the
        default for the current class is returned via `cls.create_default_url` instead.

        """
        config_url = try_none(os.getenv("SCHOLAR_FLUX_MCP_HISTORY_URL"))
        if config_url:
            try:
                cls.verify_url_string(config_url)
                return config_url
            except HistoryCacheParameterValidationException:
                storage = "SQLModel" if cls.STORAGE_TYPE == "SQL" else cls.STORAGE_TYPE
                logger.info(
                    f"The environment variable, SCHOLAR_FLUX_MCP_HISTORY_URL, is not a valid {storage} URL. "
                    f"Returning the default..."
                )

        return cls.create_default_url()

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Get default configuration with current config_settings values.

        Returns:
            dict: A dictionary configuration with the default URL and `echo` (for debugging SQL statements).

        """
        url = cls.DEFAULT_CONFIG.get("url")
        return {
            "url": url if callable(url) else lambda: url,
            "echo": cls.DEFAULT_CONFIG.get("echo") or False,
        }

    @classmethod
    def _validate_ttl(cls, ttl: int | float | str | None = None) -> int | float | None:
        """Validates TTL values, converting them into floats when possible and not already a non-negative numeric value.

        Args:
            ttl (Optional[int | float | str]):
                The value to return as a non-negative integer or float when possible. None values are returned as is.

        Returns:
            int | float | None:
                int: The original non-negative integer value if received
                float: A non-negative float, either received as such or converted from a string value
                None: TTLs are returned as None if None is received or the ttl is equal to -1 or -1.0

        Raises:
            HistoryCacheParameterValidationException: for values that are not None and cannot be converted into a
            non-negative float. While negative numbers are invalid, `-1` is the exception, signifying that TTL-based
            cache expiration should not be used.

        """
        if ttl is None or isinstance(ttl, float | int) and ttl >= 0:
            return ttl

        ttl_numeric = coerce_numeric(ttl)
        no_ttl = ttl == -1 or ttl_numeric == -1
        if ttl_numeric is None or ttl_numeric < 0 and not no_ttl:
            raise HistoryCacheParameterValidationException(
                f"The HistoryService expected the TTL to be a non-negative number, `None`, or -1 (no expiration), but "
                f"received an invalid value ({ttl})"
            )
        return None if no_ttl else ttl_numeric

    @classmethod
    def _handle_exception(
        cls, exception: BaseException, raise_exception_type: type[BaseException] | None = None, message: str = ""
    ) -> None:
        """Logs the current error, raising the provided exception type when specified.

        Args:
            exception (BaseException): The exception instance raised from the last storage cache operation
            raise_exception_type (Optional[type[BaseException]]): The exception to raise
            message (str): The error message to log and/or raise.

        """
        error_message = message or str(exception)
        logger.error(error_message)
        if raise_exception_type is not None:
            raise raise_exception_type(error_message) from exception

    async def check_health(self) -> ServiceHealth:
        """Helper used to check the health status of the HistoryService."""
        error: str | None = None

        try:
            if not sqlmodel or not history_models:
                raise SQLModelImportError()
            if self.enabled:
                self.verify_connection()
        except Exception as e:
            error = str(e)

        health_status = "unhealthy" if error else ("healthy" if self.enabled else "disabled")

        details = await self.get_stats()

        if error:
            details["error"] = error

        return ServiceHealth(
            name="history",
            status=health_status,
            details=details,
            error=error,
        )

    async def get_stats(self) -> dict[str, Any]:
        """Show the current statistics and configuration of the history service.

        Returns:
            dict: Cache configuration including backends, namespace, ttl, and status.

        """
        stats: dict[str, Any] = {
            "initialized": self.initialized,
            "ttl": self.ttl,
            "echo": self.config.get("echo", False),
            "raise_on_error": self.raise_on_error,
        }

        # If scholar-flux is available, show what masked path would be used to access the history cache
        if masker is not None and (url := self.config.get("url")):
            stats["url"] = masker.mask_object(url)

        return stats

    @classmethod
    def ping(cls, engine: Engine) -> None:
        """Verifies that the client can successfully connect to the database."""
        with engine.connect():
            pass

    @classmethod
    def is_available(cls, url: str | None = None, verbose: bool = True) -> bool:
        """Tests whether the SQL service can be accessed.

        Args:
            url (str): Indicates the location to attempt a connection
            verbose (bool): Indicates whether to log at the levels, DEBUG and lower, or to log warnings only

        Returns:
            bool: True if the SQL service is accessible and false otherwise.

        """
        db_url: str = url or cls.get_default_config()["url"]()
        try:
            engine = create_engine(url=db_url)
            cls.ping(engine)

            if verbose:
                logger.info(f"The {cls.STORAGE_TYPE} Service is available at {db_url}")
            return True

        except (exc.SQLAlchemyError, TimeoutError, ConnectionError) as e:
            logger.warning(f"An active {cls.STORAGE_TYPE} service could not be found at {db_url}: {e}")
            return False

    @classmethod
    def calculate_fuzzy_topic_similarity(
        cls,
        history_item: SupportsTopicSimilarity,
        topic: str,
        similarity_threshold: int | float | None = None,
    ) -> PartialRatioSimilarity:
        """Convenience method for calculating the similarity of the item to the current topic.

        Args:
            history_item (ResearchToolInput | ResearchHistoryOutput | SearchRecord | SearchRecordHistory):
                A research tool input, output, or record to calculate fuzzy string similarity to the topic.
            topic (str | None):
                An optional topic string used to filter and sort records by fuzzy similarity. When provided, results
                are filtered to those exceeding `similarity_threshold` and ordered by similarity score descending.
            similarity_threshold (int | float | None):
                The minimum fuzzy similarity score (0.0–1.0) required for a record to be included. Only applied
                when `topic` is provided.

        Returns:
            tuple[SupportsTopicSimilarity, PartialRatioSimilarity]:
                A  `PartialRatioSimilarity` result indicating the similarity between the history item and the topic.

        """
        similarity_result = _calculate_fuzzy_topic_similarity_worker(
            history_item, topic=topic, similarity_threshold=similarity_threshold
        )
        return similarity_result[1]


__all__ = ["HistoryService"]

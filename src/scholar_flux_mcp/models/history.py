"""SQLModel classes for storing and retrieving ScholarFlux MCP search and synthesis results."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any, ClassVar, Optional

from pydantic import ValidationError
from sqlalchemy import Index
from sqlalchemy.orm import Mapped
from sqlmodel import JSON, Column, Relationship, SQLModel
from sqlmodel import Field as SQLField
from typing_extensions import Self, TypeAliasType

from scholar_flux_mcp.models.core import TimeStampString
from scholar_flux_mcp.models.enums import APIProviders, ResearchCategory
from scholar_flux_mcp.models.schemas import (
    AgentEvidenceItem,
    EvidenceGroundingOutput,
    EvidenceGroundingStats,
    GroundedEvidenceItem,
    IndexedSearchRecord,
    PaginationInfo,
    RecordTopicSimilarity,
    RecordTopicSimilarityOutput,
    RejectedEvidenceItem,
    RelevanceSearchInput,
    RelevanceSearchOutput,
    ResearchTopic,
    SearchInput,
    SearchOutput,
    SearchRecord,
    SearchRecordEmbedding,
    SearchResponseSummary,
    SynthesisAgentOutput,
    SynthesisInput,
    SynthesisOutput,
    TopicEmbedding,
)
from scholar_flux_mcp.server.io import RelevanceSearchToolInput, SearchToolInput, SynthesisToolInput
from scholar_flux_mcp.utils.helpers import generate_datetime, generate_iso_timestamp, parse_iso_timestamp


class SearchRecordElement(SQLModel, table=True):
    """SQL Model for storing normalized academic records retrieved from academic APIs.

    Contains standardized fields across all providers after ScholarFlux normalization.

    """

    __tablename__ = "search_record_elements"

    record_hash: str = SQLField(
        description="A stable md5 hash of the current record.",
        primary_key=True,
        nullable=False,
        index=True,
        unique=True,
    )

    record_history: list["SearchRecordHistory"] = Relationship(
        back_populates="record",
    )
    record_embedding: list["RecordEmbeddingHistory"] = Relationship(
        back_populates="record",
    )
    rejected_evidence_item: list["RejectedEvidenceItemHistory"] = Relationship(
        back_populates="record",
    )
    grounded_evidence_items: list["GroundedEvidenceItemHistory"] = Relationship(
        back_populates="record",
    )
    indexed_record_history: list["IndexedSearchRecordHistory"] = Relationship(
        back_populates="record",
    )

    provider_name: str = SQLField(description="Source provider (e.g., 'pubmed', 'crossref')")
    page: int = SQLField(description="The page number associated with the record at retrieval time.")
    query: str = SQLField(description="The query used to retrieve the current record")
    doi: Optional[str] = SQLField(default=None, description="Digital Object Identifier")
    url: Optional[str] = SQLField(default=None, description="The URL link to the academic record")
    record_id: Optional[str] = SQLField(default=None, description="Provider-specific record ID")
    title: Optional[str] = SQLField(default=None, description="Academic record title")
    abstract: Optional[str] = SQLField(default=None, description="Academic record abstract")
    full_text: Optional[str] = SQLField(default=None, description="Academic record text if available")
    authors: Optional[list[str] | str] = SQLField(default=None, sa_column=Column(JSON), description="Author names")
    journal: Optional[str] = SQLField(default=None, description="Journal name")
    publisher: Optional[str] = SQLField(default=None, description="Publisher name")
    year: Optional[int] = SQLField(default=None, description="Publication year")
    date_published: Optional[str] = SQLField(default=None, description="Publication date")
    keywords: Optional[list[str] | str] = SQLField(
        default=None, sa_column=Column(JSON), description="Keywords/MeSH terms"
    )
    subjects: Optional[list[str] | str] = SQLField(
        default=None, sa_column=Column(JSON), description="Subject categories"
    )
    citation_count: Optional[int] = SQLField(default=None, description="Number of citations")
    open_access: Optional[bool] = SQLField(default=None, description="Open access status")
    cached: Optional[bool] = SQLField(default=None, description="Indicates whether the record was retrieved from cache")
    retrieval_timestamp: Optional[TimeStampString] = SQLField(default=None, description="When the record was retrieved")

    @classmethod
    def from_search_record(cls, record: SearchRecord, **kwargs: Any) -> Self:
        """Creates a SearchRecordElement from a SearchRecord model defined within the core schema."""
        record_data = record.model_dump()
        return cls(
            **record_data,
            **kwargs,
        )

    def to_search_record(self) -> SearchRecord:
        """Converts this record into a SearchRecord model defined within the core schema."""
        parsed_timestamp = parse_iso_timestamp(self.retrieval_timestamp) if self.retrieval_timestamp else None

        return SearchRecord(
            provider_name=self.provider_name,
            page=self.page,
            query=self.query,
            doi=self.doi,
            url=self.url,
            record_id=self.record_id,
            title=self.title,
            full_text=self.full_text,
            abstract=self.abstract,
            authors=self.authors,
            journal=self.journal,
            publisher=self.publisher,
            year=self.year,
            date_published=self.date_published,
            keywords=self.keywords,
            subjects=self.subjects,
            citation_count=self.citation_count,
            open_access=self.open_access,
            cached=self.cached,
            retrieval_timestamp=parsed_timestamp,
        )


class ResponseSummaryElement(SQLModel, table=True):
    """Response summary generated from a single response query and page received from an API provider."""

    __tablename__ = "response_summary_elements"

    response_hash: str = SQLField(
        description="A stable md5 hash of the current response summary.",
        primary_key=True,
        nullable=False,
        index=True,
        unique=True,
    )

    response_summary_history: list["ResponseSummaryHistory"] = Relationship(
        back_populates="response_summary",
    )

    provider_name: str = SQLField(description="The academic API provider.")
    query: str = SQLField(description="The query used for this response.")
    page: int = SQLField(description="The page number retrieved.")
    success: bool = SQLField(description="Whether a valid response was received, independent of record count.")
    status_code: int | None = SQLField(description="The status code associated with the current response.")
    error: str | None = SQLField(default=None, description="Error associated with the current response.")
    message: str | None = SQLField(default=None, description="The message associated with the current response")
    cached: bool | None = SQLField(default=None, description="Whether the response was served from session cache.")
    retrieval_timestamp: TimeStampString | None = SQLField(default=None, description="When the response was retrieved.")
    record_count: int = SQLField(default=0, description="The Number of records returned by this response.")

    @classmethod
    def from_response_summary(cls, response_summary: SearchResponseSummary, **kwargs: Any) -> Self:
        """Creates a ResponseSummaryElement from a SearchResponseSummary model defined within the core schema."""
        return cls(
            response_hash=response_summary.response_hash,
            provider_name=response_summary.provider_name,
            query=response_summary.query,
            page=response_summary.page,
            success=response_summary.success,
            status_code=response_summary.status_code,
            error=response_summary.error,
            message=response_summary.message,
            record_count=response_summary.record_count,
            cached=response_summary.cached,
            retrieval_timestamp=response_summary.retrieval_timestamp,
            **kwargs,
        )

    def to_response_summary(self) -> SearchResponseSummary:
        """Creates a SearchResponseSummary that is reconstructed from the current ResponseSummaryElement."""
        parsed_timestamp = parse_iso_timestamp(self.retrieval_timestamp) if self.retrieval_timestamp else None
        return SearchResponseSummary(
            provider_name=self.provider_name,
            query=self.query,
            page=self.page,
            success=self.success,
            status_code=self.status_code,
            error=self.error,
            message=self.message,
            record_count=self.record_count,
            cached=self.cached,
            retrieval_timestamp=parsed_timestamp,
        )


class ResponseSummaryHistory(SQLModel, table=True):
    """SQL Model for storing summaries of responses retrieved from academic APIs.

    This model maps each unique `response_hash` to a `ResponseSummaryHistory`, ensuring that the relation
    between an output and a record is clearly defined.

    """

    __tablename__ = "response_summary_history"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    response_hash: str = SQLField(
        description="A stable md5 hash of the current response.",
        foreign_key="response_summary_elements.response_hash",
        nullable=False,
    )
    response_summary: Mapped[ResponseSummaryElement] = Relationship(
        back_populates="response_summary_history",
        sa_relationship_kwargs={"foreign_keys": "[ResponseSummaryHistory.response_hash]"},
    )
    search_output_id: Optional[int] = SQLField(
        description="ID corresponding to the SearchOutput that retrieved the response.",
        foreign_key="search_outputs.id",
    )
    search_output: Mapped["SearchOutputHistory"] = Relationship(
        back_populates="response_summaries",
    )

    @classmethod
    def from_response_summary(cls, response_summary: SearchResponseSummary, **kwargs: Any) -> Self:
        """Creates a ResponseSummaryHistory from a SearchResponseSummary model defined within the core schema."""
        response_summary_element = ResponseSummaryElement.from_response_summary(response_summary)
        return cls(
            response_hash=response_summary_element.response_hash,
            response_summary=response_summary_element,
            **kwargs,
        )

    def to_response_summary(self) -> SearchResponseSummary:
        """Converts the SQLModel into a `SearchResponseSummary` instance."""
        return self.response_summary.to_response_summary()


class SearchRecordHistory(SQLModel, table=True):
    """SQL Model for storing normalized academic records retrieved from academic APIs.

    This model maps each unique `record_hash` to a `SearchOutputHistory`, ensuring that the relation
    between an output and a record is clearly defined.

    """

    __tablename__ = "search_record_history"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    record_hash: str = SQLField(
        description="A stable md5 hash of the current record.",
        foreign_key="search_record_elements.record_hash",
        nullable=False,
    )
    record: Mapped[SearchRecordElement] = Relationship(
        back_populates="record_history",
        sa_relationship_kwargs={"foreign_keys": "[SearchRecordHistory.record_hash]"},
    )
    search_output_id: Optional[int] = SQLField(
        description="ID corresponding to the SearchOutput that retrieved the record.",
        foreign_key="search_outputs.id",
    )
    search_output: Mapped["SearchOutputHistory"] = Relationship(
        back_populates="records",
    )

    @classmethod
    def from_search_record(cls, record: SearchRecord, **kwargs: Any) -> Self:
        """Creates a SearchRecordHistory from a SearchRecord model defined within the core schema."""
        record_element = SearchRecordElement.from_search_record(record)
        return cls(
            record_hash=record_element.record_hash,
            record=record_element,
            **kwargs,
        )

    def to_search_record(self) -> SearchRecord:
        """Converts the SQLModel into a `SearchRecord` instance."""
        return self.record.to_search_record()

    @cached_property
    def topic(self) -> str:
        """Identifies the topic of the current record from relevant bibliographic fields."""
        return self.to_search_record().topic


class BaseHistory(SQLModel, table=False):
    """Base SQL Model for recording timestamps and verifying cache expiration."""

    DATETIME_COMPARISON_FMT: ClassVar[str] = "%Y-%m-%d %H:%M:%S"

    id: Optional[int] = SQLField(default=None, primary_key=True, nullable=False, index=True)
    stored_at: TimeStampString = SQLField(
        default_factory=generate_iso_timestamp,
        nullable=False,
        index=True,
        description="Timestamp indicating when the SQLModel was stored",
    )

    @classmethod
    def _calculate_expiration_timestamp(cls, stored_at: str, ttl: int | float) -> datetime:
        """Calculates the time until expiration for the current ISO formatted timestamp."""
        timestamp = parse_iso_timestamp(stored_at)

        if timestamp is None:
            raise ValueError(
                f"Could not parse the `{cls.__name__}.stored_at` value ({stored_at}) as a valid timestamp."
            )
        return timestamp + timedelta(seconds=ttl)

    def is_expired(self, ttl: int | float) -> bool:
        """Indicates whether the current record has expired given the user-specified ttl."""
        if not isinstance(ttl, int | float):
            raise TypeError(
                f"Expected the provided `ttl` to be an int or float, but received type {type(ttl).__name__}"
            )

        expiration_timestamp = self._calculate_expiration_timestamp(self.stored_at, ttl)
        return expiration_timestamp < generate_datetime()


class BaseInputHistory(BaseHistory, table=False):
    """Base SQL Model for shared input history fields between search and synthesis operations."""

    input_hash: str = SQLField(
        nullable=False,
        index=True,
        description="Hash of the source input record for query resolution and history retrieval",
    )

    queries: list[str] = SQLField(
        sa_column=Column(JSON),
        description="The search query used to retrieve records from academic APIs.",
    )

    providers: list[str] = SQLField(
        sa_column=Column(JSON), description="The set of providers used to retrieve academic records"
    )

    max_records: int = SQLField(
        description="Maximum records that should be retrieved and/or analyzed per provider",
        ge=1,
        le=200,
    )
    pages: int = SQLField(
        description="The total number pages retrieved per provider",
        ge=1,
    )
    page_offset: int = SQLField(
        description="Indicates the number of pages from page 1 where retrieval begins.",
        ge=0,
    )
    year_from: Optional[int] = SQLField(
        description="Records filtered on or after this year",
        ge=1900,
        le=2100,
    )
    year_to: Optional[int] = SQLField(
        description="Records filtered on or before this year",
        ge=1900,
        le=2100,
    )
    open_access_only: bool = SQLField(
        description="Indicates that the search only retrieved open-access records",
    )

    @property
    def topic(self) -> str:
        """Identifies the research topic from core parameters available within the current SQLModel."""
        try:
            return ResearchTopic.create(self).topic
        except ValidationError:
            return ""


class BaseOutputHistory(BaseHistory, table=False):
    """Base SQL Model for shared output history fields between search and synthesis operations."""

    pass


class SearchInputHistory(BaseInputHistory, table=True):
    """Dedicated SQL Model for resolving SearchToolInput history against its output."""

    __tablename__ = "search_inputs"
    __table_args__ = (Index("ix_search_inputs_stored_at_id", "stored_at", "input_hash", "id"),)

    search_execution: Mapped[Optional["SearchExecution"]] = Relationship(
        back_populates="search_input",
        sa_relationship_kwargs={"uselist": False},
    )

    @classmethod
    def from_search_input(cls, input: SearchToolInput | SearchInput, **kwargs: Any) -> Self:
        """Initializes a new SearchInputHistory class from an existing SearchToolInput."""
        search_input = input.as_search_input() if isinstance(input, SearchToolInput) else input
        search_input_dict = (
            search_input.model_dump(include=set(cls.model_fields), mode="json")
            | {"input_hash": search_input.input_hash}
            | kwargs
        )
        return cls(**search_input_dict)

    def to_search_input(self) -> SearchInput:
        """Converts the SQLModel into a `SearchInput` instance."""
        return SearchInput.create(
            queries=self.queries,
            providers=self.providers,
            max_records=self.max_records,
            pages=self.pages,
            page_offset=self.page_offset,
            year_from=self.year_from,
            year_to=self.year_to,
            open_access_only=self.open_access_only,
        )


class RelevanceSearchInputHistory(BaseInputHistory, table=True):
    """SQL Model for storing relevance search input parameters and linking to relevance_search output."""

    __tablename__ = "relevance_search_inputs"
    __table_args__ = (Index("ix_relevance_search_inputs_stored_at_id", "stored_at", "input_hash", "id"),)

    relevance_search_execution: Mapped[Optional["RelevanceSearchExecution"]] = Relationship(
        back_populates="relevance_search_input",
    )

    providers: list[str] = SQLField(
        sa_column=Column(JSON),
        description="The set of providers from which to retrieve and rerank records by topic relevance.",
    )

    queries: list[str] = SQLField(
        default_factory=list,
        sa_column=Column(JSON),
        description="An optional search query used to retrieve search results from APIs.",
    )

    question: str = SQLField(
        ...,
        description="The research question or topic used to rerank records via record-topic embedding similarity.",
        min_length=8,
        max_length=1000,
    )

    categories: list[str] = SQLField(
        default_factory=list,
        sa_column=Column(JSON),
        description="Categories used to further determine topic relevance (e.g., depression, anxiety, meta_analysis).",
    )

    similarity_threshold: Optional[float] = SQLField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold required to keep a record during filtering.",
    )

    @classmethod
    def from_relevance_search_input(cls, input: RelevanceSearchToolInput | RelevanceSearchInput, **kwargs: Any) -> Self:
        """Initializes a new RelevanceSearchInputHistory from an existing RelevanceSearchToolInput."""
        relevance_search_input = (
            input.as_relevance_search_input() if isinstance(input, RelevanceSearchToolInput) else input
        )
        categories = ResearchCategory.as_category_names(relevance_search_input.categories)

        relevance_search_input_dict = (
            relevance_search_input.model_dump(mode="json", include=set(cls.model_fields))
            | {"categories": categories, "input_hash": relevance_search_input.input_hash}
            | kwargs
        )
        return cls(**relevance_search_input_dict)

    def to_relevance_search_input(self) -> RelevanceSearchInput:
        """Formats and processes the instance into a `RelevanceSearchInput` usable for later retrieval and reranking."""
        provider_list = APIProviders.as_provider_list(self.providers)
        category_list = ResearchCategory.as_category_list(self.categories)

        return RelevanceSearchInput(
            question=self.question,
            queries=self.queries,
            max_records=self.max_records,
            pages=self.pages,
            page_offset=self.page_offset,
            year_from=self.year_from,
            year_to=self.year_to,
            open_access_only=self.open_access_only,
            providers=provider_list,
            categories=category_list,
            similarity_threshold=self.similarity_threshold,
        )


class SynthesisInputHistory(BaseInputHistory, table=True):
    """SQL Model for storing synthesis input parameters and linking to synthesis output."""

    __tablename__ = "synthesis_inputs"
    __table_args__ = (Index("ix_synthesis_inputs_stored_at_id", "stored_at", "input_hash", "id"),)

    synthesis_execution: Mapped[Optional["SynthesisExecution"]] = Relationship(
        back_populates="synthesis_input",
    )

    providers: list[str] = SQLField(
        sa_column=Column(JSON),
        description="The set of providers from which to synthesize academic records into a research summary.",
    )

    queries: list[str] = SQLField(
        default_factory=list,
        sa_column=Column(JSON),
        description="An optional search query used to retrieve search results from APIs for later synthesis.",
    )

    question: str = SQLField(
        ...,
        description="The research question that the synthesized research summary explores.",
        min_length=10,
        max_length=1000,
    )

    categories: list[str] = SQLField(
        default_factory=list,
        sa_column=Column(JSON),
        description="Research categories to focus synthesis on (e.g., depression, anxiety, meta_analysis).",
    )

    similarity_threshold: Optional[float] = SQLField(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold required to keep a record during filtering.",
    )

    @classmethod
    def from_synthesis_input(cls, input: SynthesisToolInput | SynthesisInput, **kwargs: Any) -> Self:
        """Initializes a new SynthesisInputHistory from an existing SynthesisToolInput."""
        synthesis_input = input.as_synthesis_input() if isinstance(input, SynthesisToolInput) else input
        categories = ResearchCategory.as_category_names(synthesis_input.categories)
        synthesis_input_dict = (
            synthesis_input.model_dump(mode="json", include=set(cls.model_fields))
            | {"categories": categories, "input_hash": synthesis_input.input_hash}
            | kwargs
        )

        return cls(**synthesis_input_dict)

    def to_synthesis_input(self) -> SynthesisInput:
        """Formats and processes the instance into a `SynthesisInput` usable for later synthesis."""
        provider_list = APIProviders.as_provider_list(self.providers)
        category_list = ResearchCategory.as_category_list(self.categories)

        return SynthesisInput(
            question=self.question,
            queries=self.queries,
            max_records=self.max_records,
            pages=self.pages,
            page_offset=self.page_offset,
            year_from=self.year_from,
            year_to=self.year_to,
            open_access_only=self.open_access_only,
            providers=provider_list,
            categories=category_list,
            similarity_threshold=self.similarity_threshold,
        )


class BaseEvidenceItemHistory(SQLModel, table=False):
    """SQL Model containing core evidence item fields that are extracted during synthesis."""

    id: Optional[int] = SQLField(default=None, primary_key=True, index=True)

    record_index: int = SQLField(description="Index of the source record in the synthesis record list (0-based)")
    referenced_text: str | None = SQLField(
        default=None, description="Reference text extracted from the current record (e.g., 'CBT shows 60% efficacy')"
    )
    finding: str = SQLField(description="Key finding or claim extracted from the source record")
    relevance_score: float = SQLField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance score to the synthesis topic (0-1)",
    )


class AgentEvidenceItemHistory(BaseEvidenceItemHistory, table=True):
    """SQL Model for storing evidence items extracted during synthesis."""

    __tablename__ = "agent_evidence_items"

    synthesis_output_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the synthesis output this evidence belongs to",
        foreign_key="synthesis_outputs.id",
    )
    synthesis_output: Mapped["SynthesisOutputHistory"] = Relationship(back_populates="evidence_items")

    @classmethod
    def from_agent_evidence_item(cls, evidence_item: AgentEvidenceItem, **kwargs: Any) -> Self:
        """Generates a new AgentEvidenceItemHistory class from an `AgentEvidenceItem` object."""
        return cls(
            record_index=evidence_item.record_index,
            referenced_text=evidence_item.referenced_text,
            finding=evidence_item.finding,
            relevance_score=evidence_item.relevance_score,
            **kwargs,
        )

    def to_agent_evidence_item(self) -> AgentEvidenceItem:
        """Generates a new AgentEvidenceItem from the current SQLModel."""
        return AgentEvidenceItem(
            record_index=self.record_index,
            referenced_text=self.referenced_text,
            finding=self.finding,
            relevance_score=self.relevance_score,
        )


class RejectedEvidenceItemHistory(BaseEvidenceItemHistory, table=True):
    """SQL Model for storing evidence items that were rejected during synthesis."""

    __tablename__ = "rejected_evidence_items"

    error: str = SQLField(default=None, description="The reason why the grounding step rejected the citation")
    referenced_text_similarity: float | None = SQLField(
        default=None, description="The similarity of the record to listed citation"
    )

    synthesis_output_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the synthesis output this evidence belongs to",
        foreign_key="synthesis_outputs.id",
    )
    synthesis_output: Mapped["SynthesisOutputHistory"] = Relationship(back_populates="rejected_evidence_items")

    record_hash: Optional[str] = SQLField(
        default=None,
        description="Hash of the source SearchRecord for linking",
        foreign_key="search_record_elements.record_hash",
    )

    record: Mapped[Optional["SearchRecordElement"]] = Relationship(
        back_populates="rejected_evidence_item",
        sa_relationship_kwargs={"foreign_keys": "[RejectedEvidenceItemHistory.record_hash]", "lazy": "selectin"},
    )

    @classmethod
    def from_rejected_evidence_item(cls, rejected_evidence_item: RejectedEvidenceItem, **kwargs: Any) -> Self:
        """Generates a new RejectedEvidenceItemHistory class from an `RejectedEvidenceItem` object."""
        return cls(
            error=rejected_evidence_item.error,
            record_hash=rejected_evidence_item.record_hash,
            referenced_text_similarity=rejected_evidence_item.referenced_text_similarity,
            record_index=rejected_evidence_item.record_index,
            referenced_text=rejected_evidence_item.referenced_text,
            finding=rejected_evidence_item.finding,
            relevance_score=rejected_evidence_item.relevance_score,
            **kwargs,
        )

    def to_rejected_evidence_item(self) -> RejectedEvidenceItem:
        """Generates a new RejectedEvidenceItem from the current SQLModel."""
        record = self.record.to_search_record() if self.record else None
        return RejectedEvidenceItem(
            error=self.error,
            record=record,
            referenced_text_similarity=self.referenced_text_similarity,
            record_index=self.record_index,
            referenced_text=self.referenced_text,
            finding=self.finding,
            relevance_score=self.relevance_score,
        )


class GroundedEvidenceItemHistory(BaseEvidenceItemHistory, table=True):
    """SQL Model for storing evidence items extracted during synthesis."""

    __tablename__ = "grounded_evidence_items"

    id: Optional[int] = SQLField(default=None, primary_key=True, index=True)

    synthesis_output_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the synthesis output this evidence belongs to",
        foreign_key="synthesis_outputs.id",
    )
    synthesis_output: Mapped["SynthesisOutputHistory"] = Relationship(back_populates="grounded_evidence_items")

    record_hash: Optional[str] = SQLField(
        default=None,
        description="Hash of the source SearchRecord for linking",
        foreign_key="search_record_elements.record_hash",
    )

    record: Mapped["SearchRecordElement"] = Relationship(
        back_populates="grounded_evidence_items",
        sa_relationship_kwargs={"foreign_keys": "[GroundedEvidenceItemHistory.record_hash]", "lazy": "selectin"},
    )

    referenced_text_similarity: float | None = SQLField(
        default=None, description="The similarity of the record to listed citation"
    )

    @classmethod
    def from_grounded_evidence_item(cls, grounded_evidence_item: GroundedEvidenceItem, **kwargs: Any) -> Self:
        """Generates a new GroundedEvidenceItemHistory class from an `GroundedEvidenceItem` object."""
        return cls(
            record_hash=grounded_evidence_item.record_hash,
            record_index=grounded_evidence_item.record_index,
            referenced_text=grounded_evidence_item.referenced_text,
            referenced_text_similarity=grounded_evidence_item.referenced_text_similarity,
            finding=grounded_evidence_item.finding,
            relevance_score=grounded_evidence_item.relevance_score,
            **kwargs,
        )

    def to_grounded_evidence_item(self) -> GroundedEvidenceItem:
        """Generates a new GroundedEvidenceItem from the current SQLModel."""
        if not self.record:
            raise ValueError("The current `GroundedEvidenceItemHistory` could not resolve to a valid SearchRecord.")
        return GroundedEvidenceItem(
            record=self.record.to_search_record(),
            record_index=self.record_index,
            referenced_text=self.referenced_text,
            referenced_text_similarity=self.referenced_text_similarity,
            finding=self.finding,
            relevance_score=self.relevance_score,
        )

    def to_agent_evidence_item(self) -> AgentEvidenceItem:
        """Generates a new AgentEvidenceItem from the current SQLModel."""
        return AgentEvidenceItem(
            record_index=self.record_index,
            finding=self.finding,
            referenced_text=self.referenced_text,
            relevance_score=self.relevance_score,
        )


class RecordEmbeddingHistory(SQLModel, table=True):
    """SQLModel representing the record hash and unique embedding derived from the specified model."""

    __tablename__ = "record_embeddings"

    id: Optional[int] = SQLField(default=None, primary_key=True, nullable=False, index=True)

    record_topic_similarity: Mapped[Optional["RecordTopicSimilarityHistory"]] = Relationship(
        back_populates="record_embedding",
        sa_relationship_kwargs={"uselist": False},
    )

    record_hash: str = SQLField(
        description="A stable md5 hash of the current record.",
        foreign_key="search_record_elements.record_hash",
        nullable=False,
    )

    record: Mapped["SearchRecordElement"] = Relationship(
        back_populates="record_embedding",
        sa_relationship_kwargs={"foreign_keys": "[RecordEmbeddingHistory.record_hash]", "lazy": "selectin"},
    )
    model_name: str = SQLField(description="The embedding model used to filter records", index=True)

    embedding: Sequence[float] = SQLField(
        default_factory=list, description="Embedding for the current record", sa_column=Column(JSON)
    )

    @classmethod
    def from_record_embedding(cls, output: SearchRecordEmbedding, **kwargs: Any) -> Self:
        """Initializes a new `RecordEmbeddingHistory` instance from the provided output."""
        return cls(
            record_hash=output.record.record_hash, model_name=output.model_name, embedding=output.embedding, **kwargs
        )

    def to_record_embedding(self) -> SearchRecordEmbedding:
        """Initializes a new `SearchRecordEmbedding` instance from the current SQLModel."""
        return SearchRecordEmbedding(
            record=self.record.to_search_record(),
            embedding=self.embedding,
            model_name=self.model_name,
        )


class TopicEmbeddingHistory(SQLModel, table=True):
    """SQLModel representing the topic and unique embedding derived from the specified model."""

    __tablename__ = "topic_embeddings"

    id: Optional[int] = SQLField(default=None, primary_key=True, nullable=False, index=True)

    record_topic_similarity_output: list["RecordTopicSimilarityOutputHistory"] = Relationship(
        back_populates="topic_embedding",
    )

    topic: str = SQLField(description="The research topic to compare the record content against.", index=True)

    question: str = SQLField(description="The research question asked by the user")
    categories: list[str] = SQLField(
        default_factory=list,
        description="The research categories related to the subject matter",
        sa_column=Column(JSON),
    )
    queries: list[str] = SQLField(
        default_factory=list, description="The queries sent to the API Provider", sa_column=Column(JSON)
    )
    model_name: str = SQLField(description="The embedding model used to filter records", index=True)

    embedding: Sequence[float] = SQLField(
        default_factory=list, description="Embedding for the current record", sa_column=Column(JSON)
    )

    @classmethod
    def from_topic_embedding(cls, output: TopicEmbedding, **kwargs: Any) -> Self:
        """Initializes a new `TopicEmbeddingHistory` instance from the provided output."""
        return cls(
            topic=output.topic,
            question=output.question,
            categories=ResearchCategory.as_category_names(output.categories),
            queries=output.queries,
            model_name=output.model_name,
            embedding=output.embedding,
            **kwargs,
        )

    def to_topic_embedding(self) -> TopicEmbedding:
        """Initializes a new `TopicEmbedding` instance from the current SQLModel."""
        return TopicEmbedding(
            topic=self.topic,
            question=self.question,
            categories=ResearchCategory.as_category_list(self.categories),
            queries=self.queries,
            model_name=self.model_name,
            embedding=self.embedding,
        )


class RecordTopicSimilarityHistory(SQLModel, table=True):
    """SQLModel representing the similarity between a record embedding and topic embedding."""

    __tablename__ = "record_topic_similarities"

    id: Optional[int] = SQLField(default=None, primary_key=True, nullable=False, index=True)

    record_embedding_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the record embedding",
        foreign_key="record_embeddings.id",
    )
    record_embedding: Mapped[RecordEmbeddingHistory] = Relationship(
        back_populates="record_topic_similarity",
        sa_relationship_kwargs={"uselist": False},
    )
    topic: str = SQLField(description="The research topic to compare the record content against.", index=True)
    record_topic_similarity_output_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the record topic similarity output that the embedding corresponds to",
        foreign_key="record_topic_similarity_outputs.id",
    )
    record_topic_similarity_output: Mapped["RecordTopicSimilarityOutputHistory"] = Relationship(
        back_populates="record_topic_similarities"
    )
    topic_similarity_score: float = SQLField(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The cosine similarity between the record and topic embeddings.",
    )

    @classmethod
    def from_record_topic_embedding_similarity(cls, output: RecordTopicSimilarity, **kwargs: Any) -> Self:
        """Creates a new `RecordTopicSimilarityHistory` from an existing output model."""
        return cls(
            record_embedding=RecordEmbeddingHistory.from_record_embedding(output.record_embedding),
            topic=output.topic,
            topic_similarity_score=output.topic_similarity_score,
            **kwargs,
        )

    def to_record_topic_embedding_similarity(self) -> RecordTopicSimilarity:
        """Creates a new `RecordTopicSimilarity` instance from the current SQLModel."""
        return RecordTopicSimilarity(
            record_embedding=self.record_embedding.to_record_embedding(),
            topic=self.topic,
            topic_similarity_score=self.topic_similarity_score,
        )


class RecordTopicSimilarityOutputHistory(SQLModel, table=True):
    """SQLModel representing the record-topic similarity outputs for all retrieved records where relevant."""

    __tablename__ = "record_topic_similarity_outputs"

    id: Optional[int] = SQLField(default=None, primary_key=True, nullable=False, index=True)

    record_topic_similarities: list[RecordTopicSimilarityHistory] = Relationship(
        back_populates="record_topic_similarity_output",
    )

    topic_embedding_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the topic embedding",
        foreign_key="topic_embeddings.id",
    )
    topic_embedding: Mapped[TopicEmbeddingHistory] = Relationship(
        back_populates="record_topic_similarity_output",
        sa_relationship_kwargs={"foreign_keys": "[RecordTopicSimilarityOutputHistory.topic_embedding_id]"},
    )
    relevance_search_output: Mapped[Optional["RelevanceSearchOutputHistory"]] = Relationship(
        back_populates="record_topic_similarity_output",
        sa_relationship_kwargs={"uselist": False},
    )

    @classmethod
    def from_record_topic_similarity_output(cls, output: RecordTopicSimilarityOutput, **kwargs: Any) -> Self:
        """Creates a new `RecordTopicSimilarityOutputHistory` from an existing output model."""
        scores = [
            RecordTopicSimilarityHistory.from_record_topic_embedding_similarity(
                record_topic_similarity,
            )
            for record_topic_similarity in output.record_similarity_scores
        ]

        topic_embedding_history = TopicEmbeddingHistory.from_topic_embedding(output.topic_embedding)
        return cls(record_topic_similarities=scores, topic_embedding=topic_embedding_history, **kwargs)

    def to_record_topic_similarity_output(self) -> RecordTopicSimilarityOutput:
        """Creates a new `RecordTopicSimilarityOutput` instance from the current SQLModel."""
        record_topic_similarities = [
            similarity.to_record_topic_embedding_similarity() for similarity in self.record_topic_similarities
        ]
        topic_embedding_history = self.topic_embedding.to_topic_embedding()

        return RecordTopicSimilarityOutput(
            record_similarity_scores=record_topic_similarities,
            topic_embedding=topic_embedding_history,
        )


class IndexedSearchRecordHistory(SQLModel, table=True):
    """SQL Model for storing indexed search record references with relationships to records and embeddings."""

    __tablename__ = "indexed_search_outputs"

    id: Optional[int] = SQLField(default=None, primary_key=True, index=True)

    record_hash: str = SQLField(
        description="A stable md5 hash of the current indexed record.",
        foreign_key="search_record_elements.record_hash",
    )
    record: Mapped["SearchRecordElement"] = Relationship(
        back_populates="indexed_record_history",
        sa_relationship_kwargs={"foreign_keys": "[IndexedSearchRecordHistory.record_hash]", "lazy": "selectin"},
    )

    index: int = SQLField(description="Index of the source record in the synthesis record list (0-based)")
    relevance_search_output_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the relevance search to which the current indexed search belongs.",
        foreign_key="relevance_search_outputs.id",
    )
    relevance_search_output: Mapped["RelevanceSearchOutputHistory"] = Relationship(
        back_populates="indexed_records",
        sa_relationship_kwargs={"uselist": False},
    )
    topic_similarity_score: float | None = SQLField(
        default=None, description="The topic similarity score of the indexed record."
    )

    @classmethod
    def from_indexed_search_record(
        cls,
        output: IndexedSearchRecord,
        **kwargs: Any,
    ) -> Self:
        """Converts an `IndexedSearchRecord` into an `IndexedSearchRecordHistory` SQL model that links to its record."""
        return cls(
            record_hash=output.record_hash,
            topic_similarity_score=output.topic_similarity_score,
            index=output.index,
            **kwargs,
        )

    def to_indexed_search_record(self) -> IndexedSearchRecord:
        """Converts the current `IndexedSearchRecordHistory` instance into a `IndexedSearchRecord` model."""
        base_record = self.record.to_search_record()

        return IndexedSearchRecord.from_search_record(
            record=base_record,
            index=self.index,
            topic_similarity_score=self.topic_similarity_score,
        )


class RelevanceSearchOutputHistory(BaseOutputHistory, table=True):
    """SQL Model for storing relevance search output with relationships to input and evidence items."""

    __tablename__ = "relevance_search_outputs"

    relevance_search_execution: Mapped[Optional["RelevanceSearchExecution"]] = Relationship(
        back_populates="relevance_search_output",
        sa_relationship_kwargs={"uselist": False},
    )
    indexed_records: list["IndexedSearchRecordHistory"] = Relationship(
        back_populates="relevance_search_output",
    )
    record_topic_similarity_output_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the record topic similarity embedding output generated during the research synthesis.",
        foreign_key="record_topic_similarity_outputs.id",
    )
    record_topic_similarity_output: Mapped[Optional[RecordTopicSimilarityOutputHistory]] = Relationship(
        back_populates="relevance_search_output",
        sa_relationship_kwargs={"uselist": False},
    )
    successful: bool = SQLField(default=True, description="Whether the relevance search was completed without error.")

    @classmethod
    def from_relevance_search_output(
        cls,
        output: RelevanceSearchOutput,
        **kwargs: Any,
    ) -> Self:
        """Creates a RelevanceSearchOutputHistory from a RelevanceSearchOutput model defined within the core schema."""

        record_topic_similarity_output = (
            RecordTopicSimilarityOutputHistory.from_record_topic_similarity_output(
                output.record_topic_similarity_output
            )
            if output.record_topic_similarity_output
            else None
        )

        indexed_record_list = [
            IndexedSearchRecordHistory.from_indexed_search_record(record) for record in output.indexed_records
        ]
        return cls(
            record_topic_similarity_output=record_topic_similarity_output,
            indexed_records=indexed_record_list,
            successful=output.successful,
            **kwargs,
        )

    def to_relevance_search_output(
        self,
    ) -> RelevanceSearchOutput:
        """Creates a RelevanceSearchOutput instance from the current RelevanceSearchOutputHistory model."""
        record_topic_similarity_output = (
            self.record_topic_similarity_output.to_record_topic_similarity_output()
            if self.record_topic_similarity_output
            else None
        )

        indexed_record_list = [indexed_record.to_indexed_search_record() for indexed_record in self.indexed_records]
        if self.relevance_search_execution is None:
            raise ValueError(
                "The current `RelevanceSearchOutputHistory` does not have a corresponding RelevanceSearchInputHistory "
                "model."
            )
        if self.relevance_search_execution.search_execution is None:
            raise ValueError(
                "The current `RelevanceSearchOutputHistory` does not have a corresponding SearchExecution instance "
                "model."
            )

        relevance_search_input = self.relevance_search_execution.relevance_search_input.to_relevance_search_input()
        search_output = self.relevance_search_execution.search_execution.search_output.to_search_output()

        return RelevanceSearchOutput(
            record_topic_similarity_output=record_topic_similarity_output,
            indexed_records=indexed_record_list,
            relevance_search_input=relevance_search_input,
            search_output=search_output,
        )

    to_output = to_relevance_search_output


class SearchOutputHistory(BaseOutputHistory, table=True):
    """SQL Model for storing search output with relationships to input and records."""

    __tablename__ = "search_outputs"

    id: Optional[int] = SQLField(default=None, primary_key=True, index=True)

    records: list[SearchRecordHistory] = Relationship(back_populates="search_output")
    response_summaries: list[ResponseSummaryHistory] = Relationship(back_populates="search_output")

    search_execution: Mapped[Optional["SearchExecution"]] = Relationship(
        back_populates="search_output",
    )

    total_records: int = SQLField(description="Total number of records retrieved")
    providers_queried: int = SQLField(description="Number of providers searched")
    providers_successful: int = SQLField(description="Number of providers that returned results")
    pages_successful: int = SQLField(description="Total pages that returned a valid response across all API responses")
    pages_retrieved: int = SQLField(description="Total pages retrieved across all API responses")
    cache_hit: bool = SQLField(default=False, description="Whether results came from cache")
    successful: bool = SQLField(default=True, description="Whether records were retrieved from each API without error")

    @classmethod
    def from_search_output(
        cls,
        output: SearchOutput,
        **kwargs: Any,
    ) -> Self:
        """Creates a SearchOutputHistory from a SearchOutput model as defined in the core schema."""
        search_record_history = [
            SearchRecordHistory(record=SearchRecordElement.from_search_record(record)) for record in output.records
        ]
        response_summary_history = [
            ResponseSummaryHistory(response_summary=ResponseSummaryElement.from_response_summary(response_summary))
            for response_summary in output.response_summaries
        ]

        return cls(
            records=search_record_history,
            response_summaries=response_summary_history,
            providers_queried=output.pagination.providers_queried,
            providers_successful=output.pagination.providers_successful,
            total_records=output.pagination.total_records,
            pages_successful=output.pagination.pages_successful,
            pages_retrieved=output.pagination.pages_retrieved,
            cache_hit=output.cache_hit,
            successful=output.successful,
            **kwargs,
        )

    def to_search_output(self) -> SearchOutput:
        """Creates a `SearchOutput` from the current SQLModel."""
        search_input = (
            self.search_execution.search_input.to_search_input() if self.search_execution is not None else None
        )
        if search_input is None:
            raise ValueError(
                "The current `SearchOutputHistory` does not have a corresponding SearchInputHistory model."
            )
        records = [record_history.to_search_record() for record_history in self.records]
        response_summaries = [response_summary.to_response_summary() for response_summary in self.response_summaries]
        pagination = PaginationInfo(
            total_records=self.total_records,
            providers_queried=self.providers_queried,
            providers_successful=self.providers_successful,
            pages_successful=self.pages_successful,
            pages_retrieved=self.pages_retrieved,
        )
        return SearchOutput(
            search_input=search_input,
            records=records,
            response_summaries=response_summaries,
            pagination=pagination,
        )

    to_output = to_search_output


class SynthesisOutputHistory(BaseOutputHistory, table=True):
    """SQL Model for storing synthesis output with relationships to input and evidence items."""

    __tablename__ = "synthesis_outputs"

    id: Optional[int] = SQLField(default=None, primary_key=True, nullable=False, index=True)

    synthesis_execution: Mapped[Optional["SynthesisExecution"]] = Relationship(
        back_populates="synthesis_output",
        sa_relationship_kwargs={"uselist": False},
    )

    evidence_items: list[AgentEvidenceItemHistory] = Relationship(
        back_populates="synthesis_output", cascade_delete=True
    )
    grounded_evidence_items: list[GroundedEvidenceItemHistory] = Relationship(
        back_populates="synthesis_output", cascade_delete=True
    )
    rejected_evidence_items: list[RejectedEvidenceItemHistory] = Relationship(
        back_populates="synthesis_output", cascade_delete=True
    )
    synthesis: str = SQLField(description="Synthesized answer based on literature")

    key_findings: list[str] = SQLField(
        default_factory=list, description="Bullet-point key findings", sa_column=Column(JSON)
    )
    confidence_score: float = SQLField(
        description="Confidence in synthesis (0-1 based on evidence quality/quantity)",
        ge=0.0,
        le=1.0,
    )
    limitations: list[str] = SQLField(
        default_factory=list, description="Limitations of this synthesis", sa_column=Column(JSON)
    )
    suggested_queries: list[str] = SQLField(
        default_factory=list, description="Suggested follow-up queries", sa_column=Column(JSON)
    )
    references_grounded: int = SQLField(
        default=0, ge=0, description="The total number of grounded record references from the synthesis."
    )
    references_rejected: int = SQLField(
        default=0, ge=0, description="The total number of rejected record references from the synthesis."
    )
    grounding_error: Optional[str] = SQLField(default=None, description="Error message if grounding failed.")
    model_name: str = SQLField(default="N/A", description="The LLM used to synthesize results.")
    successful: bool = SQLField(default=True, description="Whether a compete synthesis was generated without error.")

    @classmethod
    def from_synthesis_output(
        cls,
        output: SynthesisOutput,
        **kwargs: Any,
    ) -> Self:
        """Creates a SynthesisOutputHistory from a SynthesisOutput model defined within the core schema."""
        return cls(
            synthesis=output.synthesis,
            key_findings=output.key_findings,
            confidence_score=output.confidence_score,
            evidence_items=[
                AgentEvidenceItemHistory.from_agent_evidence_item(item)
                for item in output.agent_output.evidence_summaries
            ],
            rejected_evidence_items=[
                RejectedEvidenceItemHistory.from_rejected_evidence_item(item)
                for item in output.grounding_output.rejected_evidence_items
            ],
            grounded_evidence_items=[
                GroundedEvidenceItemHistory.from_grounded_evidence_item(item) for item in output.grounded_evidence
            ],
            limitations=output.limitations,
            suggested_queries=output.suggested_queries,
            references_grounded=output.grounding_stats.references_grounded,
            references_rejected=output.grounding_stats.references_rejected,
            grounding_error=output.grounding_stats.error,
            successful=output.successful,
            model_name=output.model_name,
            **kwargs,
        )

    def to_synthesis_output(self) -> SynthesisOutput:
        """Creates a SynthesisOutput from the current SynthesisOutputHistory sqlmodel."""
        if self.synthesis_execution is None:
            raise ValueError(
                "The current `SynthesisOutputHistory` does not have a corresponding SynthesisExecution model."
            )
        if self.synthesis_execution.relevance_search_execution is None:
            raise ValueError(
                "The current `SynthesisOutputHistory` does not have a corresponding RelevanceSearchExecution model."
            )

        relevance_search_output_history = self.synthesis_execution.relevance_search_execution.relevance_search_output
        synthesis_input_history = self.synthesis_execution.synthesis_input

        agent_output = SynthesisAgentOutput(
            synthesis=self.synthesis,
            confidence_score=self.confidence_score,
            key_findings=self.key_findings,
            evidence_summaries=[item.to_agent_evidence_item() for item in self.evidence_items],
            limitations=self.limitations,
            suggested_queries=self.suggested_queries,
        )
        grounding_output = EvidenceGroundingOutput(
            grounded_evidence_items=[item.to_grounded_evidence_item() for item in self.grounded_evidence_items],
            rejected_evidence_items=[item.to_rejected_evidence_item() for item in self.rejected_evidence_items],
            grounding_stats=EvidenceGroundingStats(
                references_grounded=self.references_grounded, references_rejected=self.references_rejected
            ),
        )
        return SynthesisOutput(
            synthesis_input=synthesis_input_history.to_synthesis_input(),
            relevance_search_output=relevance_search_output_history.to_relevance_search_output(),
            agent_output=agent_output,
            grounding_output=grounding_output,
            model_name=self.model_name,
        )

    to_output = to_synthesis_output


class SearchExecution(SQLModel, table=True):
    """SQL Model used for linking core tables, recording inputs and outputs for record retrieval and processing."""

    __tablename__ = "search_executions"

    id: Optional[int] = SQLField(default=None, primary_key=True, nullable=False, index=True)

    search_input_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the search input",
        foreign_key="search_inputs.id",
    )
    search_input: Mapped[SearchInputHistory] = Relationship(
        back_populates="search_execution",
        sa_relationship_kwargs={"foreign_keys": "[SearchExecution.search_input_id]"},
    )

    search_output_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the related search output",
        foreign_key="search_outputs.id",
    )
    search_output: Mapped[SearchOutputHistory] = Relationship(
        back_populates="search_execution",
        sa_relationship_kwargs={"foreign_keys": "[SearchExecution.search_output_id]"},
    )

    relevance_search_execution: Mapped[Optional["RelevanceSearchExecution"]] = Relationship(
        back_populates="search_execution",
    )

    synthesis_execution: Mapped[Optional["SynthesisExecution"]] = Relationship(
        back_populates="search_execution",
    )

    @property
    def topic(self) -> str:
        """Convenience alias for `SearchExecution.search_input.topic."""
        return self.search_input.topic

    @property
    def stored_at(self) -> TimeStampString:
        """Retrieves the `stored_at` time associated with the current SearchExecution model."""
        return self.search_input.stored_at

    @classmethod
    def from_record_search(
        cls, search_input: SearchToolInput | SearchInput, search_output: SearchOutput, **kwargs: Any
    ) -> Self:
        """Initializes a new `SearchExecution` instance from the input and outputs."""
        return cls(
            search_input=SearchInputHistory.from_search_input(search_input),
            search_output=SearchOutputHistory.from_search_output(search_output),
            **kwargs,
        )

    def to_search_output(self) -> SearchOutput:
        """Alias for the creation of a new `SearchOutput` from the nested `SearchOutputHistory` SQLModel."""
        return self.search_output.to_search_output()

    to_output = to_search_output


class RelevanceSearchExecution(SQLModel, table=True):
    """SQL Model used for linking core tables, recording inputs and outputs for searching and reranking by relevance."""

    __tablename__ = "relevance_search_executions"

    id: Optional[int] = SQLField(default=None, primary_key=True, nullable=False, index=True)

    search_execution_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the relevance search input that the output corresponds to",
        foreign_key="search_executions.id",
    )
    search_execution: Mapped[SearchExecution] = Relationship(
        back_populates="relevance_search_execution",
        sa_relationship_kwargs={"foreign_keys": "[RelevanceSearchExecution.search_execution_id]"},
    )

    synthesis_execution: Mapped[Optional["SynthesisExecution"]] = Relationship(
        back_populates="relevance_search_execution",
    )

    relevance_search_input_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the relevance search input that the output corresponds to",
        foreign_key="relevance_search_inputs.id",
    )
    relevance_search_input: Mapped[RelevanceSearchInputHistory] = Relationship(
        back_populates="relevance_search_execution",
        sa_relationship_kwargs={"foreign_keys": "[RelevanceSearchExecution.relevance_search_input_id]"},
    )

    relevance_search_output_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the relevance search output that the output corresponds to",
        foreign_key="relevance_search_outputs.id",
    )
    relevance_search_output: Mapped[RelevanceSearchOutputHistory] = Relationship(
        back_populates="relevance_search_execution",
        sa_relationship_kwargs={"uselist": False},
    )

    @property
    def topic(self) -> str:
        """Convenience alias for `RelevanceSearchExecution.relevance_search_input.topic."""
        return self.relevance_search_input.topic

    @property
    def stored_at(self) -> TimeStampString:
        """Retrieves the `stored_at` time associated with the current RelevanceSearchExecution model."""
        return self.relevance_search_input.stored_at

    @classmethod
    def from_record_relevance_search(
        cls,
        relevance_search_input: RelevanceSearchToolInput | RelevanceSearchInput,
        relevance_search_output: RelevanceSearchOutput,
        search_execution: SearchExecution | None = None,
        **kwargs: Any,
    ) -> Self:
        """Initializes a new `RelevanceSearchExecution` instance from the input and outputs."""
        current_search_execution = search_execution or SearchExecution.from_record_search(
            relevance_search_output.search_input, relevance_search_output.search_output
        )
        return cls(
            relevance_search_input=RelevanceSearchInputHistory.from_relevance_search_input(relevance_search_input),
            relevance_search_output=RelevanceSearchOutputHistory.from_relevance_search_output(relevance_search_output),
            search_execution=current_search_execution,
            **kwargs,
        )

    def to_relevance_search_output(self) -> RelevanceSearchOutput:
        """Alias for creating a new `RelevanceSearchOutput` from the nested `RelevanceSearchOutputHistory` SQLModel."""
        return self.relevance_search_output.to_relevance_search_output()

    to_output = to_relevance_search_output


class SynthesisExecution(SQLModel, table=True):
    """SQL Model used for linking core tables, recording inputs and outputs for research synthesis."""

    __tablename__ = "synthesis_executions"

    id: Optional[int] = SQLField(default=None, primary_key=True, nullable=False, index=True)

    search_execution_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the synthesis input that the output corresponds to",
        foreign_key="search_executions.id",
    )
    search_execution: Mapped[SearchExecution] = Relationship(
        back_populates="synthesis_execution",
        sa_relationship_kwargs={"foreign_keys": "[SynthesisExecution.search_execution_id]"},
    )
    relevance_search_execution_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the relevance search input that the output corresponds to",
        foreign_key="relevance_search_executions.id",
    )
    relevance_search_execution: Mapped[RelevanceSearchExecution] = Relationship(
        back_populates="synthesis_execution",
        sa_relationship_kwargs={"foreign_keys": "[SynthesisExecution.relevance_search_execution_id]"},
    )

    synthesis_input_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the synthesis input that the output corresponds to",
        foreign_key="synthesis_inputs.id",
    )
    synthesis_input: Mapped[SynthesisInputHistory] = Relationship(
        back_populates="synthesis_execution",
        sa_relationship_kwargs={"foreign_keys": "[SynthesisExecution.synthesis_input_id]"},
    )

    synthesis_output_id: Optional[int] = SQLField(
        default=None,
        description="The ID of the synthesis output that the output corresponds to",
        foreign_key="synthesis_outputs.id",
    )
    synthesis_output: Mapped[SynthesisOutputHistory] = Relationship(
        back_populates="synthesis_execution",
        sa_relationship_kwargs={"uselist": False},
    )

    @property
    def topic(self) -> str:
        """Convenience alias for `SynthesisExecution.synthesis_input.topic."""
        return self.synthesis_input.topic

    @property
    def stored_at(self) -> TimeStampString:
        """Retrieves the `stored_at` time associated with the current SynthesisExecution model."""
        return self.synthesis_input.stored_at

    @classmethod
    def from_research_synthesis(
        cls,
        synthesis_input: SynthesisToolInput | SynthesisInput,
        synthesis_output: SynthesisOutput,
        relevance_search_execution: RelevanceSearchExecution | None = None,
        search_execution: SearchExecution | None = None,
        **kwargs: Any,
    ) -> Self:
        """Initializes a new `SynthesisExecution` instance from the input and outputs."""
        if not search_execution and (search_output := synthesis_output.search_output):
            search_execution = search_execution or SearchExecution.from_record_search(
                search_output.search_input, search_output
            )

        if not relevance_search_execution and (relevance_search_output := synthesis_output.relevance_search_output):
            relevance_search_execution = (
                relevance_search_execution
                or RelevanceSearchExecution.from_record_relevance_search(
                    relevance_search_output.relevance_search_input,
                    relevance_search_output,
                    search_execution=search_execution,
                )
            )
        return cls(
            synthesis_input=SynthesisInputHistory.from_synthesis_input(synthesis_input),
            synthesis_output=SynthesisOutputHistory.from_synthesis_output(synthesis_output),
            relevance_search_execution=relevance_search_execution,
            search_execution=search_execution,
            **kwargs,
        )

    def to_synthesis_output(self) -> SynthesisOutput:
        """Alias for creating a new `SynthesisOutput` from the nested `SynthesisOutputHistory` SQLModel."""
        return self.synthesis_output.to_synthesis_output()

    to_output = to_synthesis_output


ResearchHistoryExecution = TypeAliasType(
    "ResearchHistoryExecution", SearchExecution | RelevanceSearchExecution | SynthesisExecution
)


HISTORY_TABLES: tuple[type[SQLModel], ...] = (
    SynthesisExecution,
    RelevanceSearchExecution,
    SearchExecution,
    SynthesisOutputHistory,
    SynthesisInputHistory,
    RelevanceSearchOutputHistory,
    RelevanceSearchInputHistory,
    SearchOutputHistory,
    SearchInputHistory,
    AgentEvidenceItemHistory,
    RejectedEvidenceItemHistory,
    GroundedEvidenceItemHistory,
    IndexedSearchRecordHistory,
    TopicEmbeddingHistory,
    RecordTopicSimilarityOutputHistory,
    RecordTopicSimilarityHistory,
    RecordEmbeddingHistory,
    SearchRecordHistory,
    SearchRecordElement,
    ResponseSummaryElement,
    ResponseSummaryHistory,
)

__all__ = [
    "AgentEvidenceItemHistory",
    "BaseInputHistory",
    "BaseOutputHistory",
    "GroundedEvidenceItemHistory",
    "IndexedSearchRecordHistory",
    "RecordEmbeddingHistory",
    "RecordTopicSimilarityHistory",
    "RecordTopicSimilarityOutputHistory",
    "RejectedEvidenceItemHistory",
    "RelevanceSearchInputHistory",
    "RelevanceSearchOutputHistory",
    "RelevanceSearchExecution",
    "ResearchHistoryExecution",
    "ResponseSummaryElement",
    "ResponseSummaryHistory",
    "SearchInputHistory",
    "SearchOutputHistory",
    "SearchExecution",
    "SearchRecordElement",
    "SearchRecordHistory",
    "SynthesisExecution",
    "SynthesisInputHistory",
    "SynthesisOutputHistory",
    "TopicEmbeddingHistory",
    "HISTORY_TABLES",
]

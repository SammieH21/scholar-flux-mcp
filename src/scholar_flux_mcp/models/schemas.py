"""Pydantic schemas for the ScholarFlux MCP server.

Defines all input/output models for MCP tools with comprehensive validation and documentation for LLM agent consumption.

"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime  # noqa: TCH003
from textwrap import dedent
from typing import Annotated, Any, ClassVar, TypeVar, overload

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
    validate_call,
)
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from scholar_flux_mcp.models.core import (
    JSONDataModel,
    TimeStamp,
    TimeStampString,
    computed_cached_property,
    computed_property,
    validate_iso_timestamp_string,
)
from scholar_flux_mcp.models.enums import (
    APIProviders,
    FieldTypes,
    ProviderMetadataDescriptions,
    QueryCharacterLimit,
    QuestionCharacterLimit,
    ResearchCategory,
    SearchRecordFields,
    SubjectInfo,
)
from scholar_flux_mcp.package_metadata import ScholarFluxMCPDependencies, __version__
from scholar_flux_mcp.utils.helpers import (
    as_tuple,
    coerce_int,
    coerce_numeric,
    coerce_str,
    generate_datetime,
    truncate,
)

T = TypeVar("T", bound=object)


class BaseSearchParams(JSONDataModel):
    """Base search parameters shared by core search input models."""

    HIDDEN_FIELDS: ClassVar[set[str]] = {"input_hash"}

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore",
    )

    max_records: int = Field(
        default=25,
        description="Maximum records to retrieve per provider",
        ge=1,
        le=200,
    )
    pages: int = Field(
        default=3,
        description="Number of result pages to retrieve per provider",
        ge=1,
    )
    page_offset: int = Field(
        default=0,
        description="The page number to start the retrieval of `n` pages from.",
        ge=0,
    )
    year_from: int | None = Field(
        default=1900,
        description="Filter records published on or after this year (e.g., 2020)",
        ge=1900,
        le=2100,
    )
    year_to: int | None = Field(
        default=2100,
        description="Filter records published on or before this year",
        ge=1900,
        le=2100,
    )
    open_access_only: bool = Field(
        default=False,
        description="Only return open access records",
    )

    @property
    def topic(self) -> str:
        """Identifies the corresponding research topic from core parameters available within the current model."""
        try:
            return ResearchTopic.create(self).topic
        except ValidationError:
            return ""

    @computed_cached_property
    def input_hash(self) -> str:
        """Creates a hash for the current input model to enable efficient identification."""
        with self.__class__.serialization_context(core_fields_only=True):
            record_dict = self.model_dump(exclude={"response_format"})
        stable_str = json.dumps(record_dict, sort_keys=True, default=str)
        return hashlib.md5(stable_str.encode("utf-8")).hexdigest()[:16]

    @classmethod
    @validate_call
    def infer_queries(cls, categories: Sequence[str | SubjectInfo | ResearchCategory]) -> list[str]:
        """Helper for inferring the `queries` parameter from informational categories when a query is not provided.

        Uses provided queries if available and otherwise derives the field from known ResearchCategory fields.

        Args:
            self: The current model after preliminary field validation

        Returns:
            An updated model with inferred queries when otherwise missing.

        """
        # When available, return the `queries` parameter as is.
        inferred_queries: list[str] = []
        # Add category-specific terms
        for category in as_tuple(categories):
            research_category = ResearchCategory.get(category)
            if research_category and research_category not in (ResearchCategory.GENERAL, ResearchCategory.OTHER):
                terms = " OR ".join(research_category.value.terms) or research_category.name.title()
                inferred_queries.append(terms)

        if not inferred_queries:
            raise ValueError(
                "A valid query has not been specified and could not be inferred from the provided research categories"
            )
        return inferred_queries


class BaseRelevanceSearchParams(BaseSearchParams):
    """Base relevance search parameters shared by core relevance search input models."""

    question: str = Field(
        ...,
        description=(
            "Research question, topic, or statement used to rerank records using embeddings. Examples:"
            "'Most effective interventions for treating anxiety', "
            "'Economic outcomes of epidemics and infection rates', "
            "'ADHD Biomarkers and impact of treatment'"
        ),
    )
    queries: ResearchQueryList = Field(default_factory=list)

    similarity_threshold: float | None = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Determines the minimum similarity threshold required to keep a record (0 by default). Set this "
            "field to `None` to avoid filtering by similarity."
        ),
    )

    @field_validator("queries", mode="before")
    @classmethod
    def validate_query_list(cls, v: list[str]) -> list[str]:
        """Validates each query afterward."""
        return QueryCharacterLimit.validate_character_limits(v)

    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Converts string queries into lists if not already a list.

        Validates each query afterward.

        """
        QuestionCharacterLimit.validate_character_limit(v)
        return v


class BaseSynthesisParams(BaseRelevanceSearchParams):
    """Base synthesis parameters shared by core synthesis input models."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    question: str = Field(
        ...,
        description=(
            "Research question to synthesize. Be specific and focused. Examples: "
            "'What are the most effective interventions for treatment-resistant depression?', "
            "'How does cognitive behavioral therapy compare to medication for anxiety disorders?', "
            "'What biomarkers are associated with PTSD treatment response?'"
        ),
    )

    similarity_threshold: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Determines the minimum similarity threshold required to keep a record (0.5 by default). Set this "
            "field to `None` to avoid filtering by similarity."
        ),
    )

    max_records: int = Field(
        default=50,
        description=(
            "The Maximum number of academic_records to analyze and include within the final research "
            "synthesis (5-200 records)."
        ),
        ge=5,
        le=200,
    )

    pages: int = Field(
        default=3,
        description=(
            "Number of result pages to retrieve per provider (1-10 pages). With a higher result set, records "
            "that are more relevant to the research question and topic are likely to be retrieved."
        ),
        ge=1,
    )

    year_from: int | None = Field(
        default_factory=lambda: generate_datetime().year - 5,
        description=(
            "Filter records published on or after this year (e.g., 2021). Shows records from the last 5 years by "
            "default."
        ),
    )

    year_to: int | None = Field(
        default=2100,
        description="Filter records published on or before this year",
    )


class SearchCoordinatorConfig(BaseModel):
    """Input model to create a single SearchCoordinator.

    Use a provider to search a specific API for scholarly records with automatic schema normalization.

    Attributes:
        queries (str):
            The search query for academic papers, books, and preprints.
        provider_name (str):
            The name of the current Academic API provider.
        api_specific_fields (str):
            Additional keyword arguments/overrides to use when creating an API-specific SearchCoordinator

    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore",
    )

    query: str = Field(
        default="",
        description=(
            "Search query for academic papers, books, and preprints. Examples: "
            "'statistical methods', 'predictors of classroom success', 'anxiety biomarkers fMRI', "
            "'PTSD veteran intervention randomized controlled trial'"
        ),
        min_length=2,
    )

    provider_name: APIProviders = Field(
        description=(
            "Academic database provider to search. Options: "
            "pubmed, crossref, plos, arxiv, openalex, core, springernature. "
        ),
    )

    api_specific_fields: dict = Field(
        default_factory=dict,
        description=(
            "Additional keyword arguments/overrides to use when creating an API-specific SearchCoordinator. "
            "Examples: `filter='is_oa:true' (OpenAlex), `sort='publication_date desc` (PLOS), etc. "
        ),
    )

    @field_validator("provider_name", mode="before")
    @classmethod
    def validate_provider_name(cls, v: str | APIProviders) -> APIProviders:
        """Validates the name of the current provider against the current list of known providers."""
        return APIProviders(v.lower()) if isinstance(v, str) else v

    @field_validator("query", mode="before")
    @classmethod
    def validate_query_character_limits(cls, v: str | list[str]) -> str | list[str]:
        """Converts string queries into lists if not already a list.

        Validates each query afterward.

        """
        QueryCharacterLimit.validate_character_limits(v)
        return v


class SearchInput(BaseSearchParams):
    """Input parameters for academic record search.

    This model validates search requests across multiple academic providers. All parameters have sensible defaults for
    general research workflows.

    Attributes:
        max_records (int): Maximum records to retrieve per provider
        pages (int): Number of result pages to retrieve per provider
        page_offset (int): The page number to start the retrieval of `n` pages from.
        year_from (int | None): Filter records published on or after this year (e.g., 2020)
        year_to (int | None): Filter records published on or before this year.
        open_access_only (bool): Flag Only return open access records.
        search_coordinator_config (list[SearchCoordinatorConfig]): Provider configurations to query.

    """

    HIDDEN_FIELDS: ClassVar[set[str]] = {"search_coordinator_config"}

    search_coordinator_config: list[SearchCoordinatorConfig] = Field(
        default_factory=list, description="Provider configurations to query"
    )

    DEFAULT_PROVIDERS: ClassVar[tuple[APIProviders, ...]] = (
        APIProviders.PUBMED,
        APIProviders.PLOS,
        APIProviders.OPENALEX,
    )

    @classmethod
    def create(
        cls, queries: str | list[str], providers: str | list[str] | list[APIProviders] | None = None, **kwargs: Any
    ) -> SearchInput:
        """Creates a new `SearchInput` model instance by taking `queries` and `providers` directly."""

        query_list = [queries] if isinstance(queries, str) or not isinstance(queries, Iterable) else queries
        providers = providers or list(cls.DEFAULT_PROVIDERS)

        config = [
            SearchCoordinatorConfig(query=q, provider_name=provider)
            for provider in cls.convert_provider_strings(providers or [])
            for q in query_list
        ]
        return cls(search_coordinator_config=config, **kwargs)

    @classmethod
    def convert_provider_strings(cls, v: str | list[str] | list[APIProviders]) -> list[APIProviders]:
        """Converts provider names into its respective `APIProviders` enum."""
        if isinstance(v, list):
            return [APIProviders(p) if isinstance(p, str) else p for p in v]
        if isinstance(v, str):
            return [APIProviders(v)]
        raise ValueError(f"Expected a string or list of strings, but received type {v}")

    @classmethod
    def from_search_params(cls, params: SynthesisInput | RelevanceSearchInput) -> SearchInput:
        """Factory helper for reconstructing a `SearchInput` class from a `SynthesisInput` or `RelevanceSearchInput`."""
        # If no providers are indicated, default to the PubMed database and PLOS API
        providers = params.providers if params.providers else cls.DEFAULT_PROVIDERS

        # The Pydantic SearchInput model for the SearchService: retrieves the records used for synthesis generation
        return SearchInput.create(
            queries=params.queries,
            providers=list(providers),
            max_records=params.max_records,
            pages=params.pages,
            year_from=params.year_from,
            year_to=params.year_to,
            open_access_only=params.open_access_only,
        )

    @computed_property
    def providers(self) -> list[APIProviders]:
        """Return providers as a list for uniform handling."""
        return list(dict.fromkeys(config.provider_name for config in self.search_coordinator_config))

    @computed_property
    def queries(self) -> list[str]:
        """Return query as a list for uniform handling."""
        return list(dict.fromkeys(config.query for config in self.search_coordinator_config))

    @property
    def maximum_request_total(self) -> int:
        """Indicates the total number of requests possible given the total pages, queries, and API providers."""
        return len(self.search_coordinator_config) * self.pages


class BaseSearchRecord(BaseModel, frozen=True):
    """A Normalized academic record containing the DOI, provider name, title, and other relevant academic fields.

    Contains standardized fields across all providers after ScholarFlux normalization.

    Attributes:
        provider_name (str):
            The source provider identifier (e.g., 'pubmed', 'crossref').
        doi (str | None):
            Digital Object Identifier for the academic record.
        url (str | None):
            The URL link to the academic record.
        record_id (str | None):
            Provider-specific record identifier.
        title (str | None):
            The title of the academic record.
        abstract (str | None):
            The abstract of the academic record.
        authors (list[str] | str | None):
            Author names associated with the record.
        full_text (str | None):
            The full text of the academic record if available.
        journal (str | None):
            The journal name where the record was published.
        publisher (str | None):
            The publisher of the record.
        year (int | None):
            The publication year of the record.
        date_published (str | None):
            The publication date of the record.
        keywords (list[str] | str | None):
            Keywords or MeSH terms associated with the record.
        subjects (list[str] | str | None):
            Subject categories for the record.
        citation_count (int | None):
            The number of citations the record has received.
        open_access (bool | None):
            The open access status of the record.

    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="ignore",
    )

    DEFAULT_STRICT_VALIDATION: ClassVar[bool] = True
    DEFAULT_RECORD_STRING_FIELDS: ClassVar[list[str]] = [
        "title",
        "year",
        "authors",
        "doi",
        "abstract",
        "open_access",
        "full_text",
    ]
    DEFAULT_TOPIC_FIELDS: ClassVar[list[str]] = [
        "title",
        "authors",
        "doi",
        "abstract",
    ]

    # Core identifiers
    provider_name: str = Field(description="Source provider (e.g., 'pubmed', 'crossref')")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    url: str | None = Field(default=None, description="The URL link to the academic record")
    record_id: str | None = Field(default=None, description="Provider-specific record ID")

    # Bibliographic metadata
    title: str | None = Field(default=None, description="Academic record title")
    abstract: str | None = Field(default=None, description="Academic record abstract")
    authors: list[str] | str | None = Field(default=None, description="Author names")
    full_text: str | None = Field(default=None, description="Academic record text if available")

    # Publication metadata
    journal: str | None = Field(default=None, description="Journal name")
    publisher: str | None = Field(default=None, description="Publisher name")
    year: int | None = Field(default=None, description="Publication year")
    date_published: str | None = Field(default=None, description="Publication date")

    # Content classification
    keywords: list[str] | str | None = Field(default=None, description="Keywords/MeSH terms")
    subjects: list[str] | str | None = Field(default=None, description="Subject categories")

    # Metrics
    citation_count: int | None = Field(default=None, description="Number of citations")
    open_access: bool | None = Field(default=None, description="Open access status")

    @property
    def topic(self) -> str:
        """Identifies the corresponding research topic from core parameters available within the current model."""
        return self.to_string(self.DEFAULT_TOPIC_FIELDS)

    @property
    def display_name(self) -> str:
        """The display name of the academic database API from which the record was retrieved."""
        return (
            metadata.value.display_name
            if (metadata := ProviderMetadataDescriptions.get(self.provider_name))
            else self.provider_name.title()
        )

    @model_validator(mode="before")
    @classmethod
    def coerce_record_fields(cls, data: dict) -> dict:
        """Applies type coercion to fields if strict validation is disabled.

        Otherwise, records are validated as is.

        """
        if cls.DEFAULT_STRICT_VALIDATION:
            return data

        for field in cls.model_fields:
            field_info = SearchRecordFields.get(field)
            type_info = field_info.value.type if field_info else None

            # Raise a KeyError if a field with a type found in `SearchRecordFields` Enum is missing from `data`
            if type_info is None or (field_value := data[field]) is None:
                continue

            # Note: bool types (open_access) are left as is given potential difficulties with interpretation otherwise.

            elif type_info is FieldTypes.STR_FIELD:
                data[field] = coerce_str(field_value)

            elif type_info is FieldTypes.LIST_OR_STR_FIELD:
                # If it's already a list, coerce each element to str
                # Otherwise coerce the whole value to str
                data[field] = (
                    [coerce_str(item) for item in field_value]
                    if isinstance(field_value, list)
                    else coerce_str(field_value)
                )

            elif type_info is FieldTypes.INT_FIELD:
                data[field] = coerce_int(field_value)

        return data

    @computed_cached_property
    def record_hash(self) -> str:
        """Creates a hash for the current SearchRecord to enable later identification."""
        core_fields = self.get_record_hash_fields()
        record_dict = self.model_dump(include=core_fields)
        stable_str = json.dumps(record_dict, sort_keys=True, default=str)
        return hashlib.md5(stable_str.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def get_record_hash_fields(cls) -> set[str]:
        """Defines the core fields used to create a hash identifier for the current `BaseSearchRecord` or subclass."""
        return set(cls.model_fields)

    def __hash__(self) -> int:
        """Utilizes the content hash."""
        return hash(self.record_hash)

    @overload
    def get(self, key: str, default: None = None) -> list[str] | str | bool | int | float | None:
        """When the field cannot be found and a default is not provided, None is returned instead."""
        ...

    @overload
    def get(self, key: str, default: T) -> list[str] | str | bool | int | float | None | T:
        """When the field cannot be found, the user-specified default will be returned instead."""
        ...

    def get(self, key: str, default: T | None = None) -> list[str] | str | bool | int | float | None | T:
        """Helper method for retrieving the value of a field within the current record."""
        return getattr(self, key, default) if key in self.__class__.model_fields else None

    def to_string(
        self,
        fields: list[str] | None = None,
        record_truncation_length: int | None = None,
        *,
        field_truncation_length: int | None = None,
    ) -> str:
        """Converts the current record into its string representation."""
        if not fields or not isinstance(fields, list | tuple):
            fields = self.DEFAULT_RECORD_STRING_FIELDS

        record_field_strings = "\n".join(
            truncate(
                f"{field_metadata.value.display}: {self.get(field) or field_metadata.value.default or 'N/A'}",
                field_truncation_length,
            )
            for field in fields
            if (field_metadata := SearchRecordFields.get(field)) is not None
        )

        return truncate(record_field_strings, record_truncation_length)

    def information_content(self) -> float:
        """Extracts information content based on the importance rankings assigned to each field.

        Returns:
            float:
                The sum of the available information content for the current record. This value is computed from the
                `SearchRecordFieldInfo.relative_importance` score assigned to each individual field in the
                `SearchRecordFields`.

        """
        return sum(
            field_info.value.relative_importance
            for field, value in self.model_dump().items()
            if (value or isinstance(value, int)) and (field_info := SearchRecordFields.get(field))
        )


class SearchRecord(BaseSearchRecord, frozen=True):
    """A Normalized academic record containing the DOI, provider name, title, and other relevant academic fields.

    Contains standardized fields across all providers after ScholarFlux normalization.

    Attributes:
        page (int): The page number associated with the record at retrieval time.
        query (str): The query used to retrieve the current record
        cached (bool | None): default=None, description="Indicates whether the record was retrieved from cache
        retrieval_timestamp (TimeStamp | None): When the record was retrieved

    """

    # Page Metadata
    page: int = Field(description="The page number associated with the record at retrieval time.")
    query: str = Field(description="The query used to retrieve the current record")
    cached: bool | None = Field(default=None, description="Indicates whether the record was retrieved from cache")
    retrieval_timestamp: TimeStamp | None = Field(default=None, description="When the record was retrieved")

    @classmethod
    def get_record_hash_fields(cls) -> set[str]:
        """Defines the core fields used to create a hash identifier for the current `SearchRecord` or subclass."""
        return BaseSearchRecord.get_record_hash_fields() | {"page", "query"}

    @field_serializer("retrieval_timestamp", mode="plain")
    def serialize_timestamp(self, v: TimeStamp) -> TimeStampString | None:
        """Serializes the timestamp as a string from a datetime object."""
        return validate_iso_timestamp_string(v) if v is not None else None


class IndexedSearchRecord(SearchRecord, frozen=True):
    """An individual search record after filtering and reranking via record embeddings.

    Attributes:
        index (int): The index of the record after sorting and/or reranking via topic similarity.
        topic_similarity_score (float | None): The topic similarity score for the record.


    """

    index: int = Field(description="The index of the record after sorting and/or reranking via topic similarity.")
    topic_similarity_score: float | None = Field(default=None, description="The topic similarity score for the record.")

    @classmethod
    def from_search_record(cls, record: SearchRecord, index: int, topic_similarity_score: float | None = None) -> Self:
        """Helper method for constructing an IndexedSearchRecord from a basic SearchRecord."""
        search_record_fields = record.model_dump(include=set(SearchRecord.model_fields))
        topic_similarity_score = (
            topic_similarity_score
            if topic_similarity_score is not None
            else coerce_numeric(record.get("topic_similarity_score"))
            if isinstance(record, cls | dict)
            else None
        )
        return cls(index=index, topic_similarity_score=topic_similarity_score, **search_record_fields)


def validate_research_category_list(
    value: str | ResearchCategory | list[str] | dict[str, Any] | list[ResearchCategory],
) -> list[ResearchCategory]:
    """Verifies that the passed category is a ResearchCategory or a ResearchCategory convertible string."""
    return [ResearchCategory(category) for category in as_tuple(value)]


def validate_search_record(
    value: dict[str, Any] | Sequence[dict[str, Any]] | SearchRecord | Sequence[SearchRecord],
) -> Sequence[SearchRecord]:
    """Verifies that the passed element is a SearchRecord or list of SearchRecord instances.

    Note:
        Singular SearchRecord instances and dictionaries containing search record fields are converted into
        a list with containing the validated SearchRecord.

    """
    record_list: Sequence[SearchRecord | dict[str, Any]] = [value] if isinstance(value, SearchRecord | dict) else value
    if isinstance(record_list, Iterator | tuple):
        record_list = list(record_list)
    if not isinstance(record_list, list):
        raise ValueError(f"Expected a list of SearchRecords, but received type {type(record_list)}.")
    try:
        return [
            record if isinstance(record, SearchRecord) else SearchRecord.model_validate(record)
            for record in record_list
        ]
    except ValidationError as e:
        raise ValueError(
            f"Expected a list of SearchRecords, but at least one element has an incorrect type: {e}"
        ) from e


def validate_indexed_search_record_list(
    value: Sequence[IndexedSearchRecord],
) -> Sequence[IndexedSearchRecord]:
    """Verifies that the passed category is a list of IndexedSearchRecords."""
    record_list = (
        list(value) if isinstance(value, Sequence | Iterator | tuple) and not isinstance(value, str) else value
    )

    if not isinstance(record_list, list):
        raise ValueError(f"Expected a list of IndexedSearchRecords, but received type {type(record_list)}.")
    try:
        requires_reindex = record_list and any(not isinstance(record, IndexedSearchRecord) for record in record_list)
        return (
            [
                IndexedSearchRecord.from_search_record(
                    record=record if isinstance(record, SearchRecord) else SearchRecord.model_validate(record),
                    index=i,
                )
                for i, record in enumerate(record_list)
            ]
            if requires_reindex
            else record_list
        )

    except ValidationError as e:
        raise ValueError(
            f"Expected a list of IndexedSearchRecords, but at least one element has an incorrect type: {e}"
        ) from e


# Basic Search tool
QueryList = Annotated[
    list[str],
    Field(
        ...,
        description=(
            "Search one or more queries for academic records across all providers. Examples: "
            "'depression treatment CBT', 'anxiety biomarkers fMRI', 'PTSD veteran intervention "
            "randomized controlled trial'"
        ),
        min_length=1,
    ),
]

# Values to be inferred from categories
ResearchQueryList = Annotated[
    list[str],
    Field(
        default_factory=list,
        description=(
            "Search queries to execute against academic databases. Provide multiple queries "
            "to cover different aspects of the research question. If empty, a query is "
            "derived from user-defined research categories. Query examples: ['CBT depression efficacy RCT', "
            "'cognitive behavioral therapy major depressive disorder', 'psychotherapy vs antidepressants']"
        ),
        max_length=10,
    ),
]
ResearchCategoryList = Annotated[list[ResearchCategory], BeforeValidator(validate_research_category_list)]
SearchRecordList = Annotated[Sequence[SearchRecord], BeforeValidator(validate_search_record)]
IndexedSearchRecordList = Annotated[Sequence[IndexedSearchRecord], BeforeValidator(validate_indexed_search_record_list)]


class SearchRecordEmbedding(JSONDataModel):
    """Records the query embedding and context of the query associated with a question.

    Attributes:
        record (SearchRecord):
            The embedded record retrieved from an academic API.
        embedding (Sequence[float] | None):
            The embedding for the current record.
        model_name (str):
            The name of the model used to generate the record embedding.

    """

    HIDDEN_FIELDS: ClassVar[set[str]] = {"embedding"}

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    record: SearchRecord = Field(description="The embedded record retrieved from an academic API")
    embedding: Sequence[float] | None = Field(
        default=None, description="The embedding for the current record", repr=False
    )
    model_name: str = Field(default="N/A", description="The name of the model used to generate the record embedding.")

    def __bool__(self) -> bool:
        """Indicates whether the current SearchRecordEmbedding contains a valid record and embedding."""
        return bool(self.record and self.embedding)


class ResearchTopic(JSONDataModel):
    """Value object representing the semantic scope of a research search.

    This object encapsulates the input scope (question, queries, and categories)
    and provides methods to format the topic for different similarity algorithms
    (embeddings vs fuzzy matching).

    Attributes:
        queries (list[str]): A list of specific search queries.
        question (str): The primary research question or topic.
        categories (ResearchCategoryList): A list of research categories.

    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    queries: list[str] = Field(default_factory=list, description="The queries sent to the API Provider")
    question: str = Field(default="", description="The research question asked by the user")
    categories: ResearchCategoryList = Field(
        default_factory=list, description="The research categories related to the subject matter"
    )
    topic: str = Field(
        default="",
        description="The topic of the synthesis, including the research question, query, and specified categories.",
    )

    @model_validator(mode="after")
    def validate_topic(self) -> Self:
        """Applies type coercion to response fields when success or status code is known."""
        if not self.queries and not self.question and not self.categories:
            raise ValueError(
                "Expected at least one of `queries`, `question`, and `categories` to be populated, but each of these fields is empty"
            )
        if not self.topic:
            updated_topic = self.prepare_topic(queries=self.queries, question=self.question, categories=self.categories)
            object.__setattr__(self, "topic", updated_topic)
        return self

    @classmethod
    def create(cls, model_data: BaseModel | dict[str, Any]) -> Self:
        """Helper that extracts relevant fields and creates a new topic or subclass from model fields."""
        model_dict = (
            model_data.model_dump(include=set(cls.model_fields)) if isinstance(model_data, BaseModel) else model_data
        )
        return cls.model_validate(model_dict, extra="ignore")  # ignore is a valid field on pydantic v2

    @classmethod
    def prepare_topic(
        cls, queries: str | list[str], question: str, categories: str | list[ResearchCategory] | None = None
    ) -> str:
        """Prepares the topic string for use with later fuzzy finding and embedding approaches."""

        category_string = ", ".join(
            ", ".join(ResearchCategory(category).value.terms) for category in as_tuple(categories)
        )
        query_string = "; ".join(as_tuple(queries))

        if not category_string and not question:
            return query_string

        topic = dedent(
            f"""\
            {question}
            --
            {category_string}
            --
            {query_string}"""
        )

        return topic.replace("--\n\n--", "--\n").replace("\n\n", "\n").strip("-\n")


class TopicEmbedding(ResearchTopic):
    """Records the embedding and context of the query associated with a question.

    Attributes:
        embedding (Sequence[float]):
            The embedding for the current research topic.
        model_name (str):
            The name of the model used to generate the topic embedding.

    """

    HIDDEN_FIELDS: ClassVar[set[str]] = {"embedding"}

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )
    embedding: Sequence[float] = Field(description="The embedding for the current research topic", repr=False)
    model_name: str = Field(default="N/A", description="The name of the model used to generate the topic embedding.")

    @field_serializer("categories", mode="plain")
    def serialize_category_names(self, v: ResearchCategoryList) -> list[str]:
        """Serializes each category name given the provided `ResearchCategoryList`."""
        return ResearchCategory.as_category_names(v)

    def __bool__(self) -> bool:
        """Indicates whether the current TopicEmbedding contains a valid topic and embedding."""
        return bool(self.topic and self.embedding)


class RecordTopicSimilarity(BaseModel):
    """Helper for recording the relevance of retrieved records to the question or topic of interest.

    Attributes:
        record_embedding (SearchRecordEmbedding):
            The record and its associated embedding.
        topic (str):
            The research topic to compare the record content against.
        topic_similarity_score (float):
            The relevance of the record content to the question. Calculated via cosine similarity.

    """

    record_embedding: SearchRecordEmbedding = Field(description="The record and its associated embedding.")
    topic: str = Field(description="The research topic to compare the record content against.")
    topic_similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The relevance of the record content to the question. Calculated via cosine similarity",
    )

    @field_validator("topic_similarity_score", mode="before")
    @classmethod
    def validate_similarity_score(cls, v: float) -> float:
        """Verifies and clamps otherwise valid embedding similarity scores (rounding error) between .0 and 1.0."""
        return max(min(v, 1.0), 0.0) if isinstance(v, float | int) and (0.0 > v > -0.05 or 1.0 < v < 1.05) else v

    @property
    def record(self) -> SearchRecord:
        """The embedded record retrieved from an academic API"""
        return self.record_embedding.record

    @property
    def embedding(self) -> Sequence[float] | None:
        """The embedding for the current record."""
        return self.record_embedding.embedding

    def __bool__(self) -> bool:
        """Indicates whether the RecordTopicSimilarity instance contains valid embeddings and similarity scores."""
        return bool(self.record_embedding and self.topic and self.topic_similarity_score is not None)


class RecordTopicSimilarityOutput(BaseModel):
    """Records RecordTopicSimilarityEmbedder output, including the record and topic embeddings and their similarity."""

    topic_embedding: TopicEmbedding = Field(description="The research topic and its embedding.")
    record_similarity_scores: list[RecordTopicSimilarity] = Field(
        description="The record and its cosine similarity to the topic."
    )

    @property
    def model_name(self) -> str:
        """Embedding model used to calculate record-topic similarity scores for the original question/queries."""
        return self.topic_embedding.model_name

    def __bool__(self) -> bool:
        """Indicates whether the RecordTopicSimilarity instance contains valid embeddings and similarity scores."""
        return bool(
            self.topic_embedding
            and self.record_similarity_scores
            and all(bool(score) for score in self.record_similarity_scores)
        )


class PaginationInfo(BaseModel):
    """Pagination metadata for search results.

    Attributes:
        total_records (int):
            Total number of records retrieved.
        providers_queried (int):
            Number of providers searched.
        providers_successful (int):
            Number of providers that returned results.
        pages_successful (int):
            Total pages that returned a valid response across all pages.
        pages_retrieved (int):
            Total pages retrieved across all queries and providers.

    """

    total_records: int = Field(description="Total number of records retrieved")
    providers_queried: int = Field(description="Number of providers searched")
    providers_successful: int = Field(description="Number of providers that returned results")
    pages_successful: int = Field(description="Total pages that returned a valid response across all pages")
    pages_retrieved: int = Field(description="Total pages retrieved across all queries and providers")


class SearchResponseSummary(JSONDataModel):
    """Response summary generated from a single response query and page received from an API provider.

    Attributes:
        provider_name (str):
            The academic API provider.
        query (str):
            The query used for this response.
        page (int):
            The page number retrieved.
        success (bool):
            Whether a valid response was received, independent of record count.
        status_code (int | None):
            The status code associated with the current response.
        error (str | None):
            Error associated with the current response.
        message (str | None):
            The message associated with the current response.
        cached (bool | None):
            Whether the response was served from session cache.
        retrieval_timestamp (TimeStamp | None):
            When the response was retrieved.
        record_count (int):
            The Number of records returned by this response.

    """

    model_config = ConfigDict(frozen=True)
    provider_name: str = Field(description="The academic API provider.")
    query: str = Field(description="The query used for this response.")
    page: int = Field(description="The page number retrieved.")
    success: bool = Field(description="Whether a valid response was received, independent of record count.")
    status_code: int | None = Field(default=None, description="The status code associated with the current response.")
    error: str | None = Field(default=None, description="Error associated with the current response.")
    message: str | None = Field(default=None, description="The message associated with the current response")
    cached: bool | None = Field(default=None, description="Whether the response was served from session cache.")
    retrieval_timestamp: TimeStamp | None = Field(default=None, description="When the response was retrieved.")
    record_count: int = Field(default=0, description="The Number of records returned by this response.")

    @field_serializer("retrieval_timestamp", mode="plain")
    def serialize_timestamp(self, v: TimeStamp) -> TimeStampString | None:
        """Serializes the timestamp as a string from a datetime object."""
        return validate_iso_timestamp_string(v) if v is not None else None

    @model_validator(mode="before")
    @classmethod
    def coerce_inferred_response_fields(cls, data: dict) -> dict:
        """Applies type coercion to response fields when success or status code is known."""
        if data.get("success") is True and data.get("status_code") is None:
            data["status_code"] = 200  # assume 200 when success is provided

        return data

    @classmethod
    def get_response_hash_fields(cls) -> set[str]:
        """Defines the core fields used to create a hash identifier for the current `SearchResponseSummary`."""
        return set(cls.model_fields)

    @computed_cached_property
    def response_hash(self) -> str:
        """Creates a hash for the current SearchResponseSummary to enable later identification."""
        core_fields = self.get_response_hash_fields()
        response_dict = self.model_dump(include=core_fields)
        stable_str = json.dumps(response_dict, sort_keys=True, default=str)
        return hashlib.md5(stable_str.encode("utf-8")).hexdigest()[:16]


class SearchOutput(JSONDataModel):
    """Output model for search results.

    Contains normalized records with pagination info and per-response observability summaries.

    Attributes:
        search_input (SearchInput):
            The original search input used to produce these results.
        records (SearchRecordList):
            Normalized academic records.
        pagination (PaginationInfo):
            Pagination metadata.
        response_summaries (list[SearchResponseSummary]):
            Per-response observability summaries, one entry per provider/query/page combination.
        cache_hit (bool):
            Indicates whether all records originate from cache.

    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore",
    )

    PERCENT_RETRIEVAL_SUCCESS_THRESHOLD: ClassVar[float] = 0.75

    search_input: SearchInput = Field(description="The original search input used to produce these results", repr=False)
    records: SearchRecordList = Field(description="Normalized academic records")
    pagination: PaginationInfo = Field(description="Pagination metadata")
    response_summaries: list[SearchResponseSummary] = Field(
        default_factory=list,
        description="Per-response observability summaries, one entry per provider/query/page combination.",
    )

    @computed_property
    def cache_hit(self) -> bool:
        """Indicates whether all records originate from cache."""
        return all(record.cached for record in self.records)

    @property
    def topic(self) -> str:
        """Convenience alias for `SearchOutput.search_input.topic."""
        return self.search_input.topic

    @property
    def queries(self) -> list[str]:
        """The original search query or queries used to search for academic records from each API."""
        return self.search_input.queries

    @property
    def providers(self) -> list[APIProviders]:
        """Providers derived from the original search input."""
        return self.search_input.providers

    @property
    def successful(self) -> bool:
        """Indicates the successful retrieval of the minimum percentage of pages across all queries and providers.

        Note: The `PERCENT_RETRIEVAL_SUCCESS_THRESHOLD` class attribute defines the percentage threshold required to
        indicate the record search as successful.

        Returns:
            bool: True if the minimum percentage of pages have been retrieved, False otherwise.

        """
        if self.pagination.pages_successful < 1:
            return False

        success_threshold = min(self.PERCENT_RETRIEVAL_SUCCESS_THRESHOLD, 1)
        return self.pagination.pages_successful / self.pagination.pages_retrieved > success_threshold


# =============================================================================
# SYNTHESIS MODELS
# =============================================================================


class AgentEvidenceItem(BaseModel):
    """AI-generated evidence using record index (internal use).

    Attributes:
        record_index (int):
            Index of the record in the provided list (0-based).
        referenced_text (str | None):
            Reference text extracted from the current record (e.g., "CBT shows 60% efficacy").
        finding (str):
            Key finding or claim extracted from this record.
        relevance_score (float):
            Relevance score to the synthesis topic (0-1).

    """

    record_index: int = Field(description="Index of the record in the provided list (0-based)")
    referenced_text: str | None = Field(
        default=None, description="Reference text extracted from the current record (e.g., 'CBT shows 60% efficacy')"
    )
    finding: str = Field(description="Key finding or claim extracted from this record")
    relevance_score: float = Field(
        default=0.0,
        description="Relevance score to the synthesis topic (0-1)",
        ge=0.0,
        le=1.0,
    )


class GroundedEvidenceItem(AgentEvidenceItem):
    """A piece of evidence extracted from academic literature for synthesis."""

    record: SearchRecord = Field(description="The record referenced by the evidence item")
    referenced_text_similarity: float | None = Field(
        default=None, description="The similarity of the record to listed citation"
    )

    @property
    def title(self) -> str:
        """The title associated with the referenced record."""
        return self.record.title or "Unknown"

    @property
    def doi(self) -> str | None:
        """The DOI associated with the referenced record."""
        return self.record.doi

    @property
    def url(self) -> str | None:
        """The URL associated with the referenced record."""
        return self.record.url

    @property
    def year(self) -> int | None:
        """The year of creation/publication for the referenced record."""
        return self.record.year

    @property
    def authors(self) -> str | list[str] | None:
        """The authors of the referenced record."""
        return self.record.authors

    @property
    def display_name(self) -> str:
        """The name of the academic database provider from which the current record was retrieved."""
        return self.record.display_name

    @property
    def abstract(self) -> str:
        """The abstract of the referenced record."""
        return self.record.abstract or "Not Provided"

    @property
    def record_hash(self) -> str | None:
        """The current record hash of the referenced record."""
        return self.record.record_hash


def validate_agent_evidence_item_structure(
    value: AgentEvidenceItem | Sequence[AgentEvidenceItem],
) -> Sequence[AgentEvidenceItem]:
    """Verifies that the current value is a AgentEvidenceItem or list of AgentEvidenceItems."""
    record_list = [value] if isinstance(value, AgentEvidenceItem) else value
    if isinstance(record_list, Iterator | tuple):
        record_list = list(record_list)
    if not isinstance(record_list, list):
        raise ValueError(f"Expected a list of AgentEvidenceItem instances, but received type {type(record_list)}.")
    if not all(isinstance(record, AgentEvidenceItem) for record in record_list):
        raise ValueError("Expected a list of AgentEvidenceItems, but at least one element has an incorrect type.")
    return record_list


class RejectedEvidenceItem(AgentEvidenceItem):
    """An evidence item that was rejected due to a specific reason explained in the `error` field.

    Attributes:
        record (SearchRecord | None):
            The record referenced by the evidence item.
        error (str):
            The reason for rejection.
        referenced_text_similarity (float | None):
            The similarity of the record to listed citation.

    """

    record: SearchRecord | None = Field(default=None, description="The record referenced by the evidence item")
    error: str
    referenced_text_similarity: float | None = Field(
        default=None, description="The similarity of the record to listed citation"
    )

    @property
    def title(self) -> str | None:
        """The title associated with the referenced record."""
        return self.record.title if self.record else None

    @property
    def doi(self) -> str | None:
        """The DOI associated with the referenced record."""
        return self.record.doi if self.record else None

    @property
    def url(self) -> str | None:
        """The URL associated with the referenced record."""
        return self.record.url if self.record else None

    @property
    def year(self) -> int | None:
        """The year of creation/publication for the referenced record."""
        return self.record.year if self.record else None

    @property
    def authors(self) -> str | list[str] | None:
        """The authors of the referenced record."""
        return self.record.authors if self.record else None

    @property
    def display_name(self) -> str | None:
        """The name of the academic database provider from which the current record was retrieved."""
        return self.record.display_name if self.record else None

    @property
    def abstract(self) -> str | None:
        """The abstract of the referenced record."""
        return self.record.abstract if self.record else None

    @property
    def record_hash(self) -> str | None:
        """The current record hash of the referenced record."""
        return self.record.record_hash if self.record else None


AgentEvidenceItemList = Annotated[Sequence[AgentEvidenceItem], BeforeValidator(validate_agent_evidence_item_structure)]


class EvidenceGroundingInput(BaseModel):
    """The input for the GroundingService including the raw, AI-generated evidence items and a list of SearchRecords.

    Attributes:
        evidence_items (AgentEvidenceItemList):
            A list of evidence summaries generated by the Synthesis agent.
        source_records (SearchRecordList):
            The source SearchRecords used and referenced to synthesize the current research summary.

    """

    evidence_items: AgentEvidenceItemList = Field(
        description=(
            "A list of evidence summaries generated by the Synthesis agent. These represent the statements to be "
            "grounded based on the passed search record list."
        )
    )
    source_records: SearchRecordList = Field(
        description="The source SearchRecords used and referenced to synthesize the current research summary."
    )


class EvidenceGroundingStats(JSONDataModel):
    """Statistics from grounded indexed references from AI generated output.

    Attributes:
        references_grounded (int):
            The total number of AI generated references that were grounded in evidence.
        references_rejected (int):
            The total number of AI generated references that were rejected.
        error (str | None):
            The exception raised during grounding in case of an error.

    """

    HIDDEN_FIELDS: ClassVar[set[str]] = {"error"}

    references_grounded: int = Field(
        default=0, description="The total number of AI generated references that were grounded in evidence."
    )
    references_rejected: int = Field(
        default=0, description="The total number of AI generated references that were rejected."
    )
    error: str | None = Field(default=None, description="The exception raised during grounding in case of an error.")


class EvidenceGroundingOutput(BaseModel):
    """Returns the result set and statistics after grounding indexed record references from AI Generated Output.

    Attributes:
        grounded_evidence_items (list[GroundedEvidenceItem]):
            A list of the grounded evidence items previously generated from synthesis output.
        rejected_evidence_items (list[RejectedEvidenceItem]):
            A list of rejected evidence items added for observability and debugging. Not serialized.
        grounding_stats (EvidenceGroundingStats):
            Grounding statistics indicating the proportion of accepted/rejected AI generated references.

    """

    grounded_evidence_items: list[GroundedEvidenceItem] = Field(
        default_factory=list,
        description="A list of the grounded evidence items previously generated from synthesis output",
    )
    rejected_evidence_items: list[RejectedEvidenceItem] = Field(
        default_factory=list,
        description="A list of rejected evidence items added for observability and debugging. Not serialized",
        repr=False,
    )
    grounding_stats: EvidenceGroundingStats = Field(
        default_factory=EvidenceGroundingStats,
        description="Grounding statistics indicating the proportion of accepted/rejected AI generated references",
    )


class RelevanceSearchInput(BaseRelevanceSearchParams):
    """Relevance search parameters used to sort and rerank the record retrieved from APIs."""

    DEFAULT_PROVIDERS: ClassVar[tuple[APIProviders, ...]] = (
        APIProviders.PUBMED,
        APIProviders.PLOS,
    )
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )
    queries: ResearchQueryList = Field(default_factory=list)

    categories: ResearchCategoryList = Field(
        default=[ResearchCategory.GENERAL],
        description="A List of possible categories used for topic relevance scoring.",
        max_length=5,
    )

    providers: list[APIProviders] = Field(
        default_factory=lambda: list(RelevanceSearchInput.DEFAULT_PROVIDERS),
        description="API Providers used in the record relevance search.",
    )

    @field_serializer("categories", mode="plain")
    def serialize_category_names(self, v: ResearchCategoryList) -> list[str]:
        """Serializes each category name given the provided `ResearchCategoryList`."""
        return ResearchCategory.as_category_names(v)

    @field_serializer("providers", mode="plain")
    def serialize_providers(self, v: list[APIProviders]) -> list[str]:
        """Serializes each provider name using its corresponding code."""
        return [provider.value for provider in v]

    @classmethod
    def create(
        cls,
        question: str,
        queries: str | list[str] | None,
        categories: str | list[str] | SubjectInfo | ResearchCategory | list[ResearchCategory] | None = None,
        providers: str | list[str] | APIProviders | list[APIProviders] | None = None,
        similarity_threshold: float | None = None,
        **kwargs: Any,
    ) -> RelevanceSearchInput:
        """Creates a new `RelevanceSearchInput` instance from `question`, `queries`, 'providers and `categories` fields."""

        query_list = list(as_tuple(queries))
        api_providers_list = [APIProviders(provider) for provider in as_tuple(providers)]
        research_categories_list = [ResearchCategory(category) for category in as_tuple(categories)]
        return cls(
            question=question,
            queries=query_list,
            providers=api_providers_list,
            categories=research_categories_list,
            similarity_threshold=similarity_threshold,
            **kwargs,
        )

    @classmethod
    def from_synthesis_params(cls, params: SynthesisInput) -> Self:
        """Factory method for creating the RelevanceSearchInput from SynthesisInput parameters."""
        record_relevance_fields = params.model_dump(include=set(cls.model_fields))
        return cls.model_validate(record_relevance_fields)


class SynthesisInput(BaseSynthesisParams):
    """Input parameters for literature synthesis.

    This model is used as input for the SynthesisService that uses PydanticAI to analyze and synthesize findings from
    records across a wid range of academic subjects.

    """

    DEFAULT_PROVIDERS: ClassVar[tuple[APIProviders, ...]] = (
        APIProviders.PUBMED,
        APIProviders.PLOS,
    )

    categories: ResearchCategoryList = Field(
        default=[ResearchCategory.GENERAL],
        description="A List of possible categories to focus subsequent literature synthesis on",
        max_length=5,
    )

    providers: list[APIProviders] = Field(
        default_factory=lambda: list(SynthesisInput.DEFAULT_PROVIDERS),
        description="Providers to search for synthesis source material",
    )

    @field_serializer("categories", mode="plain")
    def serialize_category_names(self, v: ResearchCategoryList) -> list[str]:
        """Serializes each category name given the provided `ResearchCategoryList`."""
        return ResearchCategory.as_category_names(v)

    @field_serializer("providers", mode="plain")
    def serialize_providers(self, v: list[APIProviders]) -> list[str]:
        """Serializes each provider name using its corresponding code."""
        return [provider.value for provider in v]

    @classmethod
    def create(
        cls,
        question: str,
        queries: str | list[str] | None,
        providers: str | list[str] | APIProviders | list[APIProviders] | None = None,
        categories: str | list[str] | SubjectInfo | ResearchCategory | list[ResearchCategory] | None = None,
        **kwargs: Any,
    ) -> SynthesisInput:
        """Creates a new `SynthesisInput` instance from `question`, `queries`, 'providers and `categories` fields."""

        query_list = list(as_tuple(queries))
        api_providers_list = [APIProviders(provider) for provider in as_tuple(providers)]
        research_categories_list = [ResearchCategory(category) for category in as_tuple(categories)]
        return cls(
            question=question,
            queries=query_list,
            providers=api_providers_list,
            categories=research_categories_list,
            **kwargs,
        )


@dataclass
class SynthesisContext:
    """Context to be passed as input to the PydanticAI synthesis agent.

    Contains academic record data and metadata.

    Attributes:
        question (str):
            The research question for synthesis.
        categories (list[ResearchCategory]):
            The research categories for context.
        records (IndexedSearchRecordList):
            The indexed search records to use in synthesis.

    """

    question: str
    categories: list[ResearchCategory]
    records: IndexedSearchRecordList

    @computed_property
    def total_records(self) -> int:
        """Property indicating the total number of available, analyzable records."""
        return len(self.records)


class SynthesisAgentOutput(BaseModel):
    """Structured output generated from a PydanticAI synthesis agent.

    Note: The output contains a list of evidence items that reference the position of each record in the received list.
    The `GroundingService` later uses the record_index to validate record index bounds and citation matches to verify
    whether the AgentEvidenceItem references an actual record with a valid citation.

    Attributes:
        synthesis (str):
            The synthesized research summary.
        confidence_score (float):
            The confidence score assigned to the synthesis.
        key_findings (list[str]):
            Key findings extracted from the synthesized output.
        evidence_summaries (list[AgentEvidenceItem]):
            Evidence items referencing records from the input.
        limitations (list[str]):
            Limitations identified by the synthesis agent.
        suggested_queries (list[str]):
            Additional queries suggested for further exploration.

    """

    SYNTHESIS_DEFAULT: ClassVar[str] = "Synthesis not available"

    synthesis: str = Field(default_factory=lambda: SynthesisAgentOutput.SYNTHESIS_DEFAULT)
    confidence_score: float = Field(default=0.5, le=1.0, ge=0.0)
    key_findings: list[str] = Field(default_factory=list)
    evidence_summaries: list[AgentEvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)

    @property
    def successful(self) -> bool:
        """Indicates whether the `SynthesisAgentOutput` is a valid synthesis with evidence and confidence scoring."""
        return bool(
            self.synthesis != self.SYNTHESIS_DEFAULT
            and self.confidence_score > 0
            and self.key_findings
            and self.evidence_summaries
        )


class RelevanceSearchOutput(JSONDataModel):
    """Output model for retrieval, deduplication, and reranking using embeddings and record relevance.

    Attributes:
        relevance_search_input (RelevanceSearchInput):
            The raw input used to generate the embeddings and relevance scoring.
        record_topic_similarity_output (RecordTopicSimilarityOutput | None):
            Record/Topic embeddings and cosine similarity output.
        indexed_records (IndexedSearchRecordList):
            A list of indexed record where each key represents its reference location after reordering.
        search_output (SearchOutput):
            Output from the record search.

    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore",
    )

    HIDDEN_FIELDS: ClassVar[set[str]] = {
        "relevance_search_input",
        "record_topic_similarity_output",
        "indexed_records",
        "search_output",
    }

    relevance_search_input: RelevanceSearchInput = Field(
        description="The Raw input used to generate the embeddings and relevance scoring", repr=False
    )
    record_topic_similarity_output: RecordTopicSimilarityOutput | None = Field(
        default=None, description="Record/Topic embeddings and cosine similarity output", repr=False
    )
    indexed_records: IndexedSearchRecordList = Field(
        default_factory=list,
        description="A list of indexed record where each key represents its reference location after reordering.",
        repr=False,
    )
    search_output: SearchOutput = Field(description="Output from the record search", repr=False)

    @property
    def search_input(self) -> SearchInput:
        """The original search input used to retrieve records sorted by relevance."""
        return self.search_output.search_input

    @property
    def records(self) -> IndexedSearchRecordList:
        """Alias for the complete list of indexed records."""
        return self.indexed_records

    @property
    def pagination(self) -> PaginationInfo:
        """Alias for the pagination metadata extracted from the output."""
        return self.search_output.pagination

    @property
    def cache_hit(self) -> bool:
        """Indicates whether the output from the original search was retrieved from cache."""
        return self.search_output.cache_hit

    @property
    def topic(self) -> str:
        """Convenience alias for `RelevanceSearchOutput.relevance_search_input.topic."""
        return self.relevance_search_input.topic

    @computed_property
    def question(self) -> str:
        """The question/topic of interest for the current relevance search."""
        return self.relevance_search_input.question

    @computed_property
    def queries(self) -> list[str]:
        """The queries used to retrieve records related to the relevance search."""
        return self.relevance_search_input.queries

    @computed_property
    def similarity_threshold(self) -> float | None:
        """The similarity threshold for retaining a record."""
        return self.relevance_search_input.similarity_threshold

    @computed_property
    def providers(self) -> list[APIProviders]:
        """The providers queried to retrieve records related to the relevance search."""
        return self.search_input.providers

    @computed_property
    def records_analyzed(self) -> int:
        """The number of records analyzed in the relevance search."""
        return len(self.indexed_records)

    @computed_property
    def categories_analyzed(self) -> list[ResearchCategory]:
        """The categories of interest for the relevance search."""
        return self.relevance_search_input.categories

    @computed_property
    def embedding_model_name(self) -> str:
        """Embedding model used to rank/filter records based on their similarity to the original question/queries."""
        model_name = self.record_topic_similarity_output.model_name if self.record_topic_similarity_output else None
        return model_name or "N/A"

    @computed_property
    def successful(self) -> bool:
        """Indicates whether the current `RelevanceSearchOutput` has retrieved and reranked records via embeddings."""
        return bool(self.search_output.successful and self.record_topic_similarity_output and self.indexed_records)


class SynthesisOutput(JSONDataModel):
    """Output model for research literature synthesis using structured inputs and outputs.

    Contains structured synthesis with evidence citations and confidence scoring.

    Attributes:
        synthesis_input (SynthesisInput):
            The raw input used to generate the research synthesis.
        agent_output (SynthesisAgentOutput):
            The output generated by the synthesis agent.
        grounding_output (EvidenceGroundingOutput):
            Evidence grounding results and statistics.
        relevance_search_output (RelevanceSearchOutput | None):
            Output from the record search used for synthesis.
        model_name (str):
            The large language model used to synthesize results.

    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore",
    )

    HIDDEN_FIELDS: ClassVar[set[str]] = {
        "synthesis_input",
        "agent_output",
        "relevance_search_output",
        "grounding_output",
    }

    synthesis_input: SynthesisInput = Field(
        description="The Raw input used to generate the research synthesis", repr=False
    )
    agent_output: SynthesisAgentOutput = Field(
        default_factory=SynthesisAgentOutput,
        description="The output generated by the synthesis agent",
        repr=False,
    )
    grounding_output: EvidenceGroundingOutput = Field(
        default_factory=EvidenceGroundingOutput,
        description="Evidence grounding results and statistics",
        repr=False,
    )

    relevance_search_output: RelevanceSearchOutput | None = Field(
        default=None, description="Output from the record search used for synthesis", repr=False
    )

    model_name: str = Field(default="N/A", description="The large language model used to synthesize results")

    @computed_property
    def indexed_records(self) -> IndexedSearchRecordList | None:
        """A list of indexed record where each key represents its reference location after reordering."""
        return self.relevance_search_output.indexed_records if self.relevance_search_output else None

    @computed_property
    def record_topic_similarity_output(self) -> RecordTopicSimilarityOutput | None:
        """Record/Topic embeddings and cosine similarity output."""
        return self.relevance_search_output.record_topic_similarity_output if self.relevance_search_output else None

    @property
    def search_output(self) -> SearchOutput | None:
        """Alias for the processed SearchOutput prior to reranking and synthesis."""
        return self.relevance_search_output.search_output if self.relevance_search_output else None

    @property
    def topic(self) -> str:
        """Convenience alias for `SynthesisOutput.synthesis_input.topic."""
        return self.synthesis_input.topic

    @computed_property
    def question(self) -> str:
        """The question/topic of interest for the research synthesis."""
        return self.synthesis_input.question

    @computed_property
    def queries(self) -> list[str]:
        """The queries used to retrieve records related to the current research synthesis."""
        return self.synthesis_input.queries

    @computed_property
    def records_analyzed(self) -> int:
        """The number of records analyzed in the research synthesis."""
        return len(self.indexed_records or [])

    @computed_property
    def categories_analyzed(self) -> list[ResearchCategory]:
        """The question/topic of interest for the research synthesis."""
        return self.synthesis_input.categories

    @computed_property
    def synthesis(self) -> str:
        """The synthesized answer based on the available records and topic of interest."""
        return self.agent_output.synthesis

    @computed_property
    def confidence_score(self) -> float:
        """The degree of confidence that the LLM assigned to the generated output after completing the synthesis."""
        return self.agent_output.confidence_score

    @computed_property
    def key_findings(self) -> list[str]:
        """A bulleted list of key findings, each of which references a record for synthesis generation."""
        return self.agent_output.key_findings or []

    @computed_property
    def limitations(self) -> list[str]:
        """Limitations of this synthesis according to the LLM."""
        return self.agent_output.limitations or []

    @computed_property
    def suggested_queries(self) -> list[str]:
        """Additional queries to further explore the topic or other topics related to the synthesis."""
        return self.agent_output.suggested_queries or []

    # Derived from grounding_output
    @computed_property
    def grounded_evidence(self) -> list[GroundedEvidenceItem]:
        """Key evidence derived from the records referenced from the synthesized report."""
        return self.grounding_output.grounded_evidence_items

    @computed_property
    def grounding_stats(self) -> EvidenceGroundingStats:
        """Grounding statistics related to the synthesized report, including rejection and acceptance statistics."""
        return self.grounding_output.grounding_stats

    @computed_property
    def embedding_model_name(self) -> str:
        """Embedding model used to rank/filter records based on their similarity to the original question/queries."""
        model_name = embedder.model_name if (embedder := self.record_topic_similarity_output) else None
        return model_name or "N/A"

    @computed_property
    def successful(self) -> bool:
        """Indicates whether the current `SynthesisOutput` has synthesized a complete report from nonmissing records."""
        return bool(self.relevance_search_output and self.relevance_search_output.successful and self.agent_output)


# =============================================================================
# UTILITY MODELS
# =============================================================================


class ServiceHealth(BaseModel):
    """Health status of an individual service.

    Attributes:
        name (str):
            The name of the service.
        status (str):
            Health status of the service (e.g., 'healthy', 'unhealthy', 'disabled').
        details (dict[str, Any]):
            Additional details about the service health.
        error (str | None):
            Error message if the service is unhealthy.

    """

    name: str = Field(description="The name of the service.")
    status: str = Field(description="Health status of the service (e.g., 'healthy', 'unhealthy', 'disabled')")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details about the service health")
    error: str | None = Field(default=None, description="Error message if the service is unhealthy")


class HealthStatus(BaseModel):
    """Health check response model.

    status (str): "Overall Health status (e.g., 'healthy' or 'unhealthy').
    version (str): ScholarFlux version.
    services (dict[str, ServiceHealth]): Health checks for individual components.
    timestamp (datetime): The timestamp indicating when the health status was last generated/updated.

    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore",
    )

    status: str = Field(description="Overall Health status (e.g., 'healthy' or 'unhealthy')")
    version: str = Field(default=__version__, description="ScholarFlux version")
    services: dict[str, ServiceHealth] = Field(
        default_factory=dict, description="Health checks for individual components"
    )
    timestamp: datetime = Field(
        default_factory=generate_datetime,
        description="The timestamp indicating when the health status was last generated/updated.",
    )

    def update_status(self, status: str = "healthy") -> HealthStatus:
        """Helper for appending or updating the current status of a ScholarFlux MCP service."""
        self.status = status
        self.timestamp = generate_datetime()  # updates the timestamp after status update
        return self

    def update_dependencies(self) -> HealthStatus:
        """Updates the current health check to reflect missing dependencies."""
        missing_dependencies = ScholarFluxMCPDependencies.identify_missing_dependencies()
        name = "Dependencies"
        if missing_dependencies:
            err = (
                "CoreDependencyImportError: At least one dependency is missing—some services may not operate as a "
                "result."
            )
            self.services["Dependencies"] = ServiceHealth(
                name="Dependencies",
                status="unhealthy",
                details={"dependencies": missing_dependencies},
                error=err,
            )

        else:
            self.services["Dependencies"] = ServiceHealth(
                name=name,
                status="healthy",
            )
        return self

    def update_service(
        self, service: str, status: str = "healthy", details: dict[str, Any] | None = None, error: str | None = None
    ) -> HealthStatus:
        """Helper for appending or updating the current status of a ScholarFlux MCP service."""
        self.services[service] = ServiceHealth(name=service, status=status, details=details or {}, error=error)
        self.timestamp = generate_datetime()  # updates the timestamp after service status updates
        return self


__all__ = [
    "BaseRelevanceSearchParams",
    "BaseSearchParams",
    "BaseSearchRecord",
    "BaseSynthesisParams",
    "EvidenceGroundingInput",
    "EvidenceGroundingOutput",
    "EvidenceGroundingStats",
    "GroundedEvidenceItem",
    "QueryList",
    "ResearchQueryList",
    "ResearchCategoryList",
    "HealthStatus",
    "IndexedSearchRecord",
    "PaginationInfo",
    "AgentEvidenceItem",
    "RecordTopicSimilarity",
    "RecordTopicSimilarityOutput",
    "RejectedEvidenceItem",
    "RelevanceSearchInput",
    "RelevanceSearchOutput",
    "ResearchTopic",
    "SearchCoordinatorConfig",
    "SearchInput",
    "SearchOutput",
    "SearchRecord",
    "SearchResponseSummary",
    "ServiceHealth",
    "SearchRecordEmbedding",
    "SynthesisAgentOutput",
    "SynthesisContext",
    "SynthesisInput",
    "SynthesisOutput",
    "TopicEmbedding",
    "SearchRecordList",
    "IndexedSearchRecordList",
]

"""Pydantic schemas for the ScholarFlux MCP server.

Defines all input/output models for MCP tools with comprehensive validation and documentation for LLM agent consumption.

"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, overload

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
)
from pydantic.dataclasses import dataclass

from scholar_flux_mcp.models.core import computed_cached_property
from scholar_flux_mcp.utils.helpers import (
    as_tuple,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# =============================================================================
# Responses
# =============================================================================


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"

    @classmethod
    def _missing_(cls, value: object) -> ResponseFormat | None:
        """Overrides the behavior of the current enum to retrieve the normalized category when basic retrieval fails."""
        if not isinstance(value, str):
            return None

        value_fmt = value.upper()
        return next(
            (format for format in cls if format.name == value_fmt),
            None,
        )


def validate_response_format(
    value: str | ResponseFormat,
) -> str:
    """Verifies that the received string is a valid response format using case-insensitive validation."""
    response_format = ResponseFormat(value)
    return response_format.value


ResponseFormatString = Annotated[str, BeforeValidator(validate_response_format)]

# =============================================================================
# Subject Info
# =============================================================================


@dataclass(frozen=True)
class SubjectInfo:
    """Dataclass for recording research subjects/methodologies and their corresponding term/keyword mappings.

    Attributes:
        name (str):
            The subject associated with a query/question (e.g., `MATHEMATICS`, `COMPUTATION`)
        display_name (str):
            A human-readable display name for the current subject.
        terms: (list[str]):
            A list of Keywords commonly associated with the current subject. Useful for semantic searches and embeddings

    """

    name: str = Field(default=..., description="The subject associated with a query/question")
    display_name: str = Field(default="", description="A human-readable display name for the current subject")
    terms: list[str] = Field(default_factory=list, description="Keywords commonly associated with the current subject")

    def __post_init__(self) -> None:
        """Post initialization hook for setting the `display_name` field from `SubjectInfo.name` if empty."""
        if not self.display_name.strip():
            object.__setattr__(self, "display_name", self.name.replace(" ", "_").title())


class ResearchCategory(Enum):
    """Research categories for query/record search classification and filtering."""

    META_ANALYSIS = SubjectInfo(
        name="meta_analysis",
        display_name="Meta-Analysis",
        terms=["meta-analysis", "pooled analysis", "combined analysis"],
    )
    THEORETICAL = SubjectInfo(name="theoretical", terms=["theoretical", "conceptual framework"])
    EMPIRICAL = SubjectInfo(name="empirical", terms=["empirical", "RCT", "experimental"])
    QUALITATIVE = SubjectInfo(name="qualitative", terms=["qualitative", "observational", "interviews"])
    QUANTITATIVE = SubjectInfo(name="quantitative", terms=["quantitative", "analysis", "correlational", "experimental"])
    DEPRESSION = SubjectInfo(
        name="depression", terms=["major depressive disorder", "depression treatment", "antidepressant"]
    )
    ANXIETY = SubjectInfo(name="anxiety", terms=["anxiety disorder", "generalized anxiety", "panic disorder"])
    PTSD = SubjectInfo(
        name="ptsd", display_name="PTSD", terms=["post-traumatic stress", "PTSD treatment", "trauma therapy"]
    )
    STRESS = SubjectInfo(name="stress", terms=["stress", "environmental stressors", "stress therapy"])
    BIPOLAR = SubjectInfo(
        name="bipolar", display_name="Bipolar Disorder", terms=["bipolar disorder", "mood stabilizer", "manic episode"]
    )
    SCHIZOPHRENIA = SubjectInfo(name="schizophrenia", terms=["schizophrenia", "psychosis", "antipsychotic"])
    SUBSTANCE_USE = SubjectInfo(
        name="substance_use", terms=["substance use disorder", "addiction treatment", "drug dependence"]
    )
    OCD = SubjectInfo(
        name="ocd", display_name="OCD", terms=["obsessive-compulsive disorder", "OCD treatment", "compulsive behavior"]
    )
    ADHD = SubjectInfo(
        name="adhd",
        display_name="ADHD",
        terms=["attention deficit hyperactivity disorder", "ADHD", "hyperactivity disorder"],
    )
    GENERAL_WELLBEING = SubjectInfo(
        name="general_wellbeing", terms=["general wellbeing", "physical health", "wellness", "psychological health"]
    )
    INTERVENTION = SubjectInfo(name="intervention", terms=["intervention", "therapy", "treatment efficacy"])
    EPIDEMIOLOGY = SubjectInfo(name="epidemiology", terms=["epidemiology", "prevalence", "disease surveillance"])
    MENTAL_HEALTH = SubjectInfo(
        name="mental_health", terms=["mental health", "psychological health", "psychological wellness"]
    )
    MATHEMATICS = SubjectInfo(name="mathematics", terms=["mathematics", "statistics", "numerical computing"])
    COMPUTATION = SubjectInfo(
        name="computation", terms=["computer science", "optimization", "statistical", "machine learning"]
    )
    GENERAL = SubjectInfo(name="general", terms=["general"])
    OTHER = SubjectInfo(name="other", terms=["other"])

    @classmethod
    def _missing_(cls, value: object) -> ResearchCategory | None:
        """Overrides the behavior of the current enum to retrieve the normalized category when basic retrieval fails."""
        if isinstance(value, dict) and set(value.keys()) == {"terms", "display_name", "name"}:
            value = value["name"]

        if not isinstance(value, str):
            return None

        formatted_category = value.upper().replace("-", "_").replace(" ", "_").strip(" _")

        return next(
            (category for category in cls if formatted_category == category.name),
            None,
        )

    @classmethod
    @overload
    def get(cls, subject: ResearchCategory) -> ResearchCategory:
        """When a research category is given, this category is returned as is."""
        ...

    @classmethod
    @overload
    def get(cls, subject: str | SubjectInfo) -> ResearchCategory | None:
        """A string or subject input returns a `ResearchCategory` item if it exists."""
        ...

    @classmethod
    def get(cls, subject: str | SubjectInfo | ResearchCategory) -> ResearchCategory | None:
        """Helper method for extracting the current subject information from a schema."""

        try:
            return cls(subject)
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def get_terms(cls, subject: str | SubjectInfo | ResearchCategory) -> list[str]:
        """Helper method for extracting the current terms associated with a particular category if it exists."""
        return research_category.value.terms if (research_category := cls.get(subject)) else []

    @classmethod
    def as_category_names(
        cls,
        subjects: str | SubjectInfo | ResearchCategory | Sequence[ResearchCategory | str | ResearchCategory],
        display_name: bool = False,
    ) -> list[str]:
        """Coerces a list of strings, ResearchCategories, or SubjectInfo into a list of category names."""
        return [
            category.value.display_name if display_name else category.name
            for cat in as_tuple(subjects)
            if (category := ResearchCategory.get(cat))
        ]

    @classmethod
    def as_category_list(
        cls, subjects: str | SubjectInfo | ResearchCategory | Sequence[ResearchCategory | str | ResearchCategory]
    ) -> list[ResearchCategory]:
        """Coerces a list of strings, ResearchCategories, or SubjectInfo into a list of `ResearchCategory` instances."""
        return [category for cat in as_tuple(subjects) if (category := ResearchCategory.get(cat))]


# =============================================================================
# Record Search Metadata
# =============================================================================


class FieldTypes(str, Enum):
    """Defines the most common field types for normalized record fields."""

    INT_FIELD = "int_field"
    STR_FIELD = "str_field"
    LIST_OR_STR_FIELD = "list_or_str_field"
    BOOL_FIELD = "bool"


class QueryCharacterLimit(int, Enum):
    """Helper defining and validating package-level query string size limits."""

    MIN = 2
    MAX = 500

    @classmethod
    def validate_character_limits(cls, queries: str | list) -> list[str]:
        """Verifies that the provided string or list is within the correct format."""
        query_list = list(as_tuple(queries))
        if invalid_queries := [
            query
            for query in query_list
            if not (isinstance(query, str) and QueryCharacterLimit.MIN <= len(query) <= QueryCharacterLimit.MAX)
        ]:
            sep = "\n- "
            invalid_query_list = sep + sep.join(str(query) for query in invalid_queries)
            raise ValueError(
                f"Expected all queries to be strings with a character length between {cls.MIN.value} and "
                f"{cls.MAX.value} characters each. The following queries are invalid: {invalid_query_list}"
            )
        return query_list


class QuestionCharacterLimit(int, Enum):
    """Defines the expected character limits of inputted questions for embedding/synthesis tools."""

    MIN = 8
    MAX = 1000

    @classmethod
    def validate_character_limit(cls, question: str) -> str:
        """Verifies that the question falls between the expected character length."""
        if not (
            isinstance(question, str) and (QuestionCharacterLimit.MIN <= len(question) <= QuestionCharacterLimit.MAX)
        ):
            received = f"a string of length {len(question)}" if isinstance(question, str) else f"{type(question)}"
            raise ValueError(
                f"Expected a valid, string formatted question with a character length between {cls.MIN.value} and "
                f"{cls.MAX.value} characters. Instead received {received}"
            )
        return question


@dataclass(frozen=True)
class SearchRecordFieldInfo:
    """Dataclass defining additional metadata for normalized fields from academic record searches.

    Provides field-level annotations including display names, default values, type information,
    and relative importance scores used for record string representations and information
    content calculations.

    Attributes:
        name (str):
            The technical name of the field (e.g., 'title', 'authors', 'year').
        display (str):
            A human-readable display name for the field (e.g., 'Title', 'Authors', 'Year').
        default (str):
            The default value to display when the field is missing from a record.
        type (FieldTypes | None):
            The expected data type for the field (e.g., STR_FIELD, INT_FIELD, LIST_OR_STR_FIELD).
        relative_importance (float):
            Importance score on a scale 0.0 to 1.0, used for information content calculations.

    """

    name: str = Field(description="The name of the field that is annotated with additional field metadata.")
    display: str = Field(
        description="A display name to use for the current field when creating string representations of records."
    )
    default: str = Field(
        default="N/A", description="The default to use when creating record string representations of missing fields."
    )
    type: FieldTypes | None = Field(default=None, description="The expected data type for the record field.")
    relative_importance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The relative importance of the current field on a scale of 0.0 to 1.0.",
    )


class SearchRecordFields(Enum):
    """Enum containing additional record field metadata used to create string representations of normalized records."""

    PROVIDER_NAME = SearchRecordFieldInfo(
        name="provider_name",
        display="API Provider",
        default="Untitled",
        relative_importance=0.5,
        type=FieldTypes.STR_FIELD,
    )
    DOI = SearchRecordFieldInfo(name="doi", display="DOI", relative_importance=1.0, type=FieldTypes.STR_FIELD)
    URL = SearchRecordFieldInfo(name="url", display="URL", relative_importance=1.0, type=FieldTypes.STR_FIELD)
    RECORD_ID = SearchRecordFieldInfo(
        name="record_id", display="ID", relative_importance=0.3, type=FieldTypes.STR_FIELD
    )

    # Bibliographic metadata
    TITLE = SearchRecordFieldInfo(
        name="title", display="Title", default="Untitled", relative_importance=1.0, type=FieldTypes.STR_FIELD
    )
    ABSTRACT = SearchRecordFieldInfo(
        name="abstract", display="Abstract", default="No Abstract", relative_importance=1.0, type=FieldTypes.STR_FIELD
    )
    FULL_TEXT = SearchRecordFieldInfo(
        name="full_text", display="full_text", default="N/A", relative_importance=0.3, type=FieldTypes.STR_FIELD
    )
    AUTHORS = SearchRecordFieldInfo(
        name="authors", display="Authors", relative_importance=1.0, type=FieldTypes.LIST_OR_STR_FIELD
    )
    # Publication metadata
    JOURNAL = SearchRecordFieldInfo(
        name="journal", display="Journal", relative_importance=0.75, type=FieldTypes.STR_FIELD
    )
    PUBLISHER = SearchRecordFieldInfo(
        name="publisher", display="Publisher", relative_importance=0.25, type=FieldTypes.STR_FIELD
    )
    YEAR = SearchRecordFieldInfo(name="year", display="Year", relative_importance=0.9, type=FieldTypes.INT_FIELD)
    DATE_PUBLISHED = SearchRecordFieldInfo(
        name="date_published", display="Date Published", relative_importance=0.5, type=FieldTypes.STR_FIELD
    )

    # Content classification
    KEYWORDS = SearchRecordFieldInfo(
        name="keywords", display="Keywords", relative_importance=0.1, type=FieldTypes.LIST_OR_STR_FIELD
    )
    SUBJECTS = SearchRecordFieldInfo(
        name="subjects", display="Subjects", relative_importance=0.1, type=FieldTypes.LIST_OR_STR_FIELD
    )

    # Metrics
    CITATION_COUNT = SearchRecordFieldInfo(
        name="citation_count", display="Citation Count", relative_importance=0.05, type=FieldTypes.INT_FIELD
    )
    OPEN_ACCESS = SearchRecordFieldInfo(
        name="open_access",
        display="Open Access Status",
        default="Unknown",
        relative_importance=0.2,
        type=FieldTypes.BOOL_FIELD,
    )

    @classmethod
    def get(cls, field: str | SearchRecordFieldInfo) -> SearchRecordFields | None:
        """Attempts to retrieve `SearchRecordFieldInfo` metadata for the current record field.

        Returns:
            SearchRecordFields: When the current field has a relevant field metadata entry within the current enum.
            None: When the provided field cannot be found or an invalid type is provided.

        """
        if isinstance(field, cls | SearchRecordFieldInfo):
            field = field.name

        try:
            return cls[field.upper()] if isinstance(field, str) else None

        except (KeyError, TypeError, ValueError):
            return None


# =============================================================================
# API Provider Metadata
# =============================================================================


class APIProviders(str, Enum):
    """Names of currently all supported academic API providers exposed through the MCP server's public API."""

    PUBMED = "pubmed"
    CROSSREF = "crossref"
    PLOS = "plos"
    ARXIV = "arxiv"
    OPENALEX = "openalex"
    CORE = "core"
    SPRINGER_NATURE = "springer_nature"

    @classmethod
    def get(cls, subject: str | APIProviders) -> APIProviders | None:
        """Helper method for determining whether an API provider is within the range of supported, public providers."""
        if isinstance(subject, APIProviders):
            subject = subject.name
        try:
            return cls(subject.upper()) if isinstance(subject, str) else None
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def normalize_name(cls, value: object) -> None | str:
        """Helper method for normalizing provider names when possible.

        Otherwise, None is returned for non-strings.

        """
        return value.strip().upper().replace("_", "") if isinstance(value, str) else None

    @classmethod
    def _missing_(cls, value: object) -> APIProviders | None:
        """Overrides the behavior of the current enum to retrieve the normalized provider when basic retrieval fails."""
        if isinstance(value, cls):
            value = value.name
        if not isinstance(value, str):
            return None
        normalized_value = cls.normalize_name(value)
        return next((provider for provider in APIProviders if normalized_value == cls.normalize_name(provider)), None)

    @classmethod
    def as_provider_names(cls, providers: APIProviders | str | Sequence[APIProviders | str]) -> list[str]:
        """Coerces a list of strings or APIProviders into a list of provider names."""
        return [api_provider.name for p in as_tuple(providers) if (api_provider := APIProviders.get(p))]

    @classmethod
    def as_provider_list(cls, providers: APIProviders | str | Sequence[APIProviders | str]) -> list[APIProviders]:
        """Coerces a list of strings, ResearchCategories, or SubjectInfo into a list of `APIProviders` instances."""
        return [api_provider for p in as_tuple(providers) if (api_provider := APIProviders.get(p))]


class ProviderMetadata(BaseModel):
    """Information about an academic database provider.

    Attributes:
        name (str):
            The normalized name of the provider.
        display_name (str):
            The human readable display name of the API provider.
        coverage (str | None):
            The topics covered by the API.
        description (str | None):
            A brief description of the API provider.
        recommended_for (list[str] | str | None):
            Research topics that the API is recommended for.

    """

    name: str = Field(..., description="The normalized name of the provider")
    display_name: str = Field(description="The human readable display name of the API provider")
    coverage: str | None = Field(default=None, description="The topics covered by the API")
    description: str | None = Field(default=None, description="A brief description of the API provider")
    recommended_for: list[str] | str | None = Field(
        default=None, description="Research topics that the API is recommended for"
    )

    @property
    def internal_only(self) -> bool:
        """Whether the API provider entry is public or is to be used only within the internals of ScholarFluxMCP."""
        return not bool(APIProviders.get(self.name))


class ProviderInfo(ProviderMetadata):
    """Information about an academic database provider.

    Attributes:
        base_url (str): The base URL for the API provider.
        api_key_env_var (str | None): API Key environment variable for the provider.
        requires_api_key (bool): Whether API key is required.
        records_per_page (int): The total number of paginated records queried per request by default.
        rate_limit (float): The default request delay in seconds.
        docs_url (str | None): The URL for the documentation of the API provider.

    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="allow",
    )
    base_url: str = Field(description="The base URL for the API provider")
    api_key_env_var: str | None = Field(default=None, description="API Key environment variable for the provider")
    requires_api_key: bool = Field(description="Whether API key is required")
    records_per_page: int = Field(description="The total number of paginated records queried per request by default")
    rate_limit: float = Field(description="The default request delay in seconds")
    docs_url: str | None = Field(default=None, description="The URL for the documentation of the API provider")

    @computed_cached_property
    def available(self) -> bool:
        """Indicates whether the provider is available for direct use by MCP tools"""
        return bool(APIProviders.get(self.name))

    def annotate(self, metadata: ProviderMetadata | dict) -> ProviderInfo:
        """Helper method for annotating metadata fields with additional info from the original provider registry."""
        if not isinstance(metadata, ProviderMetadata | dict):
            return self.model_copy()

        metadata_dict = metadata if isinstance(metadata, dict) else metadata.model_dump()
        field_overrides = {field for field in ProviderMetadata.model_fields if field not in ("name", "display_name")}
        return self.model_copy(update=metadata_dict | self.model_dump(exclude=field_overrides))


# Provider metadata for documentation
class ProviderMetadataDescriptions(Enum):
    """Enum Containing basic API provider info for each supported provider as of ScholarFlux 0.4.0."""

    PUBMED = ProviderMetadata(
        name=APIProviders.PUBMED,
        display_name="PubMed",
        description="US National Library of Medicine database of biomedical literature",
        coverage="Biomedical, life sciences, clinical medicine",
        recommended_for=["mental health", "clinical research", "medical studies"],
    )
    PUBMED_EFETCH = ProviderMetadata(
        name="pubmed_efetch",
        display_name="PubMed (eFetch)",
        description="US National Library of Medicine database of biomedical literature (internal endpoint for resolving PubMed DOIs to records)",
        coverage="Biomedical, life sciences, clinical medicine",
        recommended_for=["mental health", "clinical research", "medical studies"],
    )
    CROSSREF = ProviderMetadata(
        name=APIProviders.CROSSREF,
        display_name="Crossref",
        description="Registration agency for scholarly metadata with DOIs",
        coverage="Cross-disciplinary scholarly works",
        recommended_for=["citation analysis", "broad searches"],
    )
    PLOS = ProviderMetadata(
        name=APIProviders.PLOS,
        display_name="PLOS",
        description="Open access multidisciplinary journal and publisher",
        coverage="Science, medicine, all disciplines (open access)",
        recommended_for=["open access research", "reproducibility studies"],
    )
    ARXIV = ProviderMetadata(
        name=APIProviders.ARXIV,
        display_name="arXiv",
        description="Preprint server for physics, mathematics, and computer science",
        coverage="Physics, math, CS, quantitative biology, statistics",
        recommended_for=[
            "preprints in physics, mathematics, and computer science",
            "early-stage frontier research",
            "open access research",
        ],
    )
    OPENALEX = ProviderMetadata(
        name=APIProviders.OPENALEX,
        display_name="OpenAlex",
        description="Open catalog of scholarly records, authors, and institutions",
        coverage="Cross-disciplinary with rich metadata",
        recommended_for=[
            "cross-disciplinary research discovery",
            "bibliometric analysis",
            "open access tracking",
            "large-scale metadata queries",
        ],
    )
    CORE = ProviderMetadata(
        name=APIProviders.CORE,
        display_name="CORE",
        description="One of the World's largest collections of open access research papers",
        coverage="Open access academic records aggregated from repositories",
        recommended_for=["open access full-text", "repository content"],
    )
    SPRINGER_NATURE = ProviderMetadata(
        name=APIProviders.SPRINGER_NATURE,
        display_name="Springer Nature",
        description="Major academic publisher covering all disciplines",
        coverage="Medicine, science, technology, humanities",
        recommended_for=["published research", "high-impact journals"],
    )

    @classmethod
    def get(cls, subject: str | ProviderMetadata | ProviderMetadataDescriptions) -> ProviderMetadataDescriptions | None:
        """Helper method for extracting the current provider information."""
        if isinstance(subject, cls | ProviderMetadata):
            subject = subject.name

        try:
            return cls(subject.upper()) if isinstance(subject, str) else None

        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def get_fields(cls, subject: str | ProviderMetadata | ProviderMetadataDescriptions) -> dict[str, Any] | None:
        """Helper method that extracts existing provider info as a dictionary if available."""
        return provider_info.value.model_dump() if (provider_info := cls.get(subject)) else None

    @classmethod
    def _missing_(
        cls, value: object | ProviderMetadata | ProviderMetadataDescriptions
    ) -> ProviderMetadataDescriptions | None:
        """Overrides the behavior of the current enum to retrieve the normalized provider when basic retrieval fails."""
        if isinstance(value, cls | ProviderMetadata):
            value = value.name
        if not isinstance(value, str):
            return None

        normalized_name = APIProviders.normalize_name(value)
        return next(
            (provider for provider in cls if normalized_name == APIProviders.normalize_name(provider.name)),
            None,
        )

    @classmethod
    def as_provider_names(
        cls,
        providers: (
            APIProviders
            | ProviderMetadataDescriptions
            | str
            | Sequence[APIProviders | ProviderMetadataDescriptions | str]
        ),
        display_name: bool = False,
    ) -> list[str]:
        """Coerces a list of strings, APIProviders, or ProviderMetadataDescriptions into a list of provider names."""
        return [
            provider_metadata.value.display_name if display_name else provider_metadata.value.name
            for p in as_tuple(providers)
            if (provider_metadata := ProviderMetadataDescriptions.get(p))
        ]

    @classmethod
    def as_provider_metadata_descriptions(
        cls,
        providers: (
            APIProviders
            | ProviderMetadataDescriptions
            | str
            | Sequence[str | APIProviders | ProviderMetadata | ProviderMetadataDescriptions]
        ),
    ) -> list[ProviderMetadataDescriptions]:
        """Coerces a list of strings, ResearchCategories, or SubjectInfo into a list of `APIProvider` instances."""
        return [
            provider_metadata for p in as_tuple(providers) if (provider_metadata := ProviderMetadataDescriptions.get(p))
        ]


__all__ = [
    "APIProviders",
    "FieldTypes",
    "ProviderInfo",
    "ProviderMetadata",
    "ProviderMetadataDescriptions",
    "QueryCharacterLimit",
    "ResearchCategory",
    "ResponseFormat",
    "ResponseFormatString",
    "SearchRecordFieldInfo",
    "SearchRecordFields",
    "SubjectInfo",
]

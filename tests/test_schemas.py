"""Defines tests for Pydantic schemas and input validation models used throughout the ScholarFluxMCP server."""

import json
from datetime import datetime
from typing import ClassVar

import pytest
from pydantic import TypeAdapter, ValidationError

from scholar_flux_mcp.models import (
    AgentEvidenceItem,
    APIProviders,
    BaseSearchParams,
    BaseSearchRecord,
    BaseSynthesisParams,
    EvidenceGroundingOutput,
    EvidenceGroundingStats,
    GroundedEvidenceItem,
    HealthStatus,
    IndexedSearchRecord,
    IndexedSearchRecordList,
    JSONDataModel,
    PaginationInfo,
    ProviderInfo,
    ProviderMetadataDescriptions,
    RecordTopicSimilarity,
    RelevanceSearchInput,
    RelevanceSearchOutput,
    ResearchCategory,
    ResponseFormat,
    SearchInput,
    SearchOutput,
    SearchRecord,
    SearchRecordEmbedding,
    SearchRecordList,
    SearchResponseSummary,
    SynthesisAgentOutput,
    SynthesisInput,
    SynthesisOutput,
    computed_property,
)
from scholar_flux_mcp.server.io.health_check import HealthCheckFormatter
from scholar_flux_mcp.utils.helpers import generate_datetime


class MockDataModel(JSONDataModel):
    """Model for verifying fields that remain available or hidden when serialized."""

    HIDDEN_FIELDS: ClassVar[set[str]] = {"c"}

    a: int = 1
    b: str = "2"
    c: list[int] = [3, 4, 5]


class MockDataModelWithComputedProperties(MockDataModel):
    """Model for verifying that computed properties are shown when not explicitly hidden."""

    @computed_property
    def d(self) -> int:
        """Computed property that sums `c`. Should be returned by default on serialization."""
        return sum(self.c)


class MockDataModelHiddenOverrides(MockDataModel):
    """Model for verifying fields that remain available or hidden when serialized."""

    HIDDEN_FIELDS: ClassVar[set[str]] = {"a", "d"}

    @computed_property
    def d(self) -> int:
        """Computed property for testing overrides."""
        return sum(self.c)


class TestJsonDataModelSerializer:
    """Tests for verifying the structure and function of the JsonDataModel as a base class."""

    def test_hidden_field_filtering(self):
        """Verifies that hidden fields are successfully hidden during model serialization."""

        mock_data_model = MockDataModel()

        json_data = mock_data_model.model_dump()
        expected = {"a", "b"}
        assert set(json_data) == expected

        json_data_str = mock_data_model.model_dump_json()
        assert set(json.loads(json_data_str)) == expected

    def test_hidden_field_identification(self):
        """Verifies that computed fields that are not hidden are also shown during serialization."""

        mock_data_model = MockDataModelWithComputedProperties()
        expected = {"a", "b", "d"}
        assert set(mock_data_model.model_dump()) == expected
        assert set(json.loads(mock_data_model.model_dump_json())) == expected

    def test_hidden_field_with_overrides(self):
        """Verifies that hidden fields with overrides can still be included during serialization."""

        mock_data_model = MockDataModelHiddenOverrides()
        expected = {"b", "c"}
        all_keys = {"a", "b", "c", "d"}
        assert set(mock_data_model.model_dump()) == expected
        assert set(json.loads(mock_data_model.model_dump_json()).keys()) == expected
        assert all_keys == set(mock_data_model.model_dump(exclude={}))

        with JSONDataModel.serialization_context(core_fields_only=True):
            all_core_fields = {"a", "b", "c"}
            assert all_core_fields == set(mock_data_model.model_dump())
        with JSONDataModel.serialization_context(serialize_hidden_fields=True):
            assert all_keys == set(mock_data_model.model_dump())

    def test_nested_model_override(self):
        """Verifies that hidden fields with overrides can still be included during serialization."""

        class MockContainerModel(JSONDataModel):
            HIDDEN_FIELDS: ClassVar[set[str]] = {"name"}

            value: MockDataModelWithComputedProperties
            name: str = "MockJsonDataModel"

        mock_model = MockDataModelWithComputedProperties(a=1, b="3", c=[5, 6, 10])
        container_mock_model = MockContainerModel(value=mock_model)

        with JSONDataModel.serialization_context(core_fields_only=False):
            default_json_container = container_mock_model.model_dump()
            assert set(default_json_container) == {"value"}
            assert set(container_mock_model.value.model_dump()) == {"a", "b", "d"}
            assert set(default_json_container["value"]) == {"a", "b", "d"}

        with JSONDataModel.serialization_context(core_fields_only=True):
            json_container = container_mock_model.model_dump()

            # core fields
            assert set(json_container) == {"name", "value"}
            assert set(json_container["value"]) == {"a", "b", "c"}


class TestProviderEnum:
    """Tests for the APIProviders enum."""

    def test_all_providers_defined(self):
        """Verifies that all expected providers are defined."""
        expected = ["pubmed", "crossref", "plos", "arxiv", "openalex", "core", "springer_nature"]

        unexpected = [name for name in expected if not APIProviders.get(name)]
        assert not unexpected

    def test_provider_case_insensitive_access(self):
        """Verifies that provider enum values are retrievable with mixed cases."""
        expected = ["PubMed", "Crossref", "PLOS", "arXiv", "OpenAlex", "Core", "SPRINGERNATURE"]

        unexpected_with_get = [name for name in expected if not APIProviders.get(name)]
        assert not unexpected_with_get

        unexpected_with_direct_idx = [name for name in expected if not APIProviders(name)]
        assert not unexpected_with_direct_idx

    def test_provider_public_vs_internal(self):
        """Verifies that the only entry with a internal status is `PubMed (eFetch)`."""
        has_internal_status = [
            metadata_info.value.name
            for metadata_info in ProviderMetadataDescriptions
            if not APIProviders.get(metadata_info.value.name)
        ]
        expected_internal_only_status = [ProviderMetadataDescriptions.PUBMED_EFETCH.value.name]
        assert expected_internal_only_status == has_internal_status
        # Verifies that all ProviderMetadata.internal_only is False for all providers except for PubMed (eFetch)
        assert all(
            metadata_info
            for metadata_info in ProviderMetadataDescriptions
            if (metadata_info.value.internal_only is False)
            ^ (metadata_info is ProviderMetadataDescriptions.PUBMED_EFETCH)
        )

    def test_provider_metadata_descriptions_idempotence(self):
        """Verifies that methods of creating a list of ProviderMetadata instances produces idempotent results."""
        metadata_from_api_providers = ProviderMetadataDescriptions.as_provider_metadata_descriptions(list(APIProviders))
        metadata_from_provider_names = ProviderMetadataDescriptions.as_provider_metadata_descriptions(
            [p.name for p in metadata_from_api_providers]
        )
        assert metadata_from_api_providers == metadata_from_provider_names
        metadata_from_metadata_values = ProviderMetadataDescriptions.as_provider_metadata_descriptions(
            [p.value for p in metadata_from_api_providers]
        )
        assert metadata_from_api_providers == metadata_from_metadata_values
        assert len(metadata_from_api_providers) == len(APIProviders)

        list_metadata_descriptions = list(ProviderMetadataDescriptions)
        metadata_from_metadata_descriptions = ProviderMetadataDescriptions.as_provider_metadata_descriptions(
            list_metadata_descriptions
        )

        assert len(metadata_from_metadata_descriptions) == len(list_metadata_descriptions)
        assert metadata_from_metadata_descriptions == list_metadata_descriptions
        metadata_from_metadata_provider_names = ProviderMetadataDescriptions.as_provider_metadata_descriptions(
            [p.name for p in metadata_from_metadata_descriptions]
        )
        assert metadata_from_metadata_descriptions == metadata_from_metadata_provider_names


class TestResearchCategory:
    """Tests for ResearchCategory enum."""

    def test_all_categories_defined(self):
        """Test all expected categories are defined."""
        subjects = [
            "quantitative",
            "qualitative",
            "empirical",
            "theoretical",
            "meta_analysis",
            "depression",
            "anxiety",
            "ptsd",
            "bipolar",
            "schizophrenia",
            "substance_use",
            "ocd",
            "adhd",
            "general_wellbeing",
            "intervention",
            "epidemiology",
            "other",
        ]

        for subject in subjects:
            assert subject and ResearchCategory.get(subject)


class TestResponseFormat:
    """Tests for ResponseFormat enum."""

    def test_formats_defined(self):
        """Test markdown and json formats are defined."""
        assert ResponseFormat.MARKDOWN.value == "markdown"
        assert ResponseFormat.JSON.value == "json"


class TestSearchInput:
    """Tests for SearchInput model."""

    def test_valid_search_input(self):
        """Test valid search input is accepted."""
        params = SearchInput.create(
            queries="depression treatment CBT",
            providers=[APIProviders.PUBMED, APIProviders.PLOS],
            max_records=25,
            pages=3,
        )

        assert set(params.queries) == {"depression treatment CBT"}
        assert len(params.providers) == 2

    def test_base_search_input_defaults(self):
        """Tests whether BaseSearchParams has defaults within the range of common sense."""
        params = BaseSearchParams()

        assert params.max_records > 5 and params.max_records <= 200
        assert params.page_offset == 0
        assert params.open_access_only is False
        assert 1 <= params.pages <= 10

    def test_search_input_defaults(self):
        """Test search input has correct defaults."""
        params = SearchInput.create(queries=["test query"])

        assert params.providers == [APIProviders.PUBMED, APIProviders.PLOS, APIProviders.OPENALEX]
        assert params.max_records == 25
        assert params.pages == 3
        assert params.open_access_only is False

    def test_search_input_query_validation(self):
        """Test query length validation."""
        # Too short
        with pytest.raises(ValidationError):
            SearchInput.create(queries="a")

        # Too long
        with pytest.raises(ValidationError):
            SearchInput.create(queries="x" * 1001)

    def test_search_input_max_records_validation(self):
        """Test maximum results per provider bounds validation."""
        # Too low
        with pytest.raises(ValidationError):
            SearchInput.create(queries="test", max_records=0)

        # Too high
        with pytest.raises(ValidationError):
            SearchInput.create(queries="test", max_records=201)

    def test_search_input_year_validation(self):
        """Test year filter validation."""
        # Valid years
        params = SearchInput.create(queries="test", year_from=2020, year_to=2024)
        assert params.year_from == 2020
        assert params.year_to == 2024

        # Invalid year (too old)
        with pytest.raises(ValidationError):
            SearchInput.create(queries="test", year_from=1800)

    def test_search_input_provider_string_conversion(self):
        """Test provider strings are converted to enums."""
        # This tests the field_validator
        params = SearchInput.create(
            queries=["test1", "test2"],
            # strings should be internally converted into enums
            providers=["pubmed", "plos"],  # type: ignore
        )

        assert all(isinstance(p, APIProviders) for p in params.providers)


class TestSearchRecord:
    """Tests for SearchRecord model."""

    def test_minimal_record(self):
        """Test record with minimal fields."""
        record = BaseSearchRecord(provider_name="pubmed")

        assert record.provider_name == "pubmed"
        assert record.title is None
        assert record.doi is None

    def test_full_record(self):
        """Test record with all fields."""
        record = SearchRecord(
            provider_name="pubmed",
            page=1,
            query="test",
            doi="10.1000/test.123",
            url="https://example.com/paper",
            record_id="12345",
            title="Test Paper Title",
            abstract="This is the abstract...",
            authors=["Smith, J.", "Jones, K."],
            journal="Test Journal",
            publisher="Test Publisher",
            year=2024,
            date_published="2024-01-15",
            keywords=["keyword1", "keyword2"],
            subjects=["subject1"],
            citation_count=10,
            open_access=True,
        )

        assert record.doi == "10.1000/test.123"
        assert record.year == 2024
        assert len(record.authors or []) == 2
        assert record.record_hash

    def test_search_record_list_validation(self, mock_search_record_list):
        """Verifies that SearchRecord validation with a type adapter covers and validates edge case."""
        adapter: TypeAdapter[SearchRecordList] = TypeAdapter(SearchRecordList)
        assert adapter.validate_python(mock_search_record_list) == mock_search_record_list
        assert adapter.validate_python(tuple(mock_search_record_list)) == mock_search_record_list
        assert adapter.validate_python(mock_search_record_list[0]) == [mock_search_record_list[0]]

    def test_search_record_field_list_validation(self, mock_search_record_list):
        """Verifies that SearchRecord validation with a type adapter covers and converts record dictionaries."""
        adapter: TypeAdapter[SearchRecordList] = TypeAdapter(SearchRecordList)
        search_record_dictionaries = [record.model_dump() for record in mock_search_record_list]
        assert adapter.validate_python(search_record_dictionaries) == mock_search_record_list
        assert adapter.validate_python(search_record_dictionaries[0]) == [mock_search_record_list[0]]

    def test_invalid_search_record_value(self):
        """Verifies that non-SearchRecord types are successfully flagged as invalid when identified"""
        adapter: TypeAdapter[SearchRecordList] = TypeAdapter(SearchRecordList)
        invalid_list: list = [1, 2, 3]
        with pytest.raises(ValidationError) as excinfo:
            _ = adapter.validate_python(invalid_list)

        assert "Expected a list of SearchRecords, but at least one element has an incorrect type" in str(excinfo.value)

        invalid_str = "4"
        with pytest.raises(ValidationError) as excinfo:
            _ = adapter.validate_python(invalid_str)
        assert f"Expected a list of SearchRecords, but received type {type(invalid_str)}" in str(excinfo.value)

    def test_invalid_search_record_list_validation(self, mock_search_record_list):
        """Verifies that SearchRecord validation with a type adapter covers and converts record dictionaries."""
        adapter: TypeAdapter[IndexedSearchRecordList] = TypeAdapter(IndexedSearchRecordList)
        search_record_dictionaries = [record.model_dump() for record in mock_search_record_list]
        indexed_record_list = adapter.validate_python(search_record_dictionaries)
        assert indexed_record_list and all(
            isinstance(indexed_record, IndexedSearchRecord)
            and record.record_hash == indexed_record.record_hash
            and indexed_record.index == i
            for i, (record, indexed_record) in enumerate(zip(mock_search_record_list, indexed_record_list, strict=True))
        )


class TestSearchResponseSummary:
    """Tests for SearchResponseSummary model."""

    def test_successful_response(self):
        """Test a successful response summary with all fields."""
        summary = SearchResponseSummary(
            provider_name="pubmed",
            query="depression treatment",
            page=1,
            success=True,
            record_count=10,
            cached=False,
        )

        assert summary.provider_name == "pubmed"
        assert summary.success is True
        assert summary.record_count == 10
        assert summary.status_code == 200  # inferred by model_validator

    def test_failed_response(self):
        """Test a failed response summary with minimal fields."""
        err_message = (
            "Server requested a 300s wait before retrying, which exceeds the configured limit of 120s. This typically "
            "means you've hit a rate limit"
        )
        summary = SearchResponseSummary(
            provider_name="core",
            query="test query",
            page=2,
            success=False,
            status_code=429,
            error="RetryAfterDelayExceededException",
            message=err_message,
        )

        assert summary.success is False
        assert summary.record_count == 0
        assert summary.error == "RetryAfterDelayExceededException"
        assert summary.message == err_message
        assert summary.status_code == 429
        assert summary.cached is None

    def test_response_hash_is_stable(self):
        """Verifies that response_hashes are deterministic for the same input."""
        summary_a = SearchResponseSummary(provider_name="plos", query="anxiety", page=1, success=True)
        summary_b = SearchResponseSummary(provider_name="plos", query="anxiety", page=1, success=True)

        assert summary_a.response_hash == summary_b.response_hash

    def test_response_hash_differs_on_different_input(self):
        """Tests whether response_hashes change when identity fields differ."""
        summary_a = SearchResponseSummary(provider_name="plos", query="anxiety", page=1, success=True)
        summary_b = SearchResponseSummary(provider_name="plos", query="anxiety", page=2, success=True)

        assert summary_a.response_hash != summary_b.response_hash

    def test_retrieval_timestamp_serialization(self):
        """Test that retrieval_timestamp serializes to an ISO string."""
        summary = SearchResponseSummary(
            provider_name="arxiv",
            query="embeddings",
            page=1,
            success=True,
            retrieval_timestamp=generate_datetime(),
        )

        dumped = summary.model_dump(mode="json")
        assert isinstance(dumped["retrieval_timestamp"], str)


class TestRecordSimilarity:
    """Tests for the RecordTopicSimilarity class."""

    def test_basic_initialization(self):
        """Verifies that a complete RecordTopicSimilarity class can be initialized as a valid and truthy value."""
        assert RecordTopicSimilarity(
            record_embedding=SearchRecordEmbedding(
                record=SearchRecord(page=1, provider_name="PLOS", query="test"), embedding=[1, 2, 3, 4]
            ),
            topic="q",
            topic_similarity_score=0.5,
        )

    def test_record_similarity_clamping(self):
        """Verifies that relevance scores between -0.05 and 1.05 are clamped between 0 and 1.0 when needed."""
        record_embedding = SearchRecordEmbedding(
            record=SearchRecord(page=2, provider_name="PLOS", query="test"), embedding=[1.0, 2.0, 3.0, 4.0]
        )
        record_similarity = RecordTopicSimilarity(
            record_embedding=record_embedding,
            topic="test query",
            topic_similarity_score=-0.04,
        )
        assert record_similarity.topic_similarity_score == 0.0

        record_embedding = SearchRecordEmbedding(record=SearchRecord(page=1, provider_name="PLOS", query="test"))
        record_similarity = RecordTopicSimilarity(
            record_embedding=record_embedding, topic="test query", topic_similarity_score=1.03
        )
        assert record_similarity.topic_similarity_score == 1.0

    def test_record_similarity_raises_out_of_bounds(self):
        """Verifies that relevance scores not between 0.05 and 1.05 (noninclusive) are identified as invalid."""
        record_embedding = SearchRecordEmbedding(record=SearchRecord(page=1, provider_name="PLOS", query="test"))

        with pytest.raises(ValidationError):
            _ = RecordTopicSimilarity(
                record_embedding=record_embedding,
                topic="random query",
                topic_similarity_score=-0.05,
            )

        with pytest.raises(ValidationError):
            _ = RecordTopicSimilarity(
                record_embedding=record_embedding,
                topic="random topic",
                topic_similarity_score=1.05,
            )

    def test_record_similarity_representation(self):
        """Verifies that the representation of a RecordTopicSimilarity model prints as intended."""
        record_embedding = SearchRecordEmbedding(record=SearchRecord(page=2, provider_name="PLOS", query="test"))

        record_similarity = RecordTopicSimilarity(
            record_embedding=record_embedding,
            topic="q",
            topic_similarity_score=0.5,
        )
        representation = repr(record_similarity)
        assert f"record={repr(record_similarity.record)}" in representation
        assert f"topic='{record_similarity.topic}'" in representation
        assert f"topic_similarity_score={record_similarity.topic_similarity_score}" in representation


class TestSynthesisInput:
    """Tests for SynthesisInput model."""

    def test_valid_synthesis_input(self):
        """Test valid synthesis input is accepted."""
        params = SynthesisInput(
            question="What is the efficacy of CBT for depression?",
            categories=[ResearchCategory.DEPRESSION, ResearchCategory.INTERVENTION],
            max_records=30,
        )

        assert "CBT" in params.question
        assert len(params.categories) == 2

    def test_base_synthesis_input_defaults(self):
        """Tests whether BaseSynthesisParams input has defaults within the range of common sense."""

        params = BaseSynthesisParams(question="Test question for synthesis?")

        assert params.max_records > 5 and params.max_records <= 200
        assert params.year_from is not None and params.year_from >= 2000
        assert params.year_to is not None and params.year_from <= 2100
        assert params.open_access_only is False
        assert 1 <= params.pages <= 10

    def test_synthesis_input_defaults(self):
        """Tests whether synthesis input has defaults within the range of common sense."""
        params = SynthesisInput(question="Test question for synthesis?")

        assert params.categories == [ResearchCategory.GENERAL]
        assert params.max_records > 5 and params.max_records <= 200
        assert params.year_from is not None and params.year_from >= 2000
        assert params.year_to is not None and params.year_from <= 2100

    def test_synthesis_input_question_validation(self):
        """Test question length validation."""

        # Too short
        with pytest.raises(ValidationError):
            SynthesisInput(question="Short?")

        # Minimum length (10 chars)
        params = SynthesisInput(question="1234567890")
        assert len(params.question) == 10


class TestSynthesisOutput:
    """Tests for SynthesisOutput model."""

    def test_synthesis_output_creation(self):
        """Test creating a synthesis output."""
        record = SearchRecord(
            page=3, query="test", provider_name="test_provider", title="Test Paper", doi="10.1000/test", year=2024
        )
        evidence_item = AgentEvidenceItem(record_index=1, finding="Key finding from paper", relevance_score=0.9)
        grounded_evidence_item = GroundedEvidenceItem(**evidence_item.model_dump(), record=record)
        synthesis_input = SynthesisInput(
            question="Test question?",
            providers=[APIProviders.ARXIV, APIProviders.CORE],
            categories=[ResearchCategory.DEPRESSION],
        )
        search_input = SearchInput.create(providers=synthesis_input.providers, queries=synthesis_input.queries)
        search_output = SearchOutput(
            search_input=search_input,
            records=[record],  # type: ignore
            pagination=PaginationInfo(
                total_records=1, providers_queried=1, providers_successful=2, pages_successful=2, pages_retrieved=2
            ),
        )
        relevance_search_input = RelevanceSearchInput(
            question=synthesis_input.question,
            providers=synthesis_input.providers,
            categories=synthesis_input.categories,
        )
        relevance_search_output = RelevanceSearchOutput(
            search_output=search_output,
            relevance_search_input=relevance_search_input,
            indexed_records=[record],  # type: ignore
        )
        synthesis_agent_output = SynthesisAgentOutput(
            synthesis="Synthesized result with the test record [1]",
            key_findings=["Finding 1", "Finding 2"],
            evidence_summaries=[evidence_item],
        )
        grounding_output = EvidenceGroundingOutput(
            grounded_evidence_items=[grounded_evidence_item],
            grounding_stats=EvidenceGroundingStats(references_grounded=1),
        )
        output = SynthesisOutput(
            relevance_search_output=relevance_search_output,
            synthesis_input=synthesis_input,
            agent_output=synthesis_agent_output,
            grounding_output=grounding_output,
            model_name="test-llm:latest",
        )

        assert output.records_analyzed == len(output.indexed_records)
        assert output.indexed_records == [IndexedSearchRecord.from_search_record(record, index=0)]
        assert len(output.grounded_evidence) == 1
        assert output.confidence_score == synthesis_agent_output.confidence_score


class TestEvidenceItem:
    """Tests for GroundedEvidenceItem model."""

    def test_evidence_item_creation(self):
        """Test creating an evidence item."""
        item = GroundedEvidenceItem(
            record=SearchRecord(
                page=4,
                query="test",
                provider_name="test_provider",
                title="Test Paper",
                doi="10.1000/test",
                year=2024,
                authors="John & Mills",
            ),
            finding="This paper found that...",
            relevance_score=0.85,
            record_index=1,
        )

        assert item.title == "Test Paper"
        assert item.relevance_score == 0.85

    def test_evidence_item_relevance_bounds(self):
        """Test relevance score bounds validation."""
        record = SearchRecord(page=2, query="test", provider_name="test_provider")
        # Valid range
        item = GroundedEvidenceItem(record=record, finding="Finding", record_index=1, relevance_score=0.5)
        assert item.relevance_score == 0.5

        # Out of bounds
        with pytest.raises(ValidationError):
            GroundedEvidenceItem(record=record, finding="Finding", record_index=1, relevance_score=1.5)

        with pytest.raises(ValidationError):
            GroundedEvidenceItem(record=record, finding="Finding", record_index=1, relevance_score=-0.1)


class TestPaginationInfo:
    """Tests for PaginationInfo model."""

    def test_pagination_info_creation(self):
        """Test creating pagination info."""
        info = PaginationInfo(
            total_records=100,
            providers_queried=3,
            providers_successful=3,
            pages_successful=9,
            pages_retrieved=9,
        )

        assert info.total_records == 100


class TestHealthStatus:
    """Tests for HealthStatus model."""

    def test_health_status_creation(self):
        """Test creating health status."""
        health_status = (
            HealthStatus(
                status="unhealthy",
                version="0.1.0",
            )
            .update_service("Session Cache")
            .update_service("Response Processing Cache", status="unhealthy")
            .update_service("History Service", "disabled")
            .update_status("healthy")
        )

        assert health_status.status == "healthy"
        assert health_status.timestamp is not None  # Auto-set
        assert health_status.services["Session Cache"].status == "healthy"
        assert health_status.services["Response Processing Cache"].status == "unhealthy"
        assert health_status.services["History Service"].status == "disabled"

        assert HealthCheckFormatter.format_health_check_json(health_status)
        assert HealthCheckFormatter.format_health_check_markdown(health_status)

    def test_health_status_timestamp_default(self):
        """Test timestamp has default."""
        status = HealthStatus(status="healthy", version="0.1.0")

        assert isinstance(status.timestamp, datetime)


class TestProviderInfo:
    """Tests for ProviderInfo model."""

    def test_provider_info_creation(self):
        """Test creating provider info."""
        info = ProviderInfo(
            name="plos",
            display_name="PLOS",
            base_url="https://plos.org",
            requires_api_key=False,
            records_per_page=50,
            rate_limit=6,
            description="PLOS: Shaping the future of Open Access Scientific Research",
        )

        assert info.name == "plos"
        assert info.display_name == "PLOS"
        assert info.base_url == "https://plos.org"
        assert info.requires_api_key is False
        assert info.rate_limit == 6
        assert info.description == "PLOS: Shaping the future of Open Access Scientific Research"

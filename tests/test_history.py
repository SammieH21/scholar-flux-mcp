"""Tests for verifying that history can be converted from MCP pydantic model inputs into SQLModels."""

from datetime import timedelta

import pytest
import sqlmodel

from scholar_flux_mcp.models import (
    JSONDataModel,
    RelevanceSearchOutput,
    SearchOutput,
    SynthesisOutput,
)
from scholar_flux_mcp.models.history import (
    RelevanceSearchExecution,
    RelevanceSearchInputHistory,
    RelevanceSearchOutputHistory,
    SearchExecution,
    SearchInputHistory,
    SearchOutputHistory,
    SynthesisExecution,
    SynthesisInputHistory,
    SynthesisOutputHistory,
)
from scholar_flux_mcp.server.io import RelevanceSearchToolInput, SynthesisToolInput
from scholar_flux_mcp.utils.helpers import generate_datetime, generate_iso_timestamp, parse_iso_timestamp


@pytest.fixture
async def mock_ai_search_output(mock_ai_synthesis_output) -> SearchOutput:
    """Fixture for verifying that the SearchOutput can be transformed into a SQLModel when needed."""
    if (search_output := mock_ai_synthesis_output.search_output) is None:
        raise ValueError("SearchOutput is not available in the mocked synthesis output")
    return search_output


@pytest.fixture
async def mock_ai_relevance_search_output(mock_ai_synthesis_output) -> RelevanceSearchOutput:
    """Fixture for verifying that the RelevanceSearchOutput can be transformed into a SQLModel when needed."""
    if (relevance_search_output := mock_ai_synthesis_output.relevance_search_output) is None:
        raise ValueError("RelevanceSearchOutput is not available in the mocked synthesis output")
    return relevance_search_output


@pytest.fixture
def mock_computer_literacy_relevance_search_execution(
    mock_computer_literacy_relevance_search_output,
) -> RelevanceSearchExecution:
    """Generates a relevance-search-executions table for testing and verifying later storage via SQLModel."""
    search_execution = SearchExecution.from_record_search(
        mock_computer_literacy_relevance_search_output.search_output.search_input,
        mock_computer_literacy_relevance_search_output.search_output,
    )
    relevance_search_execution = RelevanceSearchExecution.from_record_relevance_search(
        mock_computer_literacy_relevance_search_output.relevance_search_input,
        mock_computer_literacy_relevance_search_output,
        search_execution=search_execution,
    )

    return relevance_search_execution


@pytest.fixture
def mock_ai_synthesis_execution(mock_ai_synthesis_output) -> SynthesisExecution:
    """Generates a fully-linked synthesis-executions table for testing and verifying later storage via SQLModel."""
    search_execution = SearchExecution.from_record_search(
        mock_ai_synthesis_output.search_output.search_input, mock_ai_synthesis_output.search_output
    )
    relevance_search_execution = RelevanceSearchExecution.from_record_relevance_search(
        mock_ai_synthesis_output.relevance_search_output.relevance_search_input,
        mock_ai_synthesis_output.relevance_search_output,
        search_execution=search_execution,
    )

    synthesis_execution = SynthesisExecution.from_research_synthesis(
        mock_ai_synthesis_output.synthesis_input,
        mock_ai_synthesis_output,
        search_execution=search_execution,
        relevance_search_execution=relevance_search_execution,
        stored_at=generate_iso_timestamp(),
    )

    return synthesis_execution


@pytest.fixture
def mock_sqlmodel_output_history_session(
    mock_ai_synthesis_execution,
    mock_computer_literacy_relevance_search_execution,
):
    """Mock database fixture containing the computer literacy relevance search and AI literacy synthesis outputs."""
    engine = sqlmodel.create_engine("sqlite:///:memory:")
    sqlmodel.SQLModel.metadata.create_all(engine)

    with sqlmodel.Session(engine) as session:
        session.add(mock_computer_literacy_relevance_search_execution)
        session.commit()
        session.add(mock_computer_literacy_relevance_search_execution.search_execution)
        session.commit()
        session.refresh(mock_computer_literacy_relevance_search_execution)
        session.add(mock_ai_synthesis_execution)
        session.commit()
        session.refresh(mock_ai_synthesis_execution)

        yield session


def test_output_integrations(mock_ai_search_output, mock_ai_relevance_search_output, mock_ai_synthesis_output):
    """Verifies that each of the outputs correctly are formed in tandem for use with scholar_flux.models.history."""
    assert mock_ai_search_output and mock_ai_relevance_search_output and mock_ai_synthesis_output

    with JSONDataModel.serialization_context(core_fields_only=True):
        dumped = mock_ai_synthesis_output.model_dump()
    dumped["relevance_search_output"]["search_output"]["search_input"].keys()


def test_search_executions(mock_ai_search_markdown_input, mock_ai_search_output):
    """Verifies that SearchExecution objects correctly initialize with the required SearchInput and SearchOutput."""
    search_execution = SearchExecution.from_record_search(mock_ai_search_markdown_input, mock_ai_search_output)
    assert search_execution


def test_synthesis_executions(mock_ai_synthesis_markdown_input, mock_ai_synthesis_output):
    """Verifies that SynthesisExecution objects initializes with the required SynthesisInput and SynthesisOutput."""
    search_execution = None
    relevance_search_execution = None
    synthesis_execution = SynthesisExecution.from_research_synthesis(
        mock_ai_synthesis_markdown_input,
        mock_ai_synthesis_output,
        search_execution=search_execution,
        relevance_search_execution=relevance_search_execution,
    )
    assert synthesis_execution


def test_search_input_history_creation(mock_ai_search_input):
    """Verifies that the search history input can be efficiently created from a pydantic model after validation."""
    shared_fields = set(SearchInputHistory.model_fields)
    search_hist = SearchInputHistory.from_search_input(mock_ai_search_input)

    assert search_hist.model_dump(
        include=shared_fields, exclude={"stored_at", "id"}
    ) == mock_ai_search_input.model_dump(include=shared_fields)


def test_search_output_history_creation(mock_ai_search_output):
    """Verifies that the search history input can be efficiently created from a pydantic model after validation."""
    search_hist = SearchOutputHistory.from_search_output(mock_ai_search_output)
    assert search_hist
    assert search_hist.cache_hit is mock_ai_search_output.cache_hit
    assert len(search_hist.records) == len(mock_ai_search_output.records)


def test_relevance_search_history_creation():
    """Verifies that the relevance search history input can be efficiently created from a pydantic model."""
    relevance_search_tool_input = RelevanceSearchToolInput(
        question="What is the consensus on the effectiveness of cancer classification methods in healthcare?",
        queries=["statistical methods cancer classification", "data science cancer identification"],
        providers=["PLOS", "PUBMED", "SpringerNature"],
        max_records=100,
        pages=3,
        year_from=1900,
        year_to=2100,
        open_access_only=False,
    )
    relevance_search_input = relevance_search_tool_input.as_relevance_search_input()
    shared_fields = set(RelevanceSearchInputHistory.model_fields)
    relevance_search_hist = RelevanceSearchInputHistory.from_relevance_search_input(relevance_search_input)

    assert relevance_search_hist.model_dump(
        include=shared_fields, exclude={"stored_at", "id", "categories"}
    ) == relevance_search_input.model_dump(include=shared_fields, exclude={"categories"}, mode="json")


def test_synthesis_output_with_rejected_evidence_creation(mock_synthesis_output_with_rejected_evidence_output):
    """Verifies that evidence associated with an output can be recreated from their corresponding SQLModel."""
    synthesis_hist = SynthesisOutputHistory.from_synthesis_output(mock_synthesis_output_with_rejected_evidence_output)
    assert synthesis_hist

    grounded_count = len(mock_synthesis_output_with_rejected_evidence_output.grounding_output.grounded_evidence_items)
    rejected_count = len(mock_synthesis_output_with_rejected_evidence_output.grounding_output.rejected_evidence_items)

    assert len(synthesis_hist.grounded_evidence_items) == grounded_count
    assert len(synthesis_hist.rejected_evidence_items) == rejected_count

    evidence = [item.to_agent_evidence_item() for item in synthesis_hist.evidence_items]

    assert evidence == mock_synthesis_output_with_rejected_evidence_output.agent_output.evidence_summaries


@pytest.mark.xfail(reason="Future SQModel revision without uniqueness constraints may result in roundtrip recreation.")
def test_mock_synthesis_output_round_trip_without_session_storage(mock_synthesis_output_with_rejected_evidence_output):
    """Verifies that synthesis generally requires resolution for grounded records to resolve with unique constraints."""
    synthesis_input = mock_synthesis_output_with_rejected_evidence_output.synthesis_input
    synthesis_hist = SynthesisExecution.from_research_synthesis(
        synthesis_input, mock_synthesis_output_with_rejected_evidence_output
    )

    with pytest.raises(ValueError, match="The current `GroundedEvidenceItemHistory` could not resolve"):
        _ = mock_synthesis_output_with_rejected_evidence_output == synthesis_hist.to_synthesis_output()


def test_mock_synthesis_output_round_trip_with_session_storage(mock_synthesis_output_with_rejected_evidence_output):
    """Verifies that synthesis stores a valid representation of synthesis outputs regardless of evidence rejection."""
    synthesis_input = mock_synthesis_output_with_rejected_evidence_output.synthesis_input
    synthesis_execution = SynthesisExecution.from_research_synthesis(
        synthesis_input, mock_synthesis_output_with_rejected_evidence_output
    )

    engine = sqlmodel.create_engine("sqlite:///:memory:")
    sqlmodel.SQLModel.metadata.create_all(engine)

    with sqlmodel.Session(engine) as session:
        session.add(synthesis_execution)
        session.commit()
        session.refresh(synthesis_execution)

        assert mock_synthesis_output_with_rejected_evidence_output == synthesis_execution.to_synthesis_output()


def test_relevance_search_output_history_creation(mock_ai_relevance_search_output):
    """Verifies that the search history input can be efficiently created from a pydantic model after validation."""
    relevance_search_hist = RelevanceSearchOutputHistory.from_relevance_search_output(mock_ai_relevance_search_output)
    assert isinstance(relevance_search_hist, RelevanceSearchOutputHistory)


def test_synthesis_history_creation():
    """Verifies that the synthesis history input can be efficiently created from a pydantic model."""
    synthesis_input = SynthesisToolInput(
        question="What is the consensus on the effectiveness of cancer classification methods in healthcare?",
        queries=["statistical methods cancer classification", "data science cancer identification"],
        providers=["plos", "pubmed", "springer_nature"],
        max_records=100,
        pages=3,
        year_from=1900,
        year_to=2100,
        open_access_only=False,
    )
    shared_fields = set(SynthesisInputHistory.model_fields)
    synthesis_hist = SynthesisInputHistory.from_synthesis_input(synthesis_input)

    assert synthesis_hist.model_dump(
        include=shared_fields, mode="json", exclude={"stored_at", "id", "categories"}
    ) == synthesis_input.model_dump(include=shared_fields, exclude={"categories"}, mode="json")


def test_synthesis_full_specification_executions(
    mock_ai_synthesis_markdown_input, mock_ai_synthesis_execution, mock_ai_synthesis_output
):
    """Verifies that SynthesisExecution objects initialize with the required SynthesisInput and SynthesisOutput."""
    assert isinstance(mock_ai_synthesis_execution, SynthesisExecution)
    synthesis_execution = mock_ai_synthesis_execution

    assert mock_ai_synthesis_execution.search_execution
    assert mock_ai_synthesis_execution.relevance_search_execution
    assert synthesis_execution.search_execution == synthesis_execution.relevance_search_execution.search_execution

    sim_output = synthesis_execution.relevance_search_execution.relevance_search_output.record_topic_similarity_output
    assert sim_output and sim_output.record_topic_similarities
    sim_scores = sim_output.record_topic_similarities
    sample_score = sim_scores[0]
    assert sample_score.topic_similarity_score and sample_score.topic
    record_embedding_sample = sample_score.record_embedding
    topic_embedding_sample = sim_output.topic_embedding
    assert record_embedding_sample.model_name is not None
    assert record_embedding_sample.embedding
    assert record_embedding_sample.record_hash
    assert topic_embedding_sample.topic is not None
    assert topic_embedding_sample.model_name is not None
    assert topic_embedding_sample.embedding

    search_record_history = synthesis_execution.search_execution.search_output.records[0]
    assert search_record_history.record.record_history
    assert synthesis_execution.search_execution.search_output.records[0].search_output

    engine = sqlmodel.create_engine("sqlite:///:memory:")
    sqlmodel.SQLModel.metadata.create_all(engine)

    with sqlmodel.Session(engine) as session:
        session.add(synthesis_execution)
        session.commit()
        session.refresh(synthesis_execution)

        search_execution = synthesis_execution.search_execution
        relevance_search_execution = synthesis_execution.relevance_search_execution

        reloaded = session.get(SearchOutputHistory, search_execution.search_output.id)
        assert reloaded and mock_ai_synthesis_output.search_output
        assert len(reloaded.records) == len(mock_ai_synthesis_output.search_output.records)
        assert relevance_search_execution.relevance_search_input.similarity_threshold
        topic_similarity_output = relevance_search_execution.relevance_search_output.record_topic_similarity_output
        assert topic_similarity_output
        topic_embedding_output = topic_similarity_output.topic_embedding
        assert topic_embedding_output.to_topic_embedding()

        record_embedding_output = topic_similarity_output.record_topic_similarities
        records = [similarity.record_embedding.record for similarity in record_embedding_output]
        all_record_ids = {id(record) for record in records}

        records2 = [record_history.record for record_history in search_execution.search_output.records]
        indexed_records = relevance_search_execution.relevance_search_output.indexed_records
        assert len(records) == len(records2)
        unique_ids = all_record_ids.symmetric_difference({id(record) for record in records2})
        assert not unique_ids
        assert all(id(indexed_record.record) in all_record_ids for indexed_record in indexed_records)


def test_synthesis_execution_roundtrip(mock_ai_synthesis_execution, mock_ai_synthesis_output):
    """Verifies that SynthesisExecution objects can correctly recreate the original SynthesisOutput."""
    assert isinstance(mock_ai_synthesis_execution, SynthesisExecution)
    synthesis_execution = mock_ai_synthesis_execution

    # mock_ai_synthesis_execution.relevance_search_execution.relevance_search_output.indexed_records
    engine = sqlmodel.create_engine("sqlite:///:memory:")
    sqlmodel.SQLModel.metadata.create_all(engine)

    with sqlmodel.Session(engine) as session:
        session.add(synthesis_execution)
        session.commit()
        session.refresh(synthesis_execution)

        recreated_output = synthesis_execution.synthesis_output.to_synthesis_output()

    assert isinstance(recreated_output, SynthesisOutput)

    # Standalone string field:
    assert recreated_output.model_name == mock_ai_synthesis_output.model_name
    # Fields: 'synthesis', 'confidence_score', 'evidence_summaries', 'limitations', 'suggested_queries', 'key_findings'
    assert recreated_output.agent_output == mock_ai_synthesis_output.agent_output
    # Field: 'search_input', 'pagination', 'records'
    assert recreated_output.search_output == mock_ai_synthesis_output.search_output
    # Fields: 'search_output', 'indexed_records', 'relevance_search_input', 'record_topic_similarity_output'
    assert recreated_output.relevance_search_output == mock_ai_synthesis_output.relevance_search_output
    # Fields: 'grounded_evidence_items', 'grounding_stats'
    assert recreated_output.grounding_output == mock_ai_synthesis_output.grounding_output

    assert recreated_output.synthesis_input == mock_ai_synthesis_output.synthesis_input

    # Full set of core fields minus the input
    with JSONDataModel.serialization_context(core_fields_only=True):
        assert recreated_output.model_dump(exclude={"synthesis_input"}) == mock_ai_synthesis_output.model_dump(
            exclude={"synthesis_input"}
        )


def test_multi_output_sqlmodel_executions(
    mock_ai_synthesis_execution,
    mock_computer_literacy_relevance_search_execution,
    mock_ai_synthesis_output,
    mock_computer_literacy_relevance_search_output,
    mock_sqlmodel_output_history_session,
):
    """Verifies that the total number of executions registered does not impact previously stored capabilities."""
    synthesis_execution = mock_ai_synthesis_execution
    cl_relevance_search_execution = mock_computer_literacy_relevance_search_execution
    session = mock_sqlmodel_output_history_session

    recreated_synthesis_output = synthesis_execution.synthesis_output.to_synthesis_output()
    with JSONDataModel.serialization_context(core_fields_only=True):
        assert recreated_synthesis_output.model_dump(
            exclude={"synthesis_input"}
        ) == mock_ai_synthesis_output.model_dump(exclude={"synthesis_input"})

    cl_relevance_search_output = cl_relevance_search_execution.relevance_search_output.to_relevance_search_output()
    with JSONDataModel.serialization_context(core_fields_only=True):
        expected = cl_relevance_search_output.model_dump(exclude={"relevance_search_input"})
        observed = mock_computer_literacy_relevance_search_output.model_dump(exclude={"relevance_search_input"})
        assert expected == observed

        stmt = sqlmodel.select(SynthesisExecution)
        synthesis_execution_result = session.exec(stmt)
        assert len(list(synthesis_execution_result)) == 1

        stmt2 = sqlmodel.select(RelevanceSearchExecution)
        relevance_search_execution = session.exec(stmt2)
        assert len(list(relevance_search_execution)) == 2

        stmt3 = sqlmodel.select(SearchExecution)
        search_execution = session.exec(stmt3)
        assert len(list(search_execution)) == 2


def test_sqlmodel_timestamp_filtering(mock_ai_synthesis_output):
    """Verifies that `stored_at` enables the calculation of cached record expiration times for later filtering."""
    engine = sqlmodel.create_engine("sqlite:///:memory:")
    sqlmodel.SQLModel.metadata.create_all(engine)

    with sqlmodel.Session(engine) as session:
        synthesis_input = mock_ai_synthesis_output.synthesis_input
        synthesis_execution = SynthesisExecution.from_research_synthesis(synthesis_input, mock_ai_synthesis_output)
        session.add(synthesis_execution)
        session.commit()
        session.refresh(synthesis_execution)

    ttl = 10
    base_stmt = (
        sqlmodel.select(SynthesisExecution)
        .join(SynthesisOutputHistory)
        .join(SynthesisInputHistory)
        .where(SynthesisInputHistory.input_hash == synthesis_input.input_hash)
    )
    stored_at_utc = sqlmodel.func.datetime(SynthesisInputHistory.stored_at, "utc")
    expiration_time = sqlmodel.func.datetime(stored_at_utc, f"+{ttl} seconds")

    # DB should have 1 stored output
    synthesis_executions_results = list(session.exec(base_stmt))
    assert synthesis_executions_results and len(synthesis_executions_results) == 1
    retrieved = synthesis_executions_results[0]
    assert retrieved

    # Timestamp uses UTC under the hood
    current_timestamp_string = sqlmodel.func.strftime(SynthesisOutputHistory.DATETIME_COMPARISON_FMT, "now")

    stmt = base_stmt.where(current_timestamp_string < expiration_time)
    assert list(session.exec(stmt)) == synthesis_executions_results

    current_time = generate_datetime()
    stored_at_timestamp = parse_iso_timestamp(retrieved.stored_at)
    expiration_timestamp = stored_at_timestamp + timedelta(seconds=ttl) if stored_at_timestamp else None
    assert current_time and expiration_timestamp and current_time < expiration_timestamp

    # Verifies that smaller TTL values correctly affect downstream filtering
    ttl = 0
    current_timestamp_string = sqlmodel.func.strftime(SynthesisOutputHistory.DATETIME_COMPARISON_FMT, "now")
    expiration_time = sqlmodel.func.datetime(stored_at_utc, f"+{ttl} seconds")
    stmt = base_stmt.where(current_timestamp_string < expiration_time)
    assert not list(session.exec(stmt))  # should have 0 non-expired records

    assert retrieved.synthesis_output.is_expired(ttl)


def test_multi_output_sqlmodel_input_hash_identification(
    mock_ai_synthesis_output, mock_computer_literacy_relevance_search_output, mock_sqlmodel_output_history_session
):
    """Verifies that the history input models successfully resolve against the original input via input hashes."""
    session = mock_sqlmodel_output_history_session
    synthesis_execution_list = list(session.exec(sqlmodel.select(SynthesisExecution)))
    assert len(synthesis_execution_list) == 1

    synthesis_input_hash = mock_ai_synthesis_output.synthesis_input.input_hash
    synthesis_input_history = synthesis_execution_list[0].synthesis_input
    assert synthesis_input_history.input_hash == synthesis_input_hash

    # The following would be ideal, but sqlmodel.where doesn't seem to support computed fields
    synthesis_input_history_list2 = list(
        session.exec(
            sqlmodel.select(SynthesisInputHistory).where(SynthesisInputHistory.input_hash == synthesis_input_hash)
        )
    )
    assert synthesis_input_history_list2 and synthesis_input_history_list2[0] == synthesis_input_history

    search_input = mock_ai_synthesis_output.search_output.search_input
    matched_search_executions = list(
        session.exec(
            sqlmodel.select(SearchExecution)
            .join(SearchInputHistory)
            .where(SearchInputHistory.input_hash == search_input.input_hash)
            .order_by(sqlmodel.col(SearchInputHistory.stored_at).desc())
        )
    )
    assert len(matched_search_executions) == 1
    assert matched_search_executions[0].search_output.to_search_output() == mock_ai_synthesis_output.search_output

    relevance_search_input = mock_computer_literacy_relevance_search_output.relevance_search_input
    relevance_search_hash = relevance_search_input.input_hash
    relevance_search_history_list = list(
        session.exec(
            sqlmodel.select(RelevanceSearchInputHistory).where(
                RelevanceSearchInputHistory.input_hash == relevance_search_hash
            )
        )
    )
    assert len(relevance_search_history_list) == 1
    relevance_search_input_history = relevance_search_history_list[0]
    assert relevance_search_input_history.to_relevance_search_input() == relevance_search_input
    relevance_search_execution = relevance_search_input_history.relevance_search_execution
    assert relevance_search_execution
    relevance_search_output_history = relevance_search_execution.relevance_search_output
    assert (
        relevance_search_output_history.to_relevance_search_output() == mock_computer_literacy_relevance_search_output
    )

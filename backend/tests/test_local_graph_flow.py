"""Exercise the existing graph routes and simulation-memory path using SQLite."""

import copy
import importlib
import io
import time

import pytest

from memory_test_support import (
    FACT, ONTOLOGY, PEOPLE, SOURCE_TEXT, FixedLLM, assert_write_lock_available,
    create_graph, extraction_for, extraction_payload, ingest, memory_case,
)


def _api_client(tmp_path, monkeypatch):
    from app import create_app
    from app.api import graph as graph_api
    from app.models.project import ProjectManager
    from app.models.task import TaskManager
    from app.utils.llm_client import LLMClient

    class FixedOntology:
        def generate(self, **_kwargs):
            return copy.deepcopy(ONTOLOGY)

    fake = FixedLLM(extraction_payload())

    def chat_json(_client, messages, **kwargs):
        return fake.chat_json(messages, **kwargs)

    monkeypatch.setattr(ProjectManager, "PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(TaskManager, "_instance", None)
    monkeypatch.setattr(graph_api, "OntologyGenerator", FixedOntology)
    monkeypatch.setattr(LLMClient, "chat_json", chat_json)
    application = create_app()
    application.config.update(TESTING=True)
    return application.test_client(), fake


def _upload_and_build(client):
    uploaded = client.post(
        "/api/graph/ontology/generate",
        data={"simulation_requirement": "观察四位人物讨论咖啡涨价",
              "files": (io.BytesIO(SOURCE_TEXT.encode("utf-8")), "source.txt")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 200, uploaded.get_json()
    assert uploaded.json["success"] is True
    project_id = uploaded.json["data"]["project_id"]
    built = client.post("/api/graph/build", json={"project_id": project_id})
    assert built.status_code == 200, built.get_json()
    assert built.json["success"] is True
    return built.json["data"]["task_id"]


def _wait_for_task(client, task_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/graph/task/{task_id}")
        assert response.status_code == 200
        task = response.json["data"]
        if task["status"] in {"completed", "failed"}:
            assert task["status"] == "completed", task
            return task["result"]["graph_id"]
        time.sleep(0.01)
    pytest.fail("The fixed-LLM graph build did not finish within 10 seconds")


def _activity(platform="twitter"):
    from app.services.zep_graph_memory_updater import AgentActivity

    return AgentActivity(
        platform=platform, agent_id=1, agent_name="周宁",
        action_type="CREATE_POST", action_args={"content": SOURCE_TEXT},
        round_num=1, timestamp="2026-11-04T06:30:00+00:00",
    )


def _updater(case, graph_id, fake):
    from app.services.zep_graph_memory_updater import ZepGraphMemoryUpdater

    updater = ZepGraphMemoryUpdater(
        graph_id, simulation_id="simulation-local", store=case.store,
        extractor=case.memory.GraphExtractor(llm_client=fake),
    )
    updater.SEND_INTERVAL = 0
    return updater


def _drain_activity(updater, platform="twitter"):
    updater.start()
    try:
        updater.add_activity(_activity(platform))
    finally:
        updater.stop()


def test_upload_build_entities_activity_search_and_restart_keep_existing_shapes(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    client, build_llm = _api_client(tmp_path, monkeypatch)
    graph_id = _wait_for_task(client, _upload_and_build(client))

    response = client.get(f"/api/graph/data/{graph_id}")
    assert response.status_code == 200
    assert response.json["success"] is True
    data = response.json["data"]
    assert {"graph_id", "nodes", "edges", "node_count", "edge_count"} <= data.keys()
    assert data["node_count"] == 4
    assert data["edge_count"] == 1
    assert all({"uuid", "name", "labels", "summary", "attributes", "created_at"} <= node.keys() for node in data["nodes"])
    assert {"uuid", "name", "fact", "fact_type", "source_node_uuid", "target_node_uuid", "source_node_name", "target_node_name", "attributes", "created_at", "valid_at", "invalid_at", "expired_at", "episodes"} <= data["edges"][0].keys()
    entities_response = client.get(f"/api/simulation/entities/{graph_id}")
    assert entities_response.status_code == 200
    entities = entities_response.json["data"]["entities"]
    assert {entity["name"] for entity in entities} == set(PEOPLE)
    assert all("Person" in entity["labels"] for entity in entities)
    assert build_llm.calls

    simulation_llm = FixedLLM(extraction_payload())
    updater = _updater(case, graph_id, simulation_llm)
    updater.start()
    try:
        updater.add_activity(_activity("twitter"))
        updater.add_activity(_activity("reddit"))
    finally:
        updater.stop()

    originals = case.store.list_episodes(graph_id, source="simulation")
    assert len(originals) == 2
    assert {episode.metadata["platform"] for episode in originals} == {"twitter", "reddit"}
    assert all(SOURCE_TEXT in episode.content and episode.status == "processed" for episode in originals)
    assert all(episode.metadata["simulation_id"] == "simulation-local" for episode in originals)
    assert updater.get_stats()["items_sent"] == 2
    assert set(episode.uuid for episode in originals) <= set(case.store.list_edges(graph_id)[0].episode_uuids)
    tools = importlib.import_module("app.services.zep_tools").ZepToolsService(store=case.store)
    found = tools.search_graph(graph_id, "周宁 涨价")
    assert found.facts == [FACT]
    assert found.to_dict()["retrieval_mode"] == "keyword"
    restarted = case.memory.LocalGraphStore()
    assert restarted.list_nodes(graph_id) == case.store.list_nodes(graph_id)
    assert restarted.list_edges(graph_id) == case.store.list_edges(graph_id)
    assert restarted.list_episodes(graph_id) == case.store.list_episodes(graph_id)
    assert (case.root / f"{graph_id}.sqlite3").is_file()


def test_document_batch_is_durable_idempotent_and_resumes_only_unprocessed_chunks(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    from app.services.graph_builder import GraphBuilderService

    fake = FixedLLM(extraction_payload())
    builder = GraphBuilderService(store=case.store, extractor=case.memory.GraphExtractor(llm_client=fake))
    callbacks = []
    chunks = [SOURCE_TEXT, SOURCE_TEXT + "第二段材料。"]
    submission = builder.add_text_batches(graph_id, chunks, batch_created_callback=lambda batch, operation: callbacks.append((batch, operation)))
    assert fake.calls == []
    assert submission.batch_id == submission.operation_id
    assert callbacks == [(submission.batch_id, submission.operation_id)]
    assert submission.item_count == 2
    assert len(set(submission.episode_uuids)) == 2
    episodes = case.store.list_episodes(graph_id, operation_id=submission.operation_id)
    assert [episode.content for episode in sorted(episodes, key=lambda item: item.sequence)] == chunks
    assert all(episode.status == "pending" and episode.source == "document" for episode in episodes)
    case.store.apply_extraction(graph_id, submission.episode_uuids[0], extraction_for(case.models), ONTOLOGY)

    restarted = GraphBuilderService(store=case.memory.LocalGraphStore(), extractor=case.memory.GraphExtractor(llm_client=fake))
    assert restarted.get_batch_summary(graph_id, submission.batch_id).status == "processing"
    repeated = restarted.add_text_batches(graph_id, chunks)
    assert repeated.episode_uuids == submission.episode_uuids
    assert len(case.store.list_episodes(graph_id)) == 2
    restarted._wait_for_batch(repeated, lambda *_args: None)
    assert len(fake.calls) == 1
    assert restarted.get_batch_summary(graph_id, submission.batch_id).status == "succeeded"
    restarted._wait_for_batch(repeated, lambda *_args: None)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("source", ["document", "simulation"])
def test_llm_calls_leave_the_sqlite_write_lock_available(tmp_path, monkeypatch, source):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    fake = FixedLLM(extraction_payload(), on_call=lambda: assert_write_lock_available(case, graph_id))
    if source == "document":
        from app.services.graph_builder import GraphBuilderService

        builder = GraphBuilderService(store=case.store, extractor=case.memory.GraphExtractor(llm_client=fake))
        submission = builder.add_text_batches(graph_id, [SOURCE_TEXT])
        builder._wait_for_batch(submission, lambda *_args: None)
    else:
        _drain_activity(_updater(case, graph_id, fake))
    assert len(fake.calls) == 1
    assert case.store.list_episodes(graph_id, source=source)[0].status == "processed"


def test_document_extraction_failure_keeps_failed_original_without_invented_facts(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    from app.services.graph_builder import GraphBuilderService
    from app.utils.llm_client import LLMResponseError

    fake = FixedLLM(error=LLMResponseError("synthetic extraction failure"))
    builder = GraphBuilderService(store=case.store, extractor=case.memory.GraphExtractor(llm_client=fake))
    submission = builder.add_text_batches(graph_id, [SOURCE_TEXT])
    with pytest.raises(RuntimeError, match="synthetic extraction failure"):
        builder._wait_for_batch(submission, lambda *_args: None)
    restarted = case.memory.LocalGraphStore()
    original = restarted.get_episode(graph_id, submission.episode_uuids[0])
    assert (original.source, original.content, original.status) == ("document", SOURCE_TEXT, "failed")
    assert "synthetic extraction failure" in original.error
    assert restarted.list_nodes(graph_id) == []
    assert restarted.list_edges(graph_id) == []
    assert restarted.batch_progress(graph_id, submission.operation_id).failed == 1


def test_simulation_extraction_failure_is_durable_reported_and_not_replayed(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    from app.utils.llm_client import LLMResponseError

    fake = FixedLLM(error=LLMResponseError("synthetic activity extraction failure"))
    updater = _updater(case, graph_id, fake)
    with pytest.raises(RuntimeError):
        _drain_activity(updater)
    original = case.memory.LocalGraphStore().list_episodes(graph_id, source="simulation")
    assert len(original) == 1
    assert SOURCE_TEXT in original[0].content
    assert original[0].status == "failed"
    assert "synthetic activity extraction failure" in original[0].error
    assert len(updater._failed_batches) == 1
    assert updater.get_stats()["failed_count"] == 1
    assert len(fake.calls) == 1
    assert case.store.list_nodes(graph_id) == []
    assert case.store.list_edges(graph_id) == []


def test_entity_reader_includes_incoming_and_outgoing_edges_from_the_local_graph(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id)
    ingest(case, graph_id, episode_uuid="reverse", extraction=extraction_for(case.models, names=("林晓", "周宁"), fact="林晓回复周宁"))
    node = next(node for node in case.store.list_nodes(graph_id) if node.name == "周宁")
    reader = importlib.import_module("app.services.zep_entity_reader").ZepEntityReader(store=case.store)
    edges = reader.get_node_edges(node.uuid, graph_id=graph_id)
    assert {edge["fact"] for edge in edges} == {FACT, "林晓回复周宁"}


@pytest.mark.parametrize(("graph_id", "status"), [("missing_graph", 404), ("bad.id", 400)])
def test_graph_data_route_uses_existing_error_shape_for_missing_and_invalid_ids(tmp_path, monkeypatch, graph_id, status):
    memory_case(tmp_path, monkeypatch)
    client, fake = _api_client(tmp_path, monkeypatch)
    response = client.get(f"/api/graph/data/{graph_id}")
    assert response.status_code == status
    assert response.json["success"] is False
    assert isinstance(response.json["error"], str)
    assert fake.calls == []

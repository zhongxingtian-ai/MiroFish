"""Persistence, provenance, isolation, and input safety for local graphs."""

import sqlite3
from contextlib import closing

import pytest

from memory_test_support import (
    FACT, ONTOLOGY, SOURCE_TEXT, create_graph, extraction_for, ingest, memory_case,
)


def test_graph_and_provenance_survive_a_new_store_instance(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    document = ingest(case, graph_id, metadata={"filename": "source.txt"})
    simulation = ingest(
        case, graph_id, episode_uuid="simulation-1", source="simulation",
        operation_id="simulation-operation", content="周宁在模拟中重复表达对涨价的看法",
        metadata={"platform": "twitter", "round": 1},
    )
    original_nodes = case.store.list_nodes(graph_id)
    original_edges = case.store.list_edges(graph_id)

    restarted = case.memory.LocalGraphStore()

    assert restarted.graph_exists(graph_id) is True
    assert restarted.get_ontology(graph_id) == ONTOLOGY
    assert restarted.list_nodes(graph_id) == original_nodes
    assert restarted.list_edges(graph_id) == original_edges
    assert len(original_nodes) == 2
    assert len(original_edges) == 1
    for record in (*original_nodes, *original_edges):
        assert set(record.episode_uuids) == {document, simulation}
    episodes = {episode.uuid: episode for episode in restarted.list_episodes(graph_id)}
    assert episodes[document].content == SOURCE_TEXT
    assert episodes[document].metadata == {"filename": "source.txt"}
    assert episodes[simulation].source == "simulation"
    assert episodes[simulation].metadata == {"platform": "twitter", "round": 1}
    assert all(episode.status == "processed" and episode.processed_at for episode in episodes.values())
    with closing(sqlite3.connect(case.root / f"{graph_id}.sqlite3")) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM meta"))
        assert metadata["graph_id"] == graph_id
        assert metadata["schema_version"] == "1"
        assert metadata["name"] == "本地测试图谱"
        assert connection.execute("SELECT COUNT(*) FROM node_episodes").fetchone()[0] == 4
        assert connection.execute("SELECT COUNT(*) FROM edge_episodes").fetchone()[0] == 2


def test_graph_files_are_isolated_and_deleting_one_preserves_the_other(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    first, second = (create_graph(case, name) for name in ("graph_a", "graph_b"))
    ingest(case, first, extraction=extraction_for(case.models, names=("周宁", "林晓")))
    ingest(case, second, extraction=extraction_for(case.models, names=("周宁", "陈岚")))
    second_nodes, second_edges = case.store.list_nodes(second), case.store.list_edges(second)

    assert {node.name for node in case.store.list_nodes(first)} == {"周宁", "林晓"}
    assert {node.name for node in second_nodes} == {"周宁", "陈岚"}
    assert {path.name for path in case.root.glob("*.sqlite3")} == {"graph_a.sqlite3", "graph_b.sqlite3"}
    case.store.delete_graph(first)

    assert not (case.root / "graph_a.sqlite3").exists()
    assert case.store.graph_exists(first) is False
    with pytest.raises(case.memory.MemoryNotFoundError):
        case.store.list_nodes(first)
    with pytest.raises(case.memory.MemoryNotFoundError):
        case.store.delete_graph(first)
    assert case.memory.LocalGraphStore().list_nodes(second) == second_nodes
    assert case.memory.LocalGraphStore().list_edges(second) == second_edges
    assert not (case.root / "graph_a.sqlite3").exists()


@pytest.mark.parametrize("graph_id", ["../escape", "a/b", "a" + chr(92) + "b", ".." + chr(92) + "escape", "", ".", "..", "a" * 65, "with space", "中文", None, 123])
def test_invalid_graph_ids_are_rejected_by_every_entry_without_files(tmp_path, monkeypatch, graph_id):
    case = memory_case(tmp_path, monkeypatch)
    empty = case.models.Extraction(entities=(), relations=())
    operations = [
        lambda: case.store.create_graph(graph_id, "name", "description"),
        lambda: case.store.graph_exists(graph_id),
        lambda: case.store.delete_graph(graph_id),
        lambda: case.store.set_ontology(graph_id, ONTOLOGY),
        lambda: case.store.get_ontology(graph_id),
        lambda: case.store.add_episodes(graph_id, []),
        lambda: case.store.get_episode(graph_id, "episode"),
        lambda: case.store.list_episodes(graph_id),
        lambda: case.store.mark_episode_failed(graph_id, "episode", "error"),
        lambda: case.store.apply_extraction(graph_id, "episode", empty, ONTOLOGY),
        lambda: case.store.list_nodes(graph_id),
        lambda: case.store.list_edges(graph_id),
        lambda: case.store.get_node(graph_id, "node"),
        lambda: case.store.batch_progress(graph_id, "operation"),
    ]
    for operation in operations:
        with pytest.raises(ValueError):
            operation()
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


@pytest.mark.parametrize("graph_id", ["a", "A_Z-09", "x" * 64])
def test_valid_graph_id_boundaries_create_exactly_the_named_file(tmp_path, monkeypatch, graph_id):
    case = memory_case(tmp_path, monkeypatch)
    create_graph(case, graph_id)
    assert case.store.graph_exists(graph_id) is True
    assert (case.root / f"{graph_id}.sqlite3").is_file()


def test_missing_graph_reads_never_create_a_database(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    assert case.store.graph_exists("missing") is False
    for read in (case.store.list_nodes, case.store.list_edges, case.store.list_episodes, case.store.get_ontology):
        with pytest.raises(case.memory.MemoryNotFoundError):
            read("missing")
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_creating_an_existing_graph_does_not_overwrite_its_data(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id)
    with pytest.raises(FileExistsError):
        case.store.create_graph(graph_id, "replacement", "replacement")
    assert case.store.get_episode(graph_id, "document-1").content == SOURCE_TEXT
    assert len(case.store.list_edges(graph_id)) == 1


def test_sql_special_characters_round_trip_as_plain_data(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    name = "O'Brien'; DROP TABLE nodes; -- 100%"
    fact = "O'Brien said '50%'; DROP TABLE edges; --"
    ingest(case, graph_id, content=fact, extraction=extraction_for(case.models, names=(name, "林晓"), fact=fact))

    assert {node.name for node in case.store.list_nodes(graph_id)} == {name, "林晓"}
    assert case.store.list_edges(graph_id)[0].fact == fact
    assert case.store.get_episode(graph_id, "document-1").content == fact
    assert len(case.memory.LocalGraphStore().list_nodes(graph_id)) == 2
    assert case.store.batch_progress(graph_id, "operation-1").status == "succeeded"


def test_reapplying_an_episode_preserves_ids_and_unique_sources(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    episode_id = ingest(case, graph_id)
    nodes, edges = case.store.list_nodes(graph_id), case.store.list_edges(graph_id)

    case.store.apply_extraction(graph_id, episode_id, extraction_for(case.models), ONTOLOGY)

    assert case.store.list_nodes(graph_id) == nodes
    assert case.store.list_edges(graph_id) == edges
    assert len(case.store.list_episodes(graph_id)) == 1
    assert all(node.episode_uuids == (episode_id,) for node in nodes)
    assert edges[0].episode_uuids == (episode_id,)


def test_name_and_fact_normalization_merge_people_and_append_sources(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id, extraction=extraction_for(case.models, names=("ＡＬＩＣＥ", "林晓"), fact="Coffee UPDATE"))
    alice = next(node for node in case.store.list_nodes(graph_id) if node.name != "林晓")
    ingest(case, graph_id, episode_uuid="document-2", sequence=1,
           extraction=extraction_for(case.models, names=(" alice ", "林晓"), fact=" coffee update "))

    nodes = case.store.list_nodes(graph_id)
    assert len(nodes) == 2
    assert next(node for node in nodes if node.uuid == alice.uuid).name == "ＡＬＩＣＥ"
    assert len(case.store.list_edges(graph_id)) == 1
    assert set(case.store.list_edges(graph_id)[0].episode_uuids) == {"document-1", "document-2"}


def test_only_declared_entity_attributes_are_stored(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    entity = case.models.ExtractedEntity(name="周宁", type="Person", summary="原始简介", attributes={"role": "顾客", "undeclared": "discard"})
    ingest(case, graph_id, extraction=case.models.Extraction(entities=(entity,), relations=()))
    node = case.store.list_nodes(graph_id)[0]
    assert node.labels == ("Entity", "Person")
    assert dict(node.attributes) == {"role": "顾客"}
    assert case.store.get_node(graph_id, node.uuid) == node
    with pytest.raises(case.memory.MemoryNotFoundError):
        case.store.get_node(graph_id, "missing-node")


def test_unknown_types_and_missing_relation_endpoints_do_not_invent_nodes(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    entities = (
        case.models.ExtractedEntity(name="周宁", type="UnknownType", summary="", attributes={}),
        case.models.ExtractedEntity(name="林晓", type="Person", summary="", attributes={}),
    )
    relations = (
        case.models.ExtractedRelation(source_name="周宁", target_name="林晓", name="UNKNOWN", fact=FACT),
        case.models.ExtractedRelation(source_name="周宁", target_name="不存在的人", name="KNOWS", fact="没有有效端点"),
        case.models.ExtractedRelation(source_name="周宁", target_name="林晓", name="KNOWS", fact=""),
    )
    ingest(case, graph_id, extraction=case.models.Extraction(entities=entities, relations=relations))
    nodes = {node.name: node for node in case.store.list_nodes(graph_id)}
    assert set(nodes) == {"周宁", "林晓"}
    assert nodes["周宁"].labels == ("Entity",)
    edges = case.store.list_edges(graph_id)
    assert len(edges) == 1
    assert edges[0].name == "RELATES_TO"
    assert edges[0].fact == FACT


def test_batch_progress_and_failed_originals_persist_and_can_resume(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    missing = case.store.batch_progress(graph_id, "batch")
    assert (missing.status, missing.total, missing.processed, missing.failed) == ("failed", 0, 0, 0)
    episodes = [case.models.NewEpisode(uuid=f"chunk-{index}", source="document", content=f"原始材料{index}", metadata={}, operation_id="batch", sequence=index) for index in range(2)]
    case.store.add_episodes(graph_id, episodes)
    pending = case.store.batch_progress(graph_id, "batch")
    assert (pending.status, pending.total, pending.processed, pending.failed) == ("processing", 2, 0, 0)
    case.store.apply_extraction(graph_id, "chunk-0", extraction_for(case.models), ONTOLOGY)
    case.store.mark_episode_failed(graph_id, "chunk-1", "synthetic extraction failure")

    restarted = case.memory.LocalGraphStore()
    partial = restarted.batch_progress(graph_id, "batch")
    assert (partial.status, partial.total, partial.processed, partial.failed) == ("processing", 2, 1, 1)
    assert restarted.get_episode(graph_id, "chunk-1").content == "原始材料1"
    assert restarted.get_episode(graph_id, "chunk-1").error == "synthetic extraction failure"
    restarted.apply_extraction(graph_id, "chunk-1", extraction_for(case.models), ONTOLOGY)
    completed = restarted.batch_progress(graph_id, "batch")
    assert (completed.status, completed.total, completed.processed, completed.failed) == ("succeeded", 2, 2, 0)
    assert restarted.get_episode(graph_id, "chunk-1").error is None
    assert restarted.get_episode(graph_id, "chunk-1").processed_at


def test_readding_an_episode_does_not_overwrite_original_text_or_status(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id)
    replacement = case.models.NewEpisode(uuid="document-1", source="simulation", content="replacement", metadata={"replacement": True})
    assert case.store.add_episodes(graph_id, [replacement]) == ["document-1"]
    original = case.store.get_episode(graph_id, "document-1")
    assert (original.content, original.source, original.status) == (SOURCE_TEXT, "document", "processed")


def test_episode_filters_keep_operations_and_sources_separate(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id)
    ingest(case, graph_id, episode_uuid="simulation-1", source="simulation", operation_id="simulation-operation")
    assert [episode.uuid for episode in case.store.list_episodes(graph_id, operation_id="operation-1")] == ["document-1"]
    assert [episode.uuid for episode in case.store.list_episodes(graph_id, source="simulation")] == ["simulation-1"]
    assert case.store.list_episodes(graph_id, operation_id="operation-1", source="simulation") == []


@pytest.mark.parametrize(("key", "value"), [("graph_id", "other_graph"), ("schema_version", "999")])
def test_wrong_graph_identity_or_schema_is_rejected_without_repairing_file(tmp_path, monkeypatch, key, value):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    with closing(sqlite3.connect(case.root / f"{graph_id}.sqlite3")) as connection:
        connection.execute("UPDATE meta SET value = ? WHERE key = ?", (value, key))
        connection.commit()
    with pytest.raises(case.memory.MemoryIntegrityError):
        case.memory.LocalGraphStore().list_nodes(graph_id)
    with closing(sqlite3.connect(case.root / f"{graph_id}.sqlite3")) as connection:
        assert connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()[0] == value

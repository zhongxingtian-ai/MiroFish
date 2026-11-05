"""Foreign files and failed writes must remain intact and isolated."""

import shutil
from pathlib import Path

import pytest

from memory_test_support import ONTOLOGY, create_graph, ingest, memory_case


def test_foreign_graph_file_cannot_be_read_or_deleted_under_another_id(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id)
    foreign = case.root / "foreign.sqlite3"
    shutil.copyfile(case.root / f"{graph_id}.sqlite3", foreign)
    original_bytes = foreign.read_bytes()
    for operation in (case.store.graph_exists, case.store.list_nodes, case.store.delete_graph):
        with pytest.raises(case.memory.MemoryIntegrityError):
            operation("foreign")
    assert foreign.read_bytes() == original_bytes
    assert len(case.store.list_nodes(graph_id)) == 2


def test_an_unrelated_non_database_file_is_not_repaired_or_removed(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    case.root.mkdir(parents=True)
    foreign = case.root / "foreign.sqlite3"
    foreign.write_bytes(b"ordinary synthetic file, not a graph")
    original = foreign.read_bytes()
    for operation in (case.store.list_nodes, case.store.delete_graph):
        with pytest.raises(case.memory.MemoryIntegrityError):
            operation("foreign")
    assert foreign.read_bytes() == original


def test_resolved_directory_escape_is_rejected_before_files_are_created(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    resolve = Path.resolve
    candidate = case.root / "graph.sqlite3"
    monkeypatch.setattr(Path, "resolve", lambda path, *args, **kwargs: tmp_path / "outside.sqlite3" if path == candidate else resolve(path, *args, **kwargs))
    with pytest.raises(ValueError):
        case.store.create_graph("graph", "name", "description")
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_partial_extraction_is_rolled_back_without_losing_the_original(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    case.store.add_episodes(graph_id, [case.models.NewEpisode("episode", "document", "原始材料", {})])
    extraction = case.models.Extraction((
        case.models.ExtractedEntity("周宁", "Person", "first", {}),
        case.models.ExtractedEntity("林晓", "Person", "second", {"role": object()}),
    ), ())
    with pytest.raises(TypeError):
        case.store.apply_extraction(graph_id, "episode", extraction, ONTOLOGY)
    restarted = case.memory.LocalGraphStore()
    assert restarted.list_nodes(graph_id) == []
    assert restarted.list_edges(graph_id) == []
    assert restarted.get_episode(graph_id, "episode").status == "pending"
    assert restarted.get_episode(graph_id, "episode").content == "原始材料"

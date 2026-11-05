"""Retain read-error, contextual-edge, and deadline intent from cloud regressions."""

import importlib
from types import SimpleNamespace

import pytest

from memory_test_support import (
    SOURCE_TEXT, FixedLLM, create_graph, extraction_for, extraction_payload, ingest, memory_case,
)


@pytest.mark.parametrize("operation", ["entity", "edges", "report", "profile"])
def test_local_read_errors_are_not_reported_as_empty_or_missing(tmp_path, monkeypatch, operation):
    case = memory_case(tmp_path, monkeypatch)

    def fail(*_args, **_kwargs):
        raise case.memory.MemoryIntegrityError("synthetic unreadable memory")

    store = SimpleNamespace(get_node=fail, list_nodes=fail, list_edges=fail)
    from app.services.zep_entity_reader import ZepEntityReader
    from app.services.zep_tools import ZepToolsService
    from app.services.oasis_profile_generator import OasisProfileGenerator

    with pytest.raises(case.memory.MemoryIntegrityError, match="synthetic unreadable memory"):
        if operation == "entity":
            ZepEntityReader(store=store).get_entity_with_context("graph", "node")
        elif operation == "edges":
            ZepEntityReader(store=store).get_node_edges("node", graph_id="graph")
        elif operation == "report":
            ZepToolsService(store=store).search_graph("graph", "周宁")
        else:
            profile = object.__new__(OasisProfileGenerator)
            profile.graph_id, profile.store = "graph", store
            profile._search_memory_for_entity(SimpleNamespace(name="周宁"))


def test_entity_context_contains_both_directions_of_local_relationships(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id)
    ingest(case, graph_id, episode_uuid="reverse", extraction=extraction_for(case.models, names=("林晓", "周宁"), fact="林晓回复周宁"))
    from app.services.zep_entity_reader import ZepEntityReader

    reader = ZepEntityReader(store=case.store)
    node = next(node for node in case.store.list_nodes(graph_id) if node.name == "周宁")
    context = reader.get_entity_with_context(graph_id, node.uuid)
    assert {edge["direction"] for edge in context.related_edges} == {"incoming", "outgoing"}
    assert [other["name"] for other in context.related_nodes] == ["林晓"]
    assert reader.get_entity_with_context(graph_id, "absent") is None


def test_profile_search_uses_the_entity_name_and_bounded_keyword_queries(tmp_path, monkeypatch):
    memory_case(tmp_path, monkeypatch)
    module = importlib.import_module("app.services.oasis_profile_generator")
    calls = []

    def search(store, graph_id, query, *, scope, limit):
        calls.append((graph_id, query, scope, limit))
        return SimpleNamespace(edges=(), nodes=())

    monkeypatch.setattr(module, "keyword_search", search)
    profile = object.__new__(module.OasisProfileGenerator)
    profile.graph_id, profile.store = "graph", object()
    assert profile._search_memory_for_entity(SimpleNamespace(name="周宁"))["facts"] == []
    assert calls == [("graph", "周宁", "edges", 30), ("graph", "周宁", "nodes", 20)]


def test_extraction_deadline_leaves_original_pending_instead_of_reporting_success(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    from app.services.graph_builder import GraphBuilderService

    fake = FixedLLM(extraction_payload())
    builder = GraphBuilderService(store=case.store, extractor=case.memory.GraphExtractor(llm_client=fake))
    submission = builder.add_text_batches(graph_id, [SOURCE_TEXT])
    with pytest.raises(TimeoutError, match="deadline"):
        builder._wait_for_batch(submission, timeout=0)
    assert fake.calls == []
    assert case.store.get_episode(graph_id, submission.episode_uuids[0]).status == "pending"


@pytest.mark.parametrize(("chunks", "batch_size"), [([], 350), (["text"], 0), (["text"], 351), (["x" * 10001], 350), (["x"] * 50001, 350)])
def test_invalid_document_payload_is_rejected_before_originals_or_callbacks(tmp_path, monkeypatch, chunks, batch_size):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    from app.services.graph_builder import GraphBuilderService

    callbacks = []
    builder = GraphBuilderService(store=case.store, extractor=object())
    with pytest.raises(ValueError):
        builder.add_text_batches(graph_id, chunks, batch_size=batch_size, batch_created_callback=lambda *args: callbacks.append(args))
    assert callbacks == []
    assert case.store.list_episodes(graph_id) == []

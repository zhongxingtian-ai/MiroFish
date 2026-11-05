"""Reports must retrieve processed solo statements, with literal-match provenance."""

import importlib

import pytest

from memory_test_support import create_graph, extraction_for, ingest, memory_case


STATEMENT = "周宁现在赞成午夜营业，希望店铺将营业时间延长到午夜。"
ACTIVITY_ID = "solo-statement"
ACTIVITY_TIME = "2026-11-04T07:00:00+00:00"


def _seed_activity(case, *, status="processed", source="simulation"):
    graph_id = create_graph(case)
    ingest(case, graph_id, extraction=extraction_for(case.models, names=("周宁",)))
    before = case.store.list_nodes(graph_id)[0].summary
    case.store.add_episodes(graph_id, [case.models.NewEpisode(
        uuid=ACTIVITY_ID, source=source, content=STATEMENT,
        metadata={"platform": "twitter", "round": 2}, created_at=ACTIVITY_TIME,
    )])
    if status == "processed":
        entity = case.models.ExtractedEntity("周宁", "Person", STATEMENT, {})
        case.store.apply_extraction(
            graph_id, ACTIVITY_ID, case.models.Extraction((entity,), ()),
            case.store.get_ontology(graph_id),
        )
    elif status == "failed":
        case.store.mark_episode_failed(graph_id, ACTIVITY_ID, "synthetic failure")
    assert case.store.list_nodes(graph_id)[0].summary == before
    assert case.store.list_edges(graph_id) == []
    return graph_id


def _tools(case, monkeypatch):
    tools_type = importlib.import_module("app.services.zep_tools").ZepToolsService
    service = tools_type(store=case.memory.LocalGraphStore(), llm_client=object())
    monkeypatch.setattr(service, "_generate_sub_queries", lambda **kwargs: [kwargs["query"]])
    return service


@pytest.mark.parametrize("tool", ["quick_search", "insight_forge", "get_simulation_context"])
def test_report_tools_read_a_solo_statement_after_restart_without_changing_graph_search(tmp_path, monkeypatch, tool):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = _seed_activity(case)
    service = _tools(case, monkeypatch)
    query = "午夜营业"
    assert service.search_graph(graph_id, query, scope="both").facts == []
    assert case.memory.keyword_search(case.store, graph_id, query, scope="both", limit=10).edges == ()
    if tool == "quick_search":
        text = service.quick_search(graph_id, query).to_text()
    elif tool == "insight_forge":
        text = service.insight_forge(graph_id, query, query).to_text()
    else:
        text = "\n".join(service.get_simulation_context(graph_id, query)["related_facts"])
    assert STATEMENT in text
    assert all(value in text for value in (ACTIVITY_ID, ACTIVITY_TIME, "simulation", "模拟活动原文"))


@pytest.mark.parametrize(("status", "source"), [("pending", "simulation"), ("failed", "simulation"), ("processed", "document")])
def test_reports_do_not_present_pending_failed_or_document_text_as_completed_activity(tmp_path, monkeypatch, status, source):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = _seed_activity(case, status=status, source=source)
    result = _tools(case, monkeypatch).quick_search(graph_id, "午夜营业")
    assert result.facts == []
    assert STATEMENT not in result.to_text()


def test_activity_retrieval_never_crosses_graphs_or_fills_an_unmatched_query(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    first = _seed_activity(case)
    second = create_graph(case, "other_graph")
    service = _tools(case, monkeypatch)
    assert service.quick_search(second, "午夜营业").facts == []
    assert service.quick_search(first, "木星探测").facts == []


@pytest.mark.parametrize("tool", ["quick_search", "insight_forge"])
def test_report_agent_consumes_the_sourced_activity_text_from_its_actual_tool_entry(tmp_path, monkeypatch, tool):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = _seed_activity(case)
    from app.services.report_agent import ReportAgent

    agent = object.__new__(ReportAgent)
    agent.graph_id = graph_id
    agent.simulation_requirement = "午夜营业"
    agent.zep_tools = _tools(case, monkeypatch)
    text = agent._execute_tool(tool, {"query": "午夜营业"})
    assert STATEMENT in text
    assert ACTIVITY_ID in text
    assert "模拟活动原文" in text

"""Literal keyword retrieval must be useful in Chinese and honest on misses."""

import importlib

import pytest

from memory_test_support import FACT, create_graph, extraction_for, ingest, memory_case


@pytest.mark.parametrize("query", ["", " ", "\t\n", None, 123, [], {}])
def test_blank_or_non_string_queries_are_rejected(tmp_path, monkeypatch, query):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id)
    with pytest.raises(ValueError):
        case.memory.keyword_search(case.store, graph_id, query, scope="both", limit=10)


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_search_limit_is_rejected(tmp_path, monkeypatch, limit):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    with pytest.raises(ValueError):
        case.memory.keyword_search(case.store, graph_id, "周宁", scope="both", limit=limit)


def test_two_matching_chinese_fragments_find_a_longer_question(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id)
    hits = case.memory.keyword_search(case.store, graph_id, "周宁对涨价的态度", scope="edges", limit=10)
    assert [edge.fact for edge in hits.edges] == [FACT]
    assert hits.nodes == ()
    assert hits.retrieval_mode == "keyword"


def test_one_generic_chinese_fragment_does_not_count_as_a_hit(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id, extraction=extraction_for(case.models, fact="天气的态势有所变化"))
    hits = case.memory.keyword_search(case.store, graph_id, "周宁对涨价的态度", scope="edges", limit=10)
    assert hits.edges == ()
    assert hits.nodes == ()


@pytest.mark.parametrize("query", ["不存在的航天计划", "价格上调", "QZXV"])
def test_misses_return_empty_without_recent_records_or_raw_episode_matches(tmp_path, monkeypatch, query):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id, content="QZXV 仅保存在原始材料中，周宁认为涨价可以接受")
    hits = case.memory.keyword_search(case.store, graph_id, query, scope="both", limit=10)
    assert hits.edges == ()
    assert hits.nodes == ()
    assert hits.retrieval_mode == "keyword"


def test_chinese_name_search_returns_the_named_person(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id)
    hits = case.memory.keyword_search(case.store, graph_id, "周宁", scope="nodes", limit=10)
    assert [node.name for node in hits.nodes] == ["周宁"]
    assert hits.edges == ()


def test_full_width_and_case_variants_match_the_same_english_name(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id, extraction=extraction_for(case.models, names=("Ａｌｉｃｅ", "林晓")))
    hits = case.memory.keyword_search(case.store, graph_id, "ALICE", scope="nodes", limit=10)
    assert [node.name for node in hits.nodes] == ["Ａｌｉｃｅ"]


def test_results_never_cross_graph_files(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    first, second = (create_graph(case, name) for name in ("graph_a", "graph_b"))
    ingest(case, first)
    ingest(case, second, extraction=extraction_for(case.models, names=("陈岚", "许晴"), fact="陈岚和许晴讨论优惠"))
    hits = case.memory.keyword_search(case.store, second, "周宁", scope="both", limit=10)
    assert hits.nodes == ()
    assert hits.edges == ()


def test_limit_is_capped_at_fifty_real_hits(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    entities = extraction_for(case.models).entities
    relations = tuple(case.models.ExtractedRelation(
        source_name="周宁", target_name="林晓", name="KNOWS", fact=f"coffee update {index}"
    ) for index in range(60))
    ingest(case, graph_id, extraction=case.models.Extraction(entities=entities, relations=relations))
    hits = case.memory.keyword_search(case.store, graph_id, "coffee", scope="edges", limit=100)
    assert len(hits.edges) == 50
    assert len({edge.uuid for edge in hits.edges}) == 50


def test_exact_phrase_ranks_first_and_equal_scores_keep_creation_order(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    facts = ["alpha discusses beta", "beta discusses alpha", "alpha beta"]
    for index, fact in enumerate(facts):
        ingest(case, graph_id, episode_uuid=f"episode-{index}", sequence=index,
               extraction=extraction_for(case.models, fact=fact))
    hits = case.memory.keyword_search(case.store, graph_id, "alpha beta", scope="edges", limit=10)
    assert [edge.fact for edge in hits.edges] == [facts[2], facts[0], facts[1]]


def test_missing_or_corrupt_graph_errors_are_not_replaced_with_empty_hits(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    with pytest.raises(case.memory.MemoryNotFoundError):
        case.memory.keyword_search(case.store, "missing", "周宁", scope="both", limit=10)


def test_search_result_labels_keyword_mode_for_api_and_report_text(tmp_path, monkeypatch):
    case = memory_case(tmp_path, monkeypatch)
    graph_id = create_graph(case)
    ingest(case, graph_id)
    service = importlib.import_module("app.services.zep_tools").ZepToolsService(store=case.store)
    result = service.search_graph(graph_id, "周宁 涨价", scope="edges", limit=10)
    assert result.facts == [FACT]
    assert result.to_dict()["retrieval_mode"] == "keyword"
    assert result.to_text().splitlines()[0] == "检索方式：关键词匹配（非语义检索）"

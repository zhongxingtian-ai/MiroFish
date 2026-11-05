"""Literal keyword matching for a few hundred to a few thousand graph records.

keyword_search scans nodes and edges. A separate report entry searches processed
simulation originals with the same scoring. NFKC/casefold matching and Chinese
adjacent two-character fragments help literal retrieval; there are no embeddings,
synonyms, vector similarity, reranking, or semantic search. Misses stay empty.
"""

from dataclasses import dataclass
import re
from typing import Any, Literal
import unicodedata

from .models import MemoryEdge, MemoryNode


MAX_QUERY_CHARS = 400
MAX_RESULTS = 50
_SPLIT = re.compile(r"[\s,，。！？!?；;、:：()（）\"'“”]+")
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


@dataclass(frozen=True)
class KeywordHits:
    edges: tuple[MemoryEdge, ...]
    nodes: tuple[MemoryNode, ...]
    retrieval_mode: Literal["keyword"] = "keyword"


def _normalize(value):
    return unicodedata.normalize("NFKC", value).casefold()


def normalize_query(query: Any) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"Search query must be at most {MAX_QUERY_CHARS} characters")
    return _normalize(query.strip())


def normalize_limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Search limit must be a positive integer")
    return min(limit, MAX_RESULTS)


def query_terms(query: str) -> tuple[list[str], list[str]]:
    words = list(dict.fromkeys(word for word in _SPLIT.split(_normalize(query)) if len(word) >= 2))
    fragments = list(dict.fromkeys(
        first + second for word in words for first, second in zip(word, word[1:])
        if _HAN.fullmatch(first) and _HAN.fullmatch(second)
    ))
    return words, fragments


def _score(text, query, words, fragments):
    text = _normalize(text)
    matched = sum(fragment in text for fragment in fragments)
    score = 100 * (query in text) + 10 * sum(word in text for word in words) + matched
    return score if score >= 10 or matched >= 2 else 0


def _rank(candidates, query, limit):
    words, fragments = query_terms(query)
    scored = [(_score(text, query, words, fragments), item, kind) for item, kind, text in candidates]
    scored = [entry for entry in scored if entry[0]]
    scored.sort(key=lambda entry: (-entry[0], entry[1].created_at))
    return scored[:limit]


def keyword_search(store, graph_id: str, query: str, *, scope: Literal["edges", "nodes", "both"], limit: int) -> KeywordHits:
    normalized, limit = normalize_query(query), normalize_limit(limit)
    if scope not in {"edges", "nodes", "both"}:
        raise ValueError("Search scope must be edges, nodes, or both")
    candidates = []
    if scope in {"edges", "both"}:
        candidates.extend((edge, "edge", edge.fact + " " + edge.name) for edge in store.list_edges(graph_id))
    if scope in {"nodes", "both"}:
        candidates.extend((node, "node", node.name + " " + node.summary) for node in store.list_nodes(graph_id))
    selected = _rank(candidates, normalized, limit)
    return KeywordHits(
        edges=tuple(item for _, item, kind in selected if kind == "edge"),
        nodes=tuple(item for _, item, kind in selected if kind == "node"),
    )


def search_simulation_episodes(store, graph_id: str, query: str, *, limit: int = 10):
    """Report-only literal retrieval of successfully processed simulation originals.

    This deliberately does not change keyword_search's nodes/edges contract.
    Pending/failed originals remain available for diagnosis, not completed evidence.
    """
    normalized, limit = normalize_query(query), normalize_limit(limit)
    candidates = [(episode, "episode", episode.content)
                  for episode in store.list_episodes(graph_id, source="simulation")
                  if episode.status == "processed"]
    return tuple(item for _, item, _ in _rank(candidates, normalized, limit))

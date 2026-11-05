"""Immutable data exchanged by local storage, extraction, and retrieval."""

from dataclasses import dataclass
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class MemoryNode:
    uuid: str
    name: str
    labels: tuple[str, ...]
    summary: str
    attributes: Mapping[str, Any]
    created_at: str
    episode_uuids: tuple[str, ...]


@dataclass(frozen=True)
class MemoryEdge:
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: Mapping[str, Any]
    created_at: str
    valid_at: str | None
    invalid_at: str | None
    expired_at: str | None
    episode_uuids: tuple[str, ...]


@dataclass(frozen=True)
class NewEpisode:
    uuid: str
    source: Literal["document", "simulation"]
    content: str
    metadata: Mapping[str, Any]
    operation_id: str | None = None
    sequence: int | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class MemoryEpisode:
    uuid: str
    source: str
    content: str
    metadata: Mapping[str, Any]
    operation_id: str | None
    sequence: int | None
    status: Literal["pending", "processed", "failed"]
    error: str | None
    created_at: str
    processed_at: str | None


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    type: str | None
    summary: str
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class ExtractedRelation:
    source_name: str
    target_name: str
    name: str
    fact: str


@dataclass(frozen=True)
class Extraction:
    entities: tuple[ExtractedEntity, ...]
    relations: tuple[ExtractedRelation, ...]


@dataclass(frozen=True)
class BatchProgress:
    status: Literal["succeeded", "processing", "failed"]
    total: int
    processed: int
    failed: int


def node_to_dict(node: MemoryNode) -> dict:
    return {
        "uuid": node.uuid, "name": node.name, "labels": list(node.labels),
        "summary": node.summary, "attributes": dict(node.attributes),
        "created_at": node.created_at,
    }


def edge_to_dict(edge: MemoryEdge) -> dict:
    return {
        "uuid": edge.uuid, "name": edge.name, "fact": edge.fact,
        "source_node_uuid": edge.source_node_uuid, "target_node_uuid": edge.target_node_uuid,
        "attributes": dict(edge.attributes), "created_at": edge.created_at,
        "valid_at": edge.valid_at, "invalid_at": edge.invalid_at, "expired_at": edge.expired_at,
        "episodes": list(edge.episode_uuids),
    }

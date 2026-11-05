"""One SQLite file per graph, with short transactions and no cached connections."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
from types import MappingProxyType
from typing import Sequence
import unicodedata
import uuid

from ..config import Config
from ..utils.storage_paths import storage_path
from .errors import MemoryIntegrityError, MemoryNotFoundError
from .models import (
    BatchProgress, Extraction, MemoryEdge, MemoryEpisode, MemoryNode, NewEpisode,
)


logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE episodes(
    uuid TEXT PRIMARY KEY, source TEXT NOT NULL CHECK(source IN ('document','simulation')),
    content TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
    operation_id TEXT, sequence INTEGER,
    status TEXT NOT NULL CHECK(status IN ('pending','processed','failed')),
    error TEXT, created_at TEXT NOT NULL, processed_at TEXT);
CREATE INDEX idx_episodes_operation ON episodes(operation_id, sequence);
CREATE TABLE nodes(
    uuid TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,
    labels_json TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
    attributes_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE edges(
    uuid TEXT PRIMARY KEY, name TEXT NOT NULL, fact TEXT NOT NULL, fact_key TEXT NOT NULL,
    source_node_uuid TEXT NOT NULL REFERENCES nodes(uuid) ON DELETE CASCADE,
    target_node_uuid TEXT NOT NULL REFERENCES nodes(uuid) ON DELETE CASCADE,
    attributes_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
    valid_at TEXT, invalid_at TEXT, expired_at TEXT,
    UNIQUE(source_node_uuid, target_node_uuid, name, fact_key));
CREATE TABLE node_episodes(
    node_uuid TEXT NOT NULL REFERENCES nodes(uuid) ON DELETE CASCADE,
    episode_uuid TEXT NOT NULL REFERENCES episodes(uuid) ON DELETE CASCADE,
    PRIMARY KEY(node_uuid, episode_uuid));
CREATE TABLE edge_episodes(
    edge_uuid TEXT NOT NULL REFERENCES edges(uuid) ON DELETE CASCADE,
    episode_uuid TEXT NOT NULL REFERENCES episodes(uuid) ON DELETE CASCADE,
    PRIMARY KEY(edge_uuid, episode_uuid));
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


class LocalGraphStore:
    def __init__(self):
        # Do not create directories before a graph ID has been validated.
        self.root = Path(Config.MEMORY_DATA_DIR).expanduser().resolve()

    def _path(self, graph_id: str) -> Path:
        return storage_path(self.root, graph_id, suffix=".sqlite3")

    @contextmanager
    def _open(self, path: Path):
        connection = sqlite3.connect(
            path.as_uri() + "?mode=rw", uri=True, timeout=30, isolation_level=None,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _check_identity(connection, graph_id):
        try:
            metadata = dict(connection.execute("SELECT key, value FROM meta"))
        except sqlite3.DatabaseError as error:
            raise MemoryIntegrityError("Not a supported memory database") from error
        if metadata.get("graph_id") != graph_id:
            raise MemoryIntegrityError("Memory database belongs to a different graph")
        if metadata.get("schema_version") != str(SCHEMA_VERSION):
            raise MemoryIntegrityError("Unsupported memory schema version")

    @contextmanager
    def _connect(self, graph_id, *, write=False):
        path = self._path(graph_id)
        if not path.is_file():
            raise MemoryNotFoundError(f"Graph not found: {graph_id}")
        with self._open(path) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                self._check_identity(connection, graph_id)
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def create_graph(self, graph_id: str, name: str, description: str) -> None:
        path = self._path(graph_id)
        self.root.mkdir(parents=True, exist_ok=True)
        # Exclusive creation must not truncate an existing graph or foreign file.
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            with self._open(path) as connection:
                self._initialize(connection, graph_id, name, description)
        except BaseException:
            path.unlink()
            raise

    @staticmethod
    def _initialize(connection, graph_id, name, description):
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in _SCHEMA.strip().split(";"):
                if statement.strip():
                    connection.execute(statement)
            connection.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", [
                ("schema_version", str(SCHEMA_VERSION)), ("graph_id", graph_id),
                ("name", name), ("description", description), ("created_at", _now()),
                ("ontology_json", _json({"entity_types": [], "edge_types": []})),
            ])
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def graph_exists(self, graph_id: str) -> bool:
        path = self._path(graph_id)
        if not path.exists():
            return False
        with self._connect(graph_id):
            return True

    def delete_graph(self, graph_id: str) -> None:
        path = self._path(graph_id)
        # Validate ownership, then close the handle before unlinking on Windows.
        with self._connect(graph_id):
            pass
        path.unlink()

    def set_ontology(self, graph_id: str, ontology: dict) -> None:
        with self._connect(graph_id, write=True) as connection:
            connection.execute("UPDATE meta SET value = ? WHERE key = 'ontology_json'", (_json(ontology),))

    def get_ontology(self, graph_id: str) -> dict:
        with self._connect(graph_id) as connection:
            return json.loads(connection.execute("SELECT value FROM meta WHERE key = 'ontology_json'").fetchone()[0])

    def add_episodes(self, graph_id: str, episodes: Sequence[NewEpisode]) -> list[str]:
        with self._connect(graph_id, write=True) as connection:
            connection.executemany(
                "INSERT INTO episodes(uuid,source,content,metadata_json,operation_id,sequence,status,created_at) "
                "VALUES (?,?,?,?,?,?,'pending',?) ON CONFLICT(uuid) DO NOTHING",
                [(item.uuid, item.source, item.content, _json(dict(item.metadata)),
                  item.operation_id, item.sequence, item.created_at or _now()) for item in episodes],
            )
        return [item.uuid for item in episodes]

    @staticmethod
    def _episode(row) -> MemoryEpisode:
        return MemoryEpisode(
            uuid=row["uuid"], source=row["source"], content=row["content"],
            metadata=MappingProxyType(json.loads(row["metadata_json"])),
            operation_id=row["operation_id"], sequence=row["sequence"], status=row["status"],
            error=row["error"], created_at=row["created_at"], processed_at=row["processed_at"],
        )

    def get_episode(self, graph_id: str, uuid: str) -> MemoryEpisode:
        with self._connect(graph_id) as connection:
            row = connection.execute("SELECT * FROM episodes WHERE uuid = ?", (uuid,)).fetchone()
            if row is None:
                raise MemoryNotFoundError(f"Episode not found: {uuid}")
            return self._episode(row)

    def list_episodes(self, graph_id: str, *, operation_id=None, source=None) -> list[MemoryEpisode]:
        with self._connect(graph_id) as connection:
            clauses, parameters = [], []
            for column, value in (("operation_id", operation_id), ("source", source)):
                if value is not None:
                    clauses.append(column + " = ?")  # column names are fixed above.
                    parameters.append(value)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = connection.execute("SELECT * FROM episodes" + where + " ORDER BY sequence, created_at, rowid", parameters)
            return [self._episode(row) for row in rows]

    def mark_episode_failed(self, graph_id: str, uuid: str, error: str) -> None:
        with self._connect(graph_id, write=True) as connection:
            changed = connection.execute(
                "UPDATE episodes SET status='failed', error=?, processed_at=NULL WHERE uuid=?", (error, uuid),
            ).rowcount
            if not changed:
                raise MemoryNotFoundError(f"Episode not found: {uuid}")

    def apply_extraction(self, graph_id: str, episode_uuid: str, extraction: Extraction, ontology: dict) -> None:
        with self._connect(graph_id, write=True) as connection:
            episode = connection.execute("SELECT status FROM episodes WHERE uuid=?", (episode_uuid,)).fetchone()
            if episode is None:
                raise MemoryNotFoundError(f"Episode not found: {episode_uuid}")
            if episode["status"] == "processed":
                return
            entity_types = {item["name"]: item for item in ontology.get("entity_types", [])}
            edge_types = {item["name"] for item in ontology.get("edge_types", [])}
            for entity in extraction.entities:
                node_uuid = self._upsert_entity(connection, entity, entity_types)
                if node_uuid:
                    self._link_node(connection, node_uuid, episode_uuid)
            for relation in extraction.relations:
                self._apply_relation(connection, relation, edge_types, episode_uuid)
            connection.execute(
                "UPDATE episodes SET status='processed', processed_at=?, error=NULL WHERE uuid=?", (_now(), episode_uuid),
            )

    @staticmethod
    def _upsert_entity(connection, entity, types):
        if not isinstance(entity.name, str) or not entity.name.strip():
            return None
        name_key = _key(entity.name)
        existing = connection.execute("SELECT * FROM nodes WHERE name_key=?", (name_key,)).fetchone()
        entity_type = entity.type if entity.type in types else None
        labels = json.loads(existing["labels_json"]) if existing else ["Entity"]
        if len(labels) == 1 and entity_type:
            labels.append(entity_type)
        elif entity_type and labels[1:] != [entity_type]:
            logger.warning("Keeping existing entity type during a conflicting extraction")
        declared = types.get(labels[1], {}) if len(labels) > 1 else {}
        allowed = {item["name"] for item in declared.get("attributes", [])}
        attributes = json.loads(existing["attributes_json"]) if existing else {}
        attributes.update({key: value for key, value in entity.attributes.items() if key in allowed})
        node_uuid = existing["uuid"] if existing else str(uuid.uuid4())
        now = _now()
        connection.execute(
            "INSERT INTO nodes(uuid,name,name_key,labels_json,summary,attributes_json,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(name_key) DO UPDATE SET "
            "labels_json=excluded.labels_json,summary=excluded.summary,attributes_json=excluded.attributes_json,updated_at=excluded.updated_at",
            (node_uuid, existing["name"] if existing else entity.name.strip(), name_key, _json(labels),
             (existing["summary"] if existing else "") or entity.summary, _json(attributes),
             existing["created_at"] if existing else now, now),
        )
        return node_uuid

    @staticmethod
    def _link_node(connection, node_uuid, episode_uuid):
        connection.execute(
            "INSERT INTO node_episodes(node_uuid,episode_uuid) VALUES (?,?) ON CONFLICT DO NOTHING", (node_uuid, episode_uuid),
        )

    def _apply_relation(self, connection, relation, types, episode_uuid):
        if not isinstance(relation.fact, str) or not relation.fact.strip():
            return
        endpoints = [connection.execute("SELECT uuid FROM nodes WHERE name_key=?", (_key(name),)).fetchone()
                     for name in (relation.source_name, relation.target_name)]
        if not all(endpoints):
            logger.debug("Discarded relation with an unknown endpoint")
            return
        source, target = (row["uuid"] for row in endpoints)
        name = relation.name if relation.name in types else "RELATES_TO"
        fact_key = _key(relation.fact)
        connection.execute(
            "INSERT INTO edges(uuid,name,fact,fact_key,source_node_uuid,target_node_uuid,created_at) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(source_node_uuid,target_node_uuid,name,fact_key) DO NOTHING",
            (str(uuid.uuid4()), name, relation.fact, fact_key, source, target, _now()),
        )
        edge_uuid = connection.execute(
            "SELECT uuid FROM edges WHERE source_node_uuid=? AND target_node_uuid=? AND name=? AND fact_key=?",
            (source, target, name, fact_key),
        ).fetchone()["uuid"]
        connection.execute("INSERT INTO edge_episodes(edge_uuid,episode_uuid) VALUES (?,?) ON CONFLICT DO NOTHING", (edge_uuid, episode_uuid))
        for node_uuid in (source, target):
            self._link_node(connection, node_uuid, episode_uuid)

    @staticmethod
    def _sources(connection, record_uuid, *, edge=False):
        sql = ("SELECT episode_uuid FROM edge_episodes WHERE edge_uuid=? ORDER BY rowid" if edge
               else "SELECT episode_uuid FROM node_episodes WHERE node_uuid=? ORDER BY rowid")
        return tuple(row[0] for row in connection.execute(sql, (record_uuid,)))

    def _node(self, connection, row):
        return MemoryNode(
            uuid=row["uuid"], name=row["name"], labels=tuple(json.loads(row["labels_json"])),
            summary=row["summary"], attributes=MappingProxyType(json.loads(row["attributes_json"])),
            created_at=row["created_at"], episode_uuids=self._sources(connection, row["uuid"]),
        )

    def _edge(self, connection, row):
        return MemoryEdge(
            uuid=row["uuid"], name=row["name"], fact=row["fact"],
            source_node_uuid=row["source_node_uuid"], target_node_uuid=row["target_node_uuid"],
            attributes=MappingProxyType(json.loads(row["attributes_json"])), created_at=row["created_at"],
            valid_at=row["valid_at"], invalid_at=row["invalid_at"], expired_at=row["expired_at"],
            episode_uuids=self._sources(connection, row["uuid"], edge=True),
        )

    def list_nodes(self, graph_id: str) -> list[MemoryNode]:
        with self._connect(graph_id) as connection:
            return [self._node(connection, row) for row in connection.execute("SELECT * FROM nodes ORDER BY rowid")]

    def list_edges(self, graph_id: str) -> list[MemoryEdge]:
        with self._connect(graph_id) as connection:
            return [self._edge(connection, row) for row in connection.execute("SELECT * FROM edges ORDER BY rowid")]

    def get_node(self, graph_id: str, node_uuid: str) -> MemoryNode:
        with self._connect(graph_id) as connection:
            row = connection.execute("SELECT * FROM nodes WHERE uuid=?", (node_uuid,)).fetchone()
            if row is None:
                raise MemoryNotFoundError(f"Node not found: {node_uuid}")
            return self._node(connection, row)

    def batch_progress(self, graph_id: str, operation_id: str) -> BatchProgress:
        with self._connect(graph_id) as connection:
            row = connection.execute(
                "SELECT COUNT(*), SUM(status='processed'), SUM(status='failed') FROM episodes WHERE operation_id=?", (operation_id,),
            ).fetchone()
            total, processed, failed = (int(value or 0) for value in row)
            status = "succeeded" if total and processed == total else "processing" if total else "failed"
            return BatchProgress(status=status, total=total, processed=processed, failed=failed)

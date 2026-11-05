"""Shared synthetic data for the local-memory behavior tests."""

import copy
import importlib
import os
import socket
import sqlite3
from contextlib import closing
from types import SimpleNamespace


ONTOLOGY = {
    "entity_types": [
        {
            "name": "Person",
            "description": "明确具名的人物",
            "attributes": [{"name": "role", "type": "text", "description": "身份"}],
        }
    ],
    "edge_types": [
        {
            "name": "KNOWS",
            "description": "人物之间已知的关系",
            "attributes": [],
            "source_targets": [{"source": "Person", "target": "Person"}],
        }
    ],
}

PEOPLE = ("周宁", "林晓", "陈岚", "许晴")
FACT = "周宁告诉林晓，周宁认为涨价可以接受"
SOURCE_TEXT = (
    "林晓经营社区咖啡店。周宁告诉林晓，周宁认为涨价可以接受。"
    "陈岚询问林晓自带杯优惠。许晴和陈岚讨论优惠，四人互相认识。"
)


def deny_network(*_args, **_kwargs):
    raise AssertionError("Local-memory tests must not make network requests")


def configure_runtime(tmp_path, monkeypatch):
    from app.config import Config

    for name in tuple(os.environ):
        if name.startswith("ZEP_"):
            monkeypatch.delenv(name)
    for name in vars(Config):
        if name.startswith("ZEP_"):
            monkeypatch.setattr(Config, name, None)
    root = tmp_path / "memory"
    monkeypatch.setattr(Config, "MEMORY_DATA_DIR", str(root), raising=False)
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setattr(Config, "LLM_API_KEY", "local-memory-test-only")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(Config, "DEBUG", False)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    return root


def memory_case(tmp_path, monkeypatch):
    # Called from test bodies: a missing package must fail a test, not collection.
    memory = importlib.import_module("app.memory")
    models = importlib.import_module("app.memory.models")
    root = configure_runtime(tmp_path, monkeypatch)
    return SimpleNamespace(
        memory=memory, models=models, root=root, store=memory.LocalGraphStore()
    )


def create_graph(case, graph_id="graph_local"):
    case.store.create_graph(graph_id, "本地测试图谱", "只含合成测试数据")
    case.store.set_ontology(graph_id, copy.deepcopy(ONTOLOGY))
    return graph_id


def extraction_for(models, *, names=PEOPLE[:2], fact=FACT, relation_name="KNOWS"):
    entities = tuple(
        models.ExtractedEntity(
            name=name, type="Person", summary=f"{name}参与社区讨论", attributes={}
        )
        for name in names
    )
    relations = ()
    if fact and len(names) >= 2:
        relations = (models.ExtractedRelation(
            source_name=names[0], target_name=names[1], name=relation_name, fact=fact
        ),)
    return models.Extraction(entities=entities, relations=relations)


def ingest(case, graph_id, *, episode_uuid="document-1", source="document",
           content=SOURCE_TEXT, operation_id="operation-1", sequence=0,
           extraction=None, metadata=None):
    episode = case.models.NewEpisode(
        uuid=episode_uuid, source=source, content=content,
        metadata=metadata or {}, operation_id=operation_id, sequence=sequence,
    )
    case.store.add_episodes(graph_id, [episode])
    case.store.apply_extraction(
        graph_id, episode_uuid,
        extraction if extraction is not None else extraction_for(case.models),
        copy.deepcopy(ONTOLOGY),
    )
    return episode_uuid


def extraction_payload():
    return {
        "entities": [
            {"name": name, "type": "Person", "summary": f"{name}参与讨论",
             "attributes": {"role": "参与者"}}
            for name in PEOPLE
        ],
        "relations": [
            {"source": PEOPLE[0], "target": PEOPLE[1], "type": "KNOWS", "fact": FACT}
        ],
    }


class FixedLLM:
    def __init__(self, response=None, *, error=None, on_call=None):
        self.response = response
        self.error = error
        self.on_call = on_call
        self.calls = []

    def chat_json(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        if self.on_call is not None:
            self.on_call()
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.response)


def assert_write_lock_available(case, graph_id):
    with closing(sqlite3.connect(
        case.root / f"{graph_id}.sqlite3", timeout=0.05, isolation_level=None
    )) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.rollback()

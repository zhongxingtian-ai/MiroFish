"""
图谱构建服务
使用大模型抽取原文并保存到本地 SQLite
"""

import hashlib
import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..memory import GraphExtractor, LocalGraphStore, MEMORY_DRAIN_TIMEOUT_SECONDS
from ..memory.models import BatchProgress, NewEpisode, node_to_dict, edge_to_dict
from ..models.task import TaskManager, TaskStatus
from ..utils.ontology import (
    MAX_ONTOLOGY_TYPES, RESERVED_ONTOLOGY_ATTRIBUTE_NAMES,
    normalize_ontology_attributes, normalize_ontology_source_targets,
)
from .text_processor import TextProcessor
from ..utils.locale import t, get_locale, set_locale



def normalize_ontology_for_storage(ontology):
    """Keep the existing bounded ontology contract without generating SDK models."""
    result = dict(ontology)
    for key in ("entity_types", "edge_types"):
        definitions = []
        for item in ontology.get(key, []):
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                continue
            definition = dict(item)
            attributes = normalize_ontology_attributes(item.get("attributes"))
            definition["attributes"] = [dict(attribute, name=(
                "entity_" + attribute["name"]
                if attribute["name"].lower() in RESERVED_ONTOLOGY_ATTRIBUTE_NAMES else attribute["name"]
            )) for attribute in attributes]
            if key == "edge_types":
                definition["source_targets"] = normalize_ontology_source_targets(item.get("source_targets"))
            definitions.append(definition)
            if len(definitions) == MAX_ONTOLOGY_TYPES:
                break
        result[key] = definitions
    return result


@dataclass
class GraphInfo:
    """图谱信息"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


@dataclass(frozen=True)
class BatchSubmission:
    """Durable identity for local document ingestion."""

    batch_id: str
    operation_id: str
    episode_uuids: List[str]
    item_count: int
    graph_id: str | None = None


class GraphBuilderService:
    """
    图谱构建服务
    负责抽取实体与关系并持久化本地知识图谱
    """
    
    def __init__(self, store=None, extractor=None):
        self.store = store if store is not None else LocalGraphStore()
        self.extractor = extractor if extractor is not None else GraphExtractor()
        self.task_manager = TaskManager()
    
    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 350
    ) -> str:
        """
        异步构建图谱
        
        Args:
            text: 输入文本
            ontology: 本体定义（来自接口1的输出）
            graph_name: 图谱名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            batch_size: 每批发送的块数量
            
        Returns:
            任务ID
        """
        # 创建任务
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )
        
        # Capture locale before spawning background thread
        current_locale = get_locale()

        # 在后台线程中执行构建
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size, current_locale)
        )
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = 'zh'
    ):
        """图谱构建工作线程"""
        set_locale(locale)
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message=t('progress.startBuildingGraph')
            )
            
            # Validate the complete ingestion payload before the first local storage
            # mutation, including this legacy service entry point.
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            self.validate_batch_chunks(chunks, batch_size=batch_size)
            total_chunks = len(chunks)

            # 1. 创建图谱
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=t('progress.graphCreated', graphId=graph_id)
            )
            
            # 2. 设置本体
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message=t('progress.ontologySet')
            )
            
            # 3. 文本分块已在 local storage mutation 前完成并验证
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=t('progress.textSplit', count=total_chunks)
            )
            
            # 4. 分批发送数据
            submission = self.add_text_batches(
                graph_id, chunks, batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 40),  # 20-60%
                    message=msg
                )
            )
            
            # 5. 等待本地抽取完成
            self.task_manager.update_task(
                task_id,
                progress=60,
                message=t('progress.waitingZepProcess')
            )
            
            self._wait_for_batch(
                submission,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=60 + int(prog * 30),  # 60-90%
                    message=msg
                )
            )
            
            # 6. 获取图谱信息
            self.task_manager.update_task(
                task_id,
                progress=90,
                message=t('progress.fetchingGraphInfo')
            )
            
            graph_info = self._get_graph_info(graph_id)
            
            # 完成
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)
    
    def create_graph(self, name, *, graph_id=None, graph_id_callback=None):
        """Journal the graph identity before exclusive local creation."""
        graph_id = graph_id if graph_id is not None else f"mirofish_{uuid.uuid4().hex[:16]}"
        if graph_id_callback:
            graph_id_callback(graph_id)
        self.store.create_graph(graph_id, name, "MiroFish Social Simulation Graph")
        return graph_id

    @staticmethod
    def build_operation_id(graph_id: str, chunks: List[str]) -> str:
        payload_hash = hashlib.sha256("\0".join(chunks).encode("utf-8")).hexdigest()
        return hashlib.sha256(
            f"{graph_id}:{payload_hash}".encode("utf-8")
        ).hexdigest()

    
    def set_ontology(self, graph_id, ontology):
        self.store.set_ontology(graph_id, normalize_ontology_for_storage(ontology))
    
    def add_text_batches(self, graph_id, chunks, batch_size=350, progress_callback=None, batch_created_callback=None):
        """Persist every original before extraction; retries reuse episode IDs."""
        self.validate_batch_chunks(chunks, batch_size=batch_size)
        self.store.get_ontology(graph_id)
        operation_id = self.build_operation_id(graph_id, chunks)
        if batch_created_callback:
            batch_created_callback(operation_id, operation_id)
        episodes = [NewEpisode(
            uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, f"mirofish:{operation_id}:{index}")),
            source="document", content=chunk,
            metadata={"chunk_index": index}, operation_id=operation_id, sequence=index,
        ) for index, chunk in enumerate(chunks)]
        episode_uuids = self.store.add_episodes(graph_id, episodes)
        if progress_callback:
            progress_callback(t('progress.sendingBatch', current=1, total=1, chunks=len(chunks)), 1.0)
        return BatchSubmission(operation_id, operation_id, episode_uuids, len(chunks), graph_id)

    @staticmethod
    def validate_batch_chunks(chunks: List[str], *, batch_size: int = 350) -> None:
        """Validate every Batch API limit before the first local storage mutation."""

        if not chunks:
            raise ValueError("At least one text chunk is required")
        if not 1 <= batch_size <= 350:
            raise ValueError("batch_size must be between 1 and 350")
        if len(chunks) > 50_000:
            raise ValueError("A ingestion batch cannot contain more than 50,000 items")
        oversized = [index for index, chunk in enumerate(chunks) if len(chunk) > 10_000]
        if oversized:
            raise ValueError(
                f"ingestion batch item exceeds 10,000 characters at chunk {oversized[0]}"
            )



    def get_batch_summary(self, graph_id, batch_id) -> BatchProgress:
        return self.store.batch_progress(graph_id, batch_id)

    def _wait_for_batch(self, submission, progress_callback=None, timeout=None):
        """Extract pending/failed originals outside the store's short transactions."""
        graph_id = submission.graph_id
        if not graph_id:
            raise ValueError("A graph_id is required to resume ingestion")
        episodes = self.store.list_episodes(graph_id, operation_id=submission.operation_id)
        if len(episodes) != submission.item_count:
            raise RuntimeError("Persisted ingestion item count does not match the submission")
        ontology = self.store.get_ontology(graph_id)
        deadline = time.monotonic() + (timeout if timeout is not None else MEMORY_DRAIN_TIMEOUT_SECONDS)
        for index, episode in enumerate(episodes):
            if time.monotonic() >= deadline:
                raise TimeoutError("Local graph extraction exceeded its deadline")
            if episode.status != "processed":
                self._extract_episode(graph_id, episode, ontology, index, len(episodes))
            if progress_callback:
                progress_callback(t('progress.processingComplete', completed=index + 1, total=len(episodes)), (index + 1) / len(episodes))
        progress = self.store.batch_progress(graph_id, submission.operation_id)
        if progress.status != "succeeded" or progress.total != submission.item_count:
            raise RuntimeError("Local graph extraction is incomplete")
        return [episode.uuid for episode in episodes]

    def _extract_episode(self, graph_id, episode, ontology, index, total):
        try:
            known_names = [node.name for node in self.store.list_nodes(graph_id)][:200]
            extraction = self.extractor.extract(episode.content, ontology, known_names)
            self.store.apply_extraction(graph_id, episode.uuid, extraction, ontology)
        except Exception as error:
            self.store.mark_episode_failed(graph_id, episode.uuid, str(error))
            raise RuntimeError(f"第 {index + 1}/{total} 块抽取失败: {error}") from error
    
    
    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """获取图谱信息"""
        # 获取节点（分页）
        nodes = self.store.list_nodes(graph_id)

        # 获取边（分页）
        edges = self.store.list_edges(graph_id)

        # 统计实体类型
        entity_types = set()
        for node in nodes:
            if node.labels:
                for label in node.labels:
                    if label not in ["Entity", "Node"]:
                        entity_types.add(label)

        return GraphInfo(
            graph_id=graph_id,
            node_count=len(nodes),
            edge_count=len(edges),
            entity_types=list(entity_types)
        )
    
    def get_graph_data(self, graph_id):
        nodes, edges = self.store.list_nodes(graph_id), self.store.list_edges(graph_id)
        names = {node.uuid: node.name for node in nodes}
        edges_data = [dict(
            edge_to_dict(edge), fact_type=edge.name,
            source_node_name=names.get(edge.source_node_uuid, ""),
            target_node_name=names.get(edge.target_node_uuid, ""),
        ) for edge in edges]
        return {
            "graph_id": graph_id, "nodes": [node_to_dict(node) for node in nodes],
            "edges": edges_data, "node_count": len(nodes), "edge_count": len(edges),
        }
    
    def delete_graph(self, graph_id):
        self.store.delete_graph(graph_id)

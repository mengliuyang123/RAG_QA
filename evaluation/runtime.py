# -*- coding: utf-8 -*-
"""构建带指纹、与生产 Chroma 隔离的评估运行时。"""

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from app.application.ports.chunking import Chunker
from app.application.ports.embedding import EmbeddingProvider
from app.application.ports.keyword_index import KeywordIndex
from app.application.query.service import QueryService
from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from app.bootstrap.factories import VectorStoreAdapter
from app.config import Settings
from app.domain.documents import DocumentChunk, DocumentSegment
from evaluation.models import CorpusDataset


COLLECTION_SCHEMA_VERSION = "1"
DOCUMENT_NAMESPACE = uuid.UUID("f93b85f6-4b9a-5ed5-922f-729fea21af48")


class EvaluationContainer(Protocol):
    """声明评估装配所需的最小共享容器表面。"""

    @property
    def chunker(self) -> Chunker:
        """返回共享分块器。

        Returns:
            按应用配置创建的分块端口。
        """

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """返回共享向量 provider。

        Returns:
            同时服务索引和查询的向量端口。
        """

    @property
    def vector_store(self) -> VectorStoreAdapter:
        """返回隔离运行绑定的向量存储。

        Returns:
            具备读、写和管理能力的向量存储端口。
        """

    @property
    def keyword_index(self) -> KeywordIndex:
        """返回从已准备向量存储构建的关键词索引。

        Returns:
            共享关键词检索端口。
        """

    @property
    def baseline_query_service(self) -> QueryService:
        """返回共享查询应用服务。

        Returns:
            完成检索与生成编排的查询服务。
        """

    def close(self) -> None:
        """幂等释放已经创建的运行时资源。

        Returns:
            None。
        """


ContainerBuilder = Callable[
    [Settings, RuntimeOptions],
    EvaluationContainer,
]


class EvaluationRuntimeError(RuntimeError):
    """评估索引状态与固定指纹约定不一致时抛出。"""


class IndexConfig(BaseModel):
    """参与评估索引指纹的非敏感配置。

    Attributes:
        embedding_origin: 不含路径和凭据的 Embedding 服务 origin。
        embedding_model: Embedding 模型名。
        chunk_size: 文本块最大字符数。
        chunk_overlap: 相邻文本块重叠字符数。
        embedding_batch_size: 单次文档向量请求上限。
        collection_schema_version: collection 结构版本。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_origin: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    embedding_batch_size: int = Field(gt=0)
    collection_schema_version: str = COLLECTION_SCHEMA_VERSION

    @classmethod
    def from_settings(cls, settings: Settings) -> "IndexConfig":
        """从应用配置提取安全的索引身份字段。

        Args:
            settings: 已加载的应用配置。

        Returns:
            不包含 API Key 或完整服务路径的索引配置。
        """
        return cls(
            embedding_origin=_service_origin(settings.embedding_base_url),
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            embedding_batch_size=settings.embedding_batch_size,
        )


@dataclass(frozen=True)
class EvaluationRuntime:
    """已校验并装配完成的隔离评估组件。

    Attributes:
        collection_name: 带 12 位指纹后缀的 collection 名称。
        index_fingerprint: 规范索引清单的完整 SHA-256。
        index_manifest_json: 不含凭据的规范 JSON。
        expected_chunk_count: 固定语料按当前配置产生的块数。
        container: 生产与评估共享的应用组合根。
        vector_store: 隔离评估 Chroma 存储。
        keyword_index: 与 Chroma 内容一致的 BM25 索引。
        query_service: 支持三种检索策略的评估查询服务。
    """

    collection_name: str
    index_fingerprint: str
    index_manifest_json: str
    expected_chunk_count: int
    container: EvaluationContainer
    vector_store: VectorStoreAdapter
    keyword_index: KeywordIndex
    query_service: QueryService

    def close(self) -> None:
        """释放运行时持有的远程连接资源。

        Returns:
            None。
        """
        self.container.close()


def compute_index_fingerprint(
    config: IndexConfig,
    corpus: CorpusDataset,
) -> str:
    """计算与 manifest 顺序无关的评估索引 SHA-256。

    Args:
        config: 参与索引身份计算的非敏感配置。
        corpus: 已完成哈希校验的固定语料。

    Returns:
        规范索引 JSON 的小写十六进制 SHA-256。
    """
    manifest_json = _index_manifest_json(config, corpus)
    return hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()


def build_evaluation_runtime(
    settings: Settings,
    corpus: CorpusDataset,
    rebuild: bool = False,
    container_builder: ContainerBuilder = build_application_container,
) -> EvaluationRuntime:
    """构建或复用一个与生产路径隔离的固定评估索引。

    Args:
        settings: 应用和评估环境配置。
        corpus: 已完成数量与哈希校验的固定语料。
        rebuild: 是否只重建当前指纹对应的评估 collection。
        container_builder: 生产与评估共享的惰性容器构建函数。

    Returns:
        装配完成的隔离检索与生成运行时。

    Raises:
        ValueError: 生产与评估路径相同。
        EvaluationRuntimeError: collection 元数据或记录数不一致。
    """
    # 路径隔离必须先于 Chroma、HTTP 客户端和其他有副作用的
    # 构造，以免无效配置留下本地或远程资源。
    settings.validate_evaluation_settings(require_judge=False)
    config = IndexConfig.from_settings(settings)
    manifest_json = _index_manifest_json(config, corpus)
    fingerprint = hashlib.sha256(
        manifest_json.encode("utf-8")
    ).hexdigest()
    collection_name = (
        f"{settings.evaluation_collection_prefix}_{fingerprint[:12]}"
    )
    metadata = {
        "evaluation_fingerprint": fingerprint,
        "collection_schema_version": COLLECTION_SCHEMA_VERSION,
    }
    runtime_options = RuntimeOptions(
        chroma_path=settings.evaluation_chroma_path,
        collection_name=collection_name,
        collection_metadata=metadata,
        rebuild_collection=rebuild,
        llm_temperature=settings.evaluation_local_temperature,
    )
    container = container_builder(settings, runtime_options)
    try:
        chunks = _build_chunks(container.chunker, corpus)
        if not chunks:
            raise EvaluationRuntimeError(
                "固定评估语料没有产生任何文本块"
            )

        embedding_provider = container.embedding_provider
        vector_store = container.vector_store
        _prepare_index(
            vector_store,
            chunks,
            embedding_provider,
            settings.embedding_batch_size,
            fingerprint,
        )
        keyword_index = container.keyword_index
        query_service = container.baseline_query_service
        return EvaluationRuntime(
            collection_name=collection_name,
            index_fingerprint=fingerprint,
            index_manifest_json=manifest_json,
            expected_chunk_count=len(chunks),
            container=container,
            vector_store=vector_store,
            keyword_index=keyword_index,
            query_service=query_service,
        )
    except BaseException as exc:
        try:
            container.close()
        except BaseException as close_error:
            exc.add_note(
                "评估运行时装配失败后的容器关闭也失败："
                f"{type(close_error).__name__}"
            )
        raise


def _index_manifest_json(
    config: IndexConfig,
    corpus: CorpusDataset,
) -> str:
    documents = []
    for entry in corpus.entries:
        documents.append(
            {"filename": entry.filename, "sha256": entry.sha256}
        )
    documents.sort(key=lambda item: item["filename"])
    payload = {
        "config": config.model_dump(mode="json"),
        "documents": documents,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _build_chunks(
    chunker: Chunker,
    corpus: CorpusDataset,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    entries = sorted(corpus.entries, key=lambda entry: entry.filename)
    for entry in entries:
        document_id = str(
            uuid.uuid5(
                DOCUMENT_NAMESPACE,
                f"{entry.filename}:{entry.sha256}",
            )
        )
        segment = DocumentSegment(
            text=corpus.documents[entry.filename],
            page=None,
            filename=entry.filename,
        )
        chunks.extend(chunker.split(document_id, [segment]))
    return chunks


def _prepare_index(
    vector_store: VectorStoreAdapter,
    chunks: list[DocumentChunk],
    embedder: EmbeddingProvider,
    batch_size: int,
    fingerprint: str,
) -> None:
    metadata = vector_store.collection_metadata()
    if metadata.get("evaluation_fingerprint") != fingerprint:
        raise EvaluationRuntimeError(
            "评估 collection 指纹元数据不匹配"
        )

    stored_count = vector_store.count()
    expected_count = len(chunks)
    if stored_count > 0:
        if metadata.get("expected_chunk_count") != expected_count:
            raise EvaluationRuntimeError(
                "评估 collection 预期块数不匹配"
            )
        if stored_count != expected_count:
            raise EvaluationRuntimeError(
                "评估 collection 实际块数不匹配"
            )
        return

    try:
        for start in range(0, expected_count, batch_size):
            batch = chunks[start:start + batch_size]
            embeddings = embedder.embed_documents(
                [chunk.text for chunk in batch]
            )
            vector_store.add_chunks(batch, embeddings)
        vector_store.update_collection_metadata(
            {"expected_chunk_count": expected_count}
        )
    except Exception:
        # 完整指纹限定了删除范围，因此失败清理不会影响其他
        # 评估索引。
        vector_store.delete_collection()
        raise


def _service_origin(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if not parsed.scheme or parsed.hostname is None:
        raise ValueError("Embedding 服务地址必须包含 scheme 和 host")
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Embedding 服务地址端口无效") from exc
    is_default_port = (
        scheme == "https" and port == 443
    ) or (
        scheme == "http" and port == 80
    )
    if port is None or is_default_port:
        return f"{scheme}://{hostname}"
    return f"{scheme}://{hostname}:{port}"

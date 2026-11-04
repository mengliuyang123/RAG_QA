# -*- coding: utf-8 -*-
"""隔离评估索引指纹与运行时工厂测试。"""

from pathlib import Path

import pytest

from app.bootstrap.container import RuntimeOptions
from app.config import Settings
from app.domain.documents import DocumentChunk, DocumentSegment
from evaluation.datasets import CorpusDataset
from evaluation.models import CorpusEntry


def _entry(filename: str, digest_character: str) -> CorpusEntry:
    digest = digest_character * 64
    return CorpusEntry(
        filename=filename,
        title=filename,
        source_url=f"https://example.test/{filename}",
        source_revision="a" * 40,
        source_path=f"docs/{filename}",
        source_sha256=digest,
        license="MIT",
        sha256=digest,
    )


def _corpus(*entries: CorpusEntry) -> CorpusDataset:
    documents = {
        entry.filename: f"# {entry.title}\n\n固定正文"
        for entry in entries
    }
    return CorpusDataset(entries=list(entries), documents=documents)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_api_key": "generator-secret",
        "embedding_api_key": "embedding-secret",
        "llm_base_url": "https://generator.example/v1/private",
        "embedding_base_url": "https://embedding.example/v1/private",
        "reranker_base_url": "https://reranker.example/v1/private",
        "reranker_api_key": "reranker-secret",
        "chroma_path": tmp_path / "production",
        "evaluation_chroma_path": tmp_path / "evaluation",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


class _FakeVectorStore:
    def __init__(
        self,
        metadata: dict[str, object] | None = None,
        chunks: list[DocumentChunk] | None = None,
    ) -> None:
        self.metadata = dict(metadata or {})
        self.chunks = list(chunks or [])
        self.deleted = False

    def collection_metadata(self) -> dict[str, object]:
        return dict(self.metadata)

    def update_collection_metadata(
        self,
        metadata: dict[str, str | int],
    ) -> None:
        self.metadata.update(metadata)

    def count(self) -> int:
        return len(self.chunks)

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        assert len(chunks) == len(embeddings)
        self.chunks.extend(chunks)

    def delete_collection(self) -> None:
        self.deleted = True
        self.chunks.clear()

    def list_chunks(self) -> list[DocumentChunk]:
        return list(self.chunks)


class _FakeEmbedder:
    def __init__(self, reject_documents: bool = False) -> None:
        self.document_calls: list[list[str]] = []
        self.reject_documents = reject_documents

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.reject_documents:
            raise AssertionError(f"不应重复向量化：{len(texts)}")
        self.document_calls.append(texts)
        return [[1.0, 0.0] for _text in texts]

    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class _FakeChunker:
    def split(
        self,
        doc_id: str,
        segments: list[DocumentSegment],
    ) -> list[DocumentChunk]:
        segment = segments[0]
        return [
            DocumentChunk(
                id=f"{doc_id}:0",
                doc_id=doc_id,
                text=segment.text,
                filename=segment.filename,
                page=segment.page,
                chunk_index=0,
            )
        ]


class _FakeContainer:
    def __init__(
        self,
        vector_store: _FakeVectorStore,
        embedding_provider: _FakeEmbedder,
    ) -> None:
        self._chunker = _FakeChunker()
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._keyword_index = object()
        self._query_service = object()
        self.accesses: list[str] = []
        self.close_calls = 0
        self._closed = False

    @property
    def chunker(self) -> _FakeChunker:
        self.accesses.append("chunker")
        return self._chunker

    @property
    def embedding_provider(self) -> _FakeEmbedder:
        self.accesses.append("embedding_provider")
        return self._embedding_provider

    @property
    def vector_store(self) -> _FakeVectorStore:
        self.accesses.append("vector_store")
        return self._vector_store

    @property
    def keyword_index(self) -> object:
        assert self._vector_store.count() > 0
        self.accesses.append("keyword_index")
        return self._keyword_index

    @property
    def baseline_query_service(self) -> object:
        self.accesses.append("baseline_query_service")
        return self._query_service

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.close_calls += 1


def test_index_fingerprint_is_independent_of_manifest_order() -> None:
    """规范 JSON 应使同一语料的 manifest 顺序不影响指纹。"""
    from evaluation.runtime import IndexConfig, compute_index_fingerprint

    first = _entry("a.md", "a")
    second = _entry("b.md", "b")
    config = IndexConfig(
        embedding_origin="https://embedding.example",
        embedding_model="embedding-model",
        chunk_size=500,
        chunk_overlap=50,
        embedding_batch_size=100,
    )

    forward = compute_index_fingerprint(config, _corpus(first, second))
    reverse = compute_index_fingerprint(config, _corpus(second, first))

    assert forward == reverse
    assert len(forward) == 64


def test_index_fingerprint_changes_with_hash_or_config() -> None:
    """任何索引输入变化都必须生成不同 collection 指纹。"""
    from evaluation.runtime import IndexConfig, compute_index_fingerprint

    config = IndexConfig(
        embedding_origin="https://embedding.example",
        embedding_model="embedding-model",
        chunk_size=500,
        chunk_overlap=50,
        embedding_batch_size=100,
    )
    corpus = _corpus(_entry("a.md", "a"))

    baseline = compute_index_fingerprint(config, corpus)
    changed_hash = compute_index_fingerprint(
        config,
        _corpus(_entry("a.md", "b")),
    )
    changed_config = compute_index_fingerprint(
        config.model_copy(update={"chunk_size": 501}),
        corpus,
    )

    assert baseline != changed_hash
    assert baseline != changed_config


def test_runtime_rejects_production_path_before_component_construction(
    tmp_path: Path,
) -> None:
    """路径相同时不得打开 Chroma 或构造任何远程客户端。

    Args:
        tmp_path: pytest 临时目录。
    """
    from evaluation import runtime

    constructed: list[str] = []

    def build_container(
        settings: Settings,
        options: RuntimeOptions,
    ) -> _FakeContainer:
        del settings, options
        constructed.append("component")
        return _FakeContainer(_FakeVectorStore(), _FakeEmbedder())

    settings = _settings(
        tmp_path,
        chroma_path=tmp_path / "same",
        evaluation_chroma_path=tmp_path / "same" / ".",
    )

    with pytest.raises(ValueError, match="必须与生产路径隔离"):
        runtime.build_evaluation_runtime(
            settings,
            _corpus(_entry("a.md", "a")),
            container_builder=build_container,
        )

    assert constructed == []


def test_runtime_builds_non_sensitive_fingerprinted_collection(
    tmp_path: Path,
) -> None:
    """空评估索引应入库一次，并只暴露安全的规范清单。

    Args:
        tmp_path: pytest 临时目录。
    """
    from evaluation import runtime

    calls: list[RuntimeOptions] = []
    embedder = _FakeEmbedder()
    vector_store = _FakeVectorStore()
    container = _FakeContainer(vector_store, embedder)

    def build_container(
        settings: Settings,
        options: RuntimeOptions,
    ) -> _FakeContainer:
        del settings
        calls.append(options)
        vector_store.metadata = dict(options.collection_metadata or {})
        return container

    settings = _settings(tmp_path, llm_reasoning_effort="none")

    result = runtime.build_evaluation_runtime(
        settings,
        _corpus(_entry("a.md", "a")),
        container_builder=build_container,
    )

    assert result.collection_name == (
        f"rag_eval_{result.index_fingerprint[:12]}"
    )
    assert result.vector_store.count() == result.expected_chunk_count
    assert len(embedder.document_calls) == 1
    assert "generator-secret" not in result.index_manifest_json
    assert "embedding-secret" not in result.index_manifest_json
    assert "/v1/private" not in result.index_manifest_json
    assert calls[0].llm_temperature == 0.0


def test_runtime_reuses_matching_non_empty_index(
    tmp_path: Path,
) -> None:
    """指纹和块数都匹配时不得再次请求文档 embedding。

    Args:
        tmp_path: pytest 临时目录。
    """
    from app.bootstrap.container import ApplicationContainer
    from app.bootstrap.container import build_application_container
    from app.bootstrap.factories import AdapterFactories
    from app.infrastructure.vectorstores.chroma import ChromaVectorStore
    from evaluation import runtime

    settings = _settings(tmp_path)
    corpus = _corpus(_entry("a.md", "a"))
    config = runtime.IndexConfig.from_settings(settings)
    fingerprint = runtime.compute_index_fingerprint(config, corpus)
    stored_chunk = DocumentChunk(
        id="stable:0",
        doc_id="stable",
        text="# a.md\n\n固定正文",
        filename="a.md",
        page=None,
        chunk_index=0,
    )
    store = ChromaVectorStore(
        settings.evaluation_chroma_path,
        f"rag_eval_{fingerprint[:12]}",
        collection_metadata={
            "evaluation_fingerprint": fingerprint,
            "collection_schema_version": "1",
        },
    )
    store.add_chunks([stored_chunk], [[1.0, 0.0]])
    store.update_collection_metadata({"expected_chunk_count": 1})
    embedder = _FakeEmbedder(reject_documents=True)

    def build_container(
        current_settings: Settings,
        options: RuntimeOptions,
    ) -> ApplicationContainer:
        def create_embedding_provider(
            api_key: str | None,
            base_url: str,
            model: str,
            batch_size: int,
        ) -> _FakeEmbedder:
            del api_key, base_url, model, batch_size
            return embedder

        factories = AdapterFactories(
            embedding_provider=create_embedding_provider,
        )
        return build_application_container(
            current_settings,
            options,
            factories,
        )

    result = runtime.build_evaluation_runtime(
        settings,
        corpus,
        container_builder=build_container,
    )
    try:
        assert result.vector_store.count() == 1
        assert result.expected_chunk_count == 1
        assert embedder.document_calls == []
    finally:
        result.close()


def test_runtime_uses_shared_container_factory(tmp_path: Path) -> None:
    """评估运行时应把隔离覆盖传给共享容器工厂。

    Args:
        tmp_path: pytest 临时目录。
    """
    from evaluation.runtime import build_evaluation_runtime

    calls: list[RuntimeOptions] = []
    vector_store = _FakeVectorStore()
    container = _FakeContainer(vector_store, _FakeEmbedder())

    def build(
        settings: Settings,
        options: RuntimeOptions,
    ) -> _FakeContainer:
        del settings
        calls.append(options)
        vector_store.metadata = dict(
            options.collection_metadata or {}
        )
        return container

    settings = _settings(tmp_path)
    result = build_evaluation_runtime(
        settings,
        _corpus(_entry("a.md", "a")),
        rebuild=True,
        container_builder=build,
    )

    assert calls[0].chroma_path == tmp_path / "evaluation"
    assert calls[0].collection_name == result.collection_name
    assert calls[0].collection_metadata == {
        "evaluation_fingerprint": result.index_fingerprint,
        "collection_schema_version": "1",
    }
    assert calls[0].rebuild_collection is True
    assert calls[0].llm_temperature == 0.0
    assert result.container is container
    assert container.accesses == [
        "chunker",
        "embedding_provider",
        "vector_store",
        "keyword_index",
        "baseline_query_service",
    ]
    assert result.query_service is container.baseline_query_service

    result.close()
    result.close()
    assert container.close_calls == 1


def test_runtime_closes_container_when_preparation_fails(
    tmp_path: Path,
) -> None:
    """索引准备失败时应释放已经创建的共享容器。

    Args:
        tmp_path: pytest 临时目录。
    """
    from evaluation.runtime import EvaluationRuntimeError
    from evaluation.runtime import build_evaluation_runtime

    container = _FakeContainer(_FakeVectorStore(), _FakeEmbedder())

    def build(
        settings: Settings,
        options: RuntimeOptions,
    ) -> _FakeContainer:
        del settings, options
        return container

    expected_error = "指纹元数据不匹配"
    with pytest.raises(EvaluationRuntimeError, match=expected_error):
        build_evaluation_runtime(
            _settings(tmp_path),
            _corpus(_entry("a.md", "a")),
            container_builder=build,
        )

    assert container.close_calls == 1

from collections.abc import Iterator
from pathlib import Path
from threading import Event, RLock, Thread
from types import TracebackType

import pytest

from app.application.ports.parsing import ParserResolver
from app.application.ingestion.pipeline import IngestionPipeline
from app.domain.documents import DocumentChunk, DocumentSegment
from app.infrastructure.chunking.recursive import RecursiveDocumentChunker
from app.infrastructure.indexes.bm25 import BM25Index
from app.infrastructure.parsing.factory import ParserFactory
from app.infrastructure.vectorstores.chroma import ChromaVectorStore
from tests.helpers import FakeEmbeddingProvider


class FakeParser:
    """逐个产出测试文本片段的假解析器。"""

    def parse_iter(self, path: Path) -> Iterator[DocumentSegment]:
        """模拟保存文件名与原始文件名不同的上传场景。"""
        yield DocumentSegment(text="a", page=None, filename=path.name)
        yield DocumentSegment(text="b", page=None, filename=path.name)


class FakeParserResolver:
    """记录入库流程请求的文件路径，并返回测试解析器。"""

    def __init__(self) -> None:
        self.resolved_paths: list[Path] = []

    def resolve(self, path: Path) -> FakeParser:
        """记录路径并返回能惰性产出测试片段的解析器。"""
        self.resolved_paths.append(path)
        return FakeParser()


class FakeChunker:
    """为测试稳定产出五个文本块。"""

    def __init__(self) -> None:
        self.segment_filenames: list[str] = []

    def split_iter(self, doc_id: str, segments) -> Iterator[DocumentChunk]:
        """消费片段后产出确定性文本块。"""
        self.segment_filenames = [segment.filename for segment in segments]
        for index in range(5):
            yield DocumentChunk(
                id=f"{doc_id}:{index}",
                doc_id=doc_id,
                text=f"chunk-{index}",
                filename=self.segment_filenames[0],
                page=None,
                chunk_index=index,
            )


class FakeEmbedder:
    """记录每个向量批次并返回固定二维向量。"""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为每条文本生成可预测向量。"""
        self.batches.append(texts)
        return [[float(len(text)), 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """满足 provider 协议，本组测试不调用。"""
        return [float(len(text)), 0.0]


class FakeStore:
    """在内存中记录写入与补偿删除。"""

    def __init__(
        self,
        fail_on_call: int | None = None,
        fail_on_list: bool = False,
        fail_on_delete: bool = False,
    ) -> None:
        self.add_calls: list[list[DocumentChunk]] = []
        self.stored: list[DocumentChunk] = []
        self.deleted_doc_ids: list[str] = []
        self.fail_on_call = fail_on_call
        self.fail_on_list = fail_on_list
        self.fail_on_delete = fail_on_delete

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """记录批次，并可在指定批次模拟数据库异常。"""
        self.add_calls.append(chunks)
        if len(self.add_calls) == self.fail_on_call:
            raise RuntimeError("数据库写入失败")
        assert len(chunks) == len(embeddings)
        self.stored.extend(chunks)

    def delete_document(self, doc_id: str) -> None:
        """删除指定文档并记录补偿动作。"""
        if self.fail_on_delete:
            raise RuntimeError("删除向量失败")
        self.deleted_doc_ids.append(doc_id)
        self.stored = [chunk for chunk in self.stored if chunk.doc_id != doc_id]

    def list_chunks(self) -> list[DocumentChunk]:
        """按写入顺序返回当前保存的文本块。"""
        if self.fail_on_list:
            raise RuntimeError("读取文本块失败")
        return list(self.stored)


class RecordingBM25Index(BM25Index):
    """记录全量重建次数的真实 BM25 索引。"""

    def __init__(self) -> None:
        super().__init__()
        self.rebuild_count = 0

    def rebuild(self, chunks: list[DocumentChunk]) -> None:
        self.rebuild_count += 1
        super().rebuild(chunks)


class FailingBM25Index(BM25Index):
    """可在保留旧状态的同时模拟下一次重建失败。"""

    def __init__(self) -> None:
        super().__init__()
        self.fail_rebuild = False

    def rebuild(self, chunks: list[DocumentChunk]) -> None:
        if self.fail_rebuild:
            raise RuntimeError("关键词索引重建失败")
        super().rebuild(chunks)


def make_pipeline(
    store: FakeStore | None = None,
    keyword_index: BM25Index | None = None,
    mutation_lock=None,
):
    """创建使用全部假依赖的入库流水线。"""
    chunker = FakeChunker()
    embedder = FakeEmbedder()
    vector_store = store or FakeStore()
    pipeline = IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        embedding_batch_size=2,
        parser_resolver=FakeParserResolver(),
        mutation_lock=mutation_lock,
    )
    return pipeline, chunker, embedder, vector_store


class RecordingLock:
    """记录临界区进入次数和最大并发数的上下文锁。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self.enter_count = 0
        self.active = 0
        self.max_active = 0

    def __enter__(self) -> None:
        """获取底层锁并更新并发统计。"""
        self._lock.acquire()
        self.enter_count += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出临界区并释放底层锁。"""
        del exc_type
        del exc_value
        del traceback
        self.active -= 1
        self._lock.release()


def test_ingestion_streams_chunks_in_bounded_batches(tmp_path: Path) -> None:
    """入库不得一次收集整篇文档，应按固定大小向量化并写入。"""
    saved_path = tmp_path / "uuid_notes.txt"
    pipeline, chunker, embedder, store = make_pipeline()

    chunks_count = pipeline.ingest(saved_path, "doc-1", "notes.txt")

    assert chunks_count == 5
    assert [len(batch) for batch in embedder.batches] == [2, 2, 1]
    assert [len(batch) for batch in store.add_calls] == [2, 2, 1]
    assert chunker.segment_filenames == ["notes.txt", "notes.txt"]
    assert all(chunk.filename == "notes.txt" for chunk in store.stored)


def test_ingestion_uses_injected_parser_resolver(tmp_path: Path) -> None:
    """入库流程应通过注入的 ParserResolver 选择解析器。"""
    resolver: ParserResolver = FakeParserResolver()
    pipeline = IngestionPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
        parser_resolver=resolver,
    )
    saved_path = tmp_path / "uuid_notes.txt"

    pipeline.ingest(saved_path, "doc-1", "notes.txt")

    assert resolver.resolved_paths == [saved_path]


def test_ingest_and_delete_each_enter_shared_mutation_lock_once(
    tmp_path: Path,
) -> None:
    """两个公开写操作都应在同一个注入锁内完整执行。"""
    mutation_lock = RecordingLock()
    pipeline, _, _, _ = make_pipeline(mutation_lock=mutation_lock)

    pipeline.ingest(tmp_path / "notes.txt", "doc-1", "notes.txt")
    pipeline.delete_document("doc-1")

    assert mutation_lock.enter_count == 2
    assert mutation_lock.max_active == 1


def test_delete_cannot_snapshot_chunks_while_ingest_holds_lock(
    tmp_path: Path,
) -> None:
    """上传写入未结束时，并发删除不得开始读取 chunk 快照。"""

    class BlockingStore(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.add_started = Event()
            self.release_add = Event()
            self.list_started = Event()

        def add_chunks(
            self,
            chunks: list[DocumentChunk],
            embeddings: list[list[float]],
        ) -> None:
            """首批写入阻塞，直到测试允许上传继续。"""
            self.add_started.set()
            assert self.release_add.wait(timeout=2)
            super().add_chunks(chunks, embeddings)

        def list_chunks(self) -> list[DocumentChunk]:
            """记录任何索引快照读取的开始时刻。"""
            self.list_started.set()
            return super().list_chunks()

    store = BlockingStore()
    pipeline, _, _, _ = make_pipeline(store, BM25Index())
    errors: list[BaseException] = []

    def ingest() -> None:
        try:
            pipeline.ingest(
                tmp_path / "notes.txt",
                "doc-1",
                "notes.txt",
            )
        except BaseException as exc:
            errors.append(exc)

    def delete() -> None:
        try:
            pipeline.delete_document("doc-1")
        except BaseException as exc:
            errors.append(exc)

    ingest_thread = Thread(target=ingest)
    delete_thread = Thread(target=delete)
    ingest_thread.start()
    assert store.add_started.wait(timeout=2)
    delete_thread.start()

    assert not store.list_started.wait(timeout=0.1)
    store.release_add.set()
    ingest_thread.join(timeout=2)
    delete_thread.join(timeout=2)

    assert not ingest_thread.is_alive()
    assert not delete_thread.is_alive()
    assert errors == []


def test_ingestion_compensates_partial_writes_on_failure(tmp_path: Path) -> None:
    """任一批写入失败后必须清理该文档已经入库的文本块。"""
    store = FakeStore(fail_on_call=2)
    pipeline, _, _, _ = make_pipeline(store)

    with pytest.raises(RuntimeError, match="数据库写入失败"):
        pipeline.ingest(tmp_path / "uuid_notes.txt", "doc-1", "notes.txt")

    assert store.deleted_doc_ids == ["doc-1"]
    assert store.stored == []


def test_successful_ingestion_rebuilds_keyword_index(tmp_path: Path) -> None:
    """全部向量写入成功后，新文本块应立即可被关键词检索。"""
    index = BM25Index()
    pipeline, _, _, _ = make_pipeline(keyword_index=index)

    pipeline.ingest(tmp_path / "uuid_notes.txt", "doc-1", "notes.txt")

    assert index.search("chunk 4", 5)[0].chunk_id == "doc-1:4"


def test_failed_ingestion_does_not_rebuild_keyword_index(tmp_path: Path) -> None:
    """向量写入失败并回滚时不得发布中间状态的关键词索引。"""
    store = FakeStore(fail_on_call=2)
    index = RecordingBM25Index()
    pipeline, _, _, _ = make_pipeline(store, index)

    with pytest.raises(RuntimeError, match="数据库写入失败"):
        pipeline.ingest(tmp_path / "uuid_notes.txt", "doc-1", "notes.txt")

    assert index.rebuild_count == 0


def test_keyword_rebuild_failure_rolls_back_vector_writes(tmp_path: Path) -> None:
    """关键词索引无法发布时，上传产生的向量记录也必须回滚。"""
    store = FakeStore()
    index = FailingBM25Index()
    index.rebuild([
        DocumentChunk(
            id="old:0",
            doc_id="old",
            text="legacy_token",
            filename="old.md",
            page=None,
            chunk_index=0,
        )
    ])
    index.fail_rebuild = True
    pipeline, _, _, _ = make_pipeline(store, index)

    with pytest.raises(RuntimeError, match="关键词索引重建失败"):
        pipeline.ingest(tmp_path / "uuid_notes.txt", "doc-1", "notes.txt")

    assert store.stored == []
    assert index.search("legacy_token", 5)[0].chunk_id == "old:0"
    assert index.search("chunk 4", 5) == []


def test_chunk_listing_failure_rolls_back_vector_writes(tmp_path: Path) -> None:
    """重建前无法读取全量文本块时，不得留下本次上传的向量。"""
    store = FakeStore(fail_on_list=True)
    pipeline, _, _, _ = make_pipeline(store, BM25Index())

    with pytest.raises(RuntimeError, match="读取文本块失败"):
        pipeline.ingest(tmp_path / "uuid_notes.txt", "doc-1", "notes.txt")

    assert store.stored == []


def test_delete_document_rebuilds_keyword_index(tmp_path: Path) -> None:
    """删除文档后其文本块应从关键词检索结果中消失。"""
    index = BM25Index()
    pipeline, _, _, store = make_pipeline(keyword_index=index)
    pipeline.ingest(tmp_path / "uuid_notes.txt", "doc-1", "notes.txt")
    assert index.search("chunk 4", 5)

    pipeline.delete_document("doc-1")

    assert store.stored == []
    assert index.search("chunk 4", 5) == []


def test_delete_keeps_vectors_when_keyword_rebuild_fails() -> None:
    """删除前关键词索引无法发布时，不得先删除向量记录。"""
    chunk = DocumentChunk(
        id="old:0",
        doc_id="old",
        text="legacy_token",
        filename="old.md",
        page=None,
        chunk_index=0,
    )
    store = FakeStore()
    store.stored = [chunk]
    index = FailingBM25Index()
    index.rebuild([chunk])
    index.fail_rebuild = True
    pipeline, _, _, _ = make_pipeline(store, index)

    with pytest.raises(RuntimeError, match="关键词索引重建失败"):
        pipeline.delete_document("old")

    assert store.stored == [chunk]
    assert index.search("legacy_token", 5)[0].chunk_id == "old:0"


def test_delete_keeps_vectors_when_chunk_listing_fails() -> None:
    """无法读取删除后的目标语料时，不得改变向量或关键词索引。"""
    chunk = DocumentChunk(
        id="old:0",
        doc_id="old",
        text="legacy_token",
        filename="old.md",
        page=None,
        chunk_index=0,
    )
    store = FakeStore()
    store.stored = [chunk]
    index = BM25Index()
    index.rebuild([chunk])
    store.fail_on_list = True
    pipeline, _, _, _ = make_pipeline(store, index)

    with pytest.raises(RuntimeError, match="读取文本块失败"):
        pipeline.delete_document("old")

    assert store.stored == [chunk]
    assert index.search("legacy_token", 5)[0].chunk_id == "old:0"


def test_delete_restores_keyword_index_when_vector_delete_fails() -> None:
    """向量删除失败时应把关键词索引恢复到删除前状态。"""
    chunk = DocumentChunk(
        id="old:0",
        doc_id="old",
        text="legacy_token",
        filename="old.md",
        page=None,
        chunk_index=0,
    )
    store = FakeStore(fail_on_delete=True)
    store.stored = [chunk]
    index = BM25Index()
    index.rebuild([chunk])
    pipeline, _, _, _ = make_pipeline(store, index)

    with pytest.raises(RuntimeError, match="删除向量失败"):
        pipeline.delete_document("old")

    assert store.stored == [chunk]
    assert index.search("legacy_token", 5)[0].chunk_id == "old:0"


def test_ingestion_returns_zero_for_document_without_chunks(tmp_path: Path) -> None:
    """没有可用文本块时应返回零且不访问向量服务。"""

    class EmptyChunker(FakeChunker):
        def split_iter(self, doc_id: str, segments) -> Iterator[DocumentChunk]:
            list(segments)
            return
            yield

    embedder = FakeEmbedder()
    store = FakeStore()
    index = RecordingBM25Index()
    pipeline = IngestionPipeline(
        chunker=EmptyChunker(),
        embedder=embedder,
        vector_store=store,
        keyword_index=index,
        parser_resolver=FakeParserResolver(),
    )

    assert pipeline.ingest(tmp_path / "blank.txt", "doc-1", "blank.txt") == 0
    assert embedder.batches == []
    assert store.add_calls == []
    assert index.rebuild_count == 0


def test_ingestion_persists_chunks_in_real_chroma(tmp_path: Path) -> None:
    """完整入库后重建 Chroma 客户端仍应读取全部文本块。"""
    document_path = tmp_path / "saved_notes.txt"
    document_path.write_text("第一段内容。第二段内容。", encoding="utf-8")
    database_path = tmp_path / "chroma"
    pipeline = IngestionPipeline(
        chunker=RecursiveDocumentChunker(chunk_size=8, chunk_overlap=2),
        embedder=FakeEmbeddingProvider(),
        vector_store=ChromaVectorStore(database_path, "knowledge_test"),
        embedding_batch_size=2,
        parser_resolver=ParserFactory(),
    )

    chunks_count = pipeline.ingest(document_path, "doc-1", "notes.txt")
    reopened = ChromaVectorStore(database_path, "knowledge_test")

    assert chunks_count > 1
    assert reopened.count() == chunks_count
    assert all(chunk.filename == "notes.txt" for chunk in reopened.list_chunks())


@pytest.mark.parametrize("batch_size", [0, -1])
def test_ingestion_rejects_invalid_batch_size(batch_size: int) -> None:
    """入库批量大小必须是正整数。"""
    with pytest.raises(ValueError, match="embedding_batch_size"):
        IngestionPipeline(
            chunker=FakeChunker(),
            embedder=FakeEmbedder(),
            vector_store=FakeStore(),
            embedding_batch_size=batch_size,
            parser_resolver=FakeParserResolver(),
        )


def test_ingestion_requires_parser_resolver_as_explicit_keyword(
    tmp_path: Path,
) -> None:
    """解析器选择端口应作为显式关键字依赖注入。"""
    embedder = FakeEmbedder()
    pipeline = IngestionPipeline(
        FakeChunker(),
        embedder,
        FakeStore(),
        2,
        parser_resolver=FakeParserResolver(),
    )

    pipeline.ingest(tmp_path / "notes.txt", "doc-1", "notes.txt")

    assert [len(batch) for batch in embedder.batches] == [2, 2, 1]

# -*- coding: utf-8 -*-
"""通过私有临时分阶段暂存，产生真实批次进度的任务流水线。"""

import logging
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import AbstractContextManager
from datetime import UTC
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import TypeVar

from pydantic import BaseModel

from app.application.errors import DocumentParsingError
from app.application.errors import EmptyDocumentError
from app.application.ports.chunking import Chunker
from app.application.ports.embedding import EmbeddingProvider
from app.application.ports.ingestion import DocumentDeletion
from app.application.ports.keyword_index import KeywordIndex
from app.application.ports.parsing import ParserResolver
from app.application.ports.vector_store import VectorStoreWriter
from app.domain.documents import DocumentChunk
from app.domain.documents import DocumentSegment
from app.domain.ingestion_jobs import IngestionJobSubmission
from app.domain.ingestion_jobs import IngestionProgress


logger = logging.getLogger(__name__)
_Model = TypeVar("_Model", bound=BaseModel)


class JobIngestionPipeline:
    """分阶段处理临时载荷；只有调用方可以提交任务成功和可见性。"""

    def __init__(
        self, chunker: Chunker, embedder: EmbeddingProvider,
        vector_store: VectorStoreWriter, keyword_index: KeywordIndex,
        parser_resolver: ParserResolver, deletion: DocumentDeletion, *,
        embedding_batch_size: int = 100,
        mutation_lock: AbstractContextManager[object] | None = None,
    ) -> None:
        """装配与旧入库路径共享的组件，不创建新的索引事实来源。

        Args:
            chunker: 原有流式分块端口。
            embedder: 原有远程或本地向量化端口。
            vector_store: 原有持久化向量端口。
            keyword_index: 原有原子重建的派生索引。
            parser_resolver: 根据源文件选择原有解析器。
            deletion: 同步补偿向量和BM25的既有删除能力。
            embedding_batch_size: 有界的单批文本块数。
            mutation_lock: 与上传和删除共享的本进程变更锁。

        Returns:
            None。
        """
        if type(embedding_batch_size) is not int or embedding_batch_size <= 0:
            raise ValueError("向量批次大小必须为正整数")
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._parser_resolver = parser_resolver
        self._deletion = deletion
        self._batch_size = embedding_batch_size
        self._mutation_lock = mutation_lock or RLock()

    def ingest_job(
        self, saved_path: Path, submission: IngestionJobSubmission,
        on_progress: Callable[[IngestionProgress], None],
        *, check_active: Callable[[], None] | None = None,
    ) -> int:
        """先落临时分块取得总数，再分批向量化；不提交任务成功。

        Args:
            saved_path: 已校验完整性的受控源文件快照。
            submission: 此任务专属的文档与知识库身份。
            on_progress: 同步持久化实际进度；失败必须终止后续模型调用。
            check_active: 解析／分块／模型及写入边界的协作停止检查。

        Returns:
            实际完成向量及关键词索引的正文本块数。
        """
        validated = IngestionJobSubmission.model_validate(
            submission.model_dump(),
        )
        doc_id = str(validated.doc_id)
        check = check_active or (lambda: None)
        try:
            check()
            # 模型等待不占用整个文档写锁；任务未发布身份在SQLite中持续有效。
            with TemporaryDirectory(
                prefix="rag-ingestion-", dir=saved_path.parent,
            ) as directory:
                staging = Path(directory)
                on_progress(IngestionProgress(stage="parsing"))
                segments_path = staging / "segments.jsonl"
                self._parse(
                    saved_path, validated.filename, segments_path, check,
                )
                on_progress(IngestionProgress(stage="chunking"))
                chunks_path = staging / "chunks.jsonl"
                count = self._chunk(
                    validated, segments_path, chunks_path, check,
                )
                if count == 0:
                    raise EmptyDocumentError("文档中没有可提取的文本")
                self._embed(chunks_path, count, on_progress, check)
                on_progress(IngestionProgress(stage="indexing"))
                with self._mutation_lock:
                    check()
                    chunks = self._vector_store.list_chunks()
                    self._keyword_index.rebuild(chunks)
                return count
        except Exception:
            try:
                self._deletion.delete_document(doc_id)
            except Exception:
                # 登记仍未发布，补偿失败也不能把残留内容交给查询；不输出上游异常。
                logger.warning("任务索引补偿未完成", extra={"doc_id": doc_id})
            raise

    def _parse(
        self, path: Path, filename: str, output: Path,
        check: Callable[[], None],
    ) -> None:
        check()
        parser = self._parser_resolver.resolve(path)
        with output.open("w", encoding="utf-8") as destination:
            for segment in parser.parse_iter(path):
                check()
                normalized = segment.model_copy(update={"filename": filename})
                destination.write(normalized.model_dump_json() + "\n")

    def _chunk(
        self, submission: IngestionJobSubmission, segments_path: Path,
        output: Path,
        check: Callable[[], None],
    ) -> int:
        count = 0
        doc_id = str(submission.doc_id)
        updated_at = datetime.now(UTC)
        segments = self._read(segments_path, DocumentSegment)
        check()
        with output.open("w", encoding="utf-8") as destination:
            for chunk in self._chunker.split_iter(doc_id, segments):
                check()
                if chunk.doc_id != doc_id:
                    raise DocumentParsingError("分块返回的文档身份不匹配")
                normalized = chunk.model_copy(update={
                    "filename": submission.filename,
                    "kb_id": submission.kb_id,
                    "doc_updated_at": updated_at,
                })
                destination.write(normalized.model_dump_json() + "\n")
                count += 1
        return count

    def _embed(
        self, path: Path, chunks_count: int,
        on_progress: Callable[[IngestionProgress], None],
        check: Callable[[], None],
    ) -> None:
        total_batches = (
            chunks_count + self._batch_size - 1
        ) // self._batch_size
        completed = 0
        on_progress(IngestionProgress(
            stage="embedding", total_batches=total_batches,
        ))
        batch = []
        for chunk in self._read(path, DocumentChunk):
            batch.append(chunk)
            if len(batch) == self._batch_size:
                self._write_batch(batch, check)
                completed += 1
                on_progress(IngestionProgress(
                    stage="embedding", completed_batches=completed,
                    total_batches=total_batches,
                ))
                batch = []
        if batch:
            self._write_batch(batch, check)
            completed += 1
            on_progress(IngestionProgress(
                stage="embedding", completed_batches=completed,
                total_batches=total_batches,
            ))

    def _write_batch(
        self, batch: list[DocumentChunk], check: Callable[[], None],
    ) -> None:
        check()
        embeddings = self._embedder.embed_documents(
            [chunk.text for chunk in batch],
        )
        with self._mutation_lock:
            check()
            self._vector_store.add_chunks(batch, embeddings)

    @staticmethod
    def _read(path: Path, model: type[_Model]) -> Iterator[_Model]:
        with path.open(encoding="utf-8") as source:
            for line in source:
                yield model.model_validate_json(line)

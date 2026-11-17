# -*- coding: utf-8 -*-
"""用已保存的源文件，按当前分块规则原地重建已发布文档的文本块。

整篇文档先在内存中完成解析、分块和向量化，期间旧文本块照常可检索；
最后在一个事务内整体替换。任何一步失败或文档期间发生变化，旧文本块都保持不变。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.application.errors import (
    DocumentParsingError,
    EmbeddingError,
    IngestionSourceError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from app.application.ports.chunking import Chunker
from app.application.ports.embedding import EmbeddingProvider
from app.application.ports.ingestion_sources import IngestionSources
from app.application.ports.parsing import ParserResolver
from app.application.ports.reindex import DocumentReindexStore
from app.domain.documents import ReindexCandidate, index_text
from app.domain.ingestion_jobs import IngestionJobSubmission


ReindexOutcome = Literal["rebuilt", "skipped", "failed"]


class ReindexResult(BaseModel):
    """一篇文档的重建结果；reason 只含稳定的中文说明，不含路径或上游错误正文。"""

    model_config = ConfigDict(frozen=True)

    kb_id: str
    filename: str
    outcome: ReindexOutcome
    reason: str | None = None
    chunks: int = 0


class ReindexPlan(BaseModel):
    """一次重建将处理的文档，以及因找不到源文件而无法处理的文档。"""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[ReindexCandidate, ...]
    without_source: tuple[tuple[str, str], ...]


class DocumentReindexService:
    """编排源文件快照、解析、分块、向量化和原子替换。"""

    def __init__(
        self, store: DocumentReindexStore, sources: IngestionSources,
        parsers: ParserResolver, chunker: Chunker, embedder: EmbeddingProvider,
        *, embedding_batch_size: int = 100,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size 必须大于 0")
        self._store = store
        self._sources = sources
        self._parsers = parsers
        self._chunker = chunker
        self._embedder = embedder
        self._batch_size = embedding_batch_size
        self._clock = clock

    def plan(
        self, kb_ids: tuple[str, ...] | None = None, *, include_current: bool = False,
        limit: int | None = None,
    ) -> ReindexPlan:
        """列出将要重建的文档，不读取源文件，不调用模型。

        Args:
            kb_ids: 限定知识库；None 表示全部。
            include_current: 为 True 时连已是新格式的文档也重建。
            limit: 本次最多处理的文档数。

        Returns:
            待处理候选和缺少源文件记录的文档。
        """
        candidates = [
            candidate for candidate in self._store.candidates(kb_ids)
            if include_current or candidate.legacy_chunks
        ]
        if limit is not None:
            candidates = candidates[:limit]
        return ReindexPlan(
            candidates=tuple(candidates),
            without_source=tuple(self._store.documents_without_job(kb_ids)),
        )

    def rebuild(self, candidate: ReindexCandidate) -> ReindexResult:
        """重建一篇文档；失败时旧文本块保持不变。"""
        def result(outcome: ReindexOutcome, reason: str | None = None, chunks: int = 0) -> ReindexResult:
            return ReindexResult(
                kb_id=candidate.kb_id, filename=candidate.filename,
                outcome=outcome, reason=reason, chunks=chunks,
            )

        submission = IngestionJobSubmission(
            job_id=candidate.job_id, doc_id=candidate.doc_id, kb_id=candidate.kb_id,
            filename=candidate.filename, size_bytes=candidate.size_bytes,
            uploaded_by=candidate.uploaded_by, content_sha256=candidate.sha256,
        )
        try:
            with self._sources.verified_snapshot(submission) as path:
                parser = self._parsers.resolve(path)
                segments = [
                    segment.model_copy(update={"filename": candidate.filename})
                    for segment in parser.parse_iter(path)
                ]
        except IngestionSourceError:
            return result("skipped", "源文件缺失或与已发布内容不一致")
        except (DocumentParsingError, UnsupportedFileTypeError):
            return result("failed", "源文件解析失败")

        updated_at = self._clock()
        chunks = [
            chunk.model_copy(update={
                "filename": candidate.filename, "kb_id": candidate.kb_id,
                "doc_updated_at": updated_at,
            })
            for chunk in self._chunker.split_iter(str(candidate.doc_id), segments)
        ]
        if not chunks:
            return result("failed", "源文件没有可检索的正文")
        try:
            embeddings: list[list[float]] = []
            for start in range(0, len(chunks), self._batch_size):
                batch = chunks[start:start + self._batch_size]
                embeddings.extend(self._embedder.embed_documents([index_text(c) for c in batch]))
        except EmbeddingError:
            return result("failed", "向量化失败")
        try:
            swapped = self._store.swap_chunks(candidate, chunks, embeddings)
        except VectorStoreError:
            return result("failed", "写入文本块失败")
        if not swapped:
            return result("skipped", "重建期间文档被删除、替换或修改")
        return result("rebuilt", chunks=len(chunks))

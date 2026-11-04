# -*- coding: utf-8 -*-
"""解析、分块、向量化和持久化的流式入库流水线。"""

from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from datetime import UTC
from datetime import datetime
from threading import RLock

from app.application.ports.chunking import Chunker
from app.application.ports.embedding import EmbeddingProvider
from app.application.ports.keyword_index import KeywordIndex
from app.application.ports.parsing import ParserResolver
from app.application.ports.vector_store import VectorStoreWriter
from app.domain.documents import DocumentChunk, DocumentSegment


class IngestionPipeline:
    """以有界批次消费解析结果，并在失败时补偿已写数据。"""

    def __init__(
        self,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStoreWriter,
        embedding_batch_size: int = 100,
        *,
        parser_resolver: ParserResolver,
        keyword_index: KeywordIndex | None = None,
        mutation_lock: AbstractContextManager[object] | None = None,
    ) -> None:
        """装配入库依赖、共享写锁并校验批量大小。

        Args:
            chunker: 把解析片段切分为文本块的组件。
            embedder: 文档文本向量化组件。
            vector_store: 文本块和向量持久化组件。
            embedding_batch_size: 单次向量化的文本块上限。
            parser_resolver: 根据文件路径选择解析器的适配器。
            keyword_index: 可选 BM25 派生索引。
            mutation_lock: 上传和删除共享的可重入上下文锁。

        Raises:
            ValueError: 批量大小无效。
        """
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size 必须大于 0")
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._embedding_batch_size = embedding_batch_size
        self._parser_resolver = parser_resolver
        self._mutation_lock = mutation_lock or RLock()

    def validate_file_type(self, filename: str) -> None:
        """在源文件落盘前确认其类型有可用解析器。

        Args:
            filename: 已清理且不含客户端目录的文件名。

        Returns:
            None。

        Raises:
            UnsupportedFileTypeError: 文件扩展名没有对应解析器。
        """
        self._parser_resolver.resolve(filename)

    def ingest(
        self,
        saved_path: Path,
        doc_id: str,
        original_filename: str,
    ) -> int:
        """在共享写锁内流式处理一份已保存文档。

        Args:
            saved_path: 上传服务已经持久化的源文件路径。
            doc_id: 新文档的稳定唯一标识。
            original_filename: 进入向量元数据的安全原始文件名。

        Returns:
            成功写入的文本块数量。
        """
        with self._mutation_lock:
            return self._ingest_unlocked(
                saved_path,
                doc_id,
                original_filename,
            )

    def _ingest_unlocked(
        self,
        saved_path: Path,
        doc_id: str,
        original_filename: str,
    ) -> int:
        parser = self._parser_resolver.resolve(saved_path)
        segments = self._with_original_filename(
            parser.parse_iter(saved_path),
            original_filename,
        )
        chunks = self._chunker.split_iter(doc_id, segments)
        updated_at = datetime.now(UTC)
        chunks_count = 0

        try:
            for batch in self._batched(chunks):
                # 同一次文档版本的所有块使用同一时间，不从可变文件mtime推断。
                batch = [chunk.model_copy(update={"doc_updated_at": updated_at}) for chunk in batch]
                embeddings = self._embedder.embed_documents(
                    [chunk.text for chunk in batch]
                )
                self._vector_store.add_chunks(batch, embeddings)
                chunks_count += len(batch)

            if chunks_count == 0:
                return 0
            self._rebuild_keyword_index()
        except Exception:
            self._vector_store.delete_document(doc_id)
            raise

        return chunks_count

    def delete_document(self, doc_id: str) -> None:
        """在共享写锁内删除文档并同步重建关键词索引。

        Args:
            doc_id: 要从向量和关键词索引删除的文档标识。

        Returns:
            None。
        """
        with self._mutation_lock:
            self._delete_document_unlocked(doc_id)

    def _delete_document_unlocked(self, doc_id: str) -> None:
        if self._keyword_index is None:
            self._vector_store.delete_document(doc_id)
            return

        previous_chunks = self._vector_store.list_chunks()
        remaining_chunks = []
        for chunk in previous_chunks:
            if chunk.doc_id != doc_id:
                remaining_chunks.append(chunk)
        # 先发布过滤后的派生索引；失败时持久化向量仍保持原状。
        self._keyword_index.rebuild(remaining_chunks)
        try:
            self._vector_store.delete_document(doc_id)
        except Exception:
            # 向量删除未提交时，恢复与持久化数据一致的上一代索引。
            self._keyword_index.rebuild(previous_chunks)
            raise

    def _rebuild_keyword_index(self) -> None:
        if self._keyword_index is not None:
            self._keyword_index.rebuild(self._vector_store.list_chunks())

    @staticmethod
    def _with_original_filename(
        segments: Iterator[DocumentSegment],
        original_filename: str,
    ) -> Iterator[DocumentSegment]:
        for segment in segments:
            yield segment.model_copy(update={"filename": original_filename})

    def _batched(
        self,
        chunks: Iterator[DocumentChunk],
    ) -> Iterator[list[DocumentChunk]]:
        batch: list[DocumentChunk] = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) == self._embedding_batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

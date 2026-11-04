# -*- coding: utf-8 -*-
"""ChromaDB 持久化向量存储适配器。"""

from collections.abc import Mapping
from pathlib import Path
import re
from uuid import UUID

import chromadb

from app.application.errors import VectorStoreError
from app.domain.documents import DocumentChunk
from app.domain.knowledge_bases import DEFAULT_KB_ID
from app.domain.knowledge_bases import KB_ID_PATTERN
from app.domain.knowledge_bases import validate_kb_ids
from app.domain.retrieval import VectorSearchResult
from app.domain.knowledge_views import IndexedChunkHeader
from app.domain.knowledge_views import IndexedDocumentHeader
from app.infrastructure.vectorstores.knowledge_reader import parse_updated_at
from app.infrastructure.vectorstores.knowledge_reader import read_document_headers
from app.infrastructure.vectorstores.knowledge_reader import read_chunk_headers
from app.infrastructure.vectorstores.knowledge_reader import read_selected_chunks


class ChromaVectorStore:
    """使用本地 Chroma PersistentClient 保存和检索文本块。

    Attributes:
        _collection_name: 构造时绑定且只用于精确删除的 collection
            名称。
    """

    def __init__(
        self,
        path: Path | str,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None = None,
    ) -> None:
        """打开或创建指定路径下的 collection。

        Args:
            path: Chroma 持久化目录。
            collection_name: 要打开或创建的精确 collection 名称。
            collection_metadata: 创建时一并保存的可选元数据。

        Returns:
            None。

        Raises:
            VectorStoreError: 调用方试图覆盖固定 cosine 距离配置。
        """
        caller_metadata = dict(collection_metadata or {})
        configured_space = caller_metadata.get("hnsw:space")
        if configured_space is not None and configured_space != "cosine":
            raise VectorStoreError("hnsw:space 必须保持为 cosine")
        caller_metadata["hnsw:space"] = "cosine"

        self._collection_name = collection_name
        try:
            database_path = Path(path)
            database_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(database_path)
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata=caller_metadata,
                embedding_function=None,
            )
            self._migrate_legacy_scopes()
        except Exception as exc:
            raise VectorStoreError("向量存储初始化失败") from exc

    def close(self) -> None:
        """释放本适配器拥有的客户端引用，不删除持久化内容。

        Returns:
            None。

        Raises:
            VectorStoreError: 底层客户端无法关闭。
        """
        # SDK按路径引用计数，不能直接停止共享System而破坏其他拥有者。
        try:
            self._client.close()
        except Exception as exc:
            raise VectorStoreError("向量存储关闭失败") from exc

    def collection_metadata(self) -> dict[str, object]:
        """返回当前 collection 元数据的防御性副本。

        Returns:
            Chroma 保存的 collection 元数据。
        """
        try:
            metadata = dict(self._collection.metadata or {})
            configuration = self._collection.configuration_json
            hnsw_configuration = configuration.get("hnsw")
            if isinstance(hnsw_configuration, dict):
                space = hnsw_configuration.get("space")
                if isinstance(space, str):
                    metadata["hnsw:space"] = space
            return metadata
        except Exception as exc:
            raise VectorStoreError("向量存储元数据读取失败") from exc

    def update_collection_metadata(
        self,
        metadata: dict[str, str | int],
    ) -> None:
        """合并并持久化 collection 元数据。

        Args:
            metadata: 要覆盖或追加的字符串、整数元数据。

        Returns:
            None。

        Raises:
            VectorStoreError: 调用方试图覆盖固定 cosine 距离配置。
        """
        configured_space = metadata.get("hnsw:space")
        if configured_space is not None and configured_space != "cosine":
            raise VectorStoreError("hnsw:space 必须保持为 cosine")
        merged_metadata = self.collection_metadata()
        merged_metadata.update(metadata)
        # Chroma 把距离函数保存在只读 configuration 中；即使值未
        # 变化，modify() 也会拒绝这个保留键。
        merged_metadata.pop("hnsw:space", None)
        try:
            self._collection.modify(metadata=merged_metadata)
        except Exception as exc:
            raise VectorStoreError("向量存储元数据更新失败") from exc

    def delete_collection(self) -> None:
        """删除构造时绑定的精确 collection。

        Returns:
            None。
        """
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception as exc:
            raise VectorStoreError("向量存储集合删除失败") from exc

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """把一批等量文本块与向量写入 collection。

        Args:
            chunks: 待持久化的领域文本块。
            embeddings: 与 ``chunks`` 一一对应的向量。

        Returns:
            None。

        Raises:
            VectorStoreError: 文本块和向量数量不一致。
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError("文本块数量与向量数量不一致")
        if not chunks:
            return

        try:
            self._collection.add(
                ids=[chunk.id for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    self._encode_metadata(chunk)
                    for chunk in chunks
                ],
            )
        except Exception as exc:
            raise VectorStoreError("向量存储写入失败") from exc

    def query_by_embedding(
        self,
        vector: list[float],
        top_k: int,
        *,
        kb_ids: tuple[str, ...] | None = None,
        excluded_doc_ids: tuple[str, ...] = (),
    ) -> list[VectorSearchResult]:
        """按查询向量返回距离最近的文本块。

        Args:
            vector: 用于相似度检索的查询向量。
            top_k: 最多返回的命中数量。
            kb_ids: ANN召回之前的范围；空tuple为空结果。
            excluded_doc_ids: ANN召回前排除的未发布文档。

        Returns:
            Chroma 距离升序返回的领域命中结果。

        Raises:
            ValueError: ``top_k`` 不是正整数。
            VectorStoreError: 持久化记录的元数据无法还原为文本块。
        """
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        validate_kb_ids(kb_ids)
        if kb_ids == ():
            return []
        record_count = self.count()
        if record_count == 0:
            return []

        try:
            scope_options = {}
            filters = []
            if kb_ids is not None:
                filters.append({"kb_id": {"$in": list(kb_ids)}})
            if excluded_doc_ids:
                filters.append({"doc_id": {"$nin": list(excluded_doc_ids)}})
            if len(filters) == 1:
                scope_options["where"] = filters[0]
            elif filters:
                scope_options["where"] = {"$and": filters}
            result = self._collection.query(
                query_embeddings=[vector],
                n_results=min(top_k, record_count),
                include=["documents", "metadatas", "distances"],
                **scope_options,
            )
            ids = result["ids"][0]
            documents = (result["documents"] or [[]])[0]
            metadatas = (result["metadatas"] or [[]])[0]
            distances = (result["distances"] or [[]])[0]
            search_results = []
            records = zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
            for chunk_id, document, metadata, distance in records:
                search_results.append(
                    VectorSearchResult(
                        chunk=self._decode_chunk(
                            chunk_id,
                            document,
                            metadata,
                        ),
                        distance=float(distance),
                    )
                )
            return search_results
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("向量存储查询失败") from exc

    def list_chunks(self) -> list[DocumentChunk]:
        """按 chunk_index 稳定列出 collection 中的全部文本块。

        Returns:
            先按文档 ID、再按文本块索引排序的全部文本块。

        Raises:
            VectorStoreError: 持久化记录的元数据无法还原为文本块。
        """
        try:
            result = self._collection.get(
                include=["documents", "metadatas"]
            )
            documents = result["documents"] or []
            metadatas = result["metadatas"] or []
            chunks = []
            records = zip(
                result["ids"],
                documents,
                metadatas,
                strict=True,
            )
            for chunk_id, document, metadata in records:
                chunks.append(
                    self._decode_chunk(chunk_id, document, metadata)
                )
            return sorted(
                chunks,
                key=lambda chunk: (chunk.doc_id, chunk.chunk_index),
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("向量存储列表读取失败") from exc

    def list_document_headers(self, kb_id: str | None = None) -> tuple[IndexedDocumentHeader, ...]:
        """返回指定库的只读文档元数据。

        Args:
            kb_id: 召回前限定的知识库，None为管理元数据全量视图。

        Returns:
            稳定有序的文档头。
        """
        return read_document_headers(self._collection, kb_id)

    def list_chunk_headers(self, doc_id: UUID) -> tuple[IndexedChunkHeader, ...]:
        """仅读取指定文档元数据以选择有限原文窗口。

        Args:
            doc_id: 来源文档UUID。

        Returns:
            按实际顺序排列的块元数据。
        """
        return read_chunk_headers(self._collection, doc_id)

    def read_chunks(self, headers: tuple[IndexedChunkHeader, ...]) -> tuple[DocumentChunk, ...]:
        """仅读取已选择且经过归属核对的原文。

        Args:
            headers: 最多七块的可信元数据窗口。

        Returns:
            与窗口顺序一致的正文。
        """
        return read_selected_chunks(self._collection, headers)

    def count_documents_by_kb(
        self, *, excluded_doc_ids: tuple[str, ...] = (),
    ) -> dict[str, int]:
        """分页读取元数据后按UUID去重，不加载正文／向量或构建检索索引。

        Args:
            excluded_doc_ids: 计数前排除的未发布文档。

        Returns:
            每个有文档知识库的精确文档数。
        """
        identities = {}
        counts = {}
        offset = 0
        page_size = 1000
        filter_options = {}
        if excluded_doc_ids:
            filter_options["where"] = {
                "doc_id": {"$nin": list(excluded_doc_ids)},
            }
        try:
            while True:
                page = self._collection.get(
                    include=["metadatas"], limit=page_size, offset=offset,
                    **filter_options,
                )
                ids = page["ids"]
                metadatas = page["metadatas"] or []
                if len(ids) != len(metadatas):
                    raise ValueError("元数据数量不一致")
                for metadata in metadatas:
                    doc_id = metadata["doc_id"]
                    filename = metadata["filename"]
                    kb_id = metadata.get("kb_id", DEFAULT_KB_ID)
                    if not isinstance(doc_id, str) or not isinstance(filename, str):
                        raise ValueError("文档身份元数据无效")
                    canonical_id = str(UUID(doc_id))
                    if not isinstance(kb_id, str) or re.fullmatch(KB_ID_PATTERN, kb_id) is None:
                        raise ValueError("知识库归属元数据无效")
                    identity = (kb_id, filename)
                    existing = identities.get(canonical_id)
                    if existing is not None:
                        if existing != identity:
                            raise ValueError("同一文档的知识库或来源文件名不一致")
                        continue
                    identities[canonical_id] = identity
                    counts[kb_id] = counts.get(kb_id, 0) + 1
                if len(ids) < page_size:
                    break
                offset += len(ids)
        except Exception as error:
            raise VectorStoreError("文档元数据统计失败") from error
        return counts

    def delete_document(self, doc_id: str) -> None:
        """删除指定文档的所有文本块。

        Args:
            doc_id: 要删除的稳定文档标识。

        Returns:
            None。
        """
        try:
            self._collection.delete(where={"doc_id": doc_id})
        except Exception as exc:
            raise VectorStoreError("向量存储文档删除失败") from exc

    def count(self) -> int:
        """返回 collection 当前记录数。

        Returns:
            当前 collection 持久化的文本块数量。
        """
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError("向量存储计数失败") from exc

    @staticmethod
    def _encode_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
        """把可空页码转换为 Chroma 稳定支持的标量类型。"""
        metadata = {
            "doc_id": chunk.doc_id,
            "filename": chunk.filename,
            "page": chunk.page or 0,
            "chunk_index": chunk.chunk_index,
            "kb_id": chunk.kb_id,
        }
        if chunk.doc_updated_at is not None:
            metadata["doc_updated_at"] = chunk.doc_updated_at.isoformat()
        return metadata

    @staticmethod
    def _decode_chunk(
        chunk_id: str,
        document: str,
        metadata: dict[str, object],
    ) -> DocumentChunk:
        """把 Chroma 记录还原为领域文本块。"""
        try:
            page = int(metadata["page"])
            return DocumentChunk(
                id=chunk_id,
                doc_id=str(metadata["doc_id"]),
                text=document,
                filename=str(metadata["filename"]),
                page=page or None,
                chunk_index=int(metadata["chunk_index"]),
                kb_id=metadata.get("kb_id", DEFAULT_KB_ID),
                doc_updated_at=parse_updated_at(metadata.get("doc_updated_at")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorStoreError(
                f"向量记录元数据损坏：{chunk_id}"
            ) from exc

    def _migrate_legacy_scopes(self) -> None:
        # 只在打开适配器时分页补元数据，不查询期重建，更不重新计算向量。
        offset = 0
        page_size = 256
        while True:
            page = self._collection.get(
                include=["metadatas"], limit=page_size, offset=offset,
            )
            ids = page["ids"]
            metadatas = page["metadatas"] or []
            missing_ids = []
            updated_metadata = []
            for chunk_id, metadata in zip(ids, metadatas, strict=True):
                if metadata is None:
                    raise VectorStoreError("向量记录缺少来源元数据")
                if "kb_id" not in metadata:
                    missing_ids.append(chunk_id)
                    updated_metadata.append({**metadata, "kb_id": DEFAULT_KB_ID})
            if missing_ids:
                self._collection.update(
                    ids=missing_ids, metadatas=updated_metadata,
                )
            if len(ids) < page_size:
                return
            offset += len(ids)

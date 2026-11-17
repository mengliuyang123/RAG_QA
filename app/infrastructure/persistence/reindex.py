# -*- coding: utf-8 -*-
"""按新分块规则原地重建已发布文档：列出候选，并在单个事务内整体替换文本块。"""

from uuid import UUID

import psycopg

from app.application.errors import VectorStoreError
from app.domain.documents import DocumentChunk, ReindexCandidate
from app.domain.knowledge_bases import validate_kb_ids
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.documents import _BUSINESS_LOCK
from app.infrastructure.vectorstores.postgres import PostgresVectorStore


_LIVE = (
    "d.status='ready' AND d.deleted_at IS NULL AND NOT EXISTS ("
    "SELECT 1 FROM documents p WHERE p.replaces_doc_id=d.doc_id "
    "AND p.status='pending' AND p.deleted_at IS NULL)"
)


class PostgresDocumentReindexStore:
    """与入库共用 documents、ingestion_jobs 和 vector_chunks；写入持有同一业务写锁。"""

    def __init__(self, database: Database, vector_store: PostgresVectorStore) -> None:
        self._database = database
        self._vector_store = vector_store

    def candidates(self, kb_ids: tuple[str, ...] | None = None) -> list[ReindexCandidate]:
        validate_kb_ids(kb_ids)
        if kb_ids == ():
            return []
        query = (
            "SELECT d.doc_id,d.kb_id,d.filename,d.sha256,j.job_id,j.size_bytes,j.uploaded_by,"
            " EXISTS (SELECT 1 FROM vector_chunks c WHERE c.collection=%s AND c.doc_id=d.doc_id"
            "  AND c.metadata->>'parent_index' IS NULL) AS legacy_chunks "
            "FROM documents d JOIN ingestion_jobs j ON j.doc_id=d.doc_id "
            " AND j.status='succeeded' AND j.content_sha256=d.sha256 "
            f"WHERE {_LIVE}"
        )
        values: list[object] = [self._vector_store.collection_name]
        if kb_ids is not None:
            query += " AND d.kb_id=ANY(%s)"
            values.append(list(kb_ids))
        query += " ORDER BY d.kb_id,d.filename,d.doc_id"
        with self._database.connection(read_only=True) as connection:
            return [ReindexCandidate(
                doc_id=UUID(row["doc_id"]), kb_id=row["kb_id"], filename=row["filename"],
                sha256=row["sha256"], size_bytes=row["size_bytes"],
                uploaded_by=UUID(row["uploaded_by"]), job_id=row["job_id"],
                legacy_chunks=bool(row["legacy_chunks"]),
            ) for row in connection.execute(query, values)]

    def documents_without_job(self, kb_ids: tuple[str, ...] | None = None) -> list[tuple[str, str]]:
        validate_kb_ids(kb_ids)
        if kb_ids == ():
            return []
        query = (
            "SELECT d.kb_id,d.filename FROM documents d "
            f"WHERE {_LIVE} AND NOT EXISTS (SELECT 1 FROM ingestion_jobs j WHERE j.doc_id=d.doc_id "
            " AND j.status='succeeded' AND j.content_sha256=d.sha256)"
        )
        values: list[object] = []
        if kb_ids is not None:
            query += " AND d.kb_id=ANY(%s)"
            values.append(list(kb_ids))
        query += " ORDER BY d.kb_id,d.filename"
        with self._database.connection(read_only=True) as connection:
            return [(row["kb_id"], row["filename"]) for row in connection.execute(query, values)]

    def swap_chunks(self, candidate: ReindexCandidate, chunks: list[DocumentChunk],
                    embeddings: list[list[float]]) -> bool:
        if not chunks or len(chunks) != len(embeddings):
            raise VectorStoreError("重建的文本块与向量数量无效")
        doc_id = str(candidate.doc_id)
        if any(chunk.doc_id != doc_id or chunk.kb_id != candidate.kb_id for chunk in chunks):
            raise VectorStoreError("重建的文本块不属于该文档")
        try:
            with self._database.connection() as connection:
                # 与上传、删除、发布同一把业务写锁：期间没有其他写入能改变这篇文档的状态。
                connection.execute(_BUSINESS_LOCK)
                live = connection.execute(
                    f"SELECT d.doc_id FROM documents d WHERE d.doc_id=%s AND d.kb_id=%s AND d.sha256=%s AND {_LIVE}",
                    (doc_id, candidate.kb_id, candidate.sha256),
                ).fetchone()
                if live is None:
                    return False
                connection.execute(
                    "DELETE FROM vector_chunks WHERE collection=%s AND doc_id=%s",
                    (self._vector_store.collection_name, doc_id),
                )
                self._vector_store.insert_chunks(connection, chunks, embeddings)
                return True
        except psycopg.Error as error:
            raise VectorStoreError("文档文本块替换失败") from error

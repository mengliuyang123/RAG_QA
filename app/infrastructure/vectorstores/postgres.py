"""PostgreSQL 文本块、元数据和完整精度 pgvector 检索。"""
from collections.abc import Mapping
from contextlib import contextmanager
import math
from uuid import UUID

import psycopg
from pgvector import Vector
from psycopg.types.json import Jsonb

from app.application.errors import VectorStoreError
from app.domain.documents import DocumentChunk, index_text
from app.domain.knowledge_bases import validate_kb_ids
from app.domain.knowledge_views import IndexedChunkHeader, IndexedDocumentHeader
from app.domain.retrieval import VectorSearchResult
from app.infrastructure.persistence.database import Database
from app.infrastructure.indexes.tokenization import search_text
from app.infrastructure.vectorstores.knowledge_reader import _decode_header, _validate_group


class PostgresVectorStore:
    @property
    def database(self) -> Database:
        return self._database

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def __init__(self, database: Database, collection_name: str,
                 collection_metadata: Mapping[str, str | int] | None = None) -> None:
        self._database = database
        self._collection_name = collection_name
        metadata = dict(collection_metadata or {})
        if metadata.get('hnsw:space', 'cosine') != 'cosine':
            raise VectorStoreError('hnsw:space 必须保持为 cosine')
        metadata['hnsw:space'] = 'cosine'
        with self._connection() as connection:
            connection.execute('INSERT INTO vector_collections(name,metadata) VALUES (%s,%s) ON CONFLICT DO NOTHING',
                               (collection_name, Jsonb(metadata)))
            existing = connection.execute('SELECT metadata FROM vector_collections WHERE name=%s', (collection_name,)).fetchone()['metadata']
            if metadata.get('embedding_identity') is not None and existing.get('embedding_identity') != metadata['embedding_identity']:
                raise VectorStoreError('向量集合的 Embedding 模型身份不匹配，请重新入库')

    @contextmanager
    def _connection(self):
        try:
            with self._database.connection() as connection:
                yield connection
        except (psycopg.Error, ValueError, TypeError, KeyError) as exc:
            raise VectorStoreError('PostgreSQL 向量存储操作失败') from exc

    def close(self) -> None:
        # 连接池由应用容器拥有。
        pass

    def collection_metadata(self) -> dict[str, object]:
        with self._connection() as connection:
            row = connection.execute('SELECT metadata FROM vector_collections WHERE name=%s', (self._collection_name,)).fetchone()
            if row is None:
                raise VectorStoreError('向量集合不存在')
            return dict(row['metadata'])

    def stored_dimensions(self) -> int | None:
        """返回集合首次写入时锁定的向量维度；还没有向量时为 None。"""
        with self._connection() as connection:
            row = connection.execute('SELECT dimensions FROM vector_collections WHERE name=%s', (self._collection_name,)).fetchone()
            return None if row is None else row['dimensions']

    def update_collection_metadata(self, metadata: dict[str, str | int]) -> None:
        if metadata.get('hnsw:space', 'cosine') != 'cosine':
            raise VectorStoreError('hnsw:space 必须保持为 cosine')
        with self._connection() as connection:
            connection.execute('UPDATE vector_collections SET metadata=metadata || %s WHERE name=%s',
                               (Jsonb(metadata), self._collection_name))

    def delete_collection(self) -> None:
        with self._connection() as connection:
            connection.execute('DELETE FROM vector_collections WHERE name=%s', (self._collection_name,))

    @staticmethod
    def _vector(values: list[float]) -> str:
        if not values or any(not math.isfinite(v) for v in values) or not any(values):
            raise VectorStoreError('向量必须为非零的有限数值序列')
        return Vector(values).to_text()

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError('文本块数量与向量数量不一致')
        if not chunks:
            return
        with self._connection() as connection:
            self.insert_chunks(connection, chunks, embeddings)

    def insert_chunks(self, connection: psycopg.Connection, chunks: list[DocumentChunk],
                      embeddings: list[list[float]]) -> None:
        """在调用方的事务内写入一批文本块，校验并锁定集合维度。"""
        if len(chunks) != len(embeddings):
            raise VectorStoreError('文本块数量与向量数量不一致')
        if not chunks:
            return
        dimensions = len(embeddings[0])
        if any(len(v) != dimensions for v in embeddings):
            raise VectorStoreError('同一批次向量维度不一致')
        values = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            metadata = chunk.model_dump(mode='json', exclude={'id', 'text'})
            metadata['page'] = chunk.page or 0
            values.append((self._collection_name, chunk.id, chunk.doc_id, chunk.kb_id,
                           Jsonb(metadata), chunk.text, self._vector(vector), search_text(index_text(chunk))))
        row = connection.execute('SELECT dimensions FROM vector_collections WHERE name=%s FOR UPDATE',
                                 (self._collection_name,)).fetchone()
        if row is None or row['dimensions'] not in (None, dimensions):
            raise VectorStoreError('向量集合维度不匹配')
        connection.execute('UPDATE vector_collections SET dimensions=%s WHERE name=%s',
                           (dimensions, self._collection_name))
        connection.cursor().executemany('INSERT INTO vector_chunks(collection,id,doc_id,kb_id,metadata,text,embedding,search_text) '
                                       'VALUES (%s,%s,%s,%s,%s,%s,%s::vector,%s)', values)

    @staticmethod
    def _chunk(row) -> DocumentChunk:
        try:
            metadata = dict(row['metadata'])
            metadata['page'] = metadata['page'] or None
            return DocumentChunk(id=row['id'], text=row['text'], **metadata)
        except (KeyError, ValueError, TypeError) as exc:
            raise VectorStoreError(f"向量记录元数据损坏：{row['id']}") from exc

    def query_by_embedding(self, vector: list[float], top_k: int, *,
                           kb_ids: tuple[str, ...] | None = None,
                           excluded_doc_ids: tuple[str, ...] = ()) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError('top_k 必须大于 0')
        validate_kb_ids(kb_ids)
        if kb_ids == ():
            return []
        encoded = self._vector(vector)
        with self._connection() as connection:
            row = connection.execute('SELECT dimensions FROM vector_collections WHERE name=%s', (self._collection_name,)).fetchone()
            if row is None or row['dimensions'] is None:
                return []
            if len(vector) != row['dimensions']:
                raise VectorStoreError('查询向量维度不匹配')
            query = 'SELECT id,text,metadata,embedding <=> %s::vector AS distance FROM vector_chunks WHERE collection=%s'
            values: list[object] = [encoded, self._collection_name]
            if kb_ids is not None:
                query += ' AND kb_id=ANY(%s)'
                values.append(list(kb_ids))
            if excluded_doc_ids:
                query += ' AND NOT (doc_id=ANY(%s))'
                values.append(list(excluded_doc_ids))
            query += ' ORDER BY distance,id LIMIT %s'
            rows = connection.execute(query, [*values, top_k]).fetchall()
            return [VectorSearchResult(chunk=self._chunk(r), distance=float(r['distance'])) for r in rows]

    def read_parent_chunks(self, parents: tuple[tuple[str, int], ...], *,
                           kb_ids: tuple[str, ...] | None = None,
                           excluded_doc_ids: tuple[str, ...] = ()) -> list[DocumentChunk]:
        """按 (doc_id, parent_index) 批量读取兄弟子块，沿用召回的范围与排除条件。"""
        validate_kb_ids(kb_ids)
        if not parents or kb_ids == ():
            return []
        parents = tuple(dict.fromkeys(parents))
        doc_ids = [doc_id for doc_id, _ in parents]
        indexes = [int(index) for _, index in parents]
        # 先用 doc_id 命中 vector_chunks_doc_id 索引，再按 metadata 中的父块序号过滤。
        query = ("SELECT c.id,c.text,c.metadata FROM vector_chunks c "
                 "JOIN unnest(%s::text[], %s::int[]) AS p(doc_id, parent_index) "
                 "ON c.doc_id=p.doc_id AND (c.metadata->>'parent_index')::int=p.parent_index "
                 "WHERE c.collection=%s AND c.doc_id=ANY(%s)")
        values: list[object] = [doc_ids, indexes, self._collection_name, doc_ids]
        if kb_ids is not None:
            query += ' AND c.kb_id=ANY(%s)'
            values.append(list(kb_ids))
        if excluded_doc_ids:
            query += ' AND NOT (c.doc_id=ANY(%s))'
            values.append(list(excluded_doc_ids))
        query += " ORDER BY c.doc_id,(c.metadata->>'chunk_index')::int,c.id"
        with self._connection() as connection:
            return [self._chunk(r) for r in connection.execute(query, values)]

    def list_chunks(self) -> list[DocumentChunk]:
        with self._connection() as connection:
            return [self._chunk(r) for r in connection.execute(
                "SELECT id,text,metadata FROM vector_chunks WHERE collection=%s ORDER BY doc_id,(metadata->>'chunk_index')::int,id",
                (self._collection_name,))]

    def _headers(self, *, doc_id: str | None = None, kb_id: str | None = None):
        query = 'SELECT id,metadata FROM vector_chunks WHERE collection=%s'
        values = [self._collection_name]
        if doc_id is not None:
            query += ' AND doc_id=%s'
            values.append(doc_id)
        if kb_id is not None:
            query += ' AND kb_id=%s'
            values.append(kb_id)
        with self._connection() as connection:
            return [_decode_header(r['id'], r['metadata']) for r in connection.execute(query, values)]

    def list_chunk_headers(self, doc_id: UUID) -> tuple[IndexedChunkHeader, ...]:
        return _validate_group(self._headers(doc_id=str(doc_id)))

    def list_document_headers(self, kb_id: str | None = None) -> tuple[IndexedDocumentHeader, ...]:
        groups = {}
        for header in self._headers(kb_id=kb_id):
            groups.setdefault(header.doc_id, []).append(header)
        result = []
        for identifier in sorted(groups, key=str):
            headers = _validate_group(groups[identifier])
            first = headers[0]
            result.append(IndexedDocumentHeader(doc_id=identifier, kb_id=first.kb_id, filename=first.filename,
                                                chunks_count=len(headers), updated_at=first.doc_updated_at))
        return tuple(result)

    def read_chunks(self, headers: tuple[IndexedChunkHeader, ...]) -> tuple[DocumentChunk, ...]:
        if not 1 <= len(headers) <= 7 or len({h.chunk_id for h in headers}) != len(headers):
            raise VectorStoreError('原文窗口大小或身份无效')
        _validate_group(list(headers))
        with self._connection() as connection:
            rows = connection.execute('SELECT id,text,metadata FROM vector_chunks WHERE collection=%s AND id=ANY(%s)',
                                      (self._collection_name, [h.chunk_id for h in headers])).fetchall()
            found = {r['id']: r for r in rows}
            if len(found) != len(headers) or any(_decode_header(h.chunk_id, found[h.chunk_id]['metadata']) != h for h in headers):
                raise VectorStoreError('读取期间文档身份或版本发生变化')
            return tuple(self._chunk(found[h.chunk_id]) for h in headers)

    def count_documents_by_kb(self, *, excluded_doc_ids: tuple[str, ...] = ()) -> dict[str, int]:
        with self._connection() as connection:
            damaged = connection.execute("SELECT doc_id FROM vector_chunks WHERE collection=%s "
                "GROUP BY doc_id HAVING COUNT(DISTINCT (kb_id,metadata->>'filename')) > 1 LIMIT 1",
                (self._collection_name,)).fetchone()
            if damaged is not None:
                raise VectorStoreError('同一文档的知识库或来源文件名不一致')
            return {r['kb_id']: r['total'] for r in connection.execute(
                'SELECT kb_id,COUNT(DISTINCT doc_id) AS total FROM vector_chunks WHERE collection=%s '
                'AND NOT (doc_id=ANY(%s)) GROUP BY kb_id', (self._collection_name, list(excluded_doc_ids)))}

    def delete_document(self, doc_id: str) -> None:
        with self._connection() as connection:
            connection.execute('DELETE FROM vector_chunks WHERE collection=%s AND doc_id=%s', (self._collection_name, doc_id))

    def count(self) -> int:
        with self._connection() as connection:
            return connection.execute('SELECT COUNT(*) FROM vector_chunks WHERE collection=%s', (self._collection_name,)).fetchone()[0]

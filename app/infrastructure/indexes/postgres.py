"""共享 PostgreSQL 全文索引，无进程内语料副本。"""
import psycopg

from app.application.errors import VectorStoreError
from app.domain.documents import DocumentChunk
from app.domain.knowledge_bases import validate_kb_ids
from app.domain.retrieval import RetrievalResult
from app.infrastructure.indexes.tokenization import tokenize_search
from app.infrastructure.persistence.database import Database
from app.infrastructure.vectorstores.postgres import PostgresVectorStore


class PostgresKeywordIndex:
    maintained_with_chunks = True

    def __init__(self, database: Database, collection_name: str) -> None:
        self._database = database
        self._collection_name = collection_name

    def rebuild(self, chunks: list[DocumentChunk]) -> None:
        """兼容旧入库端口；持久化块才是事实来源，禁止发布陈旧快照。"""

    def search(self, query: str, top_k: int, *, kb_ids: tuple[str, ...] | None = None,
               excluded_doc_ids: tuple[str, ...] = ()) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError('top_k 必须大于 0')
        validate_kb_ids(kb_ids)
        if kb_ids == ():
            return []
        tokens = dict.fromkeys(tokenize_search(query, for_query=True))
        if not tokens:
            return []
        # tokenizer 只产生字母、数字、下划线和汉字；每词加引号，用户操作符不进入 tsquery。
        tsquery = ' | '.join("'" + token + "'" for token in tokens)
        statement = """WITH q AS (SELECT to_tsquery('pg_catalog.simple', %s) AS query)
            SELECT id,text,metadata,ts_rank_cd(search_vector,q.query) AS score
            FROM vector_chunks CROSS JOIN q
            WHERE collection=%s AND search_vector @@ q.query"""
        values: list[object] = [tsquery, self._collection_name]
        if kb_ids is not None:
            statement += ' AND kb_id=ANY(%s)'
            values.append(list(kb_ids))
        if excluded_doc_ids:
            statement += ' AND NOT (doc_id=ANY(%s))'
            values.append(list(excluded_doc_ids))
        statement += ' ORDER BY score DESC,id LIMIT %s'
        try:
            with self._database.connection(read_only=True) as connection:
                rows = connection.execute(statement, [*values, top_k]).fetchall()
                return [RetrievalResult(**PostgresVectorStore._chunk(row).model_dump(exclude={'id'}),
                                        chunk_id=row['id'], score=float(row['score'])) for row in rows]
        except psycopg.Error as error:
            raise VectorStoreError('PostgreSQL 全文检索暂时不可用') from error

from tests.postgres_support import pg_database, pg_connect
# -*- coding: utf-8 -*-
"""文档浏览只读元数据；上下文只取选定的不透明ID并验证归属。"""

from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest

from app.domain.documents import DocumentChunk
from app.application.errors import VectorStoreError
from app.infrastructure.vectorstores.postgres import PostgresVectorStore


DOC = UUID("11111111-1111-4111-8111-111111111111")
OTHER = UUID("22222222-2222-4222-8222-222222222222")


def test_scoped_headers_and_bounded_body_read_never_parse_chunk_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """跨库元数据、整库正文加载、按ID解析index或丢失时间都会被捕获。

    Args:
        tmp_path: 实际Chroma临时目录。
        monkeypatch: 仅阻止超出允许范围的SDK读取。
    """
    store = PostgresVectorStore(pg_database(tmp_path / "chroma"), "browse-content")
    updated = datetime(2026, 10, 21, tzinfo=UTC)
    chunks = []
    for index, identifier in ((0, "opaque-first"), (10, "opaque/target"), (20, "opaque-last")):
        chunks.append(DocumentChunk(
            id=identifier, doc_id=str(DOC), filename="部署_Plan%.MD",
            text=f"正文{index}", page=None, chunk_index=index, kb_id="ops",
            doc_updated_at=updated,
        ))
    chunks.append(DocumentChunk(id="other", doc_id=str(OTHER), filename="secret.md",
                                text="UNRELATED_PRIVATE_BODY", page=None,
                                chunk_index=0, kb_id="dev"))
    store.add_chunks(chunks, [[1.0, 0.0] for _ in chunks])
    bodies_allowed = False
    from tests.postgres_support import observe_queries
    from contextlib import ExitStack
    stack = ExitStack()
    def guard(query):
        if query.startswith('SELECT id,text'):
            assert bodies_allowed
            assert 'id=ANY' in query
        assert 'WHERE collection=' in query
    monkeypatch.setattr(store, '_query_observer', stack.enter_context(observe_queries(store._database, guard)), raising=False)
    documents = store.list_document_headers("ops")
    assert len(documents) == 1
    assert documents[0].doc_id == DOC
    assert documents[0].chunks_count == 3
    assert documents[0].updated_at == updated
    headers = store.list_chunk_headers(DOC)
    assert [header.chunk_id for header in headers] == ["opaque-first", "opaque/target", "opaque-last"]
    bodies_allowed = True
    selected = store.read_chunks(headers)
    assert [chunk.text for chunk in selected] == ["正文0", "正文10", "正文20"]
    assert selected[1].doc_updated_at == updated
    assert store.list_document_headers("unknown") == ()
    stack.close()


def test_changed_identity_between_metadata_and_body_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """即使底层返回的ID相同，正文归属变化也不能发布为原引用。

    Args:
        tmp_path: 真实Chroma目录。
        monkeypatch: 仅模拟SDK返回的版本／归属不一致。
    """
    store = PostgresVectorStore(pg_database(tmp_path / "chroma"), "changed-identity")
    store.add_chunks([DocumentChunk(
        id="opaque", doc_id=str(DOC), filename="guide.md", page=None,
        text="不能错误发布的正文", chunk_index=0,
    )], [[1.0, 0.0]])
    headers = store.list_chunk_headers(DOC)
    with store._database.connection() as connection:
        connection.execute("UPDATE vector_chunks SET metadata=jsonb_set(metadata,'{doc_id}',to_jsonb(%s::text))", (str(OTHER),))
    with pytest.raises(VectorStoreError):
        store.read_chunks(headers)

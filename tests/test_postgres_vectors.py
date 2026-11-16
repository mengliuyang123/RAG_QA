from uuid import uuid4
import pytest
from tests.postgres_support import isolated_database
from app.domain.documents import DocumentChunk

pytestmark = pytest.mark.database


def test_2560_dimension_scope_exclusions_and_persistence():
    from app.infrastructure.vectorstores.postgres import PostgresVectorStore
    with isolated_database() as database:
        store = PostgresVectorStore(database, 'knowledge')
        a, b = str(uuid4()), str(uuid4())
        chunks = [DocumentChunk(id=a+':0', doc_id=a, text='A', filename='a.md', page=None, chunk_index=0, kb_id='ops'),
                  DocumentChunk(id=b+':0', doc_id=b, text='B', filename='b.md', page=None, chunk_index=0)]
        vector = [1.0] + [0.0] * 2559
        store.add_chunks(chunks, [vector, vector])
        assert [r.chunk.doc_id for r in store.query_by_embedding(vector, 1, kb_ids=('ops',))] == [a]
        assert store.query_by_embedding(vector, 1, kb_ids=()) == []
        assert store.query_by_embedding(vector, 1, kb_ids=('ops',), excluded_doc_ids=(a,)) == []
        assert PostgresVectorStore(database, 'knowledge').count() == 2
        assert store.count_documents_by_kb() == {'ops': 1, 'default': 1}
        assert store.read_chunks(store.list_chunk_headers(__import__('uuid').UUID(a)))[0] == chunks[0]
        assert store.list_document_headers('ops')[0].chunks_count == 1
        from app.application.errors import VectorStoreError
        with pytest.raises(VectorStoreError):
            store.query_by_embedding([1.0, 0.0], 1)

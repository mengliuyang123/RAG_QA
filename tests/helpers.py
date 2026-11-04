from app.application.ingestion.pipeline import IngestionPipeline
from app.domain.documents import DocumentChunk
from app.infrastructure.chunking.recursive import RecursiveDocumentChunker
from app.infrastructure.parsing.factory import ParserFactory


class FakeEmbeddingProvider:
    """上传链路测试使用的本地确定性向量器。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """根据文本长度生成二维测试向量。"""
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """生成单条查询的测试向量。"""
        return [float(len(text)), 1.0]


class MemoryVectorStore:
    """上传链路测试使用的内存向量存储。"""

    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """保存文本块，并验证向量数量一致。"""
        assert len(chunks) == len(embeddings)
        self.chunks.extend(chunks)

    def delete_document(self, doc_id: str) -> None:
        """删除指定文档的全部测试记录。"""
        self.chunks = [chunk for chunk in self.chunks if chunk.doc_id != doc_id]

    def list_chunks(self) -> list[DocumentChunk]:
        """返回当前保存的全部测试文本块。"""
        return list(self.chunks)


def make_test_pipeline() -> tuple[IngestionPipeline, MemoryVectorStore]:
    """创建不访问网络和真实数据库的完整入库流水线。"""
    store = MemoryVectorStore()
    pipeline = IngestionPipeline(
        chunker=RecursiveDocumentChunker(chunk_size=20, chunk_overlap=5),
        embedder=FakeEmbeddingProvider(),
        vector_store=store,
        embedding_batch_size=2,
        parser_resolver=ParserFactory(),
    )
    return pipeline, store

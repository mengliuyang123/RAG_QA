import os
from pathlib import Path

import pytest

from app.application.ingestion.pipeline import IngestionPipeline
from app.infrastructure.chunking.recursive import RecursiveDocumentChunker
from app.infrastructure.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from app.infrastructure.parsing.factory import ParserFactory
from app.infrastructure.vectorstores.chroma import ChromaVectorStore


@pytest.mark.integration
def test_real_openai_embedding_can_persist_one_document(tmp_path: Path) -> None:
    """使用真实 OpenAI 向量服务完成一份小文档的端到端入库。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("未设置 OPENAI_API_KEY")

    document_path = tmp_path / "integration.txt"
    document_path.write_text("这是一段用于验证向量入库链路的文本。", encoding="utf-8")
    store = ChromaVectorStore(tmp_path / "chroma", "integration_test")
    pipeline = IngestionPipeline(
        chunker=RecursiveDocumentChunker(chunk_size=100, chunk_overlap=10),
        embedder=OpenAICompatibleEmbeddingProvider(api_key=api_key),
        vector_store=store,
        parser_resolver=ParserFactory(),
    )

    chunks_count = pipeline.ingest(
        document_path,
        "integration-doc",
        "integration.txt",
    )

    assert chunks_count == 1
    assert store.count() == 1

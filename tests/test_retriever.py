import logging
from pathlib import Path

import pytest

from app.application.errors import (
    MissingKeywordRetrieverError,
    MissingRerankerError,
)
from app.application.retrieval.service import (
    RetrievalService,
    VectorRetriever,
)
from app.domain.documents import DocumentChunk
from app.domain.errors import (
    InvalidQuestionError,
    InvalidTopKError,
    UnsupportedRetrievalStrategyError,
)
from app.domain.retrieval import (
    RetrievalResult,
    RetrievalStrategy,
    VectorSearchResult,
)
from app.infrastructure.vectorstores.chroma import ChromaVectorStore


class FakeQueryEmbedder:
    """记录查询文本并返回固定二维向量。"""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [1.0, 0.0]
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """满足向量化协议，本组测试不调用。"""
        return [self.vector for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """记录清理后的问题并返回测试向量。"""
        self.queries.append(text)
        return self.vector


class FakeVectorStore:
    """记录向量查询并返回预设命中结果。"""

    def __init__(self, results: list[VectorSearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[list[float], int]] = []

    def query_by_embedding(
        self,
        vector: list[float],
        top_k: int,
    ) -> list[VectorSearchResult]:
        """按数据库既有顺序返回测试结果。"""
        self.calls.append((vector, top_k))
        return self.results[:top_k]


class FakeKeywordRetriever:
    """记录关键词查询并返回预设的统一检索结果。"""

    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, int]] = []
        self.error: Exception | None = None

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """按预设顺序返回关键词命中，并可模拟明确服务故障。"""
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return self.results[:top_k]


class FakeReranker:
    """记录融合候选，并用 chunk_index 生成可验证的新排序。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievalResult], int]] = []
        self.error: Exception | None = None
        self.return_empty = False

    def rerank(
        self,
        question: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """按测试分数降序返回结果，并可模拟空结果或故障。"""
        self.calls.append((question, candidates, top_k))
        if self.error is not None:
            raise self.error
        if self.return_empty:
            return []
        rescored = [
            candidate.model_copy(update={"score": float(candidate.chunk_index)})
            for candidate in candidates
        ]
        return sorted(rescored, key=lambda result: -result.score)[:top_k]


def make_search_result(
    chunk_id: str,
    distance: float,
    *,
    filename: str = "fastapi.pdf",
    page: int | None = 12,
) -> VectorSearchResult:
    """创建包含完整引用元数据的向量命中。"""
    doc_id, index_text = chunk_id.split(":")
    index = int(index_text)
    return VectorSearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            doc_id=doc_id,
            text=f"文本块 {index}",
            filename=filename,
            page=page,
            chunk_index=index,
        ),
        distance=distance,
    )


def make_retrieval_result(
    chunk_id: str,
    score: float,
) -> RetrievalResult:
    """创建 BM25 或融合层使用的统一检索结果。"""
    search_result = make_search_result(chunk_id, distance=0.5)
    chunk = search_result.chunk
    return RetrievalResult(
        chunk_id=chunk.id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        filename=chunk.filename,
        page=chunk.page,
        chunk_index=chunk.chunk_index,
        score=score,
    )


def test_vector_retriever_returns_metadata_and_relevance_score() -> None:
    """检索结果应保留完整来源，并把距离转换为越大越相关的分数。"""
    embedder = FakeQueryEmbedder()
    store = FakeVectorStore(
        [
            make_search_result("doc-a:0", 0.1),
            make_search_result("doc-a:1", 0.6, page=13),
        ]
    )

    results = VectorRetriever(store, embedder).retrieve(
        "  FastAPI 路径参数  ",
        top_k=5,
    )

    assert embedder.queries == ["FastAPI 路径参数"]
    assert store.calls == [([1.0, 0.0], 5)]
    assert [result.chunk_id for result in results] == ["doc-a:0", "doc-a:1"]
    assert results[0].doc_id == "doc-a"
    assert results[0].filename == "fastapi.pdf"
    assert results[0].page == 12
    assert results[0].chunk_index == 0
    assert results[0].text == "文本块 0"
    assert results[0].score == pytest.approx(0.9)
    assert results[0].score > results[1].score


def test_retriever_dispatches_vector_strategy() -> None:
    """统一入口应把 vector 策略交给纯向量检索器。"""
    vector_retriever = VectorRetriever(
        FakeVectorStore([make_search_result("doc-a:0", 0.2)]),
        FakeQueryEmbedder(),
    )

    results = RetrievalService(vector_retriever).retrieve(
        "问题",
        strategy=RetrievalStrategy.VECTOR,
        top_k=1,
    )

    assert [result.chunk_id for result in results] == ["doc-a:0"]


def test_retriever_hybrid_uses_expanded_candidates_and_limits_output() -> None:
    """混合检索应扩大两路候选集，并只返回调用方请求的数量。"""
    store = FakeVectorStore([
        make_search_result("doc-a:0", 0.1),
        make_search_result("doc-b:0", 0.2),
        make_search_result("doc-c:0", 0.3),
    ])
    keyword = FakeKeywordRetriever([
        make_retrieval_result("doc-b:0", 100.0),
        make_retrieval_result("doc-d:0", 90.0),
        make_retrieval_result("doc-a:0", 80.0),
    ])
    retriever = RetrievalService(
        VectorRetriever(store, FakeQueryEmbedder()),
        keyword_retriever=keyword,
    )

    results = retriever.retrieve(
        "  FastAPI response_model  ",
        strategy=RetrievalStrategy.HYBRID,
        top_k=2,
    )

    assert store.calls == [([1.0, 0.0], 20)]
    assert keyword.calls == [("FastAPI response_model", 20)]
    assert len(results) == 2
    assert [result.chunk_id for result in results] == ["doc-b:0", "doc-a:0"]


def test_retriever_hybrid_allows_eighty_internal_candidates() -> None:
    """公共 top_k=20 时，两路内部召回数量应扩展到 80。"""
    store = FakeVectorStore()
    keyword = FakeKeywordRetriever()
    retriever = RetrievalService(
        VectorRetriever(store, FakeQueryEmbedder()),
        keyword_retriever=keyword,
    )

    retriever.retrieve("问题", strategy="hybrid", top_k=20)

    assert store.calls[0][1] == 80
    assert keyword.calls == [("问题", 80)]


def test_retriever_hybrid_allows_one_empty_ranking() -> None:
    """向量结果合法为空时，关键词结果仍应参与融合。"""
    keyword_result = make_retrieval_result("doc-a:0", 10.0)
    retriever = RetrievalService(
        VectorRetriever(FakeVectorStore(), FakeQueryEmbedder()),
        keyword_retriever=FakeKeywordRetriever([keyword_result]),
    )

    results = retriever.retrieve("术语", strategy="hybrid", top_k=5)

    assert [result.chunk_id for result in results] == ["doc-a:0"]
    assert results[0].score == pytest.approx(1 / 61)


def test_retriever_hybrid_propagates_keyword_failure() -> None:
    """关键词检索发生明确故障时不得静默降级为纯向量检索。"""
    keyword = FakeKeywordRetriever()
    keyword.error = RuntimeError("关键词服务故障")
    retriever = RetrievalService(
        VectorRetriever(FakeVectorStore(), FakeQueryEmbedder()),
        keyword_retriever=keyword,
    )

    with pytest.raises(RuntimeError, match="关键词服务故障"):
        retriever.retrieve("问题", strategy="hybrid", top_k=5)


def test_retriever_hybrid_requires_keyword_dependency() -> None:
    """未装配关键词检索器时不得伪装成可用的 hybrid 策略。"""
    retriever = RetrievalService(
        VectorRetriever(FakeVectorStore(), FakeQueryEmbedder())
    )

    with pytest.raises(MissingKeywordRetrieverError):
        retriever.retrieve("问题", strategy="hybrid", top_k=5)


def test_retriever_hybrid_rerank_limits_fused_candidates_and_final_output() -> None:
    """两阶段检索应只重排前 20 个融合候选，并返回最终 top-5。"""
    store = FakeVectorStore([
        make_search_result(f"doc:{index}", 0.01 * index)
        for index in range(25)
    ])
    keyword = FakeKeywordRetriever()
    reranker = FakeReranker()
    retriever = RetrievalService(
        VectorRetriever(store, FakeQueryEmbedder()),
        keyword_retriever=keyword,
        reranker=reranker,
        top_k_rerank=20,
    )

    results = retriever.retrieve(
        "  两阶段检索  ",
        strategy="hybrid_rerank",
        top_k=5,
    )

    assert store.calls == [([1.0, 0.0], 80)]
    assert keyword.calls == [("两阶段检索", 80)]
    assert len(reranker.calls) == 1
    assert reranker.calls[0][0] == "两阶段检索"
    assert len(reranker.calls[0][1]) == 20
    assert reranker.calls[0][2] == 5
    assert [result.chunk_id for result in results] == [
        "doc:19",
        "doc:18",
        "doc:17",
        "doc:16",
        "doc:15",
    ]
    assert [result.score for result in results] == [19, 18, 17, 16, 15]


def test_retriever_hybrid_rerank_allows_empty_reranker_result() -> None:
    """重排器合法返回空列表时，两阶段策略应保持空结果。"""
    reranker = FakeReranker()
    reranker.return_empty = True
    retriever = RetrievalService(
        VectorRetriever(
            FakeVectorStore([make_search_result("doc:0", 0.1)]),
            FakeQueryEmbedder(),
        ),
        keyword_retriever=FakeKeywordRetriever(),
        reranker=reranker,
    )

    results = retriever.retrieve("问题", strategy="hybrid_rerank", top_k=1)

    assert results == []
    assert len(reranker.calls) == 1


def test_retriever_close_releases_reranker_resources() -> None:
    """检索器关闭时应释放重排器自行持有的连接池。"""

    class ClosableReranker(FakeReranker):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    reranker = ClosableReranker()
    retriever = RetrievalService(
        VectorRetriever(FakeVectorStore(), FakeQueryEmbedder()),
        reranker=reranker,
    )

    retriever.close()

    assert reranker.closed is True


def test_retriever_hybrid_rerank_rejects_insufficient_candidate_limit() -> None:
    """top_k_rerank 小于最终 top_k 时不得开始任何召回。"""
    store = FakeVectorStore()
    keyword = FakeKeywordRetriever()
    reranker = FakeReranker()
    retriever = RetrievalService(
        VectorRetriever(store, FakeQueryEmbedder()),
        keyword_retriever=keyword,
        reranker=reranker,
        top_k_rerank=4,
    )

    with pytest.raises(InvalidTopKError, match="top_k_rerank"):
        retriever.retrieve("问题", strategy="hybrid_rerank", top_k=5)

    assert store.calls == []
    assert keyword.calls == []
    assert reranker.calls == []


def test_retriever_hybrid_rerank_propagates_reranker_failure() -> None:
    """远程重排失败时不得静默退化为未重排的 hybrid 结果。"""
    reranker = FakeReranker()
    reranker.error = RuntimeError("重排服务故障")
    retriever = RetrievalService(
        VectorRetriever(
            FakeVectorStore([make_search_result("doc:0", 0.1)]),
            FakeQueryEmbedder(),
        ),
        keyword_retriever=FakeKeywordRetriever(),
        reranker=reranker,
    )

    with pytest.raises(RuntimeError, match="重排服务故障"):
        retriever.retrieve("问题", strategy="hybrid_rerank", top_k=1)


def test_retriever_hybrid_rerank_requires_reranker_before_recall() -> None:
    """未装配重排器时应在昂贵的向量与 BM25 召回前失败。"""
    store = FakeVectorStore()
    keyword = FakeKeywordRetriever()
    retriever = RetrievalService(
        VectorRetriever(store, FakeQueryEmbedder()),
        keyword_retriever=keyword,
    )

    with pytest.raises(MissingRerankerError):
        retriever.retrieve("问题", strategy="hybrid_rerank", top_k=5)

    assert store.calls == []
    assert keyword.calls == []


def test_retriever_logs_all_stage_timings_for_hybrid_rerank(caplog) -> None:
    """两阶段检索成功后应输出四个可聚合的阶段耗时字段。"""
    retriever = RetrievalService(
        VectorRetriever(
            FakeVectorStore([make_search_result("doc:0", 0.1)]),
            FakeQueryEmbedder(),
        ),
        keyword_retriever=FakeKeywordRetriever(),
        reranker=FakeReranker(),
    )

    with caplog.at_level(
        logging.INFO,
        logger="app.application.retrieval.service",
    ):
        retriever.retrieve("问题", strategy="hybrid_rerank", top_k=1)

    timing_records = [
        record
        for record in caplog.records
        if record.getMessage() == "检索阶段耗时"
    ]
    assert len(timing_records) == 1
    record = timing_records[0]
    assert record.retrieval_strategy == "hybrid_rerank"
    assert record.vector_ms >= 0
    assert record.bm25_ms >= 0
    assert record.fusion_ms >= 0
    assert record.rerank_ms >= 0


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_retriever_rejects_blank_question(question: str) -> None:
    """空白问题不得调用向量服务。"""
    embedder = FakeQueryEmbedder()
    retriever = RetrievalService(
        VectorRetriever(FakeVectorStore(), embedder)
    )

    with pytest.raises(InvalidQuestionError, match="不能为空"):
        retriever.retrieve(question)

    assert embedder.queries == []


@pytest.mark.parametrize("top_k", [0, -1, 21])
def test_retriever_rejects_top_k_outside_supported_range(top_k: int) -> None:
    """top_k 必须限制在 1 到 20。"""
    retriever = RetrievalService(
        VectorRetriever(FakeVectorStore(), FakeQueryEmbedder())
    )

    with pytest.raises(InvalidTopKError, match="1 到 20"):
        retriever.retrieve("问题", top_k=top_k)


def test_retriever_rejects_unknown_strategy() -> None:
    """统一入口不得静默接受未知检索策略。"""
    retriever = RetrievalService(
        VectorRetriever(FakeVectorStore(), FakeQueryEmbedder())
    )

    with pytest.raises(UnsupportedRetrievalStrategyError, match="semantic"):
        retriever.retrieve("问题", strategy="semantic")


def test_vector_retriever_returns_empty_list_for_empty_collection() -> None:
    """空知识库应返回空列表。"""
    results = VectorRetriever(
        FakeVectorStore(),
        FakeQueryEmbedder(),
    ).retrieve("问题", top_k=5)

    assert results == []


def test_vector_retriever_preserves_database_order() -> None:
    """检索器不得根据可跨实现变化的距离值重新排序。"""
    store = FakeVectorStore(
        [
            make_search_result("doc-a:0", 0.4),
            make_search_result("doc-a:1", 0.2),
        ]
    )

    results = VectorRetriever(store, FakeQueryEmbedder()).retrieve("问题", 2)

    assert [result.chunk_id for result in results] == ["doc-a:0", "doc-a:1"]


def test_vector_retriever_queries_real_chroma_in_similarity_order(
    tmp_path: Path,
) -> None:
    """真实 Chroma 应按余弦距离返回最相近的已知向量。"""
    store = ChromaVectorStore(tmp_path / "chroma", "retrieval_test")
    chunks = [
        make_search_result("doc-a:0", 0.0).chunk,
        make_search_result("doc-b:0", 0.0, filename="b.txt", page=None).chunk,
        make_search_result("doc-c:0", 0.0, filename="c.txt", page=None).chunk,
    ]
    store.add_chunks(chunks, [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])

    results = VectorRetriever(
        store,
        FakeQueryEmbedder([1.0, 0.0]),
    ).retrieve("相关问题", top_k=2)

    assert [result.chunk_id for result in results] == ["doc-a:0", "doc-b:0"]
    assert results[0].score >= results[1].score

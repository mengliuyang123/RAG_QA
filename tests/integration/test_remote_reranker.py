from time import perf_counter

import httpx
import pytest

from app.config import Settings
from app.infrastructure.reranking.dashscope import RemoteReranker
from app.domain.retrieval import RetrievalResult


def make_candidate(chunk_id: str, text: str) -> RetrievalResult:
    """创建不包含项目数据的合成候选。"""
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":")[0],
        text=text,
        filename="synthetic.md",
        page=None,
        chunk_index=int(chunk_id.split(":")[1]),
        score=0.0,
    )


@pytest.mark.integration
def test_real_remote_reranker_prioritizes_relevant_candidate() -> None:
    """显式配置真实服务时，应把明显相关的合成候选排在第一。"""
    settings = Settings()
    if not settings.reranker_base_url:
        pytest.skip("未设置 RERANKER_BASE_URL")

    candidates = [
        make_candidate("synthetic:0", "Python 使用 async def 定义异步函数。"),
        make_candidate("synthetic:1", "今天阳光充足，适合户外散步。"),
    ]
    with httpx.Client(timeout=30.0) as client:
        reranker = RemoteReranker(
            base_url=settings.reranker_base_url,
            api_key=settings.resolved_reranker_api_key,
            model=settings.reranker_model,
            endpoint=settings.reranker_endpoint,
            client=client,
        )
        started_at = perf_counter()
        results = reranker.rerank(
            "Python 如何定义异步函数？",
            candidates,
            top_k=2,
        )
        latency_ms = (perf_counter() - started_at) * 1000

    print(
        f"reranker_model={settings.reranker_model} "
        f"latency_ms={latency_ms:.2f}"
    )
    assert results[0].chunk_id == "synthetic:0"

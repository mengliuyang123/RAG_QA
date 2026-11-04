# -*- coding: utf-8 -*-
"""非流式知识库问答 HTTP 接口测试。"""

import json
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from app.application.errors import LLMGenerationError
from app.application.errors import LLMTimeoutError
from app.application.errors import RerankerUnavailableError
from app.application.query.prompting import INSUFFICIENT_CONTEXT_ANSWER
from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from app.config import Settings
from app.domain.answers import Citation, QueryResult
from app.domain.documents import DocumentChunk
from app.domain.errors import InvalidTopKError
from app.domain.retrieval import RetrievalStrategy
from app.domain.retrieval import VectorSearchResult
from app.interfaces.http.dependencies import get_query_service
from tests.http_auth import create_test_app as create_app
from app.domain.conversations import KNOWLEDGE_REFUSAL_ANSWER
from tests.api_compat import legacy_answer_view
from tests.understanding import OfflineUnderstandingProvider


class FakeQueryService:
    """记录 API 传入参数并返回固定查询结果。

    Attributes:
        calls: HTTP 层传入的查询参数记录。
        error: 查询时需要抛出的可选异常。
    """

    def __init__(self) -> None:
        """初始化空调用记录且不注入异常。

        Returns:
            None。
        """
        self.calls: list[tuple[str, RetrievalStrategy | str, int]] = []
        self.error: Exception | None = None

    def query(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
        *, user_id: UUID | None = None,
    ) -> QueryResult:
        """模拟非流式 RAG 查询。

        Args:
            question: HTTP 层传入的问题。
            strategy: HTTP 层传入的检索策略。
            top_k: HTTP 层传入的结果上限。

        Returns:
            固定的查询结果。

        Raises:
            Exception: 测试场景注入的稳定或未知异常。
        """
        self.calls.append((question, strategy, top_k))
        if self.error is not None:
            raise self.error
        return QueryResult(
            answer="路径参数使用花括号声明[1]。",
            citations=[
                Citation(
                    position=1,
                    doc_id="doc-a",
                    filename="fastapi.pdf",
                    page=12,
                    chunk_text="使用 {item_id} 声明路径参数。",
                )
            ],
            latency_ms=12.5,
            retrieval_strategy=RetrievalStrategy(strategy),
        )


class _OfflineVectorStore:
    """为生产查询装配返回可选的确定性候选。"""

    def __init__(self, chunk: DocumentChunk | None) -> None:
        """保存测试场景中的唯一候选。

        Args:
            chunk: 需要由向量和关键词检索共同召回的候选。

        Returns:
            None。
        """
        self._chunk = chunk

    def query_by_embedding(
        self,
        vector: list[float],
        top_k: int,
        *, kb_ids: tuple[str, ...] | None = None,
    ) -> list[VectorSearchResult]:
        """保持向量分支为空，使候选只能来自默认 BM25。

        Args:
            vector: 离线 embedder 生成的查询向量。
            top_k: 向量候选上限。

        Returns:
            始终为空的向量命中列表。
        """
        del vector
        del top_k
        return []

    def list_chunks(self) -> list[DocumentChunk]:
        """返回供默认 BM25 factory 建索引的相同候选。

        Returns:
            空列表或唯一的确定性文本块。
        """
        if self._chunk is None:
            return []
        return [self._chunk]

    def close(self) -> None:
        """提供容器可关闭的离线资源边界。

        Returns:
            None。
        """


class _OfflineEmbeddingProvider:
    """为生产 VectorRetriever 返回确定性查询向量。"""

    def embed_query(self, text: str) -> list[float]:
        """返回不访问远程服务的查询向量。

        Args:
            text: 已校验的查询文本。

        Returns:
            固定二维查询向量。
        """
        del text
        return [1.0, 0.0]

    def close(self) -> None:
        """提供容器可关闭的离线资源边界。

        Returns:
            None。
        """


class _UnusedLLMProvider:
    """确保 reranker 场景不会意外进入生成阶段。"""

    def generate(self, messages: list[dict[str, str]]) -> str:
        """拒绝非流式生成调用。

        Args:
            messages: 不应到达本替身的消息。

        Returns:
            本方法不会正常返回。

        Raises:
            AssertionError: 查询错误地进入生成阶段。
        """
        del messages
        raise AssertionError("reranker 回归场景不应调用 LLM")

    def stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        """拒绝流式生成调用。

        Args:
            messages: 不应到达本替身的消息。

        Returns:
            本方法不会正常返回。

        Raises:
            AssertionError: 查询错误地进入生成阶段。
        """
        del messages
        raise AssertionError("reranker 回归场景不应调用 LLM")

    def close(self) -> None:
        """提供容器可关闭的离线资源边界。

        Returns:
            None。
        """


QueryApp = tuple[FastAPI, FakeQueryService]


def _create_production_query_app(
    tmp_path: Path,
    *,
    with_candidate: bool,
) -> FastAPI:
    """保留默认 BM25/reranker factory 并替换外部 I/O。

    Args:
        tmp_path: pytest 提供的临时应用目录。
        with_candidate: 是否让默认关键词检索返回候选。

    Returns:
        使用真实生产装配链的 FastAPI 测试应用。
    """
    chunk = None
    if with_candidate:
        chunk = DocumentChunk(
            id="doc-a:0",
            doc_id="doc-a",
            text="重排序模型用于重新打分候选。",
            filename="reranker.md",
            page=None,
            chunk_index=0,
        )
    vector_store = _OfflineVectorStore(chunk)
    embedding_provider = _OfflineEmbeddingProvider()
    llm_provider = _UnusedLLMProvider()

    def _create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> _OfflineVectorStore:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        return vector_store

    def _create_embedding_provider(
        api_key: str | None,
        base_url: str,
        model: str,
        batch_size: int,
    ) -> _OfflineEmbeddingProvider:
        del api_key
        del base_url
        del model
        del batch_size
        return embedding_provider

    def _create_llm_provider(
        api_key: str | None,
        base_url: str,
        model: str,
        temperature: float | None,
        reasoning_effort: str | None,
    ) -> _UnusedLLMProvider:
        del api_key
        del base_url
        del model
        del temperature
        del reasoning_effort
        return llm_provider

    default_factories = AdapterFactories()
    factories = replace(
        default_factories,
        vector_store=_create_vector_store,
        embedding_provider=_create_embedding_provider,
        llm_provider=_create_llm_provider,
        understanding_provider=lambda *args: OfflineUnderstandingProvider(),
    )
    settings = Settings(
        _env_file=None,
        embedding_api_key="offline",
        llm_api_key="offline",
        reranker_base_url=None,
        chroma_path=tmp_path / "chroma",
        docs_path=tmp_path / "docs",
        knowledge_base_db_path=tmp_path / "catalog.sqlite3",
        trace_db_path=tmp_path / "traces.sqlite3",
        conversation_db_path=tmp_path / "conversations.sqlite3",
        agent_checkpoint_db_path=tmp_path / "checkpoints.sqlite3",
    )
    container_factory = partial(
        build_application_container,
        factories=factories,
    )
    return create_app(
        settings=settings,
        container_factory=container_factory,
    )


@pytest.fixture
def anyio_backend() -> str:
    """API 测试固定使用 asyncio。

    Returns:
        AnyIO 使用的 asyncio 后端名称。
    """
    return "asyncio"


@pytest.fixture
def query_app() -> Iterator[QueryApp]:
    """创建注入假 RAG 流水线的测试应用。

    Yields:
        FastAPI 应用和对应的假查询服务。
    """
    application = create_app()
    pipeline = FakeQueryService()
    application.dependency_overrides[get_query_service] = lambda: pipeline
    yield application, pipeline
    application.dependency_overrides.clear()


@pytest.mark.anyio
async def test_query_returns_answer_sources_latency_and_strategy(
    query_app: QueryApp,
) -> None:
    """成功查询应返回完整稳定的响应结构。

    Args:
        query_app: 注入假查询服务的应用与服务。
    """
    app, pipeline = query_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/query",
            json={
                "question": "  FastAPI 如何定义路径参数？  ",
                "strategy": "vector",
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    assert pipeline.calls == [
        ("FastAPI 如何定义路径参数？", RetrievalStrategy.VECTOR, 3)
    ]
    assert legacy_answer_view(response.json(), (
        "answer", "citations", "latency_ms", "retrieval_strategy",
    )) == {
        "answer": "路径参数使用花括号声明[1]。",
        "citations": [
            {
                "doc_id": "doc-a",
                "filename": "fastapi.pdf",
                "page": 12,
                "chunk_text": "使用 {item_id} 声明路径参数。",
                "position": 1,
            }
        ],
        "latency_ms": 12.5,
        "retrieval_strategy": "vector",
    }


@pytest.mark.anyio
async def test_query_accepts_hybrid_strategy(
    query_app: QueryApp,
) -> None:
    """HTTP 请求模型应接受 hybrid，并把枚举传给查询服务。

    Args:
        query_app: 注入假查询服务的应用与服务。
    """
    app, pipeline = query_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/query",
            json={"question": "精确术语", "strategy": "hybrid", "top_k": 2},
        )

    assert response.status_code == 200
    assert pipeline.calls == [
        ("精确术语", RetrievalStrategy.HYBRID, 2)
    ]
    assert response.json()["retrieval_strategy"] == "hybrid"


@pytest.mark.anyio
async def test_query_accepts_hybrid_rerank_strategy(
    query_app: QueryApp,
) -> None:
    """HTTP 请求模型应接受两阶段策略并原样返回实际策略。

    Args:
        query_app: 注入假查询服务的应用与服务。
    """
    app, pipeline = query_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/query",
            json={
                "question": "什么是重排序模型？",
                "strategy": "hybrid_rerank",
                "top_k": 5,
            },
        )

    assert response.status_code == 200
    assert pipeline.calls == [
        ("什么是重排序模型？", RetrievalStrategy.HYBRID_RERANK, 5)
    ]
    assert response.json()["retrieval_strategy"] == "hybrid_rerank"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "问题", "strategy": "semantic"},
        {"question": "问题", "top_k": 0},
        {"question": "问题", "top_k": 21},
    ],
)
async def test_query_rejects_invalid_request_before_pipeline(
    query_app: QueryApp,
    payload: dict[str, object],
) -> None:
    """空问题、未知策略和越界 top_k 应由请求模型返回 422。

    Args:
        query_app: 注入假查询服务的应用与服务。
        payload: 当前参数化场景的无效请求体。
    """
    app, pipeline = query_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/query", json=payload)

    assert response.status_code == 422
    assert pipeline.calls == []


@pytest.mark.anyio
async def test_query_maps_llm_timeout_to_gateway_timeout(
    query_app: QueryApp,
) -> None:
    """语言模型超时应返回稳定的 HTTP 504。

    Args:
        query_app: 注入假查询服务的应用与服务。
    """
    app, pipeline = query_app
    pipeline.error = LLMTimeoutError("private timeout detail")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/query", json={"question": "问题"})

    assert response.status_code == 504
    assert response.json()["detail"] == "语言模型服务响应超时"


@pytest.mark.anyio
async def test_query_maps_llm_failure_to_bad_gateway(
    query_app: QueryApp,
) -> None:
    """其他语言模型故障应隐藏上游细节并返回 HTTP 502。

    Args:
        query_app: 注入假查询服务的应用与服务。
    """
    app, pipeline = query_app
    pipeline.error = LLMGenerationError("private upstream detail")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/query", json={"question": "问题"})

    assert response.status_code == 502
    assert response.json()["detail"] == "语言模型服务暂时不可用"


@pytest.mark.anyio
async def test_query_maps_reranker_failure_to_bad_gateway(
    query_app: QueryApp,
) -> None:
    """远程重排不可用时应返回稳定 502，且不暴露上游细节。

    Args:
        query_app: 注入假查询服务的应用与服务。
    """
    app, pipeline = query_app
    pipeline.error = RerankerUnavailableError("private reranker detail")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/query",
            json={"question": "问题", "strategy": "hybrid_rerank"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "重排服务暂时不可用"


@pytest.mark.anyio
async def test_query_maps_retrieval_error_to_unprocessable_content(
    query_app: QueryApp,
) -> None:
    """服务拒绝检索参数时应通过集中错误处理返回 HTTP 422。

    Args:
        query_app: 注入假查询服务的应用与服务。
    """
    app, pipeline = query_app
    pipeline.error = InvalidTopKError("top_k 必须大于 0")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/query",
            json={"question": "问题"},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "top_k 必须大于 0"


@pytest.mark.anyio
async def test_query_hides_unexpected_internal_error(
    query_app: QueryApp,
) -> None:
    """未知内部异常不得泄露实现细节。

    Args:
        query_app: 注入假查询服务的应用与服务。
    """
    app, pipeline = query_app
    pipeline.error = RuntimeError("private implementation detail")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/query", json={"question": "问题"})

    assert response.status_code == 500
    assert response.json()["detail"] == "知识库查询处理失败"


@pytest.mark.anyio
async def test_unconfigured_reranker_hybrid_empty_query_returns_stable_answer(
    tmp_path: Path,
) -> None:
    """默认生产装配在空库时应保留 hybrid_rerank 成功响应。

    Args:
        tmp_path: pytest 提供的临时应用目录。

    Returns:
        None。
    """
    application = _create_production_query_app(
        tmp_path,
        with_candidate=False,
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/query",
                json={
                    "question": "空库问题",
                    "strategy": "hybrid",
                    "top_k": 1,
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == KNOWLEDGE_REFUSAL_ANSWER
    assert body["outcome"] == "refused"
    assert body["citations"] == []
    assert body["retrieval_strategy"] == "hybrid"
    assert body["latency_ms"] >= 0


@pytest.mark.anyio
async def test_unconfigured_reranker_hybrid_empty_stream_completes(
    tmp_path: Path,
) -> None:
    """默认生产装配在空库时应发送唯一 complete 事件。

    Args:
        tmp_path: pytest 提供的临时应用目录。

    Returns:
        None。
    """
    application = _create_production_query_app(
        tmp_path,
        with_candidate=False,
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/query/stream",
                json={
                    "question": "空库问题",
                    "strategy": "hybrid",
                    "top_k": 1,
                },
            )

    assert response.status_code == 200
    frames = response.text.split("\n\n")[:-1]
    complete_frames = [frame for frame in frames if frame.startswith("event: complete\n")]
    assert len(complete_frames) == 1
    event_name, raw_data = complete_frames[0].split("\n")
    assert event_name == "event: complete"
    payload = json.loads(raw_data.removeprefix("data: "))
    assert payload["answer"] == KNOWLEDGE_REFUSAL_ANSWER
    assert payload["outcome"] == "refused"
    assert payload["citations"] == []
    assert payload["retrieval_strategy"] == "hybrid"
    assert payload["latency_ms"] >= 0


@pytest.mark.anyio
async def test_unconfigured_reranker_nonempty_query_returns_502(
    tmp_path: Path,
) -> None:
    """默认生产装配对非空候选应保留稳定 reranker 502。

    Args:
        tmp_path: pytest 提供的临时应用目录。

    Returns:
        None。
    """
    application = _create_production_query_app(
        tmp_path,
        with_candidate=True,
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/query",
                json={
                    "question": "重排序是什么？",
                    "strategy": "hybrid_rerank",
                    "top_k": 1,
                },
            )

    assert response.status_code == 502
    assert response.json()["detail"] == "重排服务暂时不可用"


@pytest.mark.anyio
async def test_unconfigured_reranker_nonempty_stream_keeps_error_frame(
    tmp_path: Path,
) -> None:
    """默认生产装配对非空候选应保留迁移前 SSE 错误帧。

    Args:
        tmp_path: pytest 提供的临时应用目录。

    Returns:
        None。
    """
    application = _create_production_query_app(
        tmp_path,
        with_candidate=True,
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/query/stream",
                json={
                    "question": "重排序是什么？",
                    "strategy": "hybrid_rerank",
                    "top_k": 1,
                },
            )

    assert response.status_code == 200
    assert response.text.endswith(
        'event: error\ndata: {"code":"retrieval_error",'
        '"message":"检索服务暂时不可用"}\n\n'
    )
    assert response.text.count("event: error") == 1
    assert "event: token" not in response.text

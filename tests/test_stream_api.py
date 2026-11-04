# -*- coding: utf-8 -*-
"""流式知识库问答 HTTP 接口测试。"""

import json
from collections.abc import Iterator
from types import SimpleNamespace
from uuid import UUID

import anyio
import httpx
import pytest
from fastapi import FastAPI
from starlette.types import Message, Scope

from tests.http_auth import allow_test_identity

from app.application.query.events import (
    CompleteStreamData,
    CompleteStreamEvent,
    ErrorStreamData,
    ErrorStreamEvent,
    StreamEvent,
    TokenStreamData,
    TokenStreamEvent,
)
from app.application.query.generator import AnswerGenerator
from app.application.query.prompting import PromptBuilder
from app.application.query.service import QueryService
from app.domain.answers import Citation
from app.domain.retrieval import RetrievalResult, RetrievalStrategy
from app.infrastructure.generation.openai_compatible import (
    OpenAICompatibleLLMProvider,
)
from app.interfaces.http.dependencies import get_query_service
from app.interfaces.http.router import router
from tests.api_compat import legacy_answer_view


class _FakeStreamingPipeline:
    """返回预设领域事件的离线流式 Pipeline。"""

    def __init__(self, events: list[StreamEvent]) -> None:
        """保存预设事件并初始化调用记录。

        Args:
            events: 依次返回给 SSE 路由的类型化领域事件。
        """
        self._events = events
        self.calls: list[
            tuple[str, RetrievalStrategy | str, int]
        ] = []

    def stream_query(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
        *, user_id: UUID | None = None,
    ) -> Iterator[StreamEvent]:
        """记录已校验参数并依次产生预设领域事件。

        Args:
            question: 已由请求模型清理的问题。
            strategy: 已由请求模型校验的检索策略。
            top_k: 已由请求模型校验的结果数量上限。

        Yields:
            测试预设的类型化流式领域事件。
        """
        self.calls.append((question, strategy, top_k))
        yield from self._events


class _EndlessClosableChatStream:
    """持续返回 token 并记录关闭状态的离线上游流。"""

    def __init__(self) -> None:
        """初始化未关闭状态。"""
        self.closed = False

    def __iter__(self) -> Iterator[SimpleNamespace]:
        """返回当前上游迭代器。

        Returns:
            当前上游迭代器。
        """
        return self

    def __next__(self) -> SimpleNamespace:
        """返回一个完整的 OpenAI 兼容响应分块。

        Returns:
            带非空文本增量的响应分块。
        """
        delta = SimpleNamespace(content="持续 token")
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice])

    def close(self) -> None:
        """记录请求链路已确定性关闭当前上游流。"""
        self.closed = True


class _OfflineStreamingCompletions:
    """返回测试预设上游流的离线聊天补全接口。"""

    def __init__(self, upstream: _EndlessClosableChatStream) -> None:
        """保存预设上游流。

        Args:
            upstream: 应由请求 teardown 关闭的离线上游流。
        """
        self._upstream = upstream

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
    ) -> _EndlessClosableChatStream:
        """验证流式调用形状并返回预设上游流。

        Args:
            model: 流式请求使用的模型名称。
            messages: PromptBuilder 构建的完整聊天消息。
            stream: 是否请求流式响应。

        Returns:
            不访问外部服务的可关闭响应流。
        """
        assert model == "offline-model"
        assert messages[0]["role"] == "system"
        assert stream is True
        return self._upstream


class _OfflineOpenAIClient:
    """暴露 Provider 所需聊天补全结构的离线客户端。"""

    def __init__(self, upstream: _EndlessClosableChatStream) -> None:
        """装配预设流式补全接口。

        Args:
            upstream: 应由请求 teardown 关闭的离线上游流。
        """
        completions = _OfflineStreamingCompletions(upstream)
        self.chat = SimpleNamespace(completions=completions)


class _OfflineRetriever:
    """为生产形状 Pipeline 返回一条确定性检索结果。"""

    def retrieve(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """返回一条无需外部服务的检索结果。

        Args:
            question: 用户问题；本测试替身不使用。
            strategy: 检索策略；本测试替身不使用。
            top_k: 返回结果数量上限。

        Returns:
            至多一条完整检索结果。
        """
        del question
        del strategy
        result = RetrievalResult(
            chunk_id="doc-a:0",
            doc_id="doc-a",
            text="FastAPI 支持原生 SSE。",
            filename="fastapi.md",
            page=None,
            chunk_index=0,
            score=0.9,
        )
        return [result][:top_k]


class _RecordingQueryService(QueryService):
    """保留真实 Pipeline 迭代器引用以排除垃圾回收关闭。"""

    stream_iterator: Iterator[StreamEvent] | None = None

    def stream_query(
        self,
        question: str,
        strategy: RetrievalStrategy | str = RetrievalStrategy.VECTOR,
        top_k: int = 5,
        *, user_id: UUID | None = None,
    ) -> Iterator[StreamEvent]:
        """保存并返回真实 Pipeline 创建的流式迭代器。

        Args:
            question: 用户问题。
            strategy: 本次使用的检索策略。
            top_k: 参与生成的文本块数量上限。

        Returns:
            真实 Pipeline 流式迭代器。
        """
        iterator = super().stream_query(question, strategy, top_k)
        self.stream_iterator = iterator
        return iterator


@pytest.fixture
def anyio_backend() -> str:
    """API 测试固定使用 asyncio。

    Returns:
        AnyIO 使用的 asyncio 后端名称。
    """
    return "asyncio"


def _create_stream_app(
    pipeline: _FakeStreamingPipeline,
) -> FastAPI:
    """创建只注入离线假 Pipeline 的测试应用。"""
    application = FastAPI()
    allow_test_identity(application)
    application.include_router(router)
    application.dependency_overrides[get_query_service] = lambda: pipeline
    return application


@pytest.mark.anyio
async def test_stream_query_returns_exact_token_and_complete_frames() -> None:
    """成功流应保持事件顺序、精确 JSON 帧和原始中文。"""
    pipeline = _FakeStreamingPipeline(
        events=[
            TokenStreamEvent(data=TokenStreamData(token="你")),
            TokenStreamEvent(data=TokenStreamData(token="好[1]")),
            CompleteStreamEvent(
                data=CompleteStreamData(
                    answer="你好[1]",
                    citations=[
                        Citation(
                            position=1,
                            doc_id="doc-a",
                            filename="a.md",
                            page=None,
                            chunk_text="文本块 0",
                        )
                    ],
                    latency_ms=12.5,
                    retrieval_strategy=RetrievalStrategy.VECTOR,
                )
            ),
        ]
    )
    application = _create_stream_app(pipeline)
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/query/stream",
            json={
                "question": "  SSE 是什么？  ",
                "strategy": "vector",
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    frames = response.text.split("\n\n")[:-1]
    assert frames[:2] == [
        'event: token\ndata: {"token":"你"}',
        'event: token\ndata: {"token":"好[1]"}',
    ]
    assert len(frames) == 3
    event_name, data = frames[-1].split("\n")
    assert event_name == "event: complete"
    assert legacy_answer_view(json.loads(data.removeprefix("data: ")), (
        "answer", "citations", "latency_ms", "retrieval_strategy",
    )) == {
        "answer": "你好[1]", "latency_ms": 12.5,
        "retrieval_strategy": "vector",
        "citations": [{
            "doc_id": "doc-a", "filename": "a.md", "page": None,
            "chunk_text": "文本块 0", "position": 1,
        }],
    }
    assert "\\u4f60" not in response.text
    assert pipeline.calls == [
        ("SSE 是什么？", RetrievalStrategy.VECTOR, 3)
    ]


@pytest.mark.anyio
async def test_stream_query_returns_exact_error_frame() -> None:
    """领域错误应编码为单个精确 SSE 错误帧。"""
    pipeline = _FakeStreamingPipeline(
        events=[
            ErrorStreamEvent(
                data=ErrorStreamData(
                    code="retrieval_error",
                    message="检索服务暂时不可用",
                )
            )
        ]
    )
    application = _create_stream_app(pipeline)
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/query/stream",
            json={"question": "问题", "strategy": "hybrid", "top_k": 2},
        )

    assert response.status_code == 200
    assert response.text == (
        'event: error\ndata: {"code":"retrieval_error",'
        '"message":"检索服务暂时不可用"}\n\n'
    )
    assert pipeline.calls == [
        ("问题", RetrievalStrategy.HYBRID, 2)
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "request_body",
    [
        {"question": "   ", "strategy": "vector", "top_k": 1},
        {"question": "问题", "strategy": "unknown", "top_k": 1},
        {"question": "问题", "strategy": "vector", "top_k": 0},
        {"question": "问题", "strategy": "vector", "top_k": 21},
    ],
    ids=["blank-question", "unknown-strategy", "top-k-zero", "top-k-21"],
)
async def test_stream_query_rejects_invalid_request_before_pipeline(
    request_body: dict[str, object],
) -> None:
    """无效流式请求应在调用 Pipeline 前返回 HTTP 422。

    Args:
        request_body: 当前参数化用例发送的无效 JSON 请求体。
    """
    pipeline = _FakeStreamingPipeline(events=[])
    application = _create_stream_app(pipeline)
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/query/stream",
            json=request_body,
        )

    assert response.status_code == 422
    assert pipeline.calls == []


@pytest.mark.anyio
async def test_disconnect_request_teardown_closes_real_stream_chain() -> None:
    """原生 SSE 断连后，请求 teardown 应关闭同步生成链。"""
    upstream = _EndlessClosableChatStream()
    provider = OpenAICompatibleLLMProvider(
        model="offline-model",
        client=_OfflineOpenAIClient(upstream),
    )
    pipeline = _RecordingQueryService(
        retriever=_OfflineRetriever(),
        answer_generator=AnswerGenerator(
            prompt_builder=PromptBuilder(),
            llm_provider=provider,
        ),
    )
    application = FastAPI()
    allow_test_identity(application)
    application.include_router(router)
    application.dependency_overrides[get_query_service] = lambda: pipeline
    request_body = json.dumps(
        {
            "question": "SSE 是什么？",
            "strategy": "vector",
            "top_k": 1,
        }
    ).encode("utf-8")
    token_sent = anyio.Event()
    request_received = False
    sent_messages: list[Message] = []

    async def receive() -> Message:
        nonlocal request_received
        if not request_received:
            request_received = True
            return {
                "type": "http.request",
                "body": request_body,
                "more_body": False,
            }
        await token_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent_messages.append(message)
        if message["type"] == "http.response.body":
            body = message.get("body", b"")
            if b"event: token" in body:
                token_sent.set()

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/query/stream",
        "raw_path": b"/api/query/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(request_body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
        "state": {},
    }

    await application(scope, receive, send)

    assert token_sent.is_set()
    assert pipeline.stream_iterator is not None
    assert upstream.closed is True
    assert any(
        message["type"] == "http.response.start"
        for message in sent_messages
    )

# -*- coding: utf-8 -*-
"""用真实SDK和离线HTTP验证四类上游调用的usage，而非根据提示词估算。"""

import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.application.tracing import TraceService
from app.application.errors import LLMTimeoutError
from app.domain.retrieval import RetrievalResult
from app.domain.routing import QueryUnderstanding
from app.infrastructure.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from app.infrastructure.generation.openai_compatible import OpenAICompatibleLLMProvider
from app.infrastructure.generation.structured_output import ModelStructuredOutputProvider
from app.infrastructure.persistence.traces import SQLiteTraceRepository
from app.infrastructure.reranking.dashscope import RemoteReranker
from tests.test_query_understanding import _payload
from app.infrastructure.conversations.langgraph_agent import LangGraphConversationAgent
from tests.test_langgraph_conversation_agent import _AcceptingQuality
from tests.test_langgraph_conversation_agent import _FakeRetriever
from tests.test_langgraph_conversation_agent import _chunk


def test_native_provider_calls_store_only_reported_usage(tmp_path: Path) -> None:
    """不能只测试一个SDK；生成、结构化、向量和重排都需接入真实调用边界。

    Args:
        tmp_path: 隔离追踪数据库。
    """
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={
                "data": [{"index": 0, "embedding": [1.0, 0.0]}],
                "model": "embed-v3", "usage": {"prompt_tokens": 7, "total_tokens": 7},
            })
        if request.url.path.endswith("/rerank"):
            return httpx.Response(200, json={
                "results": [{"index": 0, "relevance_score": 0.9}],
                "usage": {"total_tokens": 99},
            })
        payload = json.loads(request.content)
        content = "回答"
        if payload.get("response_format"):
            content = json.dumps(_payload())
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 1,
            "model": "small-v3",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        })

    repository = SQLiteTraceRepository(tmp_path / "trace.sqlite3")
    service = TraceService(repository)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        sdk = OpenAI(
            api_key="offline", base_url="https://provider.example/v1",
            http_client=client, max_retries=0,
        )
        generator = OpenAICompatibleLLMProvider(
            client=sdk, model="small", base_url="https://provider.example/v1",
        )
        embedder = OpenAICompatibleEmbeddingProvider(
            client=sdk, model="embed", base_url="https://provider.example/v1",
        )
        structured = ModelStructuredOutputProvider(ChatOpenAI(
            api_key="offline", base_url="https://provider.example/v1",
            model="small", http_client=client, max_retries=0,
        ))
        reranker = RemoteReranker(
            "https://provider.example/v1", "offline", "rerank", client=client,
        )
        with service.start("问题", request_id=uuid4(), entry="query") as trace:
            assert generator.generate([{"role": "user", "content": "问题"}]) == "回答"
            assert embedder.embed_query("问题") == [1.0, 0.0]
            structured.generate([{"role": "user", "content": "理解"}], QueryUnderstanding)
            reranker.rerank("问题", [RetrievalResult(
                chunk_id="d:0", doc_id="d", filename="d.md", page=None,
                text="资料", chunk_index=0, score=0.9,
            )], 1)
    calls = repository.get(trace.id).calls
    assert len(calls) == 4
    assert [call.purpose for call in calls] == [
        "generation", "embedding", "understanding", "reranking",
    ]
    assert calls[0].usage.input == 10
    assert calls[0].usage.output == 4
    assert calls[0].usage.cached_input == 2
    assert calls[0].usage.reasoning_output == 1
    assert calls[0].response_model == "small-v3"
    assert calls[1].usage.input == 7
    assert calls[1].usage.output == 0
    assert calls[2].usage.input == 10
    assert calls[2].usage.output == 4
    assert calls[3].usage.input is None
    assert calls[3].cost.status == "unknown"


def test_native_graph_generation_usage_reaches_parent_trace(tmp_path: Path) -> None:
    """图节点不能因线程切换或直接ChatOpenAI调用漏记Token。

    Args:
        tmp_path: 独立追踪数据库。
    """
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 1,
            "model": "small-v7", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "回答[1]"},
            }],
            "usage": {"prompt_tokens": 21, "completion_tokens": 5, "total_tokens": 26},
        })

    repository = SQLiteTraceRepository(tmp_path / "trace.sqlite3")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        model = ChatOpenAI(
            api_key="offline", base_url="https://provider.example/v1",
            model="small", http_client=client, max_retries=0,
        )
        graph = LangGraphConversationAgent(
            model, _FakeRetriever({"needle": [_chunk(1)]}), InMemorySaver(),
            top_k=1, quality_evaluator=_AcceptingQuality(),
        )
        with TraceService(repository).start(
            "问题", request_id=uuid4(), entry="conversation",
        ) as trace:
            result = graph.run(uuid4(), "问题", [], initial_query="needle")
            assert result.answer == "回答[1]"
    calls = repository.get(trace.id).calls
    assert len(calls) == 1
    assert calls[0].purpose == "generation"
    assert calls[0].origin == "https://provider.example"
    assert calls[0].response_model == "small-v7"
    assert calls[0].usage.input == 21
    assert calls[0].usage.output == 5


def test_timeout_keeps_failed_call_with_unknown_usage_not_zero(tmp_path: Path) -> None:
    """上游未返回用量时，失败调用仍要留下，但不能宣称免费。

    Args:
        tmp_path: 独立追踪数据库。
    """
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret raw error", request=request)

    repository = SQLiteTraceRepository(tmp_path / "trace.sqlite3")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        sdk = OpenAI(
            api_key="offline", base_url="https://provider.example/v1",
            http_client=client, max_retries=0,
        )
        generator = OpenAICompatibleLLMProvider(client=sdk, model="small")
        with pytest.raises(LLMTimeoutError):
            with TraceService(repository).start(
                "问题", request_id=uuid4(), entry="query",
            ) as trace:
                generator.generate([{"role": "user", "content": "问题"}])
    stored = repository.get(trace.id)
    assert stored.status == "error"
    assert len(stored.calls) == 1
    assert stored.calls[0].status == "error"
    assert stored.calls[0].usage.input is None
    assert stored.calls[0].usage.output is None
    assert stored.calls[0].cost.status == "unknown"
    assert stored.calls[0].cost.amount is None
    assert "secret raw error" not in stored.model_dump_json()

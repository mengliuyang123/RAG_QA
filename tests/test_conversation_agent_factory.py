# -*- coding: utf-8 -*-
"""生产会话 Agent 模型 factory 的凭据与本地代理边界测试。"""

import json
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.application.errors import LLMMissingAPIKeyError
from app.bootstrap.factories import _create_conversation_agent
from app.domain.retrieval import RetrievalResult
from app.domain.retrieval import RetrievalStrategy


class _UnusedRetriever:
    pass


def test_local_agent_model_ignores_system_proxy_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list[Any] = []
    model_arguments: dict[str, object] = {}

    class RecordingClient:
        def __init__(self, *, trust_env: bool) -> None:
            self.trust_env = trust_env
            self.close_calls = 0
            created_clients.append(self)

        def close(self) -> None:
            self.close_calls += 1

    class RecordingChatModel:
        def __init__(self, **kwargs: object) -> None:
            model_arguments.update(kwargs)

        def bind_tools(self, tools: object, *, strict: bool) -> object:
            del tools
            del strict
            return self

    monkeypatch.setattr(
        "app.bootstrap.factories.httpx.Client",
        RecordingClient,
    )
    monkeypatch.setattr(
        "app.bootstrap.factories.ChatOpenAI",
        RecordingChatModel,
    )

    agent = _create_conversation_agent(
        "ollama",
        "http://localhost:11434/v1",
        "qwen3",
        0.0,
        None,
        _UnusedRetriever(),  # type: ignore[arg-type]
        InMemorySaver(),
        5,
        3,
    )

    assert len(created_clients) == 1
    assert created_clients[0].trust_env is False
    assert model_arguments["http_client"] is created_clients[0]
    agent.close()
    agent.close()
    assert created_clients[0].close_calls == 1


def test_agent_factory_requires_explicit_resolved_llm_key() -> None:
    with pytest.raises(LLMMissingAPIKeyError):
        _create_conversation_agent(
            None,
            "http://localhost:11434/v1",
            "qwen3",
            0.0,
            None,
            _UnusedRetriever(),  # type: ignore[arg-type]
            InMemorySaver(),
            5,
            3,
        )


@pytest.mark.parametrize("parameter", ["max_tokens", "max_completion_tokens"])
def test_factory_quality_calls_use_unbound_model_and_real_sdk_protocol(
    monkeypatch: pytest.MonkeyPatch,
    parameter: str,
) -> None:
    """通过真实 ChatOpenAI 序列化验证默认质量审核与工具消息。

    Args:
        monkeypatch: 仅替换外部 HTTP 传输的 pytest fixture。
        parameter: 不同兼容端点明确选择的输出上限参数。
    """
    requests: list[dict[str, Any]] = []
    messages = iter([
        {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "arguments": '{"query":"A 存储"}',
                },
            }],
        },
        {"role": "assistant", "content": json.dumps({
            "sufficient": True, "relevant_numbers": [1],
            "missing_information": [], "next_query": None,
        })},
        {"role": "assistant", "content": "A 使用 SQLite[1]"},
        {"role": "assistant", "content": json.dumps({
            "supported": True, "complete": True, "issues": [],
        })},
    ])

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "completion-1", "object": "chat.completion",
            "created": 1, "model": "test-model",
            "choices": [{
                "index": 0, "message": next(messages),
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 10,
                "total_tokens": 20,
            },
        })

    class KnowledgeBase:
        """保留真实图与 SDK，只替代外部检索响应。"""

        def retrieve(
            self, question: str, strategy: RetrievalStrategy | str,
            top_k: int,
        ) -> list[RetrievalResult]:
            """返回一份确定性存储证据。

            Args:
                question: 实际检索问题。
                strategy: 固定检索策略。
                top_k: 返回数量上限。

            Returns:
                测试用的真实领域检索结果。
            """
            assert strategy is RetrievalStrategy.HYBRID
            return [RetrievalResult(
                chunk_id="a:0", doc_id="a", filename="a.md",
                text="A 使用 SQLite", page=None, chunk_index=0, score=1.0,
            )]

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(
        "app.bootstrap.factories.httpx",
        SimpleNamespace(Client=lambda **kwargs: client),
    )
    agent = _create_conversation_agent(
        "test-key", "http://localhost:11434/v1", "test-model", 0.0,
        None, KnowledgeBase(), InMemorySaver(), 5, 3,
        output_token_parameter=parameter,
    )
    try:
        result = agent.run(uuid4(), "A 存储是什么？", [])
    finally:
        agent.close()
    assert result.answer == "A 使用 SQLite[1]"
    assert result.citations[0].filename == "a.md"
    assert [bool(request.get("tools")) for request in requests] == [
        True, False, True, False,
    ]
    other = (
        "max_tokens" if parameter == "max_completion_tokens"
        else "max_completion_tokens"
    )
    assert all(request.get(parameter) == 4096 for request in requests)
    assert all(other not in request for request in requests)
    assert requests[1]["response_format"]["json_schema"]["name"] == (
        "EvidenceAssessment"
    )
    assert requests[3]["response_format"]["json_schema"]["name"] == (
        "AnswerAssessment"
    )
    tool_messages = [
        item for item in requests[2]["messages"]
        if item["role"] == "tool"
    ]
    assert tool_messages[0]["tool_call_id"] == "call-1"
    assert json.loads(tool_messages[0]["content"])[0]["text"] == (
        "A 使用 SQLite"
    )

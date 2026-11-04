# -*- coding: utf-8 -*-
"""真实SDK的在线裁判协议与远端模型客户端生命周期验证。"""

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.bootstrap.factories import _create_conversation_agent
from app.bootstrap.factories import _create_quality_evaluator
from app.domain.agent_quality import EvidenceItem


def _mock_model(monkeypatch: pytest.MonkeyPatch, content: str):
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 1,
            "model": "offline",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
        })

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(
        "app.bootstrap.factories.ChatOpenAI",
        lambda **options: ChatOpenAI(**options, http_client=client),
    )
    return client, requests


def test_quality_factory_preserves_native_schema_and_output_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新增Fast裁判必须使用真实SDK结构化协议，不是只在配置里声明。

    Args:
        monkeypatch: 只替代外部HTTP传输。
    """
    client, requests = _mock_model(monkeypatch, json.dumps({
        "sufficient": True, "relevant_numbers": [1],
        "missing_information": [], "next_query": None,
    }))
    evaluator = _create_quality_evaluator(
        "offline", "https://model.example/v1", "offline", 0.0, None,
        max_output_tokens=77,
    )
    try:
        result = evaluator.assess_evidence("问题", [], [EvidenceItem(
            number=1, chunk_id="a", doc_id="a", filename="a.md",
            page=None, text="资料",
        )])
        assert result.sufficient is True
        assert requests[0]["max_tokens"] == 77
        assert requests[0]["response_format"]["json_schema"]["name"] == "EvidenceAssessment"
        assert not requests[0].get("tools")
    finally:
        evaluator.close()
    assert client.is_closed


def test_remote_agent_factory_releases_owned_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非localhost模型同样拥有同步连接池，容器关闭时不能遗漏。

    Args:
        monkeypatch: 只替代外部模型构造使用的传输。
    """
    client, _ = _mock_model(monkeypatch, "未调用")
    agent = _create_conversation_agent(
        "offline", "https://model.example/v1", "offline", 0.0, None,
        object(), InMemorySaver(), 5, 3,
    )
    agent.close()
    assert client.is_closed

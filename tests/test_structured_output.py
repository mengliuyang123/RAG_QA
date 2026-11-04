# -*- coding: utf-8 -*-
"""通过真实聊天 SDK 与离线传输验证结构化输出协议。"""

import json
from typing import Any

import httpx
import pytest
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.application.errors import ConversationTimeoutError
from app.application.errors import LLMGenerationError
from app.application.errors import LLMTimeoutError
from app.domain.agent_quality import AnswerAssessment
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.infrastructure.conversations.quality import ModelQualityEvaluator
from app.infrastructure.generation.structured_output import (
    ModelStructuredOutputProvider,
)


def _provider(
    content: str,
    *,
    mode: str = "json_schema",
    finish_reason: str = "stop",
    refusal: str | None = None,
    timeout: bool = False,
) -> tuple[
    ModelStructuredOutputProvider, httpx.Client, list[dict[str, Any]],
]:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if timeout:
            raise httpx.ReadTimeout("private", request=request)
        message = {"role": "assistant", "content": content}
        if refusal is not None:
            message["refusal"] = refusal
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 1,
            "model": "offline", "choices": [{
                "index": 0, "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": 3, "completion_tokens": 4,
                "total_tokens": 7,
            },
        })

    client = httpx.Client(transport=httpx.MockTransport(handle))
    model = ChatOpenAI(
        api_key="offline", base_url="http://offline/v1", model="offline",
        http_client=client, max_retries=0, timeout=1,
    )
    return ModelStructuredOutputProvider(model, mode=mode), client, requests


@pytest.mark.parametrize("schema,payload", [
    (AnswerAssessment, {"supported": True, "complete": True, "issues": []}),
    (EvidenceAssessment, {
        "sufficient": True, "relevant_numbers": [2],
        "missing_information": [], "next_query": None,
    }),
])
def test_native_requests_have_strict_schema_and_validated_result(
    schema: type[BaseModel], payload: dict[str, Any],
) -> None:
    """只有提示词而没有 provider schema 时必须失败。

    Args:
        schema: 质量领域模型类型。
        payload: 手工定义的合法模型输出。
    """
    provider, client, requests = _provider(json.dumps(payload))
    try:
        result = provider.generate([{"role": "user", "content": "审核"}], schema)
    finally:
        client.close()
    assert result.model_dump() == payload
    assert len(requests) == 1
    response_format = requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    specification = response_format["json_schema"]
    assert specification["strict"] is True
    assert specification["schema"]["additionalProperties"] is False
    assert set(specification["schema"]["required"]) == set(payload)
    assert "tools" not in requests[0]


@pytest.mark.parametrize("content,finish_reason,refusal", [
    ('{"supported":true,"complete":true,"issues":[]}', "length", None),
    ('{"supported":true,"complete":true,"issues":[]}', "stop", "拒绝审核"),
    ('{"supported":"true","complete":true,"issues":[]}', "stop", None),
    ('{"supported":false,"supported":true,"complete":true,"issues":[]}',
     "stop", None),
    ('```json\n{"supported":true,"complete":true,"issues":[]}\n```',
     "stop", None),
    ('{"supported":true,"complete":true,"issues":[]', "stop", None),
])
def test_invalid_native_outputs_do_not_fall_back(
    content: str, finish_reason: str, refusal: str | None,
) -> None:
    """原生模式不应把非法协议当作兼容包装再次调用。

    Args:
        content: 原始响应文本。
        finish_reason: provider 的完成原因。
        refusal: provider 的拒绝标志。
    """
    provider, client, requests = _provider(
        content, finish_reason=finish_reason, refusal=refusal,
    )
    try:
        with pytest.raises(LLMGenerationError):
            provider.generate([], AnswerAssessment)
    finally:
        client.close()
    assert len(requests) == 1


@pytest.mark.parametrize("mode,wrapped", [
    ("json_object", False), ("prompt", True),
])
def test_compatibility_modes_are_explicit(mode: str, wrapped: bool) -> None:
    """兼容模式由配置选择，不是服务失败后偷偷切换。

    Args:
        mode: 显式选择的模式。
        wrapped: 是否模拟提示词模式返回代码围栏。
    """
    content = '{"supported":true,"complete":true,"issues":[]}'
    if wrapped:
        content = f"```json\n{content}\n```"
    provider, client, requests = _provider(content, mode=mode)
    try:
        assert provider.generate([], AnswerAssessment).supported is True
    finally:
        client.close()
    if mode == "json_object":
        assert requests[0]["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in requests[0]


def test_structured_timeout_does_not_trigger_another_request() -> None:
    """质量服务超时也遵守 SDK 重试限制并保留错误分类。"""
    provider, client, requests = _provider("", timeout=True)
    try:
        with pytest.raises(LLMTimeoutError):
            provider.generate([], AnswerAssessment)
    finally:
        client.close()
    assert len(requests) == 1


def test_quality_timeout_survives_evaluator_wrapping() -> None:
    """结构化调用超时必须穿透审核器，而非变成普通 Agent 错误。"""
    provider, client, requests = _provider("", timeout=True)
    evidence = EvidenceItem(
        number=1, chunk_id="doc:0", doc_id="doc", filename="a.md",
        page=None, text="证据",
    )
    evaluator = ModelQualityEvaluator(object(), structured_provider=provider)
    try:
        with pytest.raises(ConversationTimeoutError):
            evaluator.assess_answer("问题", [], "回答[1]", [evidence])
    finally:
        client.close()
    assert len(requests) == 1

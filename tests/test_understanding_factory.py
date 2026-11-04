# -*- coding: utf-8 -*-
"""理解模型独立配置、一次请求及客户端生命周期的真实SDK边界。"""

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.bootstrap.factories import configured_adapter_factories
from app.config import Settings
from app.domain.routing import QueryUnderstanding
from app.application.query.understanding import ModelUnderstandingService
from tests.test_quality_factory import _mock_model
from tests.test_query_understanding import _payload


def test_understanding_factory_uses_independent_config_and_native_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不能偷偷使用生成模型、重试额度或错误的输出上限。

    Args:
        monkeypatch: 将真实SDK传输替换为离线HTTP。
    """
    client, requests = _mock_model(monkeypatch, json.dumps(_payload()))
    settings = Settings(
        _env_file=None,
        llm_api_key="gen",
        llm_model="generator",
        understanding_api_key="understand",
        understanding_base_url="https://understanding.example/v1",
        understanding_model="understanding-small",
        understanding_max_output_tokens=71,
        understanding_timeout_seconds=2,
        # 显式配置的理解参数必须覆盖生成配置；未配置时才沿用 LLM_*。
        understanding_output_token_parameter="max_tokens",
        model_max_retries=2,
        llm_output_token_parameter="max_completion_tokens",
    )
    provider = configured_adapter_factories(settings).understanding_provider(
        settings.resolved_understanding_api_key,
        settings.resolved_understanding_base_url,
        settings.resolved_understanding_model,
        settings.understanding_temperature,
        settings.resolved_understanding_reasoning_effort,
    )
    try:
        result = provider.generate(
            [{"role": "user", "content": "JSON理解问题"}],
            QueryUnderstanding,
        )
        assert result.standalone_query == "SQLite checkpoint 如何清理？"
        assert requests[0]["model"] == "understanding-small"
        assert requests[0]["max_tokens"] == 71
        assert requests[0]["response_format"]["json_schema"]["name"] == "QueryUnderstanding"
    finally:
        provider.close()
        provider.close()
    assert client.is_closed


def test_explicit_understanding_endpoint_does_not_inherit_generator_key() -> None:
    """换端点必须配置自己的密钥，避免把生成服务凭据发给另一个地址。"""
    settings = Settings(
        _env_file=None, llm_api_key="gen",
        llm_base_url="https://generator.example/v1",
        understanding_base_url="https://understanding.example/v1",
    )
    assert settings.resolved_understanding_api_key is None
    inherited = Settings(
        _env_file=None, llm_api_key="gen",
        llm_base_url="https://generator.example/v1", llm_model="generator",
    )
    assert inherited.resolved_understanding_api_key == "gen"
    assert inherited.resolved_understanding_model == "generator"
    assert inherited.resolved_understanding_base_url == "https://generator.example/v1"


def test_understanding_sdk_does_not_retry_even_when_global_retries_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """500响应不能触发SDK隐藏重试，否则一次理解实际产生多次请求。

    Args:
        monkeypatch: 替换真实SDK的传输层。
    """
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "offline"}})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(
        "app.bootstrap.factories.ChatOpenAI",
        lambda **options: ChatOpenAI(**options, http_client=client),
    )
    settings = Settings(_env_file=None, model_max_retries=2)
    provider = configured_adapter_factories(settings).understanding_provider(
        "offline", "https://understanding.example/v1", "small", None, None,
    )
    try:
        output = ModelUnderstandingService(lambda: provider).understand(
            "原问题", [], [],
        )
        assert output == QueryUnderstanding.fallback("原问题")
        assert len(requests) == 1
    finally:
        provider.close()


def test_understanding_budget_must_reserve_positive_input_capacity() -> None:
    """非法理解预算应在配置校验时拒绝，不应等到首个请求才失败。"""
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None, understanding_context_tokens=1024,
            understanding_max_output_tokens=1024,
        )


def test_blank_optional_understanding_settings_keep_inheritance() -> None:
    """从模板复制的空配置不应禁用理解，或把空模型发给供应商。"""
    settings = Settings(
        _env_file=None,
        llm_api_key="gen",
        llm_model="generator",
        understanding_base_url=" ",
        understanding_model="",
        understanding_api_key="",
    )
    assert settings.resolved_understanding_api_key == "gen"
    assert settings.resolved_understanding_model == "generator"

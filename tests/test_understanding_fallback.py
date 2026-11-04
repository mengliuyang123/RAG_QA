# -*- coding: utf-8 -*-
"""未配置的理解模型参数一律沿用 LLM_* 等通用模型配置。"""

from app.bootstrap.factories import configured_adapter_factories
from app.bootstrap.trace_configuration import trace_configuration_version
from app.config import Settings


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "llm_model": "generator",
        "llm_reasoning_effort": "none",
        "llm_timeout_seconds": 45,
        "llm_max_output_tokens": 2048,
        "llm_output_token_parameter": "max_completion_tokens",
        "quality_output_mode": "json_object",
        "conversation_context_tokens": 16384,
        "conversation_reserved_tokens": 4096,
    }
    values.update(overrides)
    return Settings(**values)


def test_unset_understanding_options_inherit_llm_settings() -> None:
    """推理强度、超时、输出上限、参数名、协议和上下文都沿用通用配置。"""
    settings = _settings()
    assert settings.resolved_understanding_model == "generator"
    assert settings.resolved_understanding_reasoning_effort == "none"
    assert settings.resolved_understanding_timeout_seconds == 45
    assert settings.resolved_understanding_max_output_tokens == 2048
    assert settings.resolved_understanding_output_token_parameter == "max_completion_tokens"
    assert settings.resolved_understanding_output_mode == "json_object"
    assert settings.resolved_understanding_context_tokens == 16384


def test_blank_understanding_values_count_as_unset() -> None:
    """.env 中留空的项不应被当成显式配置。"""
    settings = _settings(
        understanding_reasoning_effort="", understanding_timeout_seconds="",
        understanding_max_output_tokens="", understanding_output_token_parameter="",
        understanding_context_tokens="", understanding_output_mode="",
    )
    assert settings.resolved_understanding_reasoning_effort == "none"
    assert settings.resolved_understanding_timeout_seconds == 45


def test_explicit_understanding_options_override_llm_settings() -> None:
    """显式配置的理解参数优先，不影响生成配置。"""
    settings = _settings(
        understanding_reasoning_effort="low", understanding_timeout_seconds=10,
        understanding_max_output_tokens=512, understanding_output_token_parameter="max_tokens",
        understanding_context_tokens=8192, understanding_output_mode="json_schema",
    )
    assert settings.resolved_understanding_reasoning_effort == "low"
    assert settings.resolved_understanding_timeout_seconds == 10
    assert settings.resolved_understanding_max_output_tokens == 512
    assert settings.resolved_understanding_output_token_parameter == "max_tokens"
    assert settings.resolved_understanding_context_tokens == 8192
    assert settings.resolved_understanding_output_mode == "json_schema"
    assert settings.llm_reasoning_effort == "none"


def test_factory_binding_uses_resolved_understanding_options() -> None:
    """工厂绑定的是沿用后的实际值。"""
    keywords = configured_adapter_factories(_settings()).understanding_provider.keywords
    assert keywords == {
        "timeout_seconds": 45, "max_output_tokens": 2048,
        "output_token_parameter": "max_completion_tokens", "output_mode": "json_object",
    }


def test_trace_fingerprint_records_effective_values() -> None:
    """显式写出与沿用得到相同的值时，配置指纹一致；值不同则指纹不同。"""
    inherited = trace_configuration_version(_settings(), None)
    explicit_same = trace_configuration_version(
        _settings(understanding_reasoning_effort="none", understanding_timeout_seconds=45), None,
    )
    different = trace_configuration_version(
        _settings(understanding_reasoning_effort="low"), None,
    )
    assert inherited == explicit_same
    assert inherited != different

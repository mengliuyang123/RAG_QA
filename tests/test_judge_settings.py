# -*- coding: utf-8 -*-
"""验证 JUDGE_* 配置的继承规则，以及快速通道和 Agent 通道都使用裁判配置。"""

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from app.bootstrap.factories import JudgeModelOptions
from app.bootstrap.factories import configured_adapter_factories
from app.bootstrap.trace_configuration import trace_configuration_version
from app.config import Settings


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "llm_api_key": "llm-key",
        "llm_base_url": "http://localhost:11434/v1",
        "llm_model": "small-model",
        "llm_reasoning_effort": "none",
        "llm_timeout_seconds": 60,
        "embedding_api_key": "embedding-key",
        "embedding_base_url": "http://localhost:11434/v1",
    }
    values.update(overrides)
    return Settings(**values)


def test_unset_judge_settings_fall_back_to_llm() -> None:
    """全部未配置时，裁判与生成完全一致，且不视为已配置。"""
    settings = _settings()
    assert settings.judge_configured is False
    assert settings.resolved_judge_api_key == "llm-key"
    assert settings.resolved_judge_base_url == "http://localhost:11434/v1"
    assert settings.resolved_judge_model == "small-model"
    assert settings.resolved_judge_reasoning_effort == "none"
    assert settings.resolved_judge_timeout_seconds == 60


def test_blank_judge_values_count_as_unset() -> None:
    """.env 中留空的 JUDGE_* 不应被当成独立配置。"""
    settings = _settings(
        judge_api_key=" ", judge_base_url="", judge_model="  ",
        judge_reasoning_effort="", judge_timeout_seconds="",
    )
    assert settings.judge_configured is False
    assert settings.resolved_judge_model == "small-model"


def test_judge_model_on_same_service_inherits_key_and_reasoning() -> None:
    """只换模型时沿用同一服务的凭据、推理强度和超时。"""
    settings = _settings(judge_model="strong-model")
    assert settings.judge_configured is True
    assert settings.resolved_judge_model == "strong-model"
    assert settings.resolved_judge_api_key == "llm-key"
    assert settings.resolved_judge_reasoning_effort == "none"
    assert settings.resolved_judge_timeout_seconds == 60


def test_judge_on_other_service_never_receives_llm_key() -> None:
    """另选服务地址而未给密钥时，不得把生成服务的密钥发过去。"""
    settings = _settings(judge_base_url="https://judge.example/v1", judge_model="m")
    assert settings.resolved_judge_api_key is None
    explicit = _settings(
        judge_base_url="https://judge.example/v1", judge_api_key="judge-key",
    )
    assert explicit.resolved_judge_api_key == "judge-key"


def test_judge_reasoning_and_timeout_override_llm() -> None:
    """裁判可以单独开启思考并放宽超时，不影响生成。"""
    settings = _settings(judge_reasoning_effort="low", judge_timeout_seconds=120)
    assert settings.resolved_judge_reasoning_effort == "low"
    assert settings.resolved_judge_timeout_seconds == 120
    assert settings.llm_reasoning_effort == "none"
    assert settings.llm_timeout_seconds == 60


def test_non_positive_judge_timeout_is_rejected() -> None:
    """超时必须为正数。"""
    with pytest.raises(ValueError):
        _settings(judge_timeout_seconds=0)


def test_fast_lane_quality_evaluator_uses_judge_settings(tmp_path: Path) -> None:
    """快速通道的证据判断与答案审核使用裁判配置，而不是生成配置。

    Args:
        tmp_path: 隔离存储目录。
    """
    calls: list[tuple[object, ...]] = []

    def record(*arguments: object) -> object:
        calls.append(arguments)
        return object()

    settings = _settings(
        judge_model="strong-model", judge_reasoning_effort="low",
        chroma_path=tmp_path / "chroma", docs_path=tmp_path / "docs",
        knowledge_base_db_path=tmp_path / "kb.sqlite3",
        conversation_db_path=tmp_path / "conv.sqlite3",
        agent_checkpoint_db_path=tmp_path / "ckpt.sqlite3",
        trace_db_path=tmp_path / "trace.sqlite3",
        auth_db_path=tmp_path / "auth.sqlite3",
    )
    container = build_application_container(
        settings, factories=AdapterFactories(quality_evaluator=record),
    )
    try:
        # 走容器的真实装配路径，而不是在测试里自己拼参数。
        container._get_knowledge_runner()
    finally:
        container.close()
    assert calls == [("llm-key", "http://localhost:11434/v1", "strong-model", 0.0, "low")]


def test_configured_factories_bind_judge_timeout_and_agent_options() -> None:
    """设置绑定：快速通道裁判超时，以及仅在配置裁判时传给 Agent。"""
    plain = configured_adapter_factories(_settings())
    assert plain.quality_evaluator.keywords["timeout_seconds"] == 60
    assert plain.conversation_agent.keywords["judge"] is None

    judged = configured_adapter_factories(
        _settings(judge_model="strong-model", judge_timeout_seconds=90),
    )
    assert judged.quality_evaluator.keywords["timeout_seconds"] == 90
    assert judged.conversation_agent.keywords["judge"] == JudgeModelOptions(
        api_key="llm-key", base_url="http://localhost:11434/v1",
        model="strong-model", reasoning_effort="none", timeout_seconds=90,
    )


def test_agent_uses_separate_judge_model_only_when_configured() -> None:
    """Agent 的证据判断在配置裁判后改用独立模型，关闭时一并释放。"""
    plain = configured_adapter_factories(_settings()).conversation_agent(
        "llm-key", "http://localhost:11434/v1", "small-model", 0.0, "none",
        object(), InMemorySaver(), 5, 3,
    )
    try:
        assert plain._quality._evaluator._model is plain._model
    finally:
        plain.close()

    judged = configured_adapter_factories(
        _settings(judge_model="strong-model", judge_timeout_seconds=90),
    ).conversation_agent(
        "llm-key", "http://localhost:11434/v1", "small-model", 0.0, "none",
        object(), InMemorySaver(), 5, 3,
    )
    judge_evaluator = judged._quality._evaluator
    try:
        assert judged._model.model_name == "small-model"
        assert judge_evaluator._model.model_name == "strong-model"
        assert judge_evaluator._model.root_client.timeout == 90
    finally:
        judged.close()
    assert judge_evaluator._closed is True


def test_trace_configuration_reflects_judge_model() -> None:
    """更换裁判模型必须改变追踪中的配置指纹，便于对比实验。"""
    assert trace_configuration_version(_settings(), None) != trace_configuration_version(
        _settings(judge_model="strong-model"), None,
    )

# -*- coding: utf-8 -*-
"""独立语义裁判配置、请求和响应测试。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_api_key": "generator-key",
        "llm_base_url": "https://generator.example/v1",
        "llm_model": "generator-model",
        "evaluation_judge_api_key": "judge-key",
        "evaluation_judge_base_url": "https://judge.example/v1",
        "evaluation_judge_model": "judge-model",
        "chroma_path": tmp_path / "production",
        "evaluation_chroma_path": tmp_path / "evaluation",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def test_judge_requires_complete_independent_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配置缺失或模型身份相同时必须在客户端构造前失败。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest 属性替换工具。
    """
    from evaluation import judge

    constructed: list[bool] = []
    monkeypatch.setattr(
        judge,
        "OpenAI",
        lambda **_kwargs: constructed.append(True),
    )
    missing = _settings(tmp_path, evaluation_judge_api_key=None)
    with pytest.raises(ValueError, match="独立裁判配置不完整"):
        judge.JudgeClient.from_settings(missing)

    equal = _settings(
        tmp_path,
        evaluation_judge_base_url="https://generator.example/other",
        evaluation_judge_model="generator-model",
    )
    with pytest.raises(ValueError, match="必须独立于生成模型"):
        judge.JudgeClient.from_settings(equal)

    assert constructed == []


def test_judge_sends_schema_request_and_parses_bounded_score(
    tmp_path: Path,
) -> None:
    """裁判应使用独立模型、固定温度并严格解析 JSON 分数。"""
    from evaluation.judge import JudgeClient

    client = _FakeClient(json.dumps({"score": 0.75, "reason": "基本一致"}))
    judge = JudgeClient.from_settings(
        _settings(
            tmp_path,
            evaluation_judge_reasoning_effort="none",
        ),
        client=client,
    )

    result = judge.score_semantic_correctness(
        question="什么是 RAG？",
        reference="RAG 会检索外部知识。",
        response="RAG 在生成前检索知识。",
    )

    assert result.score == 0.75
    assert result.reason == "基本一致"
    call = client.completions.calls[0]
    assert call["model"] == "judge-model"
    assert call["temperature"] == 0.0
    assert call["reasoning_effort"] == "none"
    assert call["response_format"]["type"] == "json_schema"
    messages = call["messages"]
    assert "什么是 RAG？" in messages[1]["content"]
    assert "judge-key" not in json.dumps(call, ensure_ascii=False)


def test_judge_sdk_has_explicit_timeout_and_retry_limits(tmp_path: Path) -> None:
    """评估裁判也应使用显式调用边界。

    Args:
        tmp_path: 生产和评估隔离路径。
    """
    from evaluation.judge import JudgeClient

    judge = JudgeClient.from_settings(_settings(
        tmp_path, llm_timeout_seconds=9, model_max_retries=0,
    ))
    try:
        assert judge._client.timeout == 9
        assert judge._client.max_retries == 0
    finally:
        judge._client.close()


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps({"score": 1.5, "reason": "越界"}),
        json.dumps({"score": float("nan"), "reason": "非有限"}),
        json.dumps({"score": 0.5, "reason": "x" * 501}),
    ],
)
def test_judge_rejects_invalid_response(tmp_path: Path, content: str) -> None:
    """非 JSON、越界、非有限或过长理由都应成为稳定结构错误。"""
    from evaluation.judge import JudgeClient, JudgeResponseError

    judge = JudgeClient.from_settings(
        _settings(tmp_path),
        client=_FakeClient(content),
    )

    with pytest.raises(JudgeResponseError, match="裁判响应格式无效"):
        judge.score_semantic_correctness("问题", "参考", "回答")

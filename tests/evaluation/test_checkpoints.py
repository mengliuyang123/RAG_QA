# -*- coding: utf-8 -*-
"""评估检查点、重试与进程锁测试。"""

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from app.application.errors import (
    EmbeddingAuthenticationError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    status: str
    value: int


def test_checkpoint_append_is_immediately_durable(tmp_path: Path) -> None:
    """追加返回后，新文件句柄应立即读到完整 JSON 行。"""
    from evaluation.checkpoints import CheckpointChannel, RunCheckpoint

    checkpoint = RunCheckpoint.create(
        tmp_path,
        "run-001",
        {"model": "generator-a", "dataset_hash": "a" * 64},
    )

    checkpoint.append(
        CheckpointChannel.LOCAL,
        _Record(key="local-001:vector", status="success", value=1),
    )

    lines = (checkpoint.run_path / "local.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(line) for line in lines] == [
        {"key": "local-001:vector", "status": "success", "value": 1}
    ]


def test_checkpoint_rejects_duplicate_success_key(tmp_path: Path) -> None:
    """同一成功 key 不得重复追加并污染恢复语义。"""
    from evaluation.checkpoints import CheckpointChannel, RunCheckpoint

    checkpoint = RunCheckpoint.create(tmp_path, "run-001", {"model": "a"})
    record = _Record(key="local-001:vector", status="success", value=1)
    checkpoint.append(CheckpointChannel.LOCAL, record)

    with pytest.raises(ValueError, match="成功 key 重复"):
        checkpoint.append(CheckpointChannel.LOCAL, record)


def test_checkpoint_resume_requires_exact_manifest(tmp_path: Path) -> None:
    """模型或输入哈希变化时必须拒绝恢复旧记录。"""
    from evaluation.checkpoints import RunCheckpoint

    RunCheckpoint.create(
        tmp_path,
        "run-001",
        {"model": "generator-a", "dataset_hash": "a" * 64},
    )

    resumed = RunCheckpoint.resume(
        tmp_path,
        "run-001",
        {"dataset_hash": "a" * 64, "model": "generator-a"},
    )
    assert resumed.run_id == "run-001"

    with pytest.raises(ValueError, match="运行清单不匹配"):
        RunCheckpoint.resume(
            tmp_path,
            "run-001",
            {"model": "generator-b", "dataset_hash": "a" * 64},
        )


def test_checkpoint_resume_loads_latest_success_records(
    tmp_path: Path,
) -> None:
    """恢复时应返回每个成功 key 的完整最新记录。"""
    from evaluation.checkpoints import CheckpointChannel, RunCheckpoint

    checkpoint = RunCheckpoint.create(tmp_path, "run-001", {"model": "a"})
    checkpoint.append(
        CheckpointChannel.LOCAL,
        _Record(key="local-001:vector", status="failed", value=1),
    )
    checkpoint.append(
        CheckpointChannel.LOCAL,
        _Record(key="local-001:vector", status="success", value=2),
    )

    resumed = RunCheckpoint.resume(tmp_path, "run-001", {"model": "a"})

    assert resumed.records(CheckpointChannel.LOCAL) == {
        "local-001:vector": {
            "key": "local-001:vector",
            "status": "success",
            "value": 2,
        }
    }


def test_retry_uses_exponential_delays_for_transient_errors() -> None:
    """超时与限流最多尝试三次，并使用 1、2 秒退避。"""
    from evaluation.checkpoints import RetryPolicy, run_with_retry

    delays: list[float] = []
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LLMTimeoutError("private timeout")
        if attempts == 2:
            raise LLMRateLimitError("private rate response")
        return "ok"

    result = run_with_retry(
        operation,
        RetryPolicy(sleeper=delays.append),
    )

    assert result == "ok"
    assert attempts == 3
    assert delays == [1.0, 2.0]


def test_retry_does_not_repeat_authentication_or_schema_errors() -> None:
    """永久错误只运行一次，且分类结果不包含原始异常正文。"""
    from evaluation.checkpoints import (
        InvalidRemoteResponseError,
        RetryPolicy,
        StableErrorCategory,
        classify_remote_error,
        run_with_retry,
    )

    errors = [
        (
            LLMAuthenticationError("secret upstream body"),
            StableErrorCategory.AUTHENTICATION,
        ),
        (
            InvalidRemoteResponseError("secret invalid payload"),
            StableErrorCategory.INVALID_RESPONSE,
        ),
    ]
    for error, expected_category in errors:
        attempts = 0

        def operation() -> None:
            nonlocal attempts
            attempts += 1
            raise error

        with pytest.raises(type(error)):
            run_with_retry(operation, RetryPolicy(sleeper=lambda _delay: None))
        assert attempts == 1
        assert classify_remote_error(error) is expected_category
        assert "secret" not in classify_remote_error(error).value


def test_embedding_errors_use_stable_retry_categories() -> None:
    """向量服务错误应与生成服务共享稳定的重试分类。"""
    from evaluation.checkpoints import (
        StableErrorCategory,
        classify_remote_error,
    )

    cases = [
        (EmbeddingTimeoutError("secret"), StableErrorCategory.TIMEOUT),
        (EmbeddingRateLimitError("secret"), StableErrorCategory.RATE_LIMIT),
        (
            EmbeddingAuthenticationError("secret"),
            StableErrorCategory.AUTHENTICATION,
        ),
        (
            EmbeddingResponseError("secret"),
            StableErrorCategory.INVALID_RESPONSE,
        ),
    ]
    for error, expected_category in cases:
        assert classify_remote_error(error) is expected_category


def test_run_lock_rejects_active_pid(tmp_path: Path) -> None:
    """仍存活进程持有的锁不得被覆盖。"""
    from evaluation.checkpoints import EvaluationRunLock

    lock_path = tmp_path / "run.lock"
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": "2026-08-20T00:00:00Z",
                "run_id": "existing",
                "manifest_hash": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="已有评估进程持有锁"):
        EvaluationRunLock.acquire(lock_path, resume=True)


def test_stale_run_lock_requires_resume_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失效 PID 只有显式恢复时可替换，并在退出时清理。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest 属性替换工具。
    """
    from evaluation import checkpoints

    lock_path = tmp_path / "run.lock"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 999999,
                "started_at": "2026-08-20T00:00:00Z",
                "run_id": "stale",
                "manifest_hash": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(checkpoints, "_process_exists", lambda _pid: False)

    with pytest.raises(RuntimeError, match="必须显式使用 resume"):
        checkpoints.EvaluationRunLock.acquire(lock_path, resume=False)

    with checkpoints.EvaluationRunLock.acquire(
        lock_path,
        resume=True,
        run_id="resumed",
        manifest_hash="b" * 64,
    ):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["run_id"] == "resumed"

    assert not lock_path.exists()

# -*- coding: utf-8 -*-
"""真实SQLite追踪应保存每步、真实usage、失败状态和隐私边界。"""

import hashlib
from pathlib import Path
from uuid import uuid4
from datetime import UTC
from datetime import datetime
from datetime import timedelta
import sqlite3

import pytest

from app.application.execution_control import StreamCancelled
from app.application.errors import TraceStorageError
from app.application.errors import TraceNotFoundError
from app.application.tracing import TraceService
from app.application.tracing import provider_call
from app.application.tracing import traced_step
from app.domain.tracing import TokenUsage
from app.infrastructure.persistence.traces import SQLiteTraceRepository


def test_trace_roundtrip_records_steps_and_redacts_content_by_default(
    tmp_path: Path,
) -> None:
    """没有生产接入也先验证真实存储，不能只断言替身收到了save。

    Args:
        tmp_path: 隔离数据库目录。
    """
    path = tmp_path / "traces.sqlite3"
    repository = SQLiteTraceRepository(path)
    service = TraceService(repository)
    question = "私密的用户问题"
    with service.start(question, request_id=uuid4(), entry="query") as trace:
        with traced_step("retrieval", {
            "query": question, "kb_ids": ["ops"], "api_key": "do-not-store",
            "history": [{"user_message": "历史正文不应复制"}],
        }) as step:
            step.set_output({
                "results": [{"chunk_id": "d:2", "score": 0.9}],
                "next_query": "敏感的补查问题",
            })
        with provider_call("embedding", "https://model.example/v1?key=private", "embed") as call:
            call.usage = TokenUsage(input=12, output=0)
    restored = SQLiteTraceRepository(path).get(trace.id)
    assert restored.status == "completed"
    assert restored.question is None
    assert restored.question_sha256 == hashlib.sha256(question.encode()).hexdigest()
    assert restored.steps[0].input["query_sha256"] == restored.question_sha256
    assert restored.steps[0].type == "retrieval"
    assert restored.steps[0].output["results"][0]["chunk_id"] == "d:2"
    assert restored.calls[0].origin == "https://model.example"
    assert restored.calls[0].usage.input == 12
    assert restored.calls[0].cost.status == "unknown"
    stored = restored.model_dump_json()
    for secret in (question, "do-not-store", "历史正文不应复制", "key=private", "敏感的补查问题"):
        assert secret not in stored


def test_capture_queries_is_explicit_but_never_captures_credentials(
    tmp_path: Path,
) -> None:
    """显式开启内容仅允许查询，不得顺便存下SDK参数或凭据。

    Args:
        tmp_path: 隔离数据库目录。
    """
    repository = SQLiteTraceRepository(tmp_path / "trace.sqlite3")
    service = TraceService(repository, capture_queries=True)
    with service.start("原问题", request_id=uuid4(), entry="query") as trace:
        with traced_step("understanding", {
            "query": "独立查询", "authorization": "private-token",
            "messages": ["完整提示词"],
        }):
            pass
    restored = repository.get(trace.id)
    assert restored.question == "原问题"
    assert restored.steps[0].input["query"] == "独立查询"
    assert "private-token" not in restored.model_dump_json()
    assert "完整提示词" not in restored.model_dump_json()


@pytest.mark.parametrize("error,status", [
    (ValueError("sensitive detail"), "error"),
    (StreamCancelled(), "cancelled"),
])
def test_failed_and_cancelled_steps_are_persisted_without_error_text(
    tmp_path: Path, error: BaseException, status: str,
) -> None:
    """异常不能丢掉执行记录，取消也不能伪装成成功。

    Args:
        tmp_path: 隔离数据库目录。
        error: 执行中出现的异常或取消。
        status: 人工预期状态。
    """
    repository = SQLiteTraceRepository(tmp_path / "trace.sqlite3")
    service = TraceService(repository)
    with pytest.raises(type(error)):
        with service.start("原问题", request_id=uuid4(), entry="query") as trace:
            with traced_step("generation"):
                raise error
    restored = repository.get(trace.id)
    assert restored.status == status
    assert restored.steps[0].status == status
    assert "sensitive detail" not in restored.model_dump_json()


def test_trace_context_is_reset_after_request(tmp_path: Path) -> None:
    """后续无追踪的调用不能串进上一条记录。

    Args:
        tmp_path: 隔离数据库目录。
    """
    repository = SQLiteTraceRepository(tmp_path / "trace.sqlite3")
    service = TraceService(repository)
    with service.start("原问题", request_id=uuid4(), entry="query") as trace:
        with traced_step("routing"):
            pass
    with traced_step("generation"):
        pass
    with provider_call("generation", "https://other.example", "other"):
        pass
    assert len(repository.get(trace.id).steps) == 1
    assert repository.get(trace.id).calls == ()


@pytest.mark.parametrize("error,status", [
    (RuntimeError("private model error"), "failed"),
    (StreamCancelled(), "cancelled"),
])
def test_failed_execution_cannot_leave_commit_pending(
    tmp_path: Path, error: BaseException, status: str,
) -> None:
    """执行已终止时不能让后台误认为还在等待正式提交。

    Args:
        tmp_path: 隔离追踪库。
        error: 业务错误或协作取消。
        status: 正式轮次没有提交的原因。
    """
    repository = SQLiteTraceRepository(tmp_path / "trace.sqlite3")
    service = TraceService(repository)
    with pytest.raises(type(error)):
        with service.start("question", request_id=uuid4(), entry="conversation", conversation_id=uuid4()) as trace:
            raise error
    snapshot = repository.get(trace.id)
    assert snapshot.commit_status == status
    assert snapshot.turn_id is None


def test_failed_trace_write_does_not_replace_original_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """取消状态写入故障不能变成普通错误并触发故障回退。

    Args:
        tmp_path: 隔离数据库目录。
        monkeypatch: 在真实存储边界模拟一次写入故障。
    """
    repository = SQLiteTraceRepository(tmp_path / "trace.sqlite3")
    original_save = repository.save

    def save(snapshot) -> None:
        if snapshot.steps and snapshot.steps[-1].status == "cancelled":
            raise TraceStorageError("private store detail")
        original_save(snapshot)

    monkeypatch.setattr(repository, "save", save)
    service = TraceService(repository)
    with pytest.raises(StreamCancelled):
        with service.start("问题", request_id=uuid4(), entry="query") as trace:
            with traced_step("generation"):
                raise StreamCancelled()
    assert repository.get(trace.id).status == "cancelled"


def test_expired_trace_is_hidden_and_explicit_pruning_deletes_only_expired_rows(
    tmp_path: Path,
) -> None:
    """到期控制不是只写一个字段，清理也不能误删仍在保留期的追踪。

    Args:
        tmp_path: 隔离SQLite目录。
    """
    now = datetime(2026, 10, 23, tzinfo=UTC)
    path = tmp_path / "trace.sqlite3"
    repository = SQLiteTraceRepository(path, clock=lambda: now)
    service = TraceService(repository, retention_days=1, wall_clock=lambda: now)
    with service.start("第一条", request_id=uuid4(), entry="query") as old:
        pass
    now += timedelta(hours=23)
    with service.start("第二条", request_id=uuid4(), entry="query") as fresh:
        pass
    now += timedelta(hours=2)
    with pytest.raises(TraceNotFoundError):
        repository.get(old.id)
    assert repository.get(fresh.id).status == "completed"
    assert repository.prune_expired() == 1
    assert repository.prune_expired() == 0
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT trace_id FROM traces").fetchall()
    assert rows == [(str(fresh.id),)]

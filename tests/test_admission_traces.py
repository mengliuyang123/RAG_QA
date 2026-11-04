# -*- coding: utf-8 -*-
"""准入失败与等待必须有真实追踪，不能被统计成知识拒答。"""

from pathlib import Path
from uuid import UUID
from uuid import uuid4

import anyio
import httpx
import pytest

from app.domain.trace_queries import TraceFilters
from app.application.execution_control import StreamCancelled
from app.application.execution_control import cancellation_scope
from app.application.tracing.queries import TraceQueryService
from app.application.tracing.service import TraceService
from app.application.tracing.admission import admitted_question
from app.application.admission import QuestionAdmissionController
from app.infrastructure.persistence.traces import SQLiteTraceRepository
from tests.http_auth import TEST_USER_ID
from tests.test_unified_query_api import _app


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("rejection", ["rate_limited", "queue_full", "queue_timeout"])
async def test_admission_rejection_has_one_private_zero_call_trace(
    tmp_path: Path, conversation: bool, streaming: bool, rejection: str,
) -> None:
    """漏掉准入追踪或误标知识拒答都会使真实HTTP的持久结果不符。

    Args:
        tmp_path: 临时真实存储。
        conversation: 两个入口。
        streaming: 两种传输方式。
        rejection: 实际控制器触发的拒绝原因。

    Returns:
        None。
    """
    app, _, quality = _app(tmp_path, settings_overrides={
        "question_max_active": 1,
        "question_max_waiting": 0 if rejection == "queue_full" else 1,
        "question_queue_timeout_seconds": 0.02,
    })
    root = app.state.container
    slot = None
    try:
        if rejection == "rate_limited":
            for _ in range(10):
                with root.question_admission.acquire(TEST_USER_ID):
                    pass
        else:
            slot = root.question_admission.acquire(uuid4())
            slot.__enter__()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            path = "/api/query"
            conversation_id = None
            if conversation:
                created = (await client.post("/api/conversations")).json()
                conversation_id = created["id"]
                path = f"/api/conversations/{conversation_id}/query"
            if streaming:
                path += "/stream"
            response = await client.post(path, json={"question": "私密问题"})
            expected = 429 if rejection == "rate_limited" else 503
            if streaming and rejection == "queue_timeout":
                expected = 200
            assert response.status_code == expected
            page = root.trace_query_service.list(TraceFilters())
            assert len(page.traces) == 1
            summary = page.traces[0]
            trace = root.trace_query_service.get(summary.trace_id)
            assert trace.admission_status == rejection
            assert summary.admission_status == rejection
            assert trace.status == "error"
            assert trace.outcome is None
            assert trace.lane is None
            assert trace.calls == ()
            assert trace.model_calls == 0
            assert trace.cost.unknown_calls == 0
            assert trace.cost.estimates == ()
            assert trace.tokens.input == 0
            assert trace.execution_duration_ms == 0
            assert trace.queue_wait_ms >= (10 if rejection == "queue_timeout" else 0)
            assert trace.duration_ms >= trace.queue_wait_ms
            assert [step.type for step in trace.steps] == ["admission"]
            assert trace.steps[0].output["status"] == rejection
            assert "私密问题" not in trace.model_dump_json()
            assert (trace.expires_at - trace.created_at).days == 30
            assert trace.user_id == TEST_USER_ID
            if conversation_id is not None:
                assert trace.conversation_id == UUID(conversation_id)
                assert trace.commit_status == "failed"
                detail = await client.get(f"/api/conversations/{conversation_id}")
                assert detail.json()["turns"] == []
            assert quality.evidence_calls == 0
    finally:
        if slot is not None:
            slot.__exit__(None, None, None)
        root.close()


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
async def test_admitted_answer_reuses_trace_and_separates_execution_time(
    tmp_path: Path, conversation: bool,
) -> None:
    """重复新建模型追踪会产生两条记录，或回答引用的不是准入追踪。

    Args:
        tmp_path: 临时真实存储。
        conversation: 两个入口。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            path = "/api/query"
            if conversation:
                created = (await client.post("/api/conversations")).json()
                path = f"/api/conversations/{created['id']}/query"
            response = await client.post(path, json={"question": "needle"})
            assert response.status_code == 200
            page = root.trace_query_service.list(TraceFilters())
            assert len(page.traces) == 1
            assert str(page.traces[0].trace_id) == response.json()["trace_id"]
            trace = root.trace_query_service.get(page.traces[0].trace_id)
            assert trace.admission_status == "admitted"
            assert trace.queue_wait_ms == 0
            assert trace.execution_duration_ms > 0
            assert trace.execution_duration_ms <= trace.duration_ms
            assert [step.type for step in trace.steps[:2]] == ["admission", "execution"]
            assert trace.steps[0].output["queued"] is False
            assert trace.commit_status == ("committed" if conversation else "not_applicable")
    finally:
        root.close()


@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_queued_execution_has_one_trace_and_distinct_wait_time(
    tmp_path: Path, conversation: bool, cancelled: bool,
) -> None:
    """在真实排队回调中取消或释放容量，不能产生重复追踪或假执行时间。

    Args:
        tmp_path: 临时真实存储。
        conversation: 两个同步应用入口。
        cancelled: 在等待期间触发协作取消。

    Returns:
        None。
    """
    app, _, quality = _app(tmp_path, settings_overrides={
        "question_max_active": 1,
    })
    root = app.state.container
    slot = root.question_admission.acquire(uuid4())
    slot.__enter__()
    released = False
    stop = False

    def check() -> None:
        if stop:
            raise StreamCancelled()

    def queued(phase: str, retrieval_calls: int | None = None) -> None:
        nonlocal released, stop
        if phase != "queued":
            return
        running = root.trace_query_service.list(TraceFilters())
        assert len(running.traces) == 1
        assert running.traces[0].admission_status == "queued"
        assert running.traces[0].status == "running"
        assert running.traces[0].queue_wait_ms is None
        if cancelled:
            stop = True
        else:
            slot.__exit__(None, None, None)
            released = True

    try:
        with cancellation_scope(check):
            if conversation:
                created = root.conversation_service.create(owner_id=TEST_USER_ID)
                stream = root.conversation_service.stream_query(
                    created.id, "needle", owner_id=TEST_USER_ID,
                )
                first = next(stream)
                assert first.data.phase == "queued"
                if cancelled:
                    stream.close()
                    # LiveEventStream关闭只发出协作取消；等待生产线程持久化。
                    assert quality.evidence_calls == 0
                else:
                    slot.__exit__(None, None, None)
                    released = True
                    list(stream)
            elif cancelled:
                with pytest.raises(StreamCancelled):
                    root.query_service.query(
                        "needle", user_id=TEST_USER_ID, on_status=queued,
                    )
            else:
                root.query_service.query(
                    "needle", user_id=TEST_USER_ID, on_status=queued,
                )
        if conversation and cancelled:
            # 直接读trace可能正处于最终保存；用有限轮询等待终态。
            anyio.run(_wait_cancelled, root.trace_query_service)
        page = root.trace_query_service.list(TraceFilters())
        assert len(page.traces) == 1
        trace = root.trace_query_service.get(page.traces[0].trace_id)
        assert trace.queue_wait_ms > 0
        assert trace.duration_ms >= trace.queue_wait_ms
        if cancelled:
            assert trace.status == "cancelled"
            assert trace.admission_status == "cancelled"
            assert trace.execution_duration_ms == 0
            assert trace.outcome is None
            assert trace.calls == ()
            assert [step.type for step in trace.steps] == ["admission"]
        else:
            assert trace.status == "completed"
            assert trace.admission_status == "admitted"
            assert trace.execution_duration_ms > 0
            assert trace.duration_ms >= trace.execution_duration_ms + trace.queue_wait_ms
            assert trace.steps[0].output["queued"] is True
            assert trace.outcome == "answered"
    finally:
        if not released:
            slot.__exit__(None, None, None)
        root.close()


async def _wait_cancelled(service: TraceQueryService) -> None:
    with anyio.fail_after(1):
        while True:
            page = service.list(TraceFilters())
            if page.traces and page.traces[0].status == "cancelled":
                return
            await anyio.sleep(0.01)


def test_cancellation_after_admission_keeps_execution_measurement(tmp_path: Path) -> None:
    """模型区间取消不能改写成等待取消并伪造零执行时间。

    Args:
        tmp_path: 真实临时SQLite。

    Returns:
        None。
    """
    repository = SQLiteTraceRepository(tmp_path / "traces.sqlite3")
    traces = TraceService(repository)
    gate = QuestionAdmissionController(max_active=1, max_waiting=0)
    execution_id = uuid4()
    with pytest.raises(StreamCancelled):
        with admitted_question(
            gate, traces, question="私密问题", execution_id=execution_id,
            entry="query", user_id=TEST_USER_ID,
        ):
            trace = traces.claim_execution_trace(
                "私密问题", execution_id=execution_id, entry="query",
                user_id=TEST_USER_ID, lane_policy="online",
            )
            assert trace is not None
            raise StreamCancelled()
    restored = repository.get(trace.id)
    assert restored.status == "cancelled"
    assert restored.admission_status == "admitted"
    assert restored.queue_wait_ms == 0
    assert restored.execution_duration_ms > 0
    assert [step.status for step in restored.steps] == ["completed", "cancelled"]
    with gate.acquire(uuid4()):
        pass


@pytest.mark.parametrize("mismatch", ["owner", "question", "execution", "entry", "policy", "service"])
def test_prepared_trace_is_bound_to_real_execution_identity(
    tmp_path: Path, mismatch: str,
) -> None:
    """任意外层trace或跨用户上下文都不能被编排器认领。

    Args:
        tmp_path: 真实临时SQLite。
        mismatch: 错误身份或策略维度。

    Returns:
        None。
    """
    repository = SQLiteTraceRepository(tmp_path / "traces.sqlite3")
    traces = TraceService(repository)
    execution_id = uuid4()
    with traces.start_execution(
        "问题", execution_id=execution_id, entry="query", user_id=TEST_USER_ID,
    ) as prepared:
        claimant = TraceService(repository) if mismatch == "service" else traces
        wrong = claimant.claim_execution_trace(
            "其他问题" if mismatch == "question" else "问题",
            execution_id=uuid4() if mismatch == "execution" else execution_id,
            entry="conversation" if mismatch == "entry" else "query",
            user_id=uuid4() if mismatch == "owner" else TEST_USER_ID,
            lane_policy="fast_only" if mismatch == "policy" else "online",
        )
        assert wrong is None
        correct = traces.claim_execution_trace(
            "问题", execution_id=execution_id, entry="query",
            user_id=TEST_USER_ID, lane_policy="online",
        )
        assert correct is prepared
        assert traces.claim_execution_trace(
            "问题", execution_id=execution_id, entry="query",
            user_id=TEST_USER_ID, lane_policy="online",
        ) is None
    assert traces.claim_execution_trace(
        "问题", execution_id=execution_id, entry="query",
        user_id=TEST_USER_ID, lane_policy="online",
    ) is None


def test_internal_traces_do_not_invent_legacy_admission_timings(tmp_path: Path) -> None:
    """未经过线上准入的追踪不能把没有测量解释为零等待。

    Args:
        tmp_path: 临时真实SQLite。

    Returns:
        None。
    """
    repository = SQLiteTraceRepository(tmp_path / "traces.sqlite3")
    traces = TraceService(repository)
    with traces.start("问题", request_id=uuid4(), entry="evaluation") as trace:
        pass
    restored = repository.get(trace.id)
    assert restored.admission_status == "not_recorded"
    assert restored.queue_wait_ms is None
    assert restored.execution_duration_ms is None

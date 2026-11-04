# -*- coding: utf-8 -*-
"""用量统计以真实追踪为依据，缺失不补零，跨库不重复计算总量。"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import UUID
from uuid import uuid4
from decimal import Decimal

import httpx
import pytest

from app.application.tracing import TraceService
from app.application.tracing import provider_call
from app.application.tracing import traced_step
from app.domain.tracing import TokenUsage
from app.domain.tracing import ModelPrice
from tests.test_unified_query_api import _app


def _seed(
    repository: object, when: datetime, user_id: UUID,
    kb_ids: tuple[str, ...], usage: TokenUsage,
) -> None:
    with TraceService(repository, wall_clock=lambda: when).start(
        "private question", request_id=uuid4(), entry="query", user_id=user_id,
    ) as trace:
        if kb_ids:
            with traced_step("retrieval", {"kb_ids": list(kb_ids)}):
                pass
        with provider_call("generation", "https://model.example", "offline") as call:
            call.usage = usage
        trace.set_result(lane="fast", outcome="answered", scope_expanded=False, escalated=False)


@pytest.mark.anyio
async def test_usage_groups_retained_traces_and_preserves_unknowns(tmp_path: Path) -> None:
    """错误的时间范围、未知补零、用户合并或跨库总量重复都会使本测试失败。

    Args:
        tmp_path: 隔离追踪和业务存储。
    """
    app, _, _ = _app(tmp_path)
    try:
        repository = app.state.container.trace_repository
        now = datetime.now(UTC)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        first = uuid4()
        second = uuid4()
        _seed(repository, today, first, ("ops",), TokenUsage(input=10, output=2))
        _seed(repository, today - timedelta(days=1), second, ("dev", "ops"), TokenUsage(input=20, output=3))
        _seed(repository, today, first, (), TokenUsage())
        _seed(repository, today - timedelta(days=31), first, ("ops",), TokenUsage(input=999, output=999))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/admin/usage", params={"days": 7, "group_by": "day"})
            assert response.status_code == 200
            body = response.json()
            assert body["totals"]["questions"] == 3
            assert body["totals"]["llm_calls"] == 3
            assert body["totals"]["tokens"]["input"] is None
            assert body["totals"]["tokens"]["known_input"] == 30
            assert body["totals"]["tokens"]["known_output"] == 5
            assert body["totals"]["tokens"]["unknown_input_calls"] == 1
            assert body["totals"]["cost"]["unknown_calls"] == 3
            rows = {row["key"]: row for row in body["rows"]}
            assert rows[today.date().isoformat()]["questions"] == 2
            assert rows[(today - timedelta(days=1)).date().isoformat()]["questions"] == 1
            assert body["rows_additive"] is True
            assert response.headers["Cache-Control"] == "no-store"

            users = (await client.get("/api/admin/usage", params={"group_by": "user"})).json()
            assert sorted(row["questions"] for row in users["rows"]) == [1, 2]
            assert str(first) not in str(users)
            assert str(second) not in str(users)
            assert "private question" not in str(users)

            by_kb = (await client.get("/api/admin/usage", params={"group_by": "kb"})).json()
            rows = {row["key"]: row for row in by_kb["rows"]}
            assert rows["ops"]["questions"] == 2
            assert rows["dev"]["questions"] == 1
            assert rows["no-retrieval:"]["questions"] == 1
            assert by_kb["totals"]["questions"] == 3
            assert by_kb["rows_additive"] is False
            assert by_kb["attribution"] == "participating_scope"

            retained = (await client.get("/api/admin/usage", params={"days": 60})).json()
            assert retained["totals"]["questions"] == 3
            assert retained["retention_limited"] is True
    finally:
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("params", [{"days": 0}, {"days": 366}, {"group_by": "strategy"}])
async def test_usage_rejects_invalid_window_or_group(tmp_path: Path, params: dict[str, object]) -> None:
    """不能用任意长范围或非法分组触发无条件全量查询。

    Args:
        tmp_path: 隔离数据。
        params: 非法输入。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/api/admin/usage", params=params)).status_code == 422
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_usage_keeps_configuration_and_price_versions_separate(tmp_path: Path) -> None:
    """配置或价格切换后的统计必须可分辨，金额不能丢失版本。

    Args:
        tmp_path: 隔离全部数据。
    """
    app, _, _ = _app(tmp_path)
    try:
        price = ModelPrice(
            origin="https://model.example", model="offline", version="manual-test-v1",
            currency="USD", input_per_million=Decimal(1), output_per_million=Decimal(2),
        )
        for version in ("config-a", "config-b"):
            service = TraceService(
                app.state.container.trace_repository,
                configuration_version=version, prices=(price,),
            )
            with service.start("question", request_id=uuid4(), entry="query"):
                with provider_call("generation", "https://model.example", "offline") as call:
                    call.usage = TokenUsage(input=10, output=2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/admin/usage")
            assert response.status_code == 200
            body = response.json()
            assert {row["key"]: row["questions"] for row in body["configurations"]} == {
                "config-a": 1, "config-b": 1,
            }
            assert body["totals"]["cost"]["status"] == "estimated"
            estimate = body["totals"]["cost"]["estimates"][0]
            assert Decimal(estimate["amount"]) == Decimal("0.000028")
            assert estimate["pricing_version"] == "manual-test-v1"
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_usage_reads_beyond_one_batch_and_does_not_merge_reserved_kb_keys(tmp_path: Path) -> None:
    """只读前100条或把保留分组当作合法知识库会使计数错误。

    Args:
        tmp_path: 隔离数据。
    """
    app, _, _ = _app(tmp_path)
    try:
        user_id = uuid4()
        service = app.state.container.trace_service
        for _ in range(105):
            with service.start("question", request_id=uuid4(), entry="query", user_id=user_id):
                pass
        _seed(app.state.container.trace_repository, datetime.now(UTC), user_id, ("unretrieved",), TokenUsage(input=1, output=0))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/admin/usage", params={"group_by": "kb"})
            assert response.status_code == 200
            body = response.json()
            assert body["totals"]["questions"] == 106
            rows = {row["key"]: row for row in body["rows"]}
            assert rows["unretrieved"]["questions"] == 1
            assert rows["no-retrieval:"]["questions"] == 105
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_empty_usage_is_zero_but_unreadable_storage_is_not(tmp_path: Path) -> None:
    """空数据库可返回零，数据库不可读时必须503而不能伪装成零用量。

    Args:
        tmp_path: 隔离数据。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            body = (await client.get("/api/admin/usage")).json()
            assert body["rows"] == []
            assert body["totals"]["questions"] == 0
            assert body["totals"]["tokens"]["input"] == 0
            assert body["totals"]["cost"]["unknown_calls"] == 0
    finally:
        app.state.container.close()
    broken, _, _ = _app(tmp_path / "broken")
    try:
        broken.state.container.settings.conversation_db_path.mkdir()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=broken), base_url="http://test") as client:
            response = await client.get("/api/admin/usage")
            assert response.status_code == 503
            assert str(tmp_path) not in response.text
    finally:
        broken.state.container.close()

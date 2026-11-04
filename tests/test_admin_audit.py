# -*- coding: utf-8 -*-
"""审计只存无正文元数据，沿用明确期限且不误删未过期记录。"""

import json
import sqlite3
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from app.application.tracing.auditing import AdminAuditService
from app.application.tracing import TraceService
from app.infrastructure.persistence.traces import SQLiteTraceRepository


def test_expired_audits_are_pruned_without_deleting_active_trace_or_audit(tmp_path: Path) -> None:
    """只写expires_at但从不清理或清理边界错误都会被本测试捕获。

    Args:
        tmp_path: 隔离追踪／审计库。
    """
    now = datetime(2026, 10, 23, tzinfo=UTC)
    path = tmp_path / "trace.sqlite3"
    repository = SQLiteTraceRepository(path, clock=lambda: now)
    one_day = AdminAuditService(repository, retention_days=1, clock=lambda: now)
    two_days = AdminAuditService(repository, retention_days=2, clock=lambda: now)
    actor = uuid4()
    old_id = uuid4()
    kept_id = uuid4()
    one_day.record(actor, "read", "feedback", (old_id,))
    two_days.record(actor, "read", "feedback", (kept_id,))
    with TraceService(repository, retention_days=2, wall_clock=lambda: now).start(
        "question", request_id=uuid4(), entry="query",
    ) as trace:
        pass
    now += timedelta(days=1)
    assert repository.prune_expired() == 0
    with sqlite3.connect(path) as connection:
        events = connection.execute("SELECT event_json FROM admin_access_audit").fetchall()
    assert len(events) == 1
    assert json.loads(events[0][0])["resource_ids"] == [str(kept_id)]
    assert repository.get(trace.id).status == "completed"

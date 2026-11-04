# -*- coding: utf-8 -*-
"""应用生命周期必须定期清理过期追踪，不删除保留期内记录。"""

import sqlite3
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from threading import Event
from uuid import uuid4

import anyio
import pytest

from app.application.tracing import TraceService
from app.bootstrap.container import build_application_container
from app.config import Settings
from app.main import create_app


@pytest.mark.anyio
async def test_lifespan_prunes_expired_traces_without_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """仅在读取时隐藏过期记录不是物理保留策略，应有后台维护。

    Args:
        tmp_path: 隔离追踪库。
        monkeypatch: 缩短维护等待，仅观察真实删除结果。
    """
    settings = Settings(
        _env_file=None, trace_db_path=tmp_path / "traces.sqlite3",
        trace_cleanup_interval_seconds=0.01,
        ingestion_worker_enabled=False,
    )
    root = build_application_container(settings)
    repository = root.trace_repository
    now = datetime.now(UTC)
    old = TraceService(repository, wall_clock=lambda: now - timedelta(days=31))
    with old.start("expired", request_id=uuid4(), entry="query") as expired:
        pass
    with root.trace_service.start("current", request_id=uuid4(), entry="query") as current:
        pass
    cleaned = Event()
    original = repository.prune_expired

    def prune() -> int:
        removed = original()
        cleaned.set()
        return removed

    monkeypatch.setattr(repository, "prune_expired", prune)
    app = create_app(settings=settings, container_factory=lambda settings: root)
    async with app.router.lifespan_context(app):
        with anyio.fail_after(1):
            while not cleaned.is_set():
                await anyio.sleep(0.005)
        with sqlite3.connect(settings.trace_db_path) as connection:
            ids = connection.execute("SELECT trace_id FROM traces").fetchall()
        assert (str(expired.id),) not in ids
        assert ids == [(str(current.id),)]

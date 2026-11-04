# -*- coding: utf-8 -*-
"""模型执行成功不等于正式轮次提交成功，两者必须可分别查询。"""

import json
import sqlite3
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from app.application.errors import ConversationStorageError
from app.application.errors import TraceStorageError
from app.bootstrap.container import build_application_container
from tests.test_unified_query_api import _app
from tests.test_unified_query_api import _events


def _last_trace(app):
    with sqlite3.connect(app.state.container.settings.trace_db_path) as connection:
        row = connection.execute(
            "SELECT snapshot_json FROM traces ORDER BY rowid DESC LIMIT 1",
        ).fetchone()
    return json.loads(row[0])


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_conversation_trace_links_only_committed_turns(tmp_path: Path, stream: bool) -> None:
    """完成响应必须对应实际业务库中的同一轮次，而不是预分配的虚构ID。

    Args:
        tmp_path: 隔离所有数据。
        stream: 两种传输外壳。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            created = (await client.post("/api/conversations")).json()
            suffix = "/stream" if stream else ""
            response = await client.post(f"/api/conversations/{created['id']}/query{suffix}", json={"question": "needle"})
            assert response.status_code == 200
            body = _events(response.text)[-1][1] if stream else response.json()
            stored = app.state.container.trace_repository.get(UUID(body["trace_id"]))
            assert stored.commit_status == "committed"
            assert str(stored.turn_id) == body["turn_id"]
            detail = (await client.get(f"/api/conversations/{created['id']}")).json()
            assert str(stored.turn_id) == detail["turns"][0]["id"]
    finally:
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_storage_failure_does_not_claim_committed_trace(
    tmp_path: Path, stream: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """业务提交失败应保留真实模型成本，但不显示为成功写入轮次。

    Args:
        tmp_path: 隔离数据。
        stream: 两种传输。
        monkeypatch: 只注入业务数据库提交故障。
    """
    app, _, _ = _app(tmp_path)
    try:
        service = app.state.container.conversation_service

        def fail(*args: object, **kwargs: object) -> None:
            raise ConversationStorageError("private storage failure")

        monkeypatch.setattr(service._repository, "save_turn", fail)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            created = (await client.post("/api/conversations")).json()
            suffix = "/stream" if stream else ""
            response = await client.post(f"/api/conversations/{created['id']}/query{suffix}", json={"question": "needle"})
            if stream:
                events = _events(response.text)
                assert events[-1][0] == "error"
                assert not any(kind == "complete" for kind, _ in events)
            else:
                assert response.status_code == 500
            trace = _last_trace(app)
            assert trace["status"] == "completed"
            assert trace["commit_status"] == "failed"
            assert trace["turn_id"] is None
            assert (await client.get(f"/api/conversations/{created['id']}")).json()["turns"] == []
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_post_commit_trace_failure_does_not_fail_saved_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个数据库无法共用事务，日志失败不能引发重试和重复正式轮次。

    Args:
        tmp_path: 隔离数据库。
        monkeypatch: 只中断提交之后的追踪状态写入。
    """
    app, _, _ = _app(tmp_path)
    try:
        def fail(*args: object, **kwargs: object) -> None:
            raise TraceStorageError("private trace disk failure")

        monkeypatch.setattr(app.state.container.trace_repository, "record_commit", fail)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            created = (await client.post("/api/conversations")).json()
            response = await client.post(f"/api/conversations/{created['id']}/query", json={"question": "needle"})
            assert response.status_code == 200
            body = response.json()
            detail = (await client.get(f"/api/conversations/{created['id']}")).json()
            assert len(detail["turns"]) == 1
            # 查询必须按业务库已提交的trace关联恢复，不信任陈旧pending快照。
            trace = await client.get(f"/api/admin/traces/{body['trace_id']}")
            assert trace.status_code == 200
            assert trace.json()["commit_status"] == "committed"
            assert trace.json()["turn_id"] == body["turn_id"]
    finally:
        app.state.container.close()


def test_cancel_during_validated_tokens_records_no_commit(tmp_path: Path) -> None:
    """模型完成后断连也不能记录正式提交或继续保存轮次。

    Args:
        tmp_path: 隔离数据。
    """
    app, _, _ = _app(tmp_path)
    try:
        service = app.state.container.conversation_service
        created = service.create()
        stream = service.stream_query(created.id, "needle")
        for event in stream:
            if event.event == "token":
                break
        stream.close()
        assert stream._finished.wait(timeout=1)
        trace = _last_trace(app)
        assert trace["status"] == "completed"
        assert trace["commit_status"] == "cancelled"
        assert trace["turn_id"] is None
        assert service.get(created.id).turns == []
    finally:
        app.state.container.close()


def test_legacy_trace_metadata_is_indexed_without_rewriting_history(tmp_path: Path) -> None:
    """新建的trace_id索引列必须恢复旧metadata中的真实关联，不重跑模型。

    Args:
        tmp_path: 隔离数据库。
    """
    app, _, _ = _app(tmp_path)
    settings = app.state.container.settings
    try:
        service = app.state.container.conversation_service
        created = service.create()
        saved = service.query(created.id, "你好")
        trace_id = UUID(saved.metadata.trace_id)
    finally:
        app.state.container.close()
    with sqlite3.connect(settings.conversation_db_path) as connection:
        before = connection.execute(
            "SELECT user_message, assistant_message, metadata_json FROM conversation_turns",
        ).fetchone()
        # 模拟已经带trace元数据、但尚未加入索引列的上个工作区版本。
        connection.execute("DROP INDEX idx_turns_trace")
        connection.execute("ALTER TABLE conversation_turns DROP COLUMN trace_id")
    root = build_application_container(settings)
    try:
        detail = root.trace_query_service.get(trace_id)
        assert detail.turn_available is True
        assert detail.turn_id == saved.turn_id
        with sqlite3.connect(settings.conversation_db_path) as connection:
            after = connection.execute(
                "SELECT user_message, assistant_message, metadata_json FROM conversation_turns",
            ).fetchone()
        assert after == before
    finally:
        root.close()

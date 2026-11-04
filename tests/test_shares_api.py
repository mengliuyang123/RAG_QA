# -*- coding: utf-8 -*-
"""单轮分享的真实鉴权、隔离、有效期和持久化契约。"""

import hashlib
import logging
import sqlite3
from contextlib import AsyncExitStack
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.application.errors import ShareExpiredError
from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from app.domain.auth import User
from app.domain.conversations import AgentResult
from app.domain.conversations import TurnCitation
from app.domain.shares import ShareRequest
from app.domain.shares import StoredShare
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository
from app.infrastructure.persistence.shares import SQLiteShareRepository
from tests.test_ingestion_job_container import _settings
from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app


async def _client(stack, app, username, role="member"):
    app.state.container.auth_service.create_user(
        username, PASSWORD, name=username, role=role,
    )
    client = await stack.enter_async_context(httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://test",
    ))
    login = await client.post("/api/auth/login", json={
        "username": username, "password": PASSWORD,
    })
    assert login.status_code == 200
    client.headers["X-CSRF-Token"] = client.cookies.get("rag_csrf")
    return client


async def _turn(client):
    conversation = (await client.post("/api/conversations")).json()
    path = f"/api/conversations/{conversation['id']}"
    response = await client.post(path + "/query", json={"question": "needle"})
    assert response.status_code == 200
    return path, response.json()


@pytest.mark.anyio
async def test_share_requires_owner_and_exposes_only_selected_turn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """缺少归属、CSRF或窄投影时泄漏其他人的历史。

    Args:
        tmp_path: 全隔离数据路径。
        caplog: 检查访问日志不泄漏分享令牌。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        async with AsyncExitStack() as stack:
            alice = await _client(stack, app, "alice")
            bob = await _client(stack, app, "bob")
            admin = await _client(stack, app, "admin", "admin")
            anonymous = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="https://test",
            ))
            path, answer = await _turn(alice)
            second = await alice.post(path + "/query", json={
                "question": "needle SECOND_PRIVATE_TURN",
            })
            assert second.status_code == 200
            endpoint = f"/api/turns/{answer['turn_id']}/shares"
            response = await alice.post(endpoint, json={"expires_in_days": 30})
            assert response.status_code == 200
            link = response.json()
            assert link["token"].startswith("shr_")
            assert len(link["token"]) >= 47
            assert link["url"] == "/s/" + link["token"]
            assert response.headers["cache-control"] == "no-store"
            for other in (bob, admin):
                denied = await other.post(endpoint, json={})
                assert denied.status_code == 404
            assert (await anonymous.post(endpoint, json={})).status_code == 401
            csrf = alice.headers.pop("X-CSRF-Token")
            assert (await alice.post(endpoint, json={})).status_code == 403
            alice.headers["X-CSRF-Token"] = csrf
            share_path = "/api/shares/" + link["token"]
            assert (await anonymous.get(share_path)).status_code == 401
            shared = await bob.get(share_path)
            assert shared.status_code == 200
            body = shared.json()
            assert set(body) == {
                "question", "answer", "citations", "shared_by", "created_at",
            }
            assert body["question"] == "needle"
            assert body["answer"] == answer["answer"]
            assert body["citations"] == answer["citations"]
            assert body["shared_by"] == "alice"
            assert "SECOND_PRIVATE_TURN" not in shared.text
            assert shared.headers["cache-control"] == "no-store"
            logging.getLogger("uvicorn.access").info(
                '%s - "%s %s HTTP/%s" %d',
                "127.0.0.1", "GET", share_path, "1.1", 200,
            )
            assert link["token"] not in caplog.text
            assert (await bob.get(path)).status_code == 404
            with sqlite3.connect(tmp_path / "conversations.sqlite3") as db:
                dump = "\n".join(db.iterdump())
            assert link["token"] not in dump
            digest = hashlib.sha256(link["token"].encode()).hexdigest()
            assert digest in dump
            assert (await alice.delete(path)).status_code == 200
            assert (await bob.get(share_path)).status_code == 404
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_share_storage_failure_is_closed_without_content(
    tmp_path: Path,
) -> None:
    """真实分享表损坏必须503，不能把故障伪装为空内容或成功。

    Args:
        tmp_path: 全部临时数据。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        async with AsyncExitStack() as stack:
            alice = await _client(stack, app, "alice")
            _, answer = await _turn(alice)
            endpoint = f"/api/turns/{answer['turn_id']}/shares"
            link = (await alice.post(endpoint, json={})).json()
            with sqlite3.connect(tmp_path / "conversations.sqlite3") as db:
                db.execute("DROP TABLE turn_shares")
            read = await alice.get("/api/shares/" + link["token"])
            write = await alice.post(endpoint, json={})
            for response in (read, write):
                assert response.status_code == 503
                assert response.json()["error"]["code"] == "share_unavailable"
                assert "needle" not in response.text
                assert "no such table" not in response.text
                assert response.headers["cache-control"] == "no-store"
    finally:
        app.state.container.close()


@pytest.mark.parametrize("explicit", [False, True])
def test_share_reopens_in_isolated_runtime_without_models(
    tmp_path: Path, explicit: bool,
) -> None:
    """丢失持久授权或组合根误用生产路径都会失败。

    Args:
        tmp_path: 生产哨兵与运行库均为临时数据。
        explicit: 是否显式覆盖业务库路径。
    """
    settings = _settings(tmp_path)
    production = SQLiteConversationRepository(settings.conversation_db_path)
    sentinel = production.create(owner_id=uuid4())
    runtime_path = None
    if explicit:
        runtime_path = tmp_path / "isolated.sqlite3"
    options = RuntimeOptions(
        chroma_path=tmp_path / "runtime", collection_name="share-test",
        conversation_db_path=runtime_path,
    )
    user = User(id=uuid4(), name="Alice")
    root = build_application_container(settings, options)
    try:
        repository = root._get_conversation_repository()
        conversation = repository.create(owner_id=user.id)
        turn = repository.save_turn(
            conversation.id, "只分享本轮", AgentResult(
                answer="事实[5]", answer_mode="retrieval", retrieval_calls=1,
                citations=[TurnCitation(
                    position=5, doc_id=str(uuid4()), filename="原文.md",
                    page=2, chunk_text="原始事实", chunk_id="opaque/5",
                    doc_updated_at=datetime(2025, 1, 2, tzinfo=UTC),
                )],
            ), 1.0,
        )
        link = root.share_service.create(turn.id, user, ShareRequest())
        other = root.share_service.create(turn.id, user, ShareRequest())
        assert other.token != link.token
    finally:
        root.close()
    reopened = build_application_container(settings, options)
    try:
        shared = reopened.share_service.read(link.token)
        assert shared.question == "只分享本轮"
        assert shared.answer == "事实[5]"
        assert shared.citations[0].position == 5
        assert shared.citations[0].doc_updated_at == datetime(2025, 1, 2, tzinfo=UTC)
        assert shared.citations[0].chunk_id == "opaque/5"
        assert shared.shared_by == "Alice"
        with sqlite3.connect(settings.conversation_db_path) as db:
            tables = db.execute(
                "SELECT name FROM sqlite_master WHERE name = 'turn_shares'",
            ).fetchall()
            assert tables == []
        assert production.get(sentinel.id).id == sentinel.id
    finally:
        reopened.close()


def test_share_expiry_is_checked_before_reading_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """到期边界不能读取聊天正文；验证真实SQL而非仓储替身。

    Args:
        tmp_path: 临时SQLite路径。
        monkeypatch: 为真实连接安装列读取保护。
    """
    path = tmp_path / "business.sqlite3"
    repository = SQLiteConversationRepository(path)
    owner = uuid4()
    turn = repository.save_turn(
        repository.create(owner_id=owner).id, "不得读取的问题",
        AgentResult(answer="问候", answer_mode="direct", retrieval_calls=0,
                    citations=[]), 1.0,
    )
    shares = SQLiteShareRepository(path)
    deadline = datetime(2026, 1, 1, tzinfo=UTC)
    shares.create(StoredShare(
        token_hash="a" * 64, turn_id=turn.id, owner_id=owner,
        shared_by="Alice", created_at=deadline - timedelta(days=1),
        expires_at=deadline,
    ))
    connect = sqlite3.connect

    def guarded(*args, **kwargs):
        connection = connect(*args, **kwargs)

        def authorize(action, table, column, database, origin):
            if action == sqlite3.SQLITE_READ:
                if column in ("user_message", "assistant_message", "chunk_text"):
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(sqlite3, "connect", guarded)
    with pytest.raises(ShareExpiredError):
        shares.read("a" * 64, deadline)


@pytest.mark.anyio
async def test_share_expiry_deleting_and_invalid_requests(tmp_path: Path) -> None:
    """过期必须失效，删除中不能读，输入不能伪造身份或无限期。

    Args:
        tmp_path: 隔离业务与认证数据库。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        async with AsyncExitStack() as stack:
            alice = await _client(stack, app, "alice")
            path, answer = await _turn(alice)
            endpoint = f"/api/turns/{answer['turn_id']}/shares"
            response = await alice.post(endpoint, json={})
            assert response.status_code == 200
            link = response.json()
            expiry = datetime.fromisoformat(link["expires_at"].replace("Z", "+00:00"))
            assert 29 < (expiry - datetime.now(UTC)).total_seconds() / 86400 <= 30
            share_path = "/api/shares/" + link["token"]
            for body in (
                {"expires_in_days": 0}, {"expires_in_days": 366},
                {"expires_in_days": True}, {"expires_in_days": "30"},
                {"owner_id": answer["turn_id"]},
            ):
                assert (await alice.post(endpoint, json=body)).status_code == 422
            with sqlite3.connect(tmp_path / "conversations.sqlite3") as db:
                db.execute("UPDATE turn_shares SET expires_at = ?", (
                    "2000-01-01T00:00:00+00:00",
                ))
            expired = await alice.get(share_path)
            assert expired.status_code == 410
            assert expired.json()["error"]["code"] == "share_expired"
            assert (await alice.get("/api/shares/shr_missing")).status_code == 404
            with sqlite3.connect(tmp_path / "conversations.sqlite3") as db:
                db.execute("UPDATE conversations SET status = 'deleting'")
            assert (await alice.get(share_path)).status_code == 404
            assert (await alice.post(endpoint, json={})).status_code == 404
    finally:
        app.state.container.close()

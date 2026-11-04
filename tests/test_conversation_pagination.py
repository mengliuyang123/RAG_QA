# -*- coding: utf-8 -*-
"""真实Cookie、SQLite和HTTP的有限会话分页与历史隔离。"""

import base64
import json
import sqlite3
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import UUID
from uuid import uuid4

import pytest

from app.domain.conversations import AgentResult
from app.domain.conversations import TurnCitation
from app.domain.feedback import TurnFeedback
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository
from tests.test_shares_api import _client
from tests.test_unified_query_api import _app


@pytest.mark.anyio
async def test_list_pages_keep_owner_total_and_tied_order(
    tmp_path: Path,
) -> None:
    """漏掉SQL分页、归属或同时间排序时，分页结果会重复或越权。

    Args:
        tmp_path: 隔离真实存储。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        async with AsyncExitStack() as stack:
            alice = await _client(stack, app, "alice")
            bob = await _client(stack, app, "bob")
            ids = []
            for _ in range(5):
                body = (await alice.post("/api/conversations")).json()
                ids.append(body["id"])
            await bob.post("/api/conversations")
            with sqlite3.connect(tmp_path / "conversations.sqlite3") as db:
                db.execute(
                    "UPDATE conversations SET updated_at = ?",
                    ("2026-10-28T01:00:00+00:00",),
                )
                db.execute(
                    "UPDATE conversations SET status = 'deleting' WHERE id = ?",
                    (ids[0],),
                )
            response = await alice.get("/api/conversations?limit=2")
            assert response.status_code == 200
            first = response.json()
            assert len(first["conversations"]) == 2
            assert first["total"] == 4
            assert [item["id"] for item in first["conversations"]] == ids[3:][::-1]
            cursor = first["next_cursor"]
            assert cursor
            second = (await alice.get("/api/conversations", params={
                "limit": 2, "cursor": cursor,
            })).json()
            assert second["total"] == 4
            assert [item["id"] for item in second["conversations"]] == ids[1:3][::-1]
            assert second["next_cursor"] is None
            denied = await bob.get("/api/conversations", params={
                "cursor": cursor,
            })
            assert denied.status_code == 422
            old = (await alice.get("/api/conversations")).json()
            assert len(old["conversations"]) == old["total"] == 4
            assert old["next_cursor"] is None
            assert response.headers["cache-control"] == "no-store"
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_turn_pages_are_bounded_and_preserve_citations_feedback(
    tmp_path: Path,
) -> None:
    """全量加载后切片会读到页外坏记录；重编号或丢反馈也必须失败。

    Args:
        tmp_path: 隔离正式轮次。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        async with AsyncExitStack() as stack:
            alice = await _client(stack, app, "alice")
            bob = await _client(stack, app, "bob")
            admin = await _client(stack, app, "admin", "admin")
            body = (await alice.post("/api/conversations")).json()
            conversation_id = UUID(body["id"])
            path = "/api/conversations/" + body["id"]
            repository = SQLiteConversationRepository(
                tmp_path / "conversations.sqlite3",
            )
            turns = []
            for index in range(5):
                turns.append(repository.save_turn(
                    conversation_id, f"question-{index}", AgentResult(
                        answer=f"answer-{index}[5]", answer_mode="retrieval",
                        retrieval_calls=1, citations=[TurnCitation(
                            position=5, doc_id=str(uuid4()), filename="a.md",
                            chunk_text=f"evidence-{index}", page=None,
                        )],
                    ), 1.0,
                ))
            owner = UUID((await alice.get("/api/me")).json()["id"])
            repository.put_feedback(turns[-1].id, owner, TurnFeedback(rating="up"))
            with sqlite3.connect(tmp_path / "conversations.sqlite3") as db:
                db.execute(
                    "UPDATE conversation_turns SET created_at = ?",
                    ("2026-10-28T01:00:00+00:00",),
                )
                db.execute(
                    "UPDATE conversation_turns SET metadata_json = 'broken' "
                    "WHERE id = ?", (str(turns[0].id),),
                )
            first = await alice.get(path, params={"turns_limit": 2})
            assert first.status_code == 200
            payload = first.json()
            assert payload["turns_count"] == 5
            assert [item["user_message"] for item in payload["turns"]] == [
                "question-3", "question-4",
            ]
            assert payload["turns"][-1]["feedback"] == "up"
            assert payload["turns"][-1]["citations"][0]["position"] == 5
            cursor = payload["turns_next_cursor"]
            assert cursor
            second = await alice.get(path, params={
                "turns_limit": 2, "turns_cursor": cursor,
            })
            assert second.status_code == 200
            assert [item["user_message"] for item in second.json()["turns"]] == [
                "question-1", "question-2",
            ]
            with sqlite3.connect(tmp_path / "conversations.sqlite3") as db:
                db.execute(
                    "UPDATE conversation_turns SET metadata_json = NULL "
                    "WHERE id = ?", (str(turns[0].id),),
                )
            last = (await alice.get(path, params={
                "turns_cursor": second.json()["turns_next_cursor"],
            })).json()
            assert [item["user_message"] for item in last["turns"]] == ["question-0"]
            assert last["turns_next_cursor"] is None
            legacy = (await alice.get(path)).json()
            assert len(legacy["turns"]) == 5
            assert legacy["turns_next_cursor"] is None
            other = (await alice.post("/api/conversations")).json()
            other_path = "/api/conversations/" + other["id"]
            assert (await alice.get(other_path, params={
                "turns_cursor": cursor,
            })).status_code == 422
            for user in (bob, admin):
                assert (await user.get(path, params={
                    "turns_cursor": cursor,
                })).status_code == 404
            empty = (await alice.get(other_path, params={"turns_limit": 2})).json()
            assert empty["turns"] == []
            assert empty["turns_next_cursor"] is None
    finally:
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": 101}, {"limit": "bad"},
    {"cursor": "not-a-cursor"}, {"cursor": ""},
])
async def test_pagination_rejects_invalid_parameters(
    tmp_path: Path, params: dict[str, object],
) -> None:
    """忽略参数或宽松游标会把错误请求伪装成全量成功。

    Args:
        tmp_path: 临时业务库。
        params: 分页非法边界。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        async with AsyncExitStack() as stack:
            alice = await _client(stack, app, "alice")
            body = (await alice.post("/api/conversations")).json()
            response = await alice.get("/api/conversations", params=params)
            assert response.status_code == 422
            detail_params = {}
            for key, value in params.items():
                detail_params["turns_" + key] = value
            detail = await alice.get(
                "/api/conversations/" + body["id"], params=detail_params,
            )
            assert detail.status_code == 422
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_cursor_integer_overflow_is_validation_error(
    tmp_path: Path,
) -> None:
    """合法JSON中的超大行号不能穿透为SQLite绑定异常。

    Args:
        tmp_path: 临时业务库。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    try:
        async with AsyncExitStack() as stack:
            alice = await _client(stack, app, "alice")
            await alice.post("/api/conversations")
            await alice.post("/api/conversations")
            page = (await alice.get("/api/conversations?limit=1")).json()
            encoded = json.loads(base64.urlsafe_b64decode(page["next_cursor"]))
            encoded["rowid"] = 2 ** 64
            cursor = base64.urlsafe_b64encode(json.dumps(encoded).encode()).decode()
            response = await alice.get("/api/conversations", params={
                "cursor": cursor,
            })
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "validation_error"
    finally:
        app.state.container.close()

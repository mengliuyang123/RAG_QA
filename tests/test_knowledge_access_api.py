# -*- coding: utf-8 -*-
"""知识库权限、库 editor、文档复审、标准答案与自助改口令。"""

from contextlib import AsyncExitStack
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.application.knowledge.access import OutdatedAwarePublication
from app.domain.knowledge_access import KbPermissions
from app.domain.knowledge_bases import KnowledgeBase
from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app


NEW_PASSWORD = "replace-me-offline-passphrase"


async def _login(stack: AsyncExitStack, app, username: str, password: str = PASSWORD) -> httpx.AsyncClient:
    client = await stack.enter_async_context(httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://test",
    ))
    login = await client.post("/api/auth/login", json={"username": username, "password": password})
    assert login.status_code == 200, login.text
    client.headers["X-CSRF-Token"] = client.cookies.get("rag_csrf")
    return client


@pytest.fixture
def secured(tmp_path: Path):
    """default 库里有 needle 文档；另建一个空的公开库 ops。"""
    app, chunk, _ = _app(
        tmp_path, legacy_identity=False, settings_overrides={"ingestion_worker_enabled": True},
    )
    container = app.state.container
    container.knowledge_base_repository.create(KnowledgeBase(id="ops", name="运维"))
    auth = container.auth_service
    admin = auth.create_user("root", PASSWORD, name="管理员", role="admin")
    alice = auth.create_user("alice", PASSWORD, name="小刘", role="member")
    try:
        yield app, chunk, admin, alice
    finally:
        container.close()


def _restrict_default(app, *members: dict) -> None:
    from app.domain.knowledge_access import KbMember
    app.state.container.knowledge_access_service.save(
        "default", restricted=True,
        members=tuple(KbMember(user_id=m["user_id"], role=m["role"]) for m in members),
    )


def test_permissions_rules() -> None:
    perms = KbPermissions(restricted=frozenset({"hr"}), member_of=frozenset({"fin"}), editable=frozenset({"fin"}))
    assert perms.can_view("ops") and not perms.can_view("hr") and perms.can_view("fin")
    assert perms.can_edit("fin") and not perms.can_edit("ops")
    assert perms.allowed(["ops", "hr", "fin"]) == frozenset({"ops", "fin"})
    assert KbPermissions(unrestricted=True).allowed(["hr"]) is None
    denied = KbPermissions(deny_all=True)
    assert not denied.can_view("ops") and not denied.can_edit("ops")


def test_outdated_documents_join_the_retrieval_exclusion_set() -> None:
    doc = uuid4()
    pending = uuid4()

    class Publication:
        def unpublished_document_ids(self):
            return (pending,)

    class Reviews:
        def outdated_doc_ids(self):
            return (str(doc), "not-a-uuid", str(pending))

    assert OutdatedAwarePublication(Publication(), Reviews()).unpublished_document_ids() == (pending, doc)


@pytest.mark.anyio
async def test_restricted_kb_is_invisible_to_non_members(secured) -> None:
    """非成员看不到受限库：目录、文档页、引用原文与问答范围都排除它。"""
    app, chunk, admin, alice = secured
    _restrict_default(app)
    async with AsyncExitStack() as stack:
        member = await _login(stack, app, "alice")
        boss = await _login(stack, app, "root")
        ids = [kb["id"] for kb in (await member.get("/api/knowledge-bases")).json()["knowledge_bases"]]
        assert "default" not in ids and "ops" in ids
        assert "default" in [kb["id"] for kb in (await boss.get("/api/knowledge-bases")).json()["knowledge_bases"]]
        assert (await member.get("/api/knowledge-bases/default/documents")).status_code == 404
        context = await member.get(f"/api/documents/{chunk.doc_id}/chunks/{chunk.id}")
        assert context.status_code == 404

        conversation = (await member.post("/api/conversations")).json()["id"]
        auto = await member.post(f"/api/conversations/{conversation}/query", json={"question": "needle"})
        assert auto.status_code == 200
        assert auto.json()["kb_scope"]["searched"] == ["ops"]
        assert auto.json()["outcome"] == "refused"
        explicit = await member.post(f"/api/conversations/{conversation}/query", json={
            "question": "needle", "kb_scope": {"mode": "explicit", "kb_ids": ["default"]},
        })
        assert explicit.status_code == 422

        # 加为 viewer 后，下一次请求即可看到并检索。
        _restrict_default(app, {"user_id": alice.id, "role": "viewer"})
        assert "default" in [kb["id"] for kb in (await member.get("/api/knowledge-bases")).json()["knowledge_bases"]]
        answered = await member.post(f"/api/conversations/{conversation}/query", json={
            "question": "needle", "kb_scope": {"mode": "explicit", "kb_ids": ["default"]},
        })
        assert answered.json()["outcome"] == "answered"
        # viewer 不能管理内容。
        assert (await member.get("/api/documents?kb_id=default")).status_code == 403


@pytest.mark.anyio
async def test_access_settings_are_admin_only(secured) -> None:
    app, _, admin, alice = secured
    async with AsyncExitStack() as stack:
        boss = await _login(stack, app, "root")
        member = await _login(stack, app, "alice")
        body = {"restricted": True, "members": [{"user_id": str(alice.id), "role": "editor"}]}
        assert (await member.put("/api/admin/knowledge-bases/default/access", json=body)).status_code == 403
        saved = await boss.put("/api/admin/knowledge-bases/default/access", json=body)
        assert saved.status_code == 200
        assert saved.json()["members"][0]["username"] == "alice"
        assert (await boss.get("/api/admin/knowledge-bases/default/access")).json()["restricted"] is True
        bad = await boss.put("/api/admin/knowledge-bases/default/access", json={
            "restricted": False, "members": [{"user_id": str(uuid4()), "role": "viewer"}],
        })
        assert bad.status_code == 422
        dup = await boss.put("/api/admin/knowledge-bases/default/access", json={
            "restricted": False, "members": [
                {"user_id": str(alice.id), "role": "viewer"}, {"user_id": str(alice.id), "role": "editor"},
            ],
        })
        assert dup.status_code == 422
        assert (await boss.get("/api/admin/knowledge-bases/nope/access")).status_code == 404


@pytest.mark.anyio
async def test_kb_editor_manages_only_their_knowledge_base(secured) -> None:
    """editor 可以管理自己的库，不能碰别的库，也不能做跨库的管理员操作。"""
    app, chunk, admin, alice = secured
    _restrict_default(app, {"user_id": alice.id, "role": "editor"})
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(app.router.lifespan_context(app))
        editor = await _login(stack, app, "alice")
        me = (await editor.get("/api/me")).json()
        assert me["role"] == "member" and me["editable_kb_ids"] == ["default"]
        boss = await _login(stack, app, "root")
        assert (await boss.get("/api/me")).json()["editable_kb_ids"] is None

        assert (await editor.patch("/api/admin/knowledge-bases/default", json={"description": "新描述"})).status_code == 200
        assert (await editor.patch("/api/admin/knowledge-bases/ops", json={"description": "x"})).status_code == 403
        assert (await editor.put("/api/admin/knowledge-bases/default/suggested-questions", json={"questions": ["needle 怎么用"]})).status_code == 200
        assert (await editor.put("/api/admin/knowledge-bases/ops/suggested-questions", json={"questions": []})).status_code == 403
        listed = await editor.get("/api/documents?kb_id=default")
        assert listed.status_code == 200 and len(listed.json()["documents"]) == 1
        assert (await editor.get("/api/documents")).status_code == 403
        assert (await editor.get("/api/documents?kb_id=ops")).status_code == 403
        upload = await editor.post("/api/admin/knowledge-bases/ops/documents", files={"file": ("a.txt", b"hello")})
        assert upload.status_code == 403
        queued = await editor.post("/api/admin/knowledge-bases/default/documents", files={"file": ("a.txt", b"hello")})
        assert queued.status_code == 202
        assert (await editor.get(f"/api/admin/ingestion-jobs/{queued.json()['job_id']}")).status_code == 200
        assert (await editor.post("/api/admin/knowledge-bases", json={"id": "new", "name": "新库"})).status_code == 403
        assert (await editor.delete("/api/admin/knowledge-bases/ops")).status_code == 403
        assert (await editor.get("/api/admin/users")).status_code == 403
        assert (await editor.get("/api/admin/knowledge-bases/default/access")).status_code == 403
        # 删除别的库的文档被拒绝；自己库的可以删。
        other = await boss.post("/api/admin/knowledge-bases/ops/documents", files={"file": ("b.txt", b"ops")})
        assert other.status_code == 202
        foreign = await editor.post("/api/documents/batch-delete", json={"doc_ids": [other.json()["doc_id"]]})
        assert foreign.status_code == 403
        own = await editor.delete(f"/api/documents/{chunk.doc_id}")
        assert own.status_code == 200


@pytest.mark.anyio
async def test_document_review_and_outdated_flag(secured) -> None:
    app, chunk, admin, alice = secured
    async with AsyncExitStack() as stack:
        boss = await _login(stack, app, "root")
        member = await _login(stack, app, "alice")
        path = f"/api/admin/knowledge-bases/default/document-reviews/{chunk.doc_id}"
        assert (await member.put(path, json={"outdated": True})).status_code == 403
        saved = await boss.put(path, json={"review_due_on": "2027-01-31", "outdated": True, "note": " 价格已调整 "})
        assert saved.status_code == 200
        assert saved.json()["outdated"] is True and saved.json()["note"] == "价格已调整"
        listed = (await boss.get("/api/admin/knowledge-bases/default/document-reviews")).json()["reviews"]
        assert listed[0]["review_due_on"] == "2027-01-31"
        missing = await boss.put(f"/api/admin/knowledge-bases/default/document-reviews/{uuid4()}", json={})
        assert missing.status_code == 404
        # 已过时的文档不再参与检索。
        conversation = (await boss.post("/api/conversations")).json()["id"]
        refused = await boss.post(f"/api/conversations/{conversation}/query", json={
            "question": "needle", "kb_scope": {"mode": "explicit", "kb_ids": ["default"]},
        })
        assert refused.json()["outcome"] == "refused"
        await boss.put(path, json={"outdated": False})
        answered = await boss.post(f"/api/conversations/{conversation}/query", json={
            "question": "needle", "kb_scope": {"mode": "explicit", "kb_ids": ["default"]},
        })
        assert answered.json()["outcome"] == "answered"


@pytest.mark.anyio
async def test_faq_entries_are_published_as_a_document(secured) -> None:
    app, _, admin, alice = secured
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(app.router.lifespan_context(app))
        boss = await _login(stack, app, "root")
        member = await _login(stack, app, "alice")
        base = "/api/admin/knowledge-bases/ops/faq"
        assert (await member.post(base, json={"question": "q", "answer": "a"})).status_code == 403
        created = await boss.post(base, json={"question": "灰度发布怎么回滚？", "answer": "执行 rollback 脚本。"})
        assert created.status_code == 201
        body = created.json()
        assert body["publication"]["status"] == "queued" and body["publication"]["job_id"]
        entry_id = body["entries"][0]["id"]
        job = await boss.get(f"/api/admin/ingestion-jobs/{body['publication']['job_id']}")
        assert job.json()["filename"] == body["filename"]

        updated = await boss.put(f"{base}/{entry_id}", json={"question": "灰度发布怎么回滚？", "answer": "先切流量再回滚。"})
        assert updated.status_code == 200
        assert updated.json()["entries"][0]["answer"] == "先切流量再回滚。"
        # 上一版仍在入库时不覆盖，提示稍后重新发布。
        assert updated.json()["publication"]["status"] in {"queued", "busy"}
        assert (await boss.put(f"{base}/{'0' * 32}", json={"question": "x", "answer": "y"})).status_code == 404

        listed = await boss.get(base)
        assert listed.json()["publication"] is None and len(listed.json()["entries"]) == 1
        deleted = await boss.delete(f"{base}/{entry_id}")
        assert deleted.status_code == 200 and deleted.json()["entries"] == []
        assert (await boss.delete(f"{base}/{entry_id}")).status_code == 404


def test_faq_markdown_keeps_each_answer_under_its_question() -> None:
    from datetime import UTC, datetime
    from app.application.knowledge.access import FaqService
    from app.domain.knowledge_access import FaqEntry
    now = datetime.now(UTC)
    text = FaqService.render([FaqEntry(
        id="a" * 32, kb_id="ops", question="怎么回滚？", answer="执行脚本。",
        created_at=now, updated_at=now, updated_by=uuid4(),
    )])
    assert "## 怎么回滚？\n\n执行脚本。" in text


@pytest.mark.anyio
async def test_user_changes_own_password_and_keeps_current_session(secured, caplog) -> None:
    app, *_ = secured
    async with AsyncExitStack() as stack:
        current = await _login(stack, app, "alice")
        other = await _login(stack, app, "alice")
        wrong = await current.post("/api/me/password", json={"current_password": "wrong-password-1234", "new_password": NEW_PASSWORD})
        assert wrong.status_code == 400 and wrong.json()["error"]["code"] == "wrong_password"
        short = await current.post("/api/me/password", json={"current_password": PASSWORD, "new_password": "short-pw"})
        assert short.status_code == 422 and "short-pw" not in short.text
        changed = await current.post("/api/me/password", json={"current_password": PASSWORD, "new_password": NEW_PASSWORD})
        assert changed.status_code == 204
        assert (await current.get("/api/me")).status_code == 200
        assert (await other.get("/api/me")).status_code == 401
        assert (await other.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})).status_code == 401
        await _login(stack, app, "alice", NEW_PASSWORD)
        no_csrf = await current.post("/api/me/password", headers={"X-CSRF-Token": ""}, json={"current_password": NEW_PASSWORD, "new_password": PASSWORD})
        assert no_csrf.status_code == 403
    assert NEW_PASSWORD not in caplog.text

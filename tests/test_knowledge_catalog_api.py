# -*- coding: utf-8 -*-
"""知识库目录可供成员读取，元数据写入必须经过真实管理员与CSRF。"""

import json
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest

from app.application.ports.structured_output import ResponseModel
from app.config import Settings
from app.main import create_app
from tests.http_auth import allow_test_identity
from tests.test_auth_service import PASSWORD
from tests.test_unified_query_api import _app
from tests.understanding import OfflineUnderstandingProvider


@pytest.mark.anyio
async def test_catalog_roles_atomic_patch_and_live_routing_metadata(tmp_path: Path) -> None:
    """绕过角色／CSRF、暴露owner或更新后仍用旧选库说明都会失败。

    Args:
        tmp_path: 全部真实存储的隔离目录。
    """
    seen = []

    class Understanding(OfflineUnderstandingProvider):
        """仅观察实际外部理解模型边界的目录数据。"""

        def generate(self, messages: list[dict[str, object]],
                     response_model: type[ResponseModel]) -> ResponseModel:
            """记录生产构造的当前目录。

            Args:
                messages: 实际理解输入。
                response_model: 严格理解schema。

            Returns:
                离线知识意图。
            """
            payload = json.loads(str(messages[-1]["content"]))
            seen.extend(payload["catalog"])
            return super().generate(messages, response_model)

    app, _, quality = _app(tmp_path, legacy_identity=False,
                          understanding_factory=lambda *args: Understanding())
    root = app.state.container
    try:
        async with AsyncExitStack() as stack:
            anonymous = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://test",
            ))
            assert (await anonymous.get("/api/knowledge-bases")).status_code == 401
            clients = []
            for role in ("member", "admin"):
                root.auth_service.create_user(role, PASSWORD, name=role, role=role)
                client = await stack.enter_async_context(httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://test",
                ))
                await client.post("/api/auth/login", json={"username": role, "password": PASSWORD})
                clients.append(client)
            member, admin = clients
            initial = await member.get("/api/knowledge-bases")
            assert initial.status_code == 200
            assert initial.headers["Cache-Control"] == "no-store"
            card = initial.json()["knowledge_bases"][0]
            assert card["id"] == "default"
            assert card["documents_count"] == 1
            assert "owner_name" not in card
            assert "guide.md" not in initial.text
            assert "chunk_text" not in initial.text
            assert quality.evidence_calls == 0
            body = {"id": "ops", "name": "运维", "description": "旧说明",
                    "icon": "server", "owner_name": "运维组"}
            member.headers["X-CSRF-Token"] = member.cookies.get("rag_csrf")
            assert (await member.post("/api/admin/knowledge-bases", json=body)).status_code == 403
            assert (await member.patch("/api/admin/knowledge-bases/default", json={"name": "越权"})).status_code == 403
            assert (await admin.post("/api/admin/knowledge-bases", json=body)).status_code == 403
            admin.headers["X-CSRF-Token"] = admin.cookies.get("rag_csrf")
            created = await admin.post("/api/admin/knowledge-bases", json=body)
            assert created.status_code == 201
            assert created.json()["owner_name"] == "运维组"
            assert created.json()["documents_count"] == 0
            conflict = await admin.post("/api/admin/knowledge-bases", json=body)
            assert conflict.status_code == 409
            assert conflict.json()["error"]["code"] == "knowledge_base_exists"
            patched = await admin.patch("/api/admin/knowledge-bases/ops", json={"description": "立即生效的新说明", "icon": None})
            assert patched.status_code == 200
            assert patched.json()["created_at"] == created.json()["created_at"]
            assert patched.json()["name"] == "运维"
            assert patched.json()["icon"] is None
            assert patched.json()["owner_name"] == "运维组"
            listed = (await member.get("/api/knowledge-bases")).json()["knowledge_bases"]
            assert [kb["id"] for kb in listed] == ["default", "ops"]
            assert listed[1]["description"] == "立即生效的新说明"
            assert "owner_name" not in listed[1]
            for invalid in ({}, {"id": "renamed"}, {"name": None}, {"name": "  "}, {"description": None}):
                assert (await admin.patch("/api/admin/knowledge-bases/ops", json=invalid)).status_code == 422
            assert (await admin.patch("/api/admin/knowledge-bases/missing", json={"name": "未知"})).status_code == 404
            assert (await admin.post("/api/query", json={"question": "needle"})).status_code == 200
            ops = [kb for kb in seen if kb["id"] == "ops"]
            assert ops[-1]["description"] == "立即生效的新说明"
    finally:
        root.close()


@pytest.mark.anyio
async def test_catalog_read_needs_no_model_credentials(tmp_path: Path) -> None:
    """目录若误构造Embedding或LLM客户端，缺凭据时会在此失败。

    Args:
        tmp_path: 真实目录和空索引的隔离路径。
    """
    settings = Settings(
        _env_file=None, openai_api_key=None, llm_api_key=None,
        embedding_api_key=None,
        reranker_base_url=None,
        http_allowed_hosts=("test",), chroma_path=tmp_path / "chroma",
        knowledge_base_db_path=tmp_path / "catalog.sqlite3",
    )
    app = create_app(settings=settings)
    allow_test_identity(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get("/api/knowledge-bases")
        assert response.status_code == 200
        assert response.json()["knowledge_bases"][0]["documents_count"] == 0
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_count_failure_is_not_empty_catalog_or_committed_metadata_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统计失败不能伪造零文档，也不能先写元数据再向管理员报失败。

    Args:
        tmp_path: 实际SQLite／Chroma隔离路径。
        monkeypatch: 只模拟底层存储读取故障。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    original_name = root.knowledge_base_repository.get("default").name

    def unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError("PRIVATE_STORAGE_DETAILS")

    monkeypatch.setattr(root.vector_store._collection, "get", unavailable)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            for method, path, body in (
                ("GET", "/api/knowledge-bases", None),
                ("POST", "/api/admin/knowledge-bases", {"id": "ops", "name": "新库"}),
                ("PATCH", "/api/admin/knowledge-bases/default", {"name": "不能提交"}),
            ):
                response = await client.request(method, path, json=body)
                assert response.status_code == 503
                assert response.json()["error"]["code"] == "knowledge_base_unavailable"
                assert "PRIVATE_STORAGE_DETAILS" not in response.text
        assert root.knowledge_base_repository.get("default").name == original_name
        assert [kb.id for kb in root.knowledge_base_repository.list()] == ["default"]
    finally:
        root.close()

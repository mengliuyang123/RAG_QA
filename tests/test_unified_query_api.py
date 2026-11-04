# -*- coding: utf-8 -*-
"""生产组合根的两个入口必须共用编排器，不以依赖替身绕过它。"""

import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.bootstrap.container import build_application_container
from app.bootstrap.factories import AdapterFactories
from app.config import Settings
from app.domain.documents import DocumentChunk
from app.domain.knowledge_bases import KnowledgeBase
from app.infrastructure.conversations.langgraph_agent import LangGraphConversationAgent
from app.main import create_app
from app.application.errors import KnowledgeBaseStorageError
from tests.test_fast_lane import _Provider
from tests.test_retriever import FakeQueryEmbedder
from tests.test_scoped_agent import _Quality
from tests.test_query_understanding import _payload
from tests.test_structured_output import _provider
from tests.understanding import OfflineUnderstandingProvider
from tests.http_auth import allow_test_identity


class _Model:
    """仅替代外部聊天生成，不替代编排器、图、持久化或HTTP。"""

    def bind_tools(self, tools: object, *, strict: bool) -> "_Model":
        """保留图工具协议。

        Args:
            tools: 标准工具定义。
            strict: 参数要求。

        Returns:
            当前离线模型。
        """
        return self

    def invoke(self, messages: list[dict[str, object]]) -> AIMessage:
        """只返回预设草稿。

        Args:
            messages: 真实图消息。

        Returns:
            带有效来源编号的完整消息。
        """
        return AIMessage(content="资料[1]")


def _app(
    tmp_path: Path, catalog_factory=None, understanding_factory=None, *,
    legacy_identity: bool = True,
    settings_overrides: dict[str, object] | None = None,
):
    quality = _Quality()
    provider = _Provider("资料[1]")

    def create_graph(key, url, model, temperature, effort, retriever, saver, top_k, max_calls):
        return LangGraphConversationAgent(
            _Model(), retriever, saver, top_k,
            max_retrieval_calls=max_calls, quality_evaluator=quality,
        )

    settings = Settings(
        _env_file=None, llm_api_key="offline",
        llm_base_url="http://offline.invalid/v1",
        embedding_api_key="offline",
        embedding_base_url="http://offline.invalid/v1",
        reranker_base_url=None, knowledge_retrieval_profile="hybrid",
        docs_path=tmp_path / "docs", chroma_path=tmp_path / "chroma",
        knowledge_base_db_path=tmp_path / "catalog.sqlite3",
        trace_db_path=tmp_path / "traces.sqlite3",
        auth_db_path=tmp_path / "auth.sqlite3",
        http_allowed_hosts=("test",),
        conversation_db_path=tmp_path / "conversations.sqlite3",
        agent_checkpoint_db_path=tmp_path / "checkpoints.sqlite3",
    )
    if settings_overrides is not None:
        settings = Settings(
            _env_file=None, **{**settings.model_dump(), **settings_overrides},
        )
    factories = replace(
        AdapterFactories(), embedding_provider=lambda *args: FakeQueryEmbedder(),
        llm_provider=lambda *args: provider,
        quality_evaluator=lambda *args: quality,
        conversation_agent=create_graph,
        understanding_provider=lambda *args: OfflineUnderstandingProvider(),
    )
    if catalog_factory is not None:
        factories = replace(factories, knowledge_base_repository=catalog_factory)
    if understanding_factory is not None:
        factories = replace(factories, understanding_provider=understanding_factory)
    container = build_application_container(settings, factories=factories)
    doc_id = str(uuid4())
    chunk = DocumentChunk(
        id=f"{doc_id}:0", doc_id=doc_id, filename="guide.md",
        page=None, text="needle 使用SQLite保存状态。", chunk_index=0,
    )
    container.vector_store.add_chunks([chunk], [[1.0, 0.0]])
    app = create_app(settings=settings, container_factory=lambda settings: container)
    if legacy_identity:
        allow_test_identity(app)
    return app, chunk, quality


@pytest.mark.anyio
async def test_query_and_conversation_use_same_routing_and_persist_metadata(tmp_path: Path) -> None:
    """相同策略和空历史下，两入口及历史详情必须一致。

    Args:
        tmp_path: 全部持久化隔离目录。
    """
    app, chunk, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            created = (await client.post("/api/conversations")).json()
            single = await client.post("/api/query", json={
                "question": "needle", "strategy": "hybrid",
            })
            turn = await client.post(f"/api/conversations/{created['id']}/query", json={
                "question": "needle",
            })
            assert single.status_code == turn.status_code == 200
            first = single.json()
            second = turn.json()
            for body in (first, second):
                assert body["lane"] == "fast"
                assert body["outcome"] == "answered"
                assert body["grounding"] == "grounded"
                assert body["answer"] == "资料[1]"
                assert body["citations"][0]["position"] == 1
                assert body["citations"][0]["chunk_id"] == chunk.id
                assert body["citations"][0]["kb_id"] == "default"
            detail = (await client.get(f"/api/conversations/{created['id']}")).json()
            saved = detail["turns"][0]
            assert saved["id"] == second["turn_id"]
            assert saved["assistant_message"] == second["answer"]
            for key in ("lane", "outcome", "grounding", "kb_scope", "evidence_stats"):
                assert saved[key] == second[key]
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_debug_and_forced_lane_never_bypass_quality(tmp_path: Path) -> None:
    """管理调试只覆盖起点，不能跳过裁判或答案审核。

    Args:
        tmp_path: 全部持久化隔离目录。
    """
    app, _, quality = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            for lane in ("fast", "agent"):
                before = quality.evidence_calls
                response = await client.post("/api/query", json={
                    "question": "needle", "strategy": "hybrid",
                    "lane": lane, "debug": True,
                })
                assert response.status_code == 200
                assert response.json()["lane"] == lane
                assert quality.evidence_calls == before + 1
    finally:
        app.state.container.close()


def _events(text: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for frame in text.split("\n\n"):
        name = None
        data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            if line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if name is not None and data is not None:
            events.append((name, data))
    return events


@pytest.mark.anyio
async def test_both_streams_publish_only_validated_body_and_same_metadata(
    tmp_path: Path,
) -> None:
    """两种流都不能重新走原始生成器，complete与正式历史共用字段。

    Args:
        tmp_path: 隔离运行目录。
    """
    app, chunk, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            conversation_id = (await client.post("/api/conversations")).json()["id"]
            for path, body in (
                ("/api/query/stream", {"question": "needle", "strategy": "hybrid"}),
                (f"/api/conversations/{conversation_id}/query/stream", {"question": "needle"}),
            ):
                response = await client.post(path, json=body)
                assert response.status_code == 200
                events = _events(response.text)
                terminals = [item for item in events if item[0] in {"complete", "error"}]
                assert [name for name, _ in terminals] == ["complete"]
                complete = terminals[0][1]
                assert complete["lane"] == "fast"
                assert complete["grounding"] == "grounded"
                assert complete["citations"][0]["chunk_id"] == chunk.id
                tokens = [data["token"] for name, data in events if name == "token"]
                assert "".join(tokens) == complete["answer"] == "资料[1]"
                assert "metadata" not in complete
                assert "execution" not in complete
                first_token = next(i for i, item in enumerate(events) if item[0] == "token")
                assert any(
                    name == "status" and data["phase"] == "validating"
                    for name, data in events[:first_token]
                )
            saved = (await client.get(f"/api/conversations/{conversation_id}")).json()["turns"][0]
            assert saved["lane"] == "fast"
            assert saved["citations"][0]["chunk_id"] == chunk.id
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_invalid_scope_is_422_before_stream_headers_and_model_calls(
    tmp_path: Path,
) -> None:
    """不存在的显式范围不能进入模型，也不能流开始后才报告。

    Args:
        tmp_path: 隔离运行目录。
    """
    app, _, quality = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            conversation_id = (await client.post("/api/conversations")).json()["id"]
            for path in (
                "/api/query", "/api/query/stream",
                f"/api/conversations/{conversation_id}/query",
                f"/api/conversations/{conversation_id}/query/stream",
            ):
                response = await client.post(path, json={
                    "question": "needle",
                    "kb_scope": {"mode": "explicit", "kb_ids": ["unknown"]},
                })
                assert response.status_code == 422
            assert quality.evidence_calls == 0
            saved = (await client.get(f"/api/conversations/{conversation_id}")).json()
            assert saved["turns_count"] == 0
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_default_direct_route_does_not_require_model_credentials(
    tmp_path: Path,
) -> None:
    """生产默认入口的规则闲聊不能提前构造任何模型或向量客户端。

    Args:
        tmp_path: 隔离数据目录。
    """
    settings = Settings(
        _env_file=None,
        openai_api_key=None,
        llm_api_key=None,
        embedding_api_key=None,
        reranker_base_url=None,
        knowledge_base_db_path=tmp_path / "catalog.sqlite3",
        chroma_path=tmp_path / "unused-chroma",
        http_allowed_hosts=("test",),
        trace_db_path=tmp_path / "traces.sqlite3",
    )
    app = create_app(settings=settings)
    allow_test_identity(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.post("/api/query", json={"question": "你好"})
        assert response.status_code == 200
        assert response.json()["lane"] == "direct"
        assert response.json()["answer_mode"] == "direct"
        assert not settings.chroma_path.exists()
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_catalog_storage_failure_is_503_for_all_entry_shells(tmp_path: Path) -> None:
    """新增目录故障不能在部分入口被包装500或流响应直接断开。

    Args:
        tmp_path: 隔离持久化目录。
    """
    class UnavailableCatalog:
        """模拟目录数据库不可用，保留全部HTTP/编排逻辑。"""

        def list(self) -> list[object]:
            """抛出稳定存储错误。

            Returns:
                不返回。
            """
            raise KnowledgeBaseStorageError("private database details")

    app, _, quality = _app(tmp_path, lambda path: UnavailableCatalog())
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            conversation_id = (await client.post("/api/conversations")).json()["id"]
            for path in (
                "/api/query", "/api/query/stream",
                f"/api/conversations/{conversation_id}/query",
                f"/api/conversations/{conversation_id}/query/stream",
            ):
                response = await client.post(path, json={"question": "needle"})
                assert response.status_code == 503
                assert "private" not in response.text
            assert quality.evidence_calls == 0
    finally:
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
async def test_catalog_failure_after_headers_has_one_error_event(
    tmp_path: Path, conversation: bool,
) -> None:
    """预检成功后目录故障也必须终止流，而非断开或发布未审核正文。

    Args:
        tmp_path: 隔离运行目录。
        conversation: 两种SSE外壳。
    """
    class Catalog:
        """只在首次范围预检可用的目录。"""

        def __init__(self) -> None:
            """初始化读取计数。"""
            self.calls = 0

        def list(self) -> list[KnowledgeBase]:
            """第二次读取模拟存储故障。

            Returns:
                预检时的真实默认库元数据。
            """
            self.calls += 1
            if self.calls > 1:
                raise KnowledgeBaseStorageError("private")
            return [KnowledgeBase(id="default", name="默认")]

    app, _, quality = _app(tmp_path, lambda path: Catalog())
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            if conversation:
                identifier = (await client.post("/api/conversations")).json()["id"]
                path = f"/api/conversations/{identifier}/query/stream"
            else:
                path = "/api/query/stream"
            response = await client.post(path, json={"question": "needle"})
            events = _events(response.text)
            assert response.status_code == 200
            assert [name for name, _ in events] == ["status", "error"]
            assert events[-1][1]["code"] == "storage_error"
            assert "private" not in response.text
            assert quality.evidence_calls == 0
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_conversation_scope_defaults_and_override_are_persisted(tmp_path: Path) -> None:
    """创建时指定的范围不能被忽略；每轮覆盖不改写会话默认范围。

    Args:
        tmp_path: 隔离运行目录。
    """
    app, _, _ = _app(tmp_path)
    try:
        app.state.container.knowledge_base_repository.create(
            KnowledgeBase(id="ops", name="空运维库"),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            created = await client.post("/api/conversations", json={
                "kb_scope": {"mode": "explicit", "kb_ids": ["ops"]},
            })
            assert created.status_code == 201
            conversation_id = created.json()["id"]
            assert created.json()["kb_scope"] == {"mode": "explicit", "kb_ids": ["ops"]}
            scoped = await client.post(f"/api/conversations/{conversation_id}/query", json={
                "question": "needle",
            })
            assert scoped.json()["outcome"] == "refused"
            assert scoped.json()["kb_scope"]["searched"] == ["ops"]
            assert scoped.json()["scope_expanded"] is False
            override = await client.post(f"/api/conversations/{conversation_id}/query", json={
                "question": "needle", "kb_scope": {"mode": "auto"},
            })
            assert override.json()["outcome"] == "answered"
            detail = (await client.get(f"/api/conversations/{conversation_id}")).json()
            assert detail["kb_scope"]["kb_ids"] == ["ops"]
            patched = await client.patch(f"/api/conversations/{conversation_id}", json={
                "title": "我的范围", "kb_scope": {"mode": "auto"},
            })
            assert patched.status_code == 200
            assert patched.json()["title"] == "我的范围"
            assert patched.json()["kb_scope"]["mode"] == "auto"
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_conversation_stream_inherits_default_explicit_scope(
    tmp_path: Path,
) -> None:
    """流式入口也必须继承默认范围，不能偷偷检索有资料的其他库。

    Args:
        tmp_path: 隔离运行目录。
    """
    app, _, _ = _app(tmp_path)
    try:
        app.state.container.knowledge_base_repository.create(
            KnowledgeBase(id="ops", name="空运维库"),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            created = (await client.post("/api/conversations", json={
                "kb_scope": {"mode": "explicit", "kb_ids": ["ops"]},
            })).json()
            response = await client.post(
                f"/api/conversations/{created['id']}/query/stream",
                json={"question": "needle"},
            )
            events = _events(response.text)
            completed = [data for name, data in events if name == "complete"]
            assert len(completed) == 1
            assert completed[0]["outcome"] == "refused"
            assert completed[0]["kb_scope"]["searched"] == ["ops"]
            assert completed[0]["scope_expanded"] is False
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_invalid_conversation_settings_do_not_mutate_saved_scope(
    tmp_path: Path,
) -> None:
    """未知范围不能落库，非法修改也不能破坏原有设置。

    Args:
        tmp_path: 隔离运行目录。
    """
    app, _, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            invalid_scope = {"mode": "explicit", "kb_ids": ["missing"]}
            rejected = await client.post("/api/conversations", json={
                "kb_scope": invalid_scope,
            })
            assert rejected.status_code == 422
            assert (await client.get("/api/conversations")).json()["total"] == 0
            created = (await client.post("/api/conversations")).json()
            url = f"/api/conversations/{created['id']}"
            for body in ({}, {"title": "   "}, {"kb_scope": invalid_scope}):
                assert (await client.patch(url, json=body)).status_code == 422
            unchanged = (await client.get(url)).json()
            assert unchanged["title"] == "新对话"
            assert unchanged["kb_scope"] == {"mode": "auto", "kb_ids": []}
    finally:
        app.state.container.close()


@pytest.mark.anyio
async def test_production_root_understands_followup_with_budgeted_history(
    tmp_path: Path,
) -> None:
    """接线不能只测独立服务；正式会话历史应到达真实SDK请求。

    Args:
        tmp_path: 隔离运行目录。
    """
    provider, client, requests = _provider(json.dumps(_payload(
        standalone_query="needle", preferred_kb_ids=["default"],
    )))
    app, _, _ = _app(
        tmp_path, understanding_factory=lambda *args: provider,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as http:
            created = (await http.post("/api/conversations")).json()
            url = f"/api/conversations/{created['id']}/query"
            first = await http.post(url, json={"question": "needle"})
            second = await http.post(url, json={"question": "它怎么用？"})
            assert first.status_code == second.status_code == 200
            assert second.json()["outcome"] == "answered"
            assert len(requests) == 2
            payload = json.loads(requests[1]["messages"][-1]["content"])
            assert payload["question"] == "它怎么用？"
            assert payload["history"] == [{
                "user_message": "needle", "assistant_message": "资料[1]",
            }]
            turns = (await http.get(
                f"/api/conversations/{created['id']}",
            )).json()["turns"]
            assert turns[1]["user_message"] == "它怎么用？"
    finally:
        app.state.container.close()
        client.close()


@pytest.fixture
def anyio_backend() -> str:
    """仅使用已安装的异步后端。

    Returns:
        asyncio后端名。
    """
    return "asyncio"

# -*- coding: utf-8 -*-
"""真实拒答快照与确定性语义向量归组，不以字符串相同代替语义。"""

import json
import sqlite3
import asyncio
import math
from contextlib import AsyncExitStack
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import httpx
import pytest

import app.application.insights.clusters as clusters_module
from app.application.errors import TraceStorageError
from app.application.errors import EmbeddingError
from app.application.insights.clustering import cluster_refusals
from app.application.insights.clustering import normalize_vector
from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from app.infrastructure.persistence.insights import SQLiteRefusalReader
from tests.http_auth import TEST_USER_ID
from tests.test_ingestion_job_container import _settings
from app.infrastructure.conversations.sqlite_repository import SQLiteConversationRepository
from tests.test_refusal_insights_api import _refusal
from tests.test_shares_api import _client
from tests.test_unified_query_api import _app


ENDPOINT = "/api/admin/insights/refusals"


def test_complete_link_prevents_cross_topic_similarity_bridges(tmp_path: Path) -> None:
    """A近B、B近C并不表示A和C同主题，完整链接不得把三者串联。

    Args:
        tmp_path: 实际正式轮次数据库。
    """
    path = tmp_path / "business.sqlite3"
    repository = SQLiteConversationRepository(path)
    for question in ("操作配置", "配置和成本的联系", "财务成本"):
        _refusal(repository, question=question)
    now = datetime.now(UTC)
    sample = SQLiteRefusalReader(path).cluster_sample(
        now - timedelta(days=1), now, None,
    )
    vectors = []
    for angle in (0, 20, 40):
        radians = math.radians(angle)
        vectors.append(normalize_vector([math.cos(radians), math.sin(radians)]))
    groups = cluster_refusals(sample.items, vectors, 0.9)
    assert [group.count for group in groups] == [2, 1]
    assert set(groups[0].sample_turn_ids) == {
        sample.items[0].turn_id, sample.items[1].turn_id,
    }


def test_empty_cluster_runtime_needs_no_model_or_production_paths(tmp_path: Path) -> None:
    """空快照也要审计，但不能为此读取模型配置、索引或生产业务库。

    Args:
        tmp_path: 生产哨兵和隔离运行路径。
    """
    settings = _settings(tmp_path)
    root = build_application_container(settings, RuntimeOptions(
        chroma_path=tmp_path / "isolated", collection_name="cluster-test",
    ))
    try:
        result = root.refusal_cluster_service.list(TEST_USER_ID)
        assert result.total_refusals == 0
        assert result.clusters == ()
        assert not settings.conversation_db_path.exists()
        assert not settings.trace_db_path.exists()
        assert not settings.chroma_path.exists()
    finally:
        root.close()


@pytest.mark.anyio
async def test_cluster_busy_and_model_deadline_release_execution_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """并发请求快速409，超时不发布部分结果且必须释放执行槽。

    Args:
        tmp_path: 全隔离实际存储。
        monkeypatch: 阻塞外部Embedding和推进协作时钟。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    _refusal(repository)
    entered, release = Event(), Event()
    ticks = [100.0]
    requests = []

    def embed(texts: list[str]) -> list[list[float]]:
        requests.append(texts)
        entered.set()
        assert release.wait(5)
        if len(requests) == 1:
            ticks[0] += 61
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(root.embedding_provider, "embed_documents", embed)
    monkeypatch.setattr(clusters_module, "time", SimpleNamespace(monotonic=lambda: ticks[0]))
    pending = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            pending = asyncio.create_task(client.get(ENDPOINT, params={"group": "cluster"}))
            assert await asyncio.to_thread(entered.wait, 3)
            busy = await asyncio.wait_for(
                client.get(ENDPOINT, params={"group": "cluster"}), 1,
            )
            assert busy.status_code == 409
            assert busy.json()["error"]["code"] == "refusal_clusters_busy"
            release.set()
            expired = await pending
            assert expired.status_code == 504
            assert expired.json()["error"]["code"] == "refusal_clusters_timeout"
            recovered = await client.get(ENDPOINT, params={"group": "cluster"})
            assert recovered.status_code == 200
            cached = await client.get(ENDPOINT, params={"group": "cluster"})
            assert cached.status_code == 200
            assert len(requests) == 2
            ticks[0] += 301
            refreshed = await client.get(ENDPOINT, params={"group": "cluster"})
            assert refreshed.status_code == 200
            assert len(requests) == 3
    finally:
        release.set()
        if pending is not None:
            await pending
        root.close()


@pytest.mark.anyio
async def test_semantic_clusters_paginate_cache_and_invalidate_deleted_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同词异义分离、异词同义归组，分页不可重复计费或返回已删除问题。

    Args:
        tmp_path: 全部实际存储隔离。
        monkeypatch: 只替代外部向量化边界。
    """
    app, _, _ = _app(tmp_path, legacy_identity=False)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    first = _refusal(repository, question="Redis缓存如何设置", kb_ids=("ops",))
    second = _refusal(repository, question="怎样启用键值缓存", kb_ids=("ops",))
    third = _refusal(repository, question="Redis与关系数据库性能比较", kb_ids=("ops",))
    vectors = {
        first.user_message: [1.0, 0.0],
        second.user_message: [0.99, 0.01],
        third.user_message: [0.0, 1.0],
    }
    requested = []

    def embed(texts: list[str]) -> list[list[float]]:
        requested.extend(texts)
        return [vectors[text] for text in texts]

    monkeypatch.setattr(root.embedding_provider, "embed_documents", embed)
    try:
        async with AsyncExitStack() as stack:
            admin = await _client(stack, app, "admin", "admin")
            member = await _client(stack, app, "member")
            assert (await member.get(ENDPOINT, params={"group": "cluster"})).status_code == 403
            response = await admin.get(ENDPOINT, params={
                "group": "cluster", "kb_id": "ops", "limit": 1,
            })
            assert response.status_code == 200
            page = response.json()
            assert page["total_refusals"] == 3
            assert page["clustered_refusals"] == 3
            assert page["truncated"] is False
            assert page["total_clusters"] == 2
            group = page["clusters"][0]
            assert group["count"] == 2
            assert group["representative_question"] in (first.user_message, second.user_message)
            assert set(group["sample_turn_ids"]) == {str(first.id), str(second.id)}
            assert group["kb_ids"] == ["ops"]
            assert len(requested) == 3
            cursor = page["next_cursor"]
            assert cursor is not None
            next_response = await admin.get(ENDPOINT, params={
                "group": "cluster", "kb_id": "ops", "limit": 1, "cursor": cursor,
            })
            assert next_response.status_code == 200
            assert next_response.json()["clusters"][0]["count"] == 1
            assert next_response.json()["clusters"][0]["sample_turn_ids"] == [str(third.id)]
            assert next_response.json()["next_cursor"] is None
            assert len(requested) == 3
            changed_scope = await admin.get(ENDPOINT, params={
                "group": "cluster", "kb_id": "other", "cursor": cursor,
            })
            assert changed_scope.status_code == 422
            with sqlite3.connect(root.settings.trace_db_path) as db:
                audits = db.execute("SELECT event_json FROM admin_access_audit").fetchall()
            assert len(audits) >= 2
            encoded = json.dumps(audits, ensure_ascii=False)
            assert "Redis" not in encoded
            assert str(first.id) in encoded
            repository.mark_deleting(first.conversation_id)
            stale = await admin.get(ENDPOINT, params={
                "group": "cluster", "kb_id": "ops", "limit": 1, "cursor": cursor,
            })
            assert stale.status_code == 409
            assert stale.json()["error"]["code"] == "refusal_clusters_changed"
            fresh = (await admin.get(ENDPOINT, params={"group": "cluster"})).json()
            assert fresh["clustered_refusals"] == 2
            assert first.user_message not in json.dumps(fresh, ensure_ascii=False)
    finally:
        root.close()


@pytest.mark.anyio
async def test_failed_embedding_batch_cannot_grow_cache_without_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """部分批次成功后失败也必须保持缓存上限，不能靠最后成功路径淘汰。

    Args:
        tmp_path: 隔离数据库与服务。
        monkeypatch: 外部模型第二批失败。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    failures = False

    def embed(texts: list[str]) -> list[list[float]]:
        if failures and len(texts) == 1:
            raise EmbeddingError("外部模型失败")
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(root.embedding_provider, "embed_documents", embed)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            for batch in range(2):
                for index in range(500):
                    _refusal(repository, question=f"缓存{batch}组{index}项")
                response = await client.get(ENDPOINT, params={"group": "cluster"})
                assert response.status_code == 200
            for index in range(21):
                _refusal(repository, question=f"新问题{index}")
            failures = True
            failed = await client.get(ENDPOINT, params={"group": "cluster"})
            assert failed.status_code == 502
            assert len(root.refusal_cluster_service._cache) <= 1000
    finally:
        root.close()


@pytest.mark.anyio
@pytest.mark.parametrize("vectors", [
    [[], []], [[0.0, 0.0], [1.0, 0.0]],
    [[float("nan"), 1.0], [1.0, 0.0]],
    [[1.0, 0.0], [1.0]], [[1.0, 0.0]],
])
async def test_cluster_rejects_invalid_vectors_without_lexical_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    vectors: list[list[float]],
) -> None:
    """空、零、非有限、维度或数量错误不能降级为关键词分组。

    Args:
        tmp_path: 全隔离数据。
        monkeypatch: 外部Embedding边界。
        vectors: 不合法模型结果。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    _refusal(repository, question="第一项缓存配置")
    _refusal(repository, question="第二项缓存配置")
    monkeypatch.setattr(root.embedding_provider, "embed_documents", lambda texts: vectors)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get(ENDPOINT, params={"group": "cluster"})
            assert response.status_code == 502
            assert response.json()["error"]["code"] == "retrieval_unavailable"
            assert "clusters" not in response.json()
    finally:
        root.close()


@pytest.mark.anyio
async def test_cluster_rechecks_deletion_after_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型执行期间的删除不能使已撤销内容出现在生成后的报告。

    Args:
        tmp_path: 临时正式业务库。
        monkeypatch: 真实删除发生于外部模型返回之前。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    turn = _refusal(repository)

    def embed(texts: list[str]) -> list[list[float]]:
        repository.mark_deleting(turn.conversation_id)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(root.embedding_provider, "embed_documents", embed)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get(ENDPOINT, params={"group": "cluster"})
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "refusal_clusters_changed"
            assert "私密拒答问题" not in response.text
    finally:
        root.close()


@pytest.mark.anyio
async def test_cluster_window_cap_counts_and_batches_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """筛选晚于LIMIT、样本冒充全量、超量模型批次均会失败。

    Args:
        tmp_path: 实际SQLite数据与索引临时目录。
        monkeypatch: 仅替换外部Embedding并观察批次。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    for index in range(505):
        _refusal(repository, question=f"配置缓存编号{index}", kb_ids=("ops",))
    _refusal(repository, question="别库问题不参与", kb_ids=("other",))
    old = _refusal(repository, question="过窗口问题不参与", kb_ids=("ops",))
    deleting = _refusal(repository, question="删除问题不参与", kb_ids=("ops",))
    repository.mark_deleting(deleting.conversation_id)
    with sqlite3.connect(root.settings.conversation_db_path) as db:
        db.execute("UPDATE conversation_turns SET created_at=? WHERE id=?", (
            (datetime.now(UTC) - timedelta(days=40)).isoformat(), str(old.id),
        ))
    requests = []

    def embed(texts: list[str]) -> list[list[float]]:
        requests.append(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(root.embedding_provider, "embed_documents", embed)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get(ENDPOINT, params={
                "group": "cluster", "kb_id": "ops",
            })
            assert response.status_code == 200
            page = response.json()
            assert page["total_refusals"] == 505
            assert page["clustered_refusals"] == 500
            assert page["truncated"] is True
            assert page["clusters"][0]["count"] == 500
            assert len(page["clusters"][0]["sample_turn_ids"]) == 5
            assert sum(len(batch) for batch in requests) == 500
            for batch in requests:
                assert len(batch) <= 20
                assert sum(len(text) for text in batch) <= 16000
                assert all("不参与" not in text for text in batch)
            with sqlite3.connect(root.settings.trace_db_path) as db:
                audits = db.execute("SELECT event_json FROM admin_access_audit").fetchall()
            assert len(audits) == 5
            assert all(len(json.loads(row[0])["resource_ids"]) == 100 for row in audits)
    finally:
        root.close()


@pytest.mark.anyio
async def test_cluster_audit_failure_prevents_sending_questions_to_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """审计失败必须在模型调用之前截止，不能静默绕过访问记录。

    Args:
        tmp_path: 全隔离数据。
        monkeypatch: 外部向量化与审计存储故障边界。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    _refusal(repository)
    requested = []

    def embed(texts: list[str]) -> list[list[float]]:
        requested.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def fail(event: object) -> None:
        raise TraceStorageError("private audit path")

    monkeypatch.setattr(root.embedding_provider, "embed_documents", embed)
    monkeypatch.setattr(root.trace_repository, "record_admin_access", fail)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get(ENDPOINT, params={"group": "cluster"})
            assert response.status_code == 503
            assert requested == []
            assert "私密拒答问题" not in response.text
    finally:
        root.close()


@pytest.mark.anyio
async def test_oversized_legacy_question_is_not_sent_or_silently_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧库异常长问题不能绕过批次字符预算，也不能偷截断改变分组含义。

    Args:
        tmp_path: 含旧格式长问题的临时业务库。
        monkeypatch: 观察外部模型输入。
    """
    app, _, _ = _app(tmp_path)
    root = app.state.container
    repository = SQLiteConversationRepository(root.settings.conversation_db_path)
    _refusal(repository, question="长" * 16001)
    requested = []

    def embed(texts: list[str]) -> list[list[float]]:
        requested.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(root.embedding_provider, "embed_documents", embed)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get(ENDPOINT, params={"group": "cluster"})
            assert response.status_code == 503
            assert requested == []
    finally:
        root.close()

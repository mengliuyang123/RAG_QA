# -*- coding: utf-8 -*-
"""两个真实HTTP入口必须返回可读取的实际执行追踪，而非占位trace_id。"""

from pathlib import Path
from uuid import UUID

import httpx
import pytest

from tests.test_unified_query_api import _app
from tests.test_unified_query_api import _events
from app.application.errors import TraceStorageError
from tests.http_auth import TEST_USER_ID


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("agent", [False, True])
async def test_answer_trace_has_real_ordered_steps_and_retrieved_sources(
    tmp_path: Path, conversation: bool, agent: bool,
) -> None:
    """测试生产组合根与真实图，只有外部模型端口使用离线替身。

    Args:
        tmp_path: 隔离所有数据。
        conversation: 两个入口。
        agent: 两个知识通道。
    """
    app, chunk, _ = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            path = "/api/query"
            if conversation:
                created = (await client.post("/api/conversations")).json()
                path = f"/api/conversations/{created['id']}/query"
            request = {"question": "比较needle" if agent else "needle"}
            if agent and not conversation:
                request["lane"] = "agent"
            response = await client.post(path, json=request)
            assert response.status_code == 200
            body = response.json()
            trace_id = UUID(body["trace_id"])
            stored = app.state.container.trace_repository.get(trace_id)
            assert stored.status == "completed"
            assert stored.user_id == TEST_USER_ID
            assert stored.entry == ("conversation" if conversation else "query")
            assert stored.lane == ("agent" if agent else "fast")
            kinds = [step.type for step in stored.steps]
            expected = [
                "admission", "execution",
                "understanding", "routing", "retrieval", "evidence_judge",
                "generation", "citation_validation",
            ]
            if agent:
                expected.append("answer_judge")
            assert kinds == expected
            assert [step.order for step in stored.steps] == list(range(1, len(expected) + 1))
            retrieval = stored.steps[4]
            assert retrieval.input["kb_ids"] == ["default"]
            assert retrieval.output["results"][0]["chunk_id"] == chunk.id
            assert "text" not in retrieval.output["results"][0]
            assert stored.steps[5].output["sufficient"] is True
            assert stored.question is None
            if conversation:
                detail = (await client.get(
                    f"/api/conversations/{created['id']}",
                )).json()
                assert detail["turns"][0]["trace_id"] == body["trace_id"]
    finally:
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_trace_storage_failure_is_typed_and_prevents_model_calls(
    tmp_path: Path, conversation: bool, stream: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """追踪不可写时不能返回一个不存在的trace_id，也不能变成笼统500。

    Args:
        tmp_path: 隔离数据库目录。
        conversation: 两种入口。
        stream: 两种传输。
        monkeypatch: 模拟真实追踪数据库写入失败。
    """
    app, _, quality = _app(tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            created = (await client.post("/api/conversations")).json()
            repository = app.state.container.trace_repository

            def fail_save(snapshot) -> None:
                raise TraceStorageError("private database detail")

            monkeypatch.setattr(repository, "save", fail_save)
            path = f"/api/conversations/{created['id']}/query" if conversation else "/api/query"
            if stream:
                path += "/stream"
            response = await client.post(path, json={"question": "needle"})
            if stream:
                assert response.status_code == 200
                events = _events(response.text)
                assert events[-1] == ("error", {
                    "code": "storage_error", "message": "追踪存储暂时不可用",
                })
                assert not any(name in {"token", "complete"} for name, _ in events)
            else:
                assert response.status_code == 503
            assert "private" not in response.text
            assert quality.evidence_calls == 0
    finally:
        app.state.container.close()

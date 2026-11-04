# -*- coding: utf-8 -*-
"""后台只读已发布评估，不误报旧文件、预览或未完成运行。"""

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest

from app.application.errors import EvaluationReportNotFoundError
from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from evaluation.reporting import EvaluationReport
from evaluation.reporting import publish_report
from tests.evaluation.test_reporting import _report
from tests.test_shares_api import _client
from tests.test_ingestion_job_container import _settings
from tests.test_unified_query_api import _app


@pytest.mark.anyio
async def test_latest_uses_real_publication_and_matching_schema(
    tmp_path: Path,
) -> None:
    """替换成占位报告、绕过管理员或漏掉原字段均会失败。

    Args:
        tmp_path: 隔离全部数据与发布目录。
    """
    results = tmp_path / "results"
    app, _, quality = _app(tmp_path, legacy_identity=False, settings_overrides={
        "evaluation_results_path": results,
    })
    try:
        async with AsyncExitStack() as stack:
            admin = await _client(stack, app, "admin", "admin")
            member = await _client(stack, app, "member")
            anonymous = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="https://test",
            ))
            endpoint = "/api/admin/evaluations/latest"
            empty = await admin.get(endpoint)
            assert empty.status_code == 404
            assert empty.json()["error"]["code"] == "no_evaluation"
            report = _report("complete", 150, 150)
            publish_report(report, results)
            response = await admin.get(endpoint)
            assert response.status_code == 200
            assert response.json() == report.model_dump(mode="json")
            assert response.headers["cache-control"] == "no-store"
            schema_path = "/api/admin/evaluations/schema"
            schema = await admin.get(schema_path)
            assert schema.status_code == 200
            assert schema.json() == EvaluationReport.model_json_schema()
            openapi = (await admin.get("/openapi.json")).json()
            descriptor = openapi["paths"][endpoint]["get"]["responses"]["200"]
            assert descriptor["content"]["application/json"]["schema"]["$ref"] == schema_path
            for path in (endpoint, schema_path):
                assert (await member.get(path)).status_code == 403
                assert (await anonymous.get(path)).status_code == 401
            assert quality.evidence_calls == 0
    finally:
        app.state.container.close()


@pytest.mark.anyio
@pytest.mark.parametrize("kind", [
    "broken", "preview", "incomplete", "oversize", "directory", "symlink", "fifo",
])
async def test_invalid_latest_is_unavailable_not_empty(
    tmp_path: Path, kind: str,
) -> None:
    """坏发布不能被当成不存在，也不能读取任意链接或等待FIFO。

    Args:
        tmp_path: 临时发布目录。
        kind: 不合法的文件形态或内容。
    """
    results = tmp_path / "results"
    results.mkdir()
    target = results / "latest.json"
    if kind == "broken":
        target.write_text('{"api_key":"must_not_escape"}', encoding="utf-8")
    elif kind == "preview":
        report = _report("preview", 1, 1)
        target.write_text(report.model_dump_json(), encoding="utf-8")
    elif kind == "incomplete":
        data = _report("preview", 1, 1).model_dump(mode="json")
        data["status"] = "complete"
        target.write_text(json.dumps(data), encoding="utf-8")
    elif kind == "oversize":
        with target.open("wb") as source:
            source.truncate(32 * 1024 * 1024 + 1)
    elif kind == "directory":
        target.mkdir()
    elif kind == "symlink":
        actual = tmp_path / "elsewhere.json"
        actual.write_text(json.dumps({"secret": "must_not_escape"}), encoding="utf-8")
        target.symlink_to(actual)
    else:
        os.mkfifo(target)
    app, _, _ = _app(tmp_path, legacy_identity=False, settings_overrides={
        "evaluation_results_path": results,
    })
    try:
        async with AsyncExitStack() as stack:
            admin = await _client(stack, app, "admin", "admin")
            response = await admin.get("/api/admin/evaluations/latest")
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "evaluation_unavailable"
            assert "must_not_escape" not in response.text
    finally:
        app.state.container.close()


@pytest.mark.parametrize("explicit", [False, True])
def test_evaluation_reader_never_falls_back_to_production_in_runtime(
    tmp_path: Path, explicit: bool,
) -> None:
    """隔离运行缺报告不能偷偷返回生产结果，且读取不初始化业务库。

    Args:
        tmp_path: 生产哨兵与隔离报告目录。
        explicit: 是否显式指定运行专属目录。
    """
    production = tmp_path / "published"
    report = _report("complete", 150, 150)
    publish_report(report, production)
    settings = _settings(tmp_path).model_copy(update={
        "evaluation_results_path": production,
    })
    override = None
    if explicit:
        override = tmp_path / "override"
    options = RuntimeOptions(
        chroma_path=tmp_path / "isolated", collection_name="reports",
        evaluation_results_path=override,
    )
    root = build_application_container(settings, options)
    try:
        with pytest.raises(EvaluationReportNotFoundError):
            root.evaluation_report_service.latest()
        assert root.evaluation_report_service.schema()["title"] == "EvaluationReport"
        if override is not None:
            publish_report(report, override)
            assert root.evaluation_report_service.latest()["run_id"] == report.run_id
        assert not settings.conversation_db_path.exists()
        assert not settings.auth_db_path.exists()
        assert not settings.chroma_path.exists()
        assert not settings.trace_db_path.exists()
    finally:
        root.close()

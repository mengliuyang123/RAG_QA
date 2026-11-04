# -*- coding: utf-8 -*-
"""管理员已发布评估报告与同源Schema接口。"""

from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Response
from pydantic import JsonValue

from app.application.evaluation_reports import EvaluationReportService


router = APIRouter(prefix="/admin/evaluations", tags=["evaluations"])


def get_evaluation_reports(request: Request) -> EvaluationReportService:
    """从唯一组合根获取只读用例。

    Args:
        request: 当前管理员请求。

    Returns:
        隔离的报告用例。
    """
    return request.app.state.container.evaluation_report_service


@router.get("/latest", responses={
    200: {"content": {"application/json": {"schema": {
        "$ref": "/api/admin/evaluations/schema",
    }}}},
})
def latest_evaluation(
    response: Response,
    service: Annotated[EvaluationReportService, Depends(get_evaluation_reports)],
) -> dict[str, JsonValue]:
    """返回完整已发布报告，权限由管理员路由依赖强制。

    Args:
        response: 禁止缓存报告。
        service: 已装配的只读服务。

    Returns:
        与发布文件同结构的报告。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.latest()


@router.get("/schema")
def evaluation_schema(
    response: Response,
    service: Annotated[EvaluationReportService, Depends(get_evaluation_reports)],
) -> dict[str, JsonValue]:
    """即使还未发布报告，也可获得准确结构。

    Args:
        response: 禁止缓存旧版本契约。
        service: 当前读取服务。

    Returns:
        发布器的JSON Schema。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.schema()

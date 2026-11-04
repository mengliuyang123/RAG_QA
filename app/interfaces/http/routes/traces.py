# -*- coding: utf-8 -*-
"""仅管理员可用的追踪查询；不开放普通用户跨会话读取。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Response
from fastapi import Query
from fastapi import HTTPException
from pydantic import AwareDatetime
from pydantic import ConfigDict
from pydantic import Field

from app.application.tracing.queries import TraceQueryService
from app.domain.auth import User
from app.domain.trace_queries import TraceDetail
from app.domain.trace_queries import TraceFilters
from app.domain.trace_queries import TracePage
from app.interfaces.http.auth import require_admin


router = APIRouter(prefix="/admin/traces", tags=["traces"])


class TraceListRequest(TraceFilters):
    """HTTP查询字段别名不进入应用层。

    Attributes:
        from_at: 前端from参数。
        to_at: 前端to参数。
        limit: 有界页大小。
        cursor: 不透明分页游标。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    from_at: AwareDatetime | None = Field(default=None, alias="from")
    to_at: AwareDatetime | None = Field(default=None, alias="to")
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=2048)


def get_trace_queries(request: Request) -> TraceQueryService:
    """只从组合根取得追踪查询用例。

    Args:
        request: 当前HTTP请求。

    Returns:
        同一运行的只读查询服务。
    """
    return request.app.state.container.trace_query_service


@router.get("", response_model=TracePage)
def list_traces(
    query: Annotated[TraceListRequest, Query()], response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[TraceQueryService, Depends(get_trace_queries)],
) -> TracePage:
    """管理员过滤后分页读取脱敏概要。

    Args:
        query: 时间、通道、知识库、反馈和游标参数。
        response: 禁止缓存的响应。
        user: 已验证管理员。
        service: 只读查询用例。

    Returns:
        追踪概要和下一页游标。
    """
    response.headers["Cache-Control"] = "no-store"
    filters = TraceFilters.model_validate(query.model_dump(exclude={"limit", "cursor"}))
    try:
        return service.list(filters, limit=query.limit, cursor=query.cursor)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="追踪游标或查询条件无效") from error


@router.get("/{trace_id}", response_model=TraceDetail)
def get_trace(
    trace_id: UUID, response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[TraceQueryService, Depends(get_trace_queries)],
) -> TraceDetail:
    """返回真实执行步骤、用量及独立提交状态。

    Args:
        trace_id: 追踪标识。
        response: 用于禁止缓存。
        user: 后端强制验证的管理员。
        service: 组合根提供的只读查询用例。

    Returns:
        默认不含提问者身份的执行详情。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.get(trace_id)

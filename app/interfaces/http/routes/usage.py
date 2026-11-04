# -*- coding: utf-8 -*-
"""仅管理员可查询的分组用量，不触发模型或假装完整账单。"""

from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Query
from fastapi import Request
from fastapi import Response

from app.application.tracing.usage import UsageQueryService
from app.domain.auth import User
from app.domain.usage import UsageGroup
from app.domain.usage import UsageReport
from app.interfaces.http.auth import require_admin


router = APIRouter(prefix="/admin/usage", tags=["usage"])


def get_usage_service(request: Request) -> UsageQueryService:
    """从组合根取得只读统计用例。

    Args:
        request: 当前请求。

    Returns:
        与追踪列表共享数据源的用量服务。
    """
    return request.app.state.container.usage_query_service


@router.get("", response_model=UsageReport)
def get_usage(
    response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[UsageQueryService, Depends(get_usage_service)],
    days: Annotated[int, Query(ge=1, le=365)] = 7,
    group_by: UsageGroup = "day",
) -> UsageReport:
    """返回UTC窗口内的去重总量和用户／知识库／日期分组。

    Args:
        response: 禁止中间缓存的响应。
        user: 已验证的管理员。
        service: 组合根提供的统计用例。
        days: 含今天的UTC日历天数。
        group_by: 用户化名、知识库参与或UTC日期。

    Returns:
        明确保留范围、未知用量和跨库非可加性的报告。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.get(days, group_by)

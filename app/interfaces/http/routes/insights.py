# -*- coding: utf-8 -*-
"""管理员拒答洞察协议适配，不允许借此访问完整私有会话。"""

from typing import Annotated
from typing import Literal

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.application.insights.refusals import RefusalInsightService
from app.application.insights.overview import InsightOverviewService
from app.domain.auth import User
from app.domain.insights import RefusalPage
from app.domain.insights import InsightOverview
from app.domain.refusal_clusters import RefusalClusterPage
from app.domain.knowledge_bases import KB_ID_PATTERN
from app.interfaces.http.auth import require_admin


router = APIRouter(prefix="/admin/insights", tags=["insights"])


class RefusalListRequest(BaseModel):
    """限制分页及当前支持的列表模式，不静默伪装聚类。

    Attributes:
        limit: 每页数量。
        kb_id: 实际检索范围筛选。
        cursor: 上页返回的有界游标。
        group: 普通列表或真实语义聚类。
        days: 聚类的UTC日历窗口，list保持原全历史分页。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    limit: int = Field(default=20, ge=1, le=100)
    kb_id: str | None = Field(default=None, pattern=KB_ID_PATTERN)
    cursor: str | None = Field(default=None, max_length=2048)
    group: Literal["list", "cluster"] = "list"
    days: int = Field(default=30, ge=1, le=365)


def get_insight_overview(request: Request) -> InsightOverviewService:
    """只通过组合根取得有审计的概览用例。

    Args:
        request: 当前请求。

    Returns:
        同一运行的统计用例。
    """
    return request.app.state.container.insight_overview_service


@router.get("/overview", response_model=InsightOverview)
def get_overview(
    response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[InsightOverviewService, Depends(get_insight_overview)],
    days: Annotated[int, Query(ge=1, le=365)] = 7,
) -> InsightOverview:
    """管理员读取UTC日历窗口内的正式问题统计。

    Args:
        response: 禁止缓存。
        user: 已认证管理员。
        service: 无模型统计用例。
        days: 包含今天的日历天数。

    Returns:
        有明确分母和未知标记的统计。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.get(user.id, days)


def get_refusal_insights(request: Request) -> RefusalInsightService:
    """只通过组合根获取同一运行的洞察用例。

    Args:
        request: 当前请求。

    Returns:
        带审计的拒答洞察服务。
    """
    return request.app.state.container.refusal_insight_service


@router.get("/refusals", response_model=RefusalPage | RefusalClusterPage)
def list_refusals(
    query: Annotated[RefusalListRequest, Query()], response: Response,
    request: Request,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[RefusalInsightService, Depends(get_refusal_insights)],
) -> RefusalPage | RefusalClusterPage:
    """后台授权与访问审计后返回有限拒答原问题。

    Args:
        query: 已验证的筛选与分页参数。
        response: 禁止缓存的响应。
        request: 仅聚类分支才获取模型相关用例。
        user: 认证后的管理员身份。
        service: 组合根装配的用例。

    Returns:
        不含提问者身份的单轮拒答列表。
    """
    response.headers["Cache-Control"] = "no-store"
    try:
        if query.group == "cluster":
            service = request.app.state.container.refusal_cluster_service
            return service.list(
                user.id, days=query.days, limit=query.limit,
                kb_id=query.kb_id, cursor=query.cursor,
            )
        return service.list(
            user.id, limit=query.limit, kb_id=query.kb_id, cursor=query.cursor,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail="拒答游标或参数无效") from error

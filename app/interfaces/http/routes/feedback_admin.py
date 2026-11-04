# -*- coding: utf-8 -*-
"""管理员反馈单轮投影与分类，不提供普通会话权限绕过。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.application.feedback.admin import FeedbackAdminService
from app.domain.auth import User
from app.domain.feedback import AdminFeedbackItem
from app.domain.feedback import AdminFeedbackPage
from app.domain.feedback import FeedbackTriage
from app.domain.feedback import FeedbackTriageFilter
from app.domain.knowledge_bases import KB_ID_PATTERN
from app.interfaces.http.auth import require_admin


router = APIRouter(prefix="/admin/insights/feedback", tags=["insights"])


class FeedbackTriageRequest(BaseModel):
    """分类不接受客户端指定操作者身份。

    Attributes:
        triage: 处理分类或清空。
        expected_revision: 客户端看到的用户反馈版本，建议始终传入。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    triage: FeedbackTriage | None
    expected_revision: int | None = Field(default=None, ge=1)


def get_feedback_admin(request: Request) -> FeedbackAdminService:
    """只通过组合根获取有审计的后台用例。

    Args:
        request: 当前请求。

    Returns:
        后台反馈服务。
    """
    return request.app.state.container.feedback_admin_service


@router.get("", response_model=AdminFeedbackPage)
def list_feedback(
    response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[FeedbackAdminService, Depends(get_feedback_admin)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    kb_id: Annotated[str | None, Query(pattern=KB_ID_PATTERN)] = None,
    triage: FeedbackTriageFilter | None = None,
    cursor: Annotated[str | None, Query(max_length=2048)] = None,
) -> AdminFeedbackPage:
    """管理员审计后读取有限点踩记录。

    Args:
        response: 禁止缓存。
        user: 已验证管理员。
        service: 带审计的用例。
        limit: 每页数量。
        kb_id: 知识库过滤。
        triage: 分类过滤。
        cursor: 上页返回的游标。

    Returns:
        不含提问者身份的单轮投影。
    """
    response.headers["Cache-Control"] = "no-store"
    try:
        return service.list(user.id, limit=limit, kb_id=kb_id, triage=triage, cursor=cursor)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="反馈游标或参数无效") from error


@router.patch("/{feedback_id}", response_model=AdminFeedbackItem)
def triage_feedback(
    feedback_id: UUID, body: FeedbackTriageRequest, response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[FeedbackAdminService, Depends(get_feedback_admin)],
) -> AdminFeedbackItem:
    """CSRF及管理员检查后分类，过期反馈版本返回409。

    Args:
        feedback_id: 后端返回的规范资源ID。
        body: 分类与可选版本。
        response: 禁止缓存。
        user: 已验证管理员。
        service: 带审计的用例。

    Returns:
        已保存的分类状态与原单轮投影。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.triage(feedback_id, user.id, body.triage, body.expected_revision)

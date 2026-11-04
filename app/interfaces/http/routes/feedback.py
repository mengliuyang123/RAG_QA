# -*- coding: utf-8 -*-
"""本人正式轮次的幂等反馈和当前反馈查询。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Response

from app.application.feedback.service import FeedbackService
from app.domain.auth import User
from app.domain.feedback import TurnFeedback
from app.interfaces.http.auth import require_user


router = APIRouter(prefix="/turns", tags=["feedback"])


def get_feedback_service(request: Request) -> FeedbackService:
    """只从组合根获取用例，不在路由创建存储。

    Args:
        request: 当前请求。

    Returns:
        共享反馈服务。
    """
    return request.app.state.container.feedback_service


@router.put("/{turn_id}/feedback", status_code=204)
def put_feedback(
    turn_id: UUID, body: TurnFeedback,
    user: Annotated[User, Depends(require_user)],
    service: Annotated[FeedbackService, Depends(get_feedback_service)],
) -> Response:
    """覆盖当前用户反馈，CSRF由认证依赖强制检查。

    Args:
        turn_id: 正式轮次标识。
        body: 当前选择和可选原因／备注。
        user: 已验证身份，不接受请求体指定。
        service: 组合根提供的反馈用例。

    Returns:
        无正文204响应。
    """
    service.put(turn_id, user.id, body)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


@router.get("/{turn_id}/feedback", response_model=TurnFeedback)
def get_feedback(
    turn_id: UUID, response: Response,
    user: Annotated[User, Depends(require_user)],
    service: Annotated[FeedbackService, Depends(get_feedback_service)],
) -> TurnFeedback:
    """返回本人反馈详情，不暴露他人反馈存在性。

    Args:
        turn_id: 正式轮次标识。
        response: 禁止中间缓存的响应。
        user: 已验证身份。
        service: 组合根提供的反馈用例。

    Returns:
        当前反馈，未反馈或取消时rating为null。
    """
    response.headers["Cache-Control"] = "no-store"
    return service.get(turn_id, user.id)

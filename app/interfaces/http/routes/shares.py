# -*- coding: utf-8 -*-
"""团队内分享入口，读取也必须登录。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Response

from app.application.shares import ShareService
from app.domain.auth import User
from app.domain.shares import ShareLink
from app.domain.shares import ShareRequest
from app.domain.shares import SharedTurn
from app.interfaces.http.auth import require_user


router = APIRouter(tags=["shares"])


def get_share_service(request: Request) -> ShareService:
    """只从唯一组合根取分享用例。

    Args:
        request: 当前请求。

    Returns:
        惰性共享的用例。
    """
    return request.app.state.container.share_service


@router.post("/turns/{turn_id}/shares", response_model=ShareLink)
def create_share(
    turn_id: UUID, body: ShareRequest, response: Response,
    user: Annotated[User, Depends(require_user)],
    service: Annotated[ShareService, Depends(get_share_service)],
) -> ShareLink:
    """本人显式分享一轮，写操作受Cookie和CSRF保护。

    Args:
        turn_id: 正式轮次。
        body: 有效期。
        response: 禁止缓存链接。
        user: 已认证身份。
        service: 已装配用例。

    Returns:
        成功落库的链接。
    """
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return service.create(turn_id, user, body)


@router.get("/shares/{token}", response_model=SharedTurn)
def read_share(
    token: str, response: Response,
    user: Annotated[User, Depends(require_user)],
    service: Annotated[ShareService, Depends(get_share_service)],
) -> SharedTurn:
    """登录后读取被主动分享的一轮，无管理员会话读取特权。

    Args:
        token: 随机链接令牌。
        response: 禁止缓存正文。
        user: 当前登录的团队成员。
        service: 已装配用例。

    Returns:
        一轮正式问答与原始引用。
    """
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return service.read(token)

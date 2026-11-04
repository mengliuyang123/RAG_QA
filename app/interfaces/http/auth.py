# -*- coding: utf-8 -*-
"""服务器Cookie认证、角色和浏览器写操作来源校验。"""

import secrets
from typing import Annotated

from fastapi import Depends
from fastapi import Request

from app.application.auth.service import AuthService
from app.application.auth.service import token_digest
from app.application.errors import CSRFError
from app.application.errors import ForbiddenError
from app.application.errors import UnauthenticatedError
from app.domain.auth import AuthenticatedSession
from app.domain.auth import User


SESSION_COOKIE = "rag_session"
CSRF_COOKIE = "rag_csrf"
CSRF_HEADER = "X-CSRF-Token"


def get_auth_service(request: Request) -> AuthService:
    """获取当前应用组合根的认证服务。

    Args:
        request: 当前HTTP请求。

    Returns:
        共享认证服务。
    """
    return request.app.state.container.auth_service


def check_request_origin(request: Request) -> None:
    """只接受同源或显式配置来源，SameSite不能替代此检查。

    Args:
        request: 浏览器或可信API客户端请求。

    Returns:
        None。
    """
    origin = request.headers.get("origin")
    if origin is None:
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise CSRFError("请求来源不受信任")
        return
    settings = request.app.state.container.settings
    current_origin = f"{request.url.scheme}://{request.url.netloc}"
    if origin != current_origin and origin not in settings.cors_allowed_origins:
        raise CSRFError("请求来源不受信任")


def require_session(request: Request) -> AuthenticatedSession:
    """检查服务器会话，所有写操作还要求会话绑定的CSRF令牌。

    Args:
        request: 当前请求。

    Returns:
        当前有效会话，不包含可用认证令牌。
    """
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        # 匿名请求不创建数据库或做密码计算，避免廉价请求消耗安全资源。
        raise UnauthenticatedError("请先登录")
    session = get_auth_service(request).authenticate(token)
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        check_request_origin(request)
        supplied = request.headers.get(CSRF_HEADER)
        cookie = request.cookies.get(CSRF_COOKIE)
        if (
            not supplied or not cookie or len(supplied) > 256
            or len(cookie) > 256 or not supplied.isascii() or not cookie.isascii()
        ):
            raise CSRFError("请求缺少有效的CSRF令牌")
        if not secrets.compare_digest(supplied, cookie):
            raise CSRFError("请求缺少有效的CSRF令牌")
        if not secrets.compare_digest(token_digest(supplied), session.csrf_hash):
            raise CSRFError("请求缺少有效的CSRF令牌")
    return session


def require_user(
    session: Annotated[AuthenticatedSession, Depends(require_session)],
) -> User:
    """返回数据库中的身份，不允许请求体选择角色。

    Args:
        session: 已验证会话。

    Returns:
        公开用户身份。
    """
    return session.user


def require_admin(user: Annotated[User, Depends(require_user)]) -> User:
    """由后端强制检查管理员权限。

    Args:
        user: 已验证身份。

    Returns:
        管理员身份。
    """
    if user.role != "admin":
        raise ForbiddenError("此操作仅限管理员")
    return user

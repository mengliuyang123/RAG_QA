# -*- coding: utf-8 -*-
"""认证错误兼容旧detail并追加稳定code，不泄漏账号存在性或令牌。"""

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from app.application.errors import AuthStorageError
from app.application.errors import CSRFError
from app.application.errors import ForbiddenError
from app.application.errors import InvalidCredentialsError
from app.application.errors import LoginRateLimitError
from app.application.errors import UnauthenticatedError


async def _handle_auth_error(request: Request, error: Exception) -> JSONResponse:
    code = "unauthenticated"
    message = "请先登录"
    status = 401
    headers = {"Cache-Control": "no-store"}
    details: dict[str, object] = {}
    if isinstance(error, InvalidCredentialsError):
        code, message = "invalid_credentials", "账号或密码错误"
    elif isinstance(error, CSRFError):
        code, message, status = "csrf_failed", "请求来源或安全令牌无效", 403
    elif isinstance(error, ForbiddenError):
        code, message, status = "forbidden", "此操作仅限管理员", 403
    elif isinstance(error, LoginRateLimitError):
        code, message, status = "rate_limited", "登录尝试过于频繁，请稍后重试", 429
        headers["Retry-After"] = str(error.retry_after)
        details["retry_after"] = error.retry_after
    elif isinstance(error, AuthStorageError):
        code, message, status = "auth_unavailable", "认证服务暂时不可用", 503
    return JSONResponse(
        status_code=status, headers=headers,
        content={"detail": message, "error": {
            "code": code, "message": message, "details": details,
        }},
    )


def register_auth_error_handlers(application: FastAPI) -> None:
    """注册认证错误处理器，存储故障不回退为匿名身份。

    Args:
        application: 当前HTTP应用。

    Returns:
        None。
    """
    for error in (
        UnauthenticatedError, InvalidCredentialsError, CSRFError,
        ForbiddenError, LoginRateLimitError, AuthStorageError,
    ):
        application.add_exception_handler(error, _handle_auth_error)

# -*- coding: utf-8 -*-
"""账号密码登录、服务端注销及当前用户接口。"""

from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr

from app.domain.auth import User
from app.interfaces.http.auth import CSRF_COOKIE
from app.interfaces.http.auth import SESSION_COOKIE
from app.interfaces.http.auth import check_request_origin
from app.interfaces.http.auth import get_auth_service
from app.interfaces.http.auth import require_user


router = APIRouter(tags=["auth"])


class LoginRequest(BaseModel):
    """只接受账号和口令，不接受角色或用户ID。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    username: str = Field(min_length=1, max_length=64)
    password: SecretStr


@router.post("/auth/login")
def login(
    request: Request, body: LoginRequest, response: Response,
) -> dict[str, object]:
    """登录成功后只通过HttpOnly Cookie返回认证令牌。

    Args:
        request: 当前请求。
        body: 账号和受保护的口令字段。
        response: 用于设置安全cookie的响应。

    Returns:
        公开用户和非认证用途的CSRF令牌。
    """
    check_request_origin(request)
    service = get_auth_service(request)
    result = service.login(
        body.username, body.password.get_secret_value(),
        client_id=request.client.host if request.client else "unknown",
        previous_token=request.cookies.get(SESSION_COOKIE),
    )
    settings = request.app.state.container.settings
    for name, value, http_only in (
        (SESSION_COOKIE, result.session_token, True),
        (CSRF_COOKIE, result.csrf_token, False),
    ):
        response.set_cookie(
            name, value, max_age=settings.auth_session_seconds,
            expires=result.expires_at, path="/",
            secure=settings.auth_secure_cookie,
            httponly=http_only, samesite="lax",
        )
    response.headers["Cache-Control"] = "no-store"
    return {
        "user": result.user.model_dump(mode="json"),
        "csrf_token": result.csrf_token,
    }


@router.post("/auth/logout", status_code=204)
def logout(
    request: Request, user: Annotated[User, Depends(require_user)],
) -> Response:
    """验证CSRF后撤销会话，并移除两种cookie。

    Args:
        request: 当前请求。
        user: 已验证身份。

    Returns:
        不含正文的注销响应。
    """
    get_auth_service(request).logout(request.cookies[SESSION_COOKIE])
    response = Response(status_code=204, headers={"Cache-Control": "no-store"})
    for name in (SESSION_COOKIE, CSRF_COOKIE):
        response.delete_cookie(
            name, path="/",
            secure=request.app.state.container.settings.auth_secure_cookie,
        )
    return response


@router.get("/me")
def me(
    request: Request, user: Annotated[User, Depends(require_user)],
    response: Response,
) -> dict[str, object]:
    """返回当前角色与两个入口共享的实际滑动窗口额度。

    Args:
        user: 服务器身份。
        request: 当前应用请求，不从客户端读取配额身份。
        response: 用于禁止缓存的响应。

    Returns:
        公开用户与真实提问次数额度，不冒充日限额或模型Token余额。
    """
    response.headers["Cache-Control"] = "no-store"
    quota = request.app.state.container.question_admission.quota(user.id)
    return {**user.model_dump(mode="json"), "quota": quota.model_dump()}

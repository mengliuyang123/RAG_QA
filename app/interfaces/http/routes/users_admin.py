# -*- coding: utf-8 -*-
"""管理员账号管理：列出、创建、改角色、禁用/启用和重置口令。

口令只在请求体中出现一次，从不返回或写入日志。会话规则见 AuthService。
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Response
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator
from pydantic import model_validator

from app.application.auth.service import AuthService
from app.domain.auth import ManagedUser
from app.domain.auth import User
from app.domain.auth import UserRole
from app.interfaces.http.auth import get_auth_service
from app.interfaces.http.auth import require_admin


router = APIRouter(prefix="/admin/users", tags=["users"])
logger = logging.getLogger(__name__)

USERNAME_PATTERN = r"^[a-z0-9][a-z0-9_.-]{2,63}$"


def _plain_password(value: SecretStr) -> str:
    # 不用字段校验器：FastAPI 的 422 详情会回显原始输入，口令不能出现在响应里。
    password = value.get_secret_value()
    if not 15 <= len(password) <= 128:
        raise HTTPException(422, "口令长度必须为15–128字符")
    return password


class CreateUserRequest(BaseModel):
    """新账号；登录名大小写不敏感，口令不截断或去空白。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    username: str = Field(pattern=USERNAME_PATTERN)
    name: str = Field(min_length=1, max_length=100)
    password: SecretStr = Field(repr=False)
    role: UserRole = "member"

    @field_validator("username", mode="before")
    @classmethod
    def normalize_username(cls, value: object) -> object:
        return value.strip().casefold() if isinstance(value, str) else value

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("展示名不能为空")
        return value


class UpdateUserRequest(BaseModel):
    """至少修改角色或禁用状态之一。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: UserRole | None = None
    disabled: bool | None = None

    @model_validator(mode="after")
    def require_change(self) -> "UpdateUserRequest":
        if self.role is None and self.disabled is None:
            raise ValueError("至少提供角色或禁用状态")
        return self


class ResetPasswordRequest(BaseModel):
    """新的原始口令。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    password: SecretStr = Field(repr=False)


@router.get("")
def list_users(
    response: Response,
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> dict[str, list[ManagedUser]]:
    """列出全部账号，不含口令哈希。"""
    response.headers["Cache-Control"] = "no-store"
    return {"users": service.list_users()}


@router.post("", status_code=201)
def create_user(
    body: CreateUserRequest, response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> ManagedUser:
    """创建账号；登录名重复返回409 user_exists。"""
    response.headers["Cache-Control"] = "no-store"
    created = service.admin_create_user(
        body.username, _plain_password(body.password), name=body.name, role=body.role,
    )
    logger.info("管理员 %s 创建了账号 %s（%s）", user.id, created.id, created.role)
    return created


@router.patch("/{user_id}")
def update_user(
    user_id: UUID, body: UpdateUserRequest, response: Response,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> ManagedUser:
    """修改角色或禁用状态；禁用会立即撤销该账号全部会话。"""
    response.headers["Cache-Control"] = "no-store"
    updated = service.update_user(user.id, user_id, role=body.role, disabled=body.disabled)
    logger.info(
        "管理员 %s 修改了账号 %s：role=%s disabled=%s",
        user.id, user_id, updated.role, updated.disabled,
    )
    return updated


@router.post("/{user_id}/password", status_code=204)
def reset_password(
    user_id: UUID, body: ResetPasswordRequest,
    user: Annotated[User, Depends(require_admin)],
    service: Annotated[AuthService, Depends(get_auth_service)],
) -> Response:
    """重置口令并撤销该账号全部会话。"""
    service.reset_password(user_id, _plain_password(body.password))
    logger.info("管理员 %s 重置了账号 %s 的口令", user.id, user_id)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})

# -*- coding: utf-8 -*-
"""账号、公开身份和不透明会话的冻结领域契约。"""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


UserRole = Literal["member", "admin"]


class User(BaseModel):
    """可返回前端的公开身份。

    Attributes:
        id: 后端生成的用户标识。
        name: 展示名。
        role: 后端维护的角色，不能从请求指定或cookie猜测。
        avatar_url: 可选头像地址。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    name: str = Field(min_length=1, max_length=100)
    role: UserRole = "member"
    avatar_url: str | None = None


class UserAccount(User):
    """持久化账号记录，密码哈希不能被公开序列化或打印。

    Attributes:
        username: 规范化的登录标识。
        password_hash: 带随机盐的算法编码。
        disabled: 服务器禁用状态。
        created_at: UTC创建时间。
    """

    username: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,63}$")
    password_hash: str = Field(min_length=1, exclude=True, repr=False)
    disabled: bool = False
    created_at: AwareDatetime

    def public(self) -> User:
        """显式移除认证内部字段，避免继承模型意外泄漏。

        Returns:
            可公开身份。
        """
        return User(id=self.id, name=self.name, role=self.role, avatar_url=self.avatar_url)


class StoredSession(BaseModel):
    """服务器只保存令牌摘要和时限，不保存可直接使用的cookie。

    Attributes:
        user_id: 账号归属。
        token_hash: 会话令牌SHA256摘要。
        csrf_hash: CSRF令牌摘要。
        created_at: 创建时间。
        expires_at: 绝对过期时间。
        last_seen_at: 最近认证时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: UUID
    token_hash: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    csrf_hash: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    created_at: AwareDatetime
    expires_at: AwareDatetime
    last_seen_at: AwareDatetime


class AuthenticatedSession(BaseModel):
    """当前已通过服务端状态检查的身份和CSRF校验资料。

    Attributes:
        user: 当前数据库角色的公开身份。
        csrf_hash: 只供HTTP写操作校验。
        expires_at: 绝对过期时间。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    user: User
    csrf_hash: str = Field(exclude=True, repr=False)
    expires_at: AwareDatetime


@dataclass(frozen=True)
class LoginResult:
    """仅在登录响应设置cookie时短暂使用的明文令牌。

    Attributes:
        user: 公开身份。
        session_token: 不可打印的会话令牌。
        csrf_token: 不可打印的CSRF令牌。
        expires_at: Cookie对应的服务器时限。
    """

    user: User
    session_token: str = field(repr=False)
    csrf_token: str = field(repr=False)
    expires_at: datetime

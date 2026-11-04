# -*- coding: utf-8 -*-
"""编排账号登录、会话撤销和有限密码计算。"""

import hashlib
import secrets
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from threading import BoundedSemaphore
from uuid import uuid4

from app.application.errors import InvalidCredentialsError
from app.application.errors import LoginRateLimitError
from app.application.errors import UnauthenticatedError
from app.application.ports.auth import AuthRepository
from app.application.ports.auth import PasswordHasher
from app.domain.auth import AuthenticatedSession
from app.domain.auth import LoginResult
from app.domain.auth import StoredSession
from app.domain.auth import User
from app.domain.auth import UserAccount
from app.domain.auth import UserRole


def _utc_now() -> datetime:
    return datetime.now(UTC)


def token_digest(token: str) -> str:
    """为随机令牌计算存储摘要，不用于密码存储。

    Args:
        token: 高熵随机令牌。

    Returns:
        SHA256摘要。
    """
    return hashlib.sha256(token.encode()).hexdigest()


class AuthService:
    """不自动创建账号、不提供默认口令，持久化每次登录的服务端状态。"""

    def __init__(
        self, repository: AuthRepository, passwords: PasswordHasher, *,
        clock: Callable[[], datetime] = _utc_now,
        session_seconds: int = 43200, idle_seconds: int = 1800,
        login_attempts: int = 10, login_window_seconds: int = 60,
    ) -> None:
        """保存认证端口与明确时限。

        Args:
            repository: 独立认证存储。
            passwords: 安全口令算法。
            clock: UTC时钟。
            session_seconds: 绝对登录有效期。
            idle_seconds: 闲置有效期。
            login_attempts: 账号及来源各自的窗口额度。
            login_window_seconds: 限流窗口秒数。

        Returns:
            None。
        """
        if min(session_seconds, idle_seconds, login_attempts, login_window_seconds) < 1:
            raise ValueError("认证时限与额度必须大于零")
        self._repository = repository
        self._passwords = passwords
        self._clock = clock
        self._session_seconds = session_seconds
        self._idle_seconds = idle_seconds
        self._login_attempts = login_attempts
        self._login_window_seconds = login_window_seconds
        self._hash_slots = BoundedSemaphore(4)
        # 不存在的账号仍做同等强度验证，避免快速失败暴露账号是否存在。
        self._dummy_hash = passwords.hash(secrets.token_urlsafe(32))

    def create_user(
        self, username: str, password: str, *, name: str,
        role: UserRole = "member",
    ) -> User:
        """供可信管理工具创建账号，不是匿名注册接口。

        Args:
            username: 登录标识，大小写不敏感。
            password: 原始口令，15–128字符，不截断或去空白。
            name: 展示名。
            role: 明确角色，默认普通成员。

        Returns:
            不含密码信息的公开身份。
        """
        if not 15 <= len(password) <= 128:
            raise ValueError("口令长度必须为15–128字符")
        account = UserAccount(
            id=uuid4(), username=username.strip().casefold(),
            name=name.strip(), role=role, password_hash=self._passwords.hash(password),
            created_at=self._clock(),
        )
        self._repository.create_user(account)
        return account.public()

    def login(
        self, username: str, password: str, *, client_id: str,
        previous_token: str | None = None,
    ) -> LoginResult:
        """限流后验证口令，签发新令牌并撤销本浏览器旧会话。

        Args:
            username: 原始登录标识。
            password: 原始口令。
            client_id: 服务器观察的来源，不信任客户端自行传入的IP头。
            previous_token: 当前浏览器先前持有的令牌。

        Returns:
            供HTTP层设置cookie的短生命周期令牌结果。
        """
        normalized = username.strip().casefold()
        now = self._clock()
        self._repository.consume_login_attempts(
            (token_digest(f"account:{normalized}"), token_digest(f"source:{client_id}")),
            now, self._login_attempts, self._login_window_seconds,
        )
        if not 1 <= len(password) <= 128 or len(normalized) > 64:
            raise InvalidCredentialsError("账号或密码错误")
        if not self._hash_slots.acquire(blocking=False):
            raise LoginRateLimitError(1)
        try:
            account = self._repository.user_by_name(normalized)
            encoded = account.password_hash if account is not None else self._dummy_hash
            verified = self._passwords.verify(password, encoded)
        finally:
            self._hash_slots.release()
        if account is None or not verified or account.disabled:
            raise InvalidCredentialsError("账号或密码错误")
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        expires_at = now + timedelta(seconds=self._session_seconds)
        self._repository.save_session(StoredSession(
            user_id=account.id, token_hash=token_digest(token),
            csrf_hash=token_digest(csrf), created_at=now,
            expires_at=expires_at, last_seen_at=now,
        ))
        if previous_token:
            self.logout(previous_token)
        return LoginResult(account.public(), token, csrf, expires_at)

    def authenticate(self, token: str | None) -> AuthenticatedSession:
        """校验令牌的服务器状态，不信任cookie中的身份或角色。

        Args:
            token: 不透明cookie令牌。

        Returns:
            当前仍有效的服务器身份。
        """
        if not token or len(token) > 256:
            raise UnauthenticatedError("请先登录")
        session = self._repository.resolve_session(
            token_digest(token), self._clock(), self._idle_seconds,
        )
        if session is None:
            raise UnauthenticatedError("请先登录")
        return session

    def logout(self, token: str) -> None:
        """撤销服务器记录，不只是让浏览器忘记cookie。

        Args:
            token: 当前浏览器令牌。

        Returns:
            None。
        """
        self._repository.revoke_session(token_digest(token))

# -*- coding: utf-8 -*-
"""隔离认证用例与密码算法、SQLite存储。"""

from datetime import datetime
from typing import Protocol

from app.domain.auth import AuthenticatedSession
from app.domain.auth import StoredSession
from app.domain.auth import UserAccount


class PasswordHasher(Protocol):
    """密码算法端口，不由应用层创建具体算法。"""

    def hash(self, password: str) -> str:
        """创建带盐编码。

        Args:
            password: 完整原始口令。

        Returns:
            编码哈希。
        """

    def verify(self, password: str, encoded: str) -> bool:
        """验证原始口令。

        Args:
            password: 原始口令。
            encoded: 已存编码。

        Returns:
            是否匹配。
        """


class AuthRepository(Protocol):
    """账号、可撤销会话和登录限流的事务边界。"""

    def create_user(self, account: UserAccount) -> None:
        """创建账号。

        Args:
            account: 已验证并哈希的账号。

        Returns:
            None。
        """

    def user_by_name(self, username: str) -> UserAccount | None:
        """查找规范化账号。

        Args:
            username: 登录标识。

        Returns:
            账号或None。
        """

    def save_session(self, session: StoredSession) -> None:
        """保存摘要会话。

        Args:
            session: 仅含令牌摘要的记录。

        Returns:
            None。
        """

    def resolve_session(
        self, token_hash: str, now: datetime, idle_seconds: int,
    ) -> AuthenticatedSession | None:
        """检查账号与时限，并原子更新最后访问时间。

        Args:
            token_hash: 请求令牌摘要。
            now: UTC当前时间。
            idle_seconds: 闲置时限。

        Returns:
            有效身份或None。
        """

    def revoke_session(self, token_hash: str) -> None:
        """幂等撤销一条会话。

        Args:
            token_hash: 精确摘要。

        Returns:
            None。
        """

    def consume_login_attempts(
        self, keys: tuple[str, ...], now: datetime,
        max_attempts: int, window_seconds: int,
    ) -> None:
        """在密码计算前原子消耗账号与来源限额。

        Args:
            keys: 不含原始账号或IP的计数键。
            now: UTC当前时间。
            max_attempts: 每个键的窗口额度。
            window_seconds: 固定窗口时长。

        Returns:
            None，超限抛出稳定限流异常。
        """

# -*- coding: utf-8 -*-
"""真实认证存储、可撤销会话和防暴力破解规则，不依赖HTTP替身。"""

import hashlib
import sqlite3
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pytest

from app.application.auth.service import AuthService
from app.application.errors import InvalidCredentialsError
from app.application.errors import LoginRateLimitError
from app.application.errors import UnauthenticatedError
from app.infrastructure.auth.passwords import Argon2PasswordHasher
from app.infrastructure.auth.sqlite_repository import SQLiteAuthRepository


PASSWORD = "offline-passphrase-12345"


def _service(tmp_path: Path, **options: object) -> AuthService:
    return AuthService(
        SQLiteAuthRepository(tmp_path / "auth.sqlite3"),
        Argon2PasswordHasher(), **options,
    )


def test_login_uses_salted_hash_and_stores_only_session_digests(tmp_path: Path) -> None:
    """数据库泄露不能直接提供可用的浏览器会话令牌。

    Args:
        tmp_path: 隔离认证数据库。
    """
    service = _service(tmp_path)
    user = service.create_user("alice", PASSWORD, name="小刘", role="member")
    logged = service.login(" ALICE ", PASSWORD, client_id="offline-client")
    authenticated = service.authenticate(logged.session_token)
    assert authenticated.user.id == user.id
    assert authenticated.user.name == "小刘"
    assert authenticated.user.role == "member"
    with sqlite3.connect(tmp_path / "auth.sqlite3") as connection:
        encoded = connection.execute("SELECT password_hash FROM users").fetchone()[0]
        stored = connection.execute("SELECT token_hash, csrf_hash FROM auth_sessions").fetchone()
    assert encoded.startswith("$argon2id$")
    assert PASSWORD not in encoded
    assert stored == (
        hashlib.sha256(logged.session_token.encode()).hexdigest(),
        hashlib.sha256(logged.csrf_token.encode()).hexdigest(),
    )
    assert logged.session_token not in repr(logged)


def test_relogin_rotates_and_logout_revokes_server_session(tmp_path: Path) -> None:
    """重新登录必须换令牌，注销不能只清理浏览器cookie。

    Args:
        tmp_path: 隔离认证数据库。
    """
    service = _service(tmp_path)
    service.create_user("alice", PASSWORD, name="Alice")
    first = service.login("alice", PASSWORD, client_id="offline")
    second = service.login(
        "alice", PASSWORD, client_id="offline", previous_token=first.session_token,
    )
    assert second.session_token != first.session_token
    with pytest.raises(UnauthenticatedError):
        service.authenticate(first.session_token)
    service.logout(second.session_token)
    with pytest.raises(UnauthenticatedError):
        service.authenticate(second.session_token)


def test_unknown_user_and_wrong_password_share_generic_failure(tmp_path: Path) -> None:
    """不能用错误类型或文案枚举用户名。

    Args:
        tmp_path: 隔离认证数据库。
    """
    service = _service(tmp_path)
    service.create_user("alice", PASSWORD, name="Alice")
    for username in ("alice", "unknown"):
        with pytest.raises(InvalidCredentialsError):
            service.login(username, "incorrect-passphrase", client_id="offline")


def test_expiry_idle_timeout_and_disabled_user_are_enforced(tmp_path: Path) -> None:
    """不能只依靠cookie过期，服务器必须强制执行会话与账户状态。

    Args:
        tmp_path: 隔离认证数据库。
    """
    now = datetime(2026, 10, 23, tzinfo=UTC)
    service = _service(
        tmp_path, clock=lambda: now, session_seconds=100,
        idle_seconds=20,
    )
    user = service.create_user("alice", PASSWORD, name="Alice")
    logged = service.login("alice", PASSWORD, client_id="offline")
    now += timedelta(seconds=20)
    with pytest.raises(UnauthenticatedError):
        service.authenticate(logged.session_token)
    fresh = service.login("alice", PASSWORD, client_id="offline")
    with sqlite3.connect(tmp_path / "auth.sqlite3") as connection:
        connection.execute("UPDATE users SET disabled = 1 WHERE id = ?", (str(user.id),))
    with pytest.raises(UnauthenticatedError):
        service.authenticate(fresh.session_token)


def test_login_rate_limit_is_persistent_and_expires(tmp_path: Path) -> None:
    """进程重建不能清空失败次数并允许无限密码猜测。

    Args:
        tmp_path: 隔离认证数据库。
    """
    now = datetime(2026, 10, 23, tzinfo=UTC)
    service = _service(tmp_path, clock=lambda: now, login_attempts=2)
    for _ in range(2):
        with pytest.raises(InvalidCredentialsError):
            service.login("unknown", PASSWORD, client_id="offline")
    reopened = _service(tmp_path, clock=lambda: now, login_attempts=2)
    with pytest.raises(LoginRateLimitError):
        reopened.login("another-user", PASSWORD, client_id="offline")
    now += timedelta(seconds=61)
    with pytest.raises(InvalidCredentialsError):
        reopened.login("unknown", PASSWORD, client_id="offline")


def test_active_requests_cannot_extend_absolute_expiry(tmp_path: Path) -> None:
    """不断刷新闲置时间也不能绕过绝对截止时间。

    Args:
        tmp_path: 隔离认证库。
    """
    now = datetime(2026, 10, 23, tzinfo=UTC)
    service = _service(tmp_path, clock=lambda: now, session_seconds=60, idle_seconds=30)
    service.create_user("alice", PASSWORD, name="Alice")
    logged = service.login("alice", PASSWORD, client_id="offline")
    for _ in range(2):
        now += timedelta(seconds=20)
        assert service.authenticate(logged.session_token).user.name == "Alice"
    now += timedelta(seconds=20)
    with pytest.raises(UnauthenticatedError):
        service.authenticate(logged.session_token)

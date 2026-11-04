# -*- coding: utf-8 -*-
"""可信本地账号初始化不从命令行参数、环境变量或默认口令读取密码。"""

from pathlib import Path

import pytest

from app.config import Settings
from app.bootstrap.container import build_application_container
from app.interfaces.cli import users
from tests.test_auth_service import PASSWORD


def test_local_provisioning_prompts_for_password_and_creates_real_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """必须真正持久化账号且输出不包含口令，不能只打印创建成功。

    Args:
        tmp_path: 隔离认证数据库。
        monkeypatch: 只替换终端隐藏输入。
        capsys: 检查公开输出。
    """
    settings = Settings(_env_file=None, auth_db_path=tmp_path / "auth.sqlite3")
    monkeypatch.setattr(users, "getpass", lambda prompt: PASSWORD)
    monkeypatch.setattr(users, "get_settings", lambda: settings)
    assert users.main([
        "create", "--username", "alice", "--name", "Alice", "--role", "admin",
    ]) == 0
    container = build_application_container(settings)
    try:
        logged = container.auth_service.login("alice", PASSWORD, client_id="offline")
        assert logged.user.role == "admin"
        output = capsys.readouterr()
        assert PASSWORD not in output.out + output.err
        assert logged.session_token not in output.out + output.err
    finally:
        container.close()


def test_password_confirmation_mismatch_does_not_create_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """二次输入不一致时不写账号，也不能回显密码。

    Args:
        tmp_path: 隔离认证数据库。
        monkeypatch: 隐藏输入序列。
    """
    settings = Settings(_env_file=None, auth_db_path=tmp_path / "auth.sqlite3")
    answers = iter((PASSWORD, "different-passphrase"))
    monkeypatch.setattr(users, "getpass", lambda prompt: next(answers))
    assert users.main(["create", "--username", "alice", "--name", "Alice"], settings=settings) == 2
    assert not settings.auth_db_path.exists()

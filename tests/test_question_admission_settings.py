# -*- coding: utf-8 -*-
"""保护同步入口的线程池余量，配置与追踪指纹应反映实际准入策略。"""

from pathlib import Path

import pytest

from app.config import Settings
from tests.test_trace_configuration import _version


def test_waiting_and_execution_capacity_leave_default_pool_headroom() -> None:
    """不能配置48个阻塞槽并声称默认40线程池仍有健康检查余量。
    """
    with pytest.raises(ValueError):
        Settings(_env_file=None, question_max_active=16, question_max_waiting=32)
    settings = Settings(_env_file=None, question_max_active=16, question_max_waiting=16)
    assert settings.question_max_waiting == 16


def test_admission_policy_changes_trace_configuration_version(tmp_path: Path) -> None:
    """不同排队与个人窗口配置不能在后台报告里混成同一版本。

    Args:
        tmp_path: 隔离追踪目录。
    """
    first = _version(tmp_path / "first.sqlite3")
    changed = _version(tmp_path / "second.sqlite3", question_max_active=2)
    assert first != changed

# -*- coding: utf-8 -*-
"""配置指纹必须区分执行行为，不纳入凭据和本地数据路径。"""

from pathlib import Path
from uuid import uuid4

from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from app.config import Settings


def _version(path: Path, *, temperature: float | None = None, **options: object) -> str:
    root = build_application_container(
        Settings(_env_file=None, trace_db_path=path, **options),
        RuntimeOptions(llm_temperature=temperature),
    )
    try:
        with root.trace_service.start("你好", request_id=uuid4(), entry="query") as trace:
            pass
        return root.trace_repository.get(trace.id).configuration_version
    finally:
        root.close()


def test_execution_settings_change_configuration_version(tmp_path: Path) -> None:
    """固定字符串不能支撑按配置比较，影响行为的参数必须改变版本指纹。

    Args:
        tmp_path: 隔离所有追踪库。
    """
    original = _version(tmp_path / "first.sqlite3")
    changed = _version(tmp_path / "second.sqlite3", understanding_confidence_threshold=0.9)
    overridden = _version(tmp_path / "third.sqlite3", temperature=0.0)
    assert original != changed
    assert original != overridden


def test_credentials_and_storage_paths_do_not_change_version(tmp_path: Path) -> None:
    """不能通过哈希全部Settings间接保存凭据指纹，也不应按目录分裂配置。

    Args:
        tmp_path: 隔离所有追踪库。
    """
    original = _version(tmp_path / "first.sqlite3", llm_api_key="first-test-credential")
    changed = _version(
        tmp_path / "other.sqlite3", llm_api_key="another-test-credential",
        docs_path=tmp_path / "other-docs", auth_db_path=tmp_path / "auth.sqlite3",
    )
    assert changed == original

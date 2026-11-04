# -*- coding: utf-8 -*-
"""以真实子进程验证文件锁互斥与进程退出后的自动释放。"""

import subprocess
import sys
from pathlib import Path

from app.infrastructure.persistence.ingestion_execution import LocalIngestionExecutionGuard


CHILD = """
import sys
from pathlib import Path
from app.application.errors import IngestionWorkerUnavailableError
from app.infrastructure.persistence.ingestion_execution import LocalIngestionExecutionGuard
try:
    with LocalIngestionExecutionGuard(Path(sys.argv[1])).hold():
        print('acquired')
except IngestionWorkerUnavailableError:
    print('busy')
    sys.exit(3)
"""


def test_another_process_cannot_enter_held_execution_guard(tmp_path: Path) -> None:
    """仅使用RLock或进程内标记无法阻止这个独立解释器。

    Args:
        tmp_path: 隔离锁文件。
    """
    path = tmp_path / "worker.lock"
    with LocalIngestionExecutionGuard(path).hold():
        before = path.stat().st_ino
        blocked = subprocess.run(
            [sys.executable, "-c", CHILD, str(path)], capture_output=True,
            text=True, timeout=5, check=False,
        )
        assert blocked.returncode == 3
        assert blocked.stdout.strip() == "busy"
    acquired = subprocess.run(
        [sys.executable, "-c", CHILD, str(path)], capture_output=True,
        text=True, timeout=5, check=False,
    )
    assert acquired.returncode == 0
    assert acquired.stdout.strip() == "acquired"
    assert path.stat().st_ino == before


def test_process_exit_releases_guard_even_without_context_cleanup(tmp_path: Path) -> None:
    """恢复不能依赖旧进程执行finally；退出后操作系统必须能释放完整执行权。

    Args:
        tmp_path: 隔离锁文件。
    """
    path = tmp_path / "worker.lock"
    code = """
import os
import sys
from pathlib import Path
from app.infrastructure.persistence.ingestion_execution import LocalIngestionExecutionGuard
with LocalIngestionExecutionGuard(Path(sys.argv[1])).hold():
    os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], timeout=5, check=False,
    )
    assert result.returncode == 0
    with LocalIngestionExecutionGuard(path).hold():
        assert path.is_file()

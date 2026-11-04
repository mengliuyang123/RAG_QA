# -*- coding: utf-8 -*-
"""用操作系统文件锁排斥同一物理索引的后台执行器。"""

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.application.errors import IngestionWorkerUnavailableError


class LocalIngestionExecutionGuard:
    """锁文件不含正文，不能在锁仍持有时删除或替换该文件。"""

    def __init__(self, path: Path) -> None:
        """保存由组合根按Chroma目录和collection派生的锁路径。

        Args:
            path: 同一物理索引共享的固定文件路径。

        Returns:
            None。
        """
        self._path = path

    @contextmanager
    def hold(self) -> Iterator[None]:
        """非阻塞取得排他锁，整个执行退出才释放，不依赖租约过期。

        Yields:
            完整任务的执行权。
        """
        descriptor = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self._path, flags, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            raise IngestionWorkerUnavailableError("入库执行器暂时不可用") from error
        try:
            yield
        finally:
            os.close(descriptor)

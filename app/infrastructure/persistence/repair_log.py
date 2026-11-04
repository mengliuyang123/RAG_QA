# -*- coding: utf-8 -*-
"""持久化文档删除部分失败的安全 JSONL 修复记录。"""

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID


class RepairLog:
    """持久化不含路径、正文和原始异常的删除修复记录。"""

    def __init__(
        self,
        path: Path,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """保存 repair JSONL 路径和可替换时钟。

        Args:
            path: 文档删除 repair JSONL 文件路径。
            now: 返回带时区当前时间的函数。
        """
        self._path = path
        self._now = now or (lambda: datetime.now(timezone.utc))

    def append_document_deletion(
        self,
        doc_id: UUID,
        filename: str,
    ) -> None:
        """同步追加并 fsync 一条稳定文件清理失败记录。

        Args:
            doc_id: 已完成索引删除的文档 UUID。
            filename: 不含目录的安全原始文件名。

        Returns:
            None。

        Raises:
            OSError: repair 目录或文件无法写入。
            ValueError: 时钟返回无时区时间。
        """
        occurred_at = self._now()
        if occurred_at.tzinfo is None:
            raise ValueError("repair 时间必须包含时区")
        utc_time = occurred_at.astimezone(timezone.utc)
        timestamp = utc_time.isoformat().replace("+00:00", "Z")
        payload = {
            "doc_id": str(doc_id),
            "filename": filename,
            "occurred_at": timestamp,
            "error_category": "source_file_cleanup_failed",
        }
        content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(content + "\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())

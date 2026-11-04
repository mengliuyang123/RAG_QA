# -*- coding: utf-8 -*-
"""源文件持久化与恢复快照端口，不在任务表保存文件路径。"""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from app.domain.ingestion_jobs import IngestionJobSubmission


class IngestionSources(Protocol):
    """入队前持久化，执行前从受控身份取得已验证私有快照。"""

    def persist(self, submission: IngestionJobSubmission, content: bytes) -> None:
        """排他保存并同步源文件，不覆盖已有身份。

        Args:
            submission: 服务端生成的稳定身份与完整性资料。
            content: 实际上传内容。

        Returns:
            None。
        """

    def discard(self, submission: IngestionJobSubmission) -> None:
        """仅清理确定未入队的该份源文件。

        Args:
            submission: 本次提交拥有的文件身份。

        Returns:
            None。
        """

    def verified_snapshot(
        self, submission: IngestionJobSubmission,
    ) -> AbstractContextManager[Path]:
        """取得大小和摘要一致的私有快照，退出时清理快照。

        Args:
            submission: 持久登记的身份与完整性资料。

        Returns:
            不受原文件随后变动影响的解析路径上下文。
        """

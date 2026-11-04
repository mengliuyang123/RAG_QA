# -*- coding: utf-8 -*-
"""有界读取原子发布文件，并复用发布器的严格格式校验。"""

import os
import stat
from pathlib import Path

from pydantic import JsonValue

from app.application.errors import EvaluationReportNotFoundError
from app.application.errors import EvaluationReportStorageError
from evaluation.reporting import EvaluationReport


MAX_REPORT_BYTES = 32 * 1024 * 1024


class FileEvaluationReportReader:
    """不遍历运行目录、不补造指标，只读取latest.json。"""

    def __init__(self, results_path: Path) -> None:
        """保存管理员部署时选定的发布目录。

        Args:
            results_path: 可信配置，不能通过请求传入路径。

        Returns:
            None。
        """
        self._path = results_path / "latest.json"

    def latest(self) -> dict[str, JsonValue]:
        """在同一文件描述符内检查类型、限制大小并校验完整性。

        Returns:
            与发布器一致的完整报告。
        """
        try:
            # NONBLOCK避免FIFO挂起；NOFOLLOW拒绝链接，打开后再检查同一inode。
            descriptor = os.open(
                self._path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            )
        except FileNotFoundError as error:
            raise EvaluationReportNotFoundError("尚无已发布评估") from error
        except OSError as error:
            raise EvaluationReportStorageError("评估报告不可读") from error
        try:
            with os.fdopen(descriptor, "rb") as source:
                metadata = os.fstat(source.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("评估报告不是普通文件")
                if metadata.st_size > MAX_REPORT_BYTES:
                    raise ValueError("评估报告超过读取上限")
                content = source.read(MAX_REPORT_BYTES + 1)
                if len(content) > MAX_REPORT_BYTES:
                    raise ValueError("评估报告超过读取上限")
            report = EvaluationReport.model_validate_json(content)
            if report.status != "complete":
                raise ValueError("latest不是完整发布报告")
            return report.model_dump(mode="json")
        except (OSError, ValueError) as error:
            raise EvaluationReportStorageError("评估报告不可读") from error

    def schema(self) -> dict[str, JsonValue]:
        """直接从发布模型提供Schema，避免手写契约漂移。

        Returns:
            发布器同源JSON Schema。
        """
        return EvaluationReport.model_json_schema()

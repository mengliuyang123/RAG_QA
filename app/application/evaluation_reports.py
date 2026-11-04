# -*- coding: utf-8 -*-
"""后台报告只读用例，不依赖评估运行时或模型。"""

from pydantic import JsonValue

from app.application.ports.evaluation_reports import EvaluationReportReader


class EvaluationReportService:
    """通过发布存储端口提供报告和协议描述。"""

    def __init__(self, reader: EvaluationReportReader) -> None:
        """保存唯一组合根选定的读取器。

        Args:
            reader: 已发布文件的读取端口。

        Returns:
            None。
        """
        self._reader = reader

    def latest(self) -> dict[str, JsonValue]:
        """返回完整报告，缺失与损坏由稳定异常区分。

        Returns:
            发布时定义的报告对象。
        """
        return self._reader.latest()

    def schema(self) -> dict[str, JsonValue]:
        """提供发布格式的机器可读契约。

        Returns:
            当前JSON Schema。
        """
        return self._reader.schema()

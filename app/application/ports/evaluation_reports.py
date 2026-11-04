# -*- coding: utf-8 -*-
"""已发布报告读取契约，与具体评估执行器解耦。"""

from typing import Protocol

from pydantic import JsonValue


class EvaluationReportReader(Protocol):
    """返回经发布格式校验的JSON与其同源Schema。"""

    def latest(self) -> dict[str, JsonValue]:
        """读取最近完整发布报告，不启动评估。

        Returns:
            发布格式的完整JSON对象。
        """
        ...

    def schema(self) -> dict[str, JsonValue]:
        """提供与当前读取验证器一致的结构。

        Returns:
            JSON Schema。
        """
        ...

# -*- coding: utf-8 -*-
"""声明与具体模型 SDK 无关的结构化输出端口。"""

from typing import Protocol
from typing import TypeVar

from pydantic import BaseModel


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class StructuredOutputProvider(Protocol):
    """生成通过指定业务 schema 校验的模型结果。"""

    def generate(
        self,
        messages: list[dict[str, object]],
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        """调用模型并校验结构化结果。

        Args:
            messages: 完整聊天消息。
            response_model: 需要满足的业务模型类型。

        Returns:
            通过字段及跨字段业务校验的模型。

        Raises:
            LLMTimeoutError: 请求超时。
            LLMGenerationError: 调用或响应协议无效。
        """

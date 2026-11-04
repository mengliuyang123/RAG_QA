# -*- coding: utf-8 -*-
"""以显式协议约束聊天模型，不在失败后静默降级。"""

from collections.abc import Callable
from typing import Any
from typing import Protocol

from langchain_core.utils.function_calling import convert_to_openai_function
from openai import APITimeoutError

from app.application.errors import LLMGenerationError
from app.application.errors import LLMTimeoutError
from app.application.ports.structured_output import ResponseModel
from app.application.tracing import provider_call
from app.infrastructure.generation.usage import extract_token_usage
from app.infrastructure.conversations.json_protocol import parse_quality_object


class _ChatModel(Protocol):
    def bind(self, **options: object) -> Any:
        """绑定本次请求的输出协议。

        Args:
            options: provider 请求选项。

        Returns:
            未绑定检索工具的可调用模型。
        """

    def invoke(self, messages: list[dict[str, object]]) -> object:
        """返回模型的完整消息。

        Args:
            messages: 完整聊天消息。

        Returns:
            带正文与完成元数据的消息。
        """


class ModelStructuredOutputProvider:
    """复用聊天客户端，以原生 schema 或显式兼容模式生成结果。"""

    def __init__(
        self, model: _ChatModel, mode: str = "json_schema",
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        """保存由组合根装配的模型和协议。

        Args:
            model: 未绑定检索工具的聊天模型。
            mode: json_schema、json_object 或 prompt；不自动探测。
            close_callback: 适配器拥有的客户端释放函数。

        Returns:
            None。

        Raises:
            ValueError: 模式不受支持。
        """
        if mode not in {"json_schema", "json_object", "prompt"}:
            raise ValueError("不支持的结构化输出模式")
        self._model = model
        self._origin = getattr(model, "openai_api_base", None) or "unknown"
        self._model_name = getattr(model, "model_name", None) or "unknown"
        self._mode = mode
        self._close_callback = close_callback

    def close(self) -> None:
        """幂等释放由工厂转移所有权的客户端。

        Returns:
            None。
        """
        callback = self._close_callback
        self._close_callback = None
        if callback is not None:
            callback()

    def generate(
        self,
        messages: list[dict[str, object]],
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        """请求唯一完整结果，provider 约束之后仍执行本地严格校验。

        Args:
            messages: 完整聊天消息。
            response_model: Pydantic 业务契约。

        Returns:
            满足业务契约的模型结果。

        Raises:
            LLMTimeoutError: provider 请求超时。
            LLMGenerationError: 请求失败、拒绝、截断或 schema 不合法。
        """
        try:
            model = self._model
            if self._mode == "json_schema":
                # Python 默认值不代表 provider 允许省略字段；原生严格协议
                # 要求所有属性显式返回，nullable 字段也不能缺失。
                specification = convert_to_openai_function(
                    response_model, strict=True,
                )
                model = model.bind(response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__, "strict": True,
                        "schema": specification["parameters"],
                    },
                })
            elif self._mode == "json_object":
                model = model.bind(response_format={"type": "json_object"})
            purpose = {
                "QueryUnderstanding": "understanding",
                "EvidenceAssessment": "evidence_judge",
                "AnswerAssessment": "answer_judge",
                "LaneJudgement": "evaluation_judge",
            }.get(response_model.__name__, "structured")
            with provider_call(purpose, self._origin, self._model_name) as call:
                response = model.invoke(messages)
                usage = getattr(response, "usage_metadata", None)
                metadata = getattr(response, "response_metadata", {})
                call.usage = extract_token_usage(usage or metadata.get("token_usage"))
                call.response_model = metadata.get("model_name")
            metadata = getattr(response, "response_metadata", {})
            additional = getattr(response, "additional_kwargs", {})
            if metadata.get("finish_reason") in {"length", "content_filter"}:
                raise ValueError("结构化响应未完整完成")
            if additional.get("refusal") or getattr(response, "tool_calls", []):
                raise ValueError("模型拒绝或返回了非预期工具调用")
            content = getattr(response, "content", None)
            if not isinstance(content, str):
                raise ValueError("结构化响应不是文本")
            payload = parse_quality_object(
                content, allow_wrapping=self._mode == "prompt",
            )
            return response_model.model_validate(payload)
        except APITimeoutError as exc:
            raise LLMTimeoutError("结构化模型请求超时") from exc
        except Exception as exc:
            raise LLMGenerationError("结构化模型响应无效或调用失败") from exc

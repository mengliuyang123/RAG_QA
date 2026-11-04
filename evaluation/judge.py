# -*- coding: utf-8 -*-
"""使用独立 OpenAI 兼容模型评估回答语义正确性。"""

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings
from evaluation.checkpoints import InvalidRemoteResponseError


class JudgeResponseError(InvalidRemoteResponseError):
    """独立裁判响应不是严格有限分数 JSON 时抛出。"""


class JudgeScore(BaseModel):
    """独立裁判给出的一项有界语义正确性评分。

    Attributes:
        score: 零到一之间的有限正确性分数。
        reason: 不超过 500 字符的简短判定理由。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)


class JudgeClient:
    """只持有独立裁判模型配置的语义评分客户端。"""

    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float,
        reasoning_effort: str | None = None,
    ) -> None:
        """保存已经通过独立性预检的客户端参数。

        Args:
            client: 兼容 OpenAI chat completions 的客户端。
            model: 独立裁判模型名。
            temperature: 固定裁判温度。
            reasoning_effort: 可选裁判推理强度。
        """
        self._client = client
        self._model = model
        self._temperature = temperature
        self._request_options: dict[str, object] = {}
        if reasoning_effort is not None:
            self._request_options["reasoning_effort"] = reasoning_effort

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        client: Any | None = None,
    ) -> "JudgeClient":
        """在任何客户端构造前验证裁判完整性和独立性。

        Args:
            settings: 生成模型、评估路径和裁判配置。
            client: 测试或外部生命周期管理的兼容客户端。

        Returns:
            已绑定独立裁判模型的客户端。

        Raises:
            ValueError: 裁判配置缺失、路径冲突或模型身份不独立。
        """
        settings.validate_evaluation_settings(require_judge=True)
        api_key = settings.evaluation_judge_api_key
        base_url = settings.evaluation_judge_base_url
        model = settings.evaluation_judge_model
        if api_key is None or base_url is None or model is None:
            raise ValueError("独立裁判配置不完整")
        if client is None:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.model_max_retries,
            )
        return cls(
            client=client,
            model=model,
            temperature=settings.evaluation_judge_temperature,
            reasoning_effort=(
                settings.evaluation_judge_reasoning_effort
            ),
        )

    def score_semantic_correctness(
        self,
        question: str,
        reference: str,
        response: str,
    ) -> JudgeScore:
        """根据问题、参考答案和回答请求严格语义评分。

        Args:
            question: 原始问题。
            reference: 人工或官方参考答案。
            response: 待评分模型回答。

        Returns:
            有界分数和简短理由。

        Raises:
            JudgeResponseError: 裁判未返回严格、有限、有界 JSON。
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an independent evaluator. Score semantic "
                    "correctness from 0 to 1. Return only the requested JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Reference answer:\n{reference}\n\n"
                    f"Response:\n{response}"
                ),
            },
        ]
        result = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            **self._request_options,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_correctness",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number"},
                            "reason": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 500,
                            },
                        },
                        "required": ["score", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        try:
            content = result.choices[0].message.content
            if not isinstance(content, str):
                raise TypeError("裁判 content 不是字符串")
            payload = json.loads(content)
            return JudgeScore.model_validate(payload)
        except (
            AttributeError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
            ValidationError,
        ) as exc:
            raise JudgeResponseError("裁判响应格式无效") from exc

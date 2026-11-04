# -*- coding: utf-8 -*-
"""在调用模型前约束包含工具和证据的完整输入。"""

import json
from collections.abc import Mapping
from collections.abc import Sequence

from app.application.ports.token_counting import TokenCounter
from app.domain.errors import PromptContextTooLargeError


class PromptBudgetGuard:
    """以组合根提供的计数器检查完整提示词，不篡改用户或证据。"""

    def __init__(self, counter: TokenCounter, max_input_tokens: int) -> None:
        """保存已为输出预留空间的输入预算。

        Args:
            counter: 与历史窗口一致的明确计数器。
            max_input_tokens: 总上下文减去最大输出后的输入上限。
        """
        if max_input_tokens <= 0:
            raise ValueError("完整提示词预算必须为正数")
        self._counter = counter
        self._limit = max_input_tokens

    def ensure(
        self, messages: Sequence[Mapping[str, object]], *,
        extra_payload: Mapping[str, object] | None = None,
    ) -> None:
        """检查消息和附加协议，不能只统计裸问题长度。

        Args:
            messages: 即将发往模型的全部消息。
            extra_payload: 工具或结构化输出 schema 等非消息负载。

        Returns:
            None。

        Raises:
            PromptContextTooLargeError: 完整输入超出约定预算。
        """
        serialized = json.dumps(
            {"messages": list(messages), "extra": dict(extra_payload or {})},
            ensure_ascii=False, separators=(",", ":"),
        )
        # 协议角色标记也占上下文，不能把正文计数伪称为完整输入 usage。
        estimated = self._counter.count_text(serialized) + 16 * len(messages)
        if estimated > self._limit:
            raise PromptContextTooLargeError(
                "系统、历史与检索证据超过模型上下文预算",
            )

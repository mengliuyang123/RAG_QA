# -*- coding: utf-8 -*-
"""按输入预算选择连续历史后缀，不改动权威历史原文。"""

from collections.abc import Iterable

from app.application.ports.token_counting import TokenCounter
from app.domain.conversations import ConversationTurn
from app.domain.errors import HistoryContextTooLargeError


class HistoryWindowBuilder:
    """优先保留最近用户全文，在预算内裁剪助手回答副本。"""

    def __init__(
        self,
        counter: TokenCounter,
        token_budget: int,
        assistant_token_limit: int = 512,
        message_overhead_tokens: int = 16,
    ) -> None:
        """注入计数口径与扣除其他模型上下文后的可用预算。

        Args:
            counter: 模型匹配或明确的保守计数适配器。
            token_budget: 当前问题与历史可共享的预算。
            assistant_token_limit: 每条助手历史的最大预算。
            message_overhead_tokens: 每条消息的角色／分隔符保留量。

        Returns:
            None。
        """
        if token_budget <= 0 or assistant_token_limit <= 0:
            raise ValueError("输入和助手消息预算必须大于零")
        if message_overhead_tokens < 0:
            raise ValueError("消息开销不能为负数")
        self._counter = counter
        self._budget = token_budget
        self._assistant_limit = assistant_token_limit
        self._overhead = message_overhead_tokens

    def validate_question(self, question: str) -> int:
        """先为当前用户全文保留额度，超限时明确报错。

        Args:
            question: 完整原问题。

        Returns:
            留给历史的剩余额度。

        Raises:
            HistoryContextTooLargeError: 问题本身已超过可用上下文。
        """
        remaining = self._budget - self._counter.count_text(question)
        remaining -= self._overhead
        if remaining < 0:
            raise HistoryContextTooLargeError("问题超过模型输入预算")
        return remaining

    def build(
        self,
        question: str,
        newest_first: Iterable[ConversationTurn],
    ) -> list[ConversationTurn]:
        """消费按倒序分页的历史，只返回正序的连续最近窗口。

        Args:
            question: 当前问题，始终完整保留。
            newest_first: 最新在前的惰性历史迭代器。

        Returns:
            仅助手回答可能被截断的消息副本，按对话正序排列。
        """
        remaining = self.validate_question(question)
        selected = []
        for turn in newest_first:
            user_cost = self._counter.count_text(turn.user_message)
            pair_overhead = self._overhead * 2
            available = remaining - user_cost - pair_overhead
            if available <= 0:
                break
            assistant_budget = min(available, self._assistant_limit)
            assistant = self._truncate_assistant(
                turn.assistant_message, assistant_budget,
            )
            if not assistant:
                break
            assistant_cost = self._counter.count_text(assistant)
            # Counter 适配器须遵守同口径上限；宁可拒绝旧轮次，也不超预算。
            if assistant_cost > assistant_budget:
                raise ValueError("计数器返回的前缀超过预算")
            selected.append(turn.model_copy(update={
                "assistant_message": assistant,
            }))
            remaining -= user_cost + assistant_cost + pair_overhead
        selected.reverse()
        return selected

    def _truncate_assistant(self, text: str, limit: int) -> str:
        if self._counter.count_text(text) <= limit:
            return text
        marker = "..."
        marker_cost = self._counter.count_text(marker)
        if marker_cost >= limit:
            return self._counter.truncate_text(text, limit)
        prefix = self._counter.truncate_text(text, limit - marker_cost)
        return prefix + marker

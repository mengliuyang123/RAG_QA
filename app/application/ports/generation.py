# -*- coding: utf-8 -*-
"""声明语言模型生成端口。"""

from collections.abc import Iterator
from typing import Protocol

from app.domain.agent_quality import EvidenceItem


class GroundedAnswerGenerator(Protocol):
    """定义只能使用已批准、保留原编号证据的生成能力。"""

    def generate_grounded(
        self, question: str, history: list[dict[str, str]],
        evidence: list[EvidenceItem],
    ) -> str:
        """基于批准证据生成待规则审核的草稿。

        Args:
            question: 当前原始问题。
            history: 已按预算选取的历史。
            evidence: 只能用于回答的批准证据。

        Returns:
            尚未发布的完整草稿。
        """


class LLMProvider(Protocol):
    """定义答案生成所依赖的最小语言模型能力。"""

    def generate(self, messages: list[dict[str, str]]) -> str:
        """根据消息返回完整回答。

        Args:
            messages: 已构建的聊天消息。

        Returns:
            语言模型生成的完整回答。
        """

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """逐个返回回答增量。

        Args:
            messages: 已构建的聊天消息。

        Returns:
            逐个产生回答增量的迭代器。
        """

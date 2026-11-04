# -*- coding: utf-8 -*-
"""隔离模型输入计数与安全截断方式。"""

from typing import Protocol


class TokenCounter(Protocol):
    """提供模型匹配计数或明确的保守上界，不代表计费 usage。"""

    def count_text(self, text: str) -> int:
        """统计文本的输入预算消耗。

        Args:
            text: 原始文本。

        Returns:
            非负 token 数或已声明的保守上界。
        """

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """在相同计数口径下安全取前缀。

        Args:
            text: 待截断的助手文本。
            max_tokens: 前缀预算上限。

        Returns:
            不超过预算、Unicode 完整的文本前缀。
        """

# -*- coding: utf-8 -*-
"""声明管理员预设问题的持久化边界，不访问私人会话。"""

from typing import Protocol

from app.domain.suggestions import CuratedQuestionsUpdate
from app.domain.suggestions import SuggestedQuestion


class CuratedQuestionRepository(Protocol):
    """仅提供显式预设的查询和原子替换。"""

    def list_curated(self, kb_id: str | None, limit: int) -> tuple[SuggestedQuestion, ...]:
        """读取当前仍注册知识库的预设。

        Args:
            kb_id: 可选范围。
            limit: 返回上限。

        Returns:
            稳定有序的已发布问题。
        """

    def replace_curated(
        self, kb_id: str, update: CuratedQuestionsUpdate,
    ) -> tuple[SuggestedQuestion, ...]:
        """原子替换单库预设，同一内容重试保留稳定ID。

        Args:
            kb_id: 既有知识库。
            update: 管理员明确提供的规范问题。

        Returns:
            本次发布后的有序集合。
        """

# -*- coding: utf-8 -*-
"""隔离规则／轻量模型理解与确定性路由。"""

from typing import Protocol

from app.domain.knowledge_bases import KnowledgeBase
from app.domain.routing import QueryUnderstanding


class QuestionUnderstanding(Protocol):
    """理解输出必须在执行前按领域契约再次验证。"""

    def understand(
        self, question: str, history: list[dict[str, str]],
        catalog: list[KnowledgeBase],
    ) -> QueryUnderstanding:
        """在最多一次轻量模型调用内理解问题。

        Args:
            question: 原问题。
            history: 预算内的历史。
            catalog: 后端目录快照，描述仅作为数据。

        Returns:
            待确定性路由验证的理解结果。
        """

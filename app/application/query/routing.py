# -*- coding: utf-8 -*-
"""以纯应用规则决定执行起点，隔离模型输出与执行权限。"""

import math
import re

from app.domain.routing import QueryUnderstanding
from app.domain.routing import RoutingDecision


class DeterministicRouter:
    """仅根据已校验数据和明确配置选择起始通道。"""

    def __init__(self, confidence_threshold: float = 0.8) -> None:
        """保存低置信度回退阈值。

        Args:
            confidence_threshold: (0, 1] 内的有限值，后续由评估校准。

        Returns:
            None。
        """
        if not math.isfinite(confidence_threshold):
            raise ValueError("阈值必须是有限值")
        if not 0 < confidence_threshold <= 1:
            raise ValueError("置信度阈值必须位于 (0, 1]")
        self._threshold = confidence_threshold

    def decide(
        self,
        understanding: QueryUnderstanding,
        requested_lane: str = "auto",
    ) -> RoutingDecision:
        """自动路由或设置管理员测试的起点，不改变检索策略和范围。

        Args:
            understanding: 已校验的理解结果。
            requested_lane: auto、fast、agent；接口层另行检查管理员权限。

        Returns:
            满足知识问题必须检索的起始决策。

        Raises:
            ValueError: 试图强制指定不支持的通道。
        """
        if requested_lane not in {"auto", "fast", "agent"}:
            raise ValueError("起始通道只能是 auto、fast 或 agent")
        reason = "intent"
        if self.fallback_required(understanding):
            understanding = QueryUnderstanding.fallback(
                understanding.standalone_query,
            )
            reason = "low_confidence"
        if requested_lane != "auto":
            return RoutingDecision(
                understanding=understanding, lane=requested_lane,
                reason="requested_lane",
            )
        if understanding.intent == "ambiguous":
            lane = "clarify"
        elif understanding.intent in {"chitchat", "meta", "out_of_scope"}:
            lane = "direct"
        elif understanding.complexity == "single_hop":
            lane = "fast"
        else:
            lane = "agent"
            reason = "complexity"
        return RoutingDecision(
            understanding=understanding, lane=lane, reason=reason,
        )

    def fallback_required(self, understanding: QueryUnderstanding) -> bool:
        """使理解层和路由共用同一置信度阈值。

        Args:
            understanding: 已校验的结构化结果。

        Returns:
            是否必须回到原问题执行保守检索。
        """
        return understanding.confidence < self._threshold


class RuleQueryUnderstanding:
    """P1 的保守理解器：只对白名单完整问题走无检索路径。"""

    _CHITCHAT = frozenset({
        "你好", "您好", "hello", "hi", "谢谢", "再见",
    })
    _META = frozenset({"你是谁", "你能做什么", "如何使用这个系统"})

    def understand(self, question: str) -> QueryUnderstanding:
        """完整匹配闲聊与系统介绍，其余保守作为知识问题。

        Args:
            question: 原始用户问题。

        Returns:
            不依赖模型或网络的理解结果。

        Raises:
            ValueError: 问题为空白。
        """
        normalized = question.strip()
        if not normalized:
            raise ValueError("问题不能为空")
        exact = normalized.rstrip("。！？!? ").casefold()
        if exact in self._CHITCHAT:
            return QueryUnderstanding(
                intent="chitchat", standalone_query=normalized,
                confidence=1.0,
            )
        if exact in self._META:
            return QueryUnderstanding(
                intent="meta", standalone_query=normalized, confidence=1.0,
            )
        complexity = "single_hop"
        if re.search(r"比较|对比|区别|差异", normalized):
            complexity = "comparison"
        elif re.search(r"汇总|统计|列出所有", normalized):
            complexity = "aggregation"
        elif re.search(r"先.+再", normalized):
            complexity = "multi_hop"
        if complexity == "single_hop":
            return QueryUnderstanding.fallback(normalized)
        return QueryUnderstanding(
            standalone_query=normalized, complexity=complexity,
            confidence=0.8,
        )

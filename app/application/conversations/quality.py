# -*- coding: utf-8 -*-
"""声明在线质量端口及与模型无关的证据、查询策略。"""

from typing import Protocol

from app.application.errors import ConversationAgentError
from app.domain.agent_quality import AnswerAssessment
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem


class ConversationQualityEvaluator(Protocol):
    """隔离图编排与具体模型质量判断。"""

    def assess_evidence(
        self,
        question: str,
        history: list[dict[str, str]],
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment:
        """判断本轮累计证据能否回答原问题。

        Args:
            question: 原始用户问题。
            history: 最近正式对话。
            evidence: 稳定编号的累计候选证据。

        Returns:
            充分性、相关编号、信息缺口和可选补查查询。
        """

    def assess_answer(
        self,
        question: str,
        history: list[dict[str, str]],
        answer: str,
        evidence: list[EvidenceItem],
    ) -> AnswerAssessment:
        """判断回答是否获得引用证据支持且覆盖问题。

        Args:
            question: 原始用户问题。
            history: 最近正式对话。
            answer: 已通过引用编号校验的草稿。
            evidence: 草稿实际引用且已批准的证据。

        Returns:
            支持性、完整性和修复反馈。
        """


def normalize_query(query: str) -> str:
    """统一空白和大小写，用于阻止重复执行查询。

    Args:
        query: 模型提出的检索文本。

    Returns:
        只用于比较的规范查询。
    """
    return " ".join(query.split()).casefold()


def select_evidence(
    evidence: list[EvidenceItem],
    assessment: EvidenceAssessment,
) -> list[EvidenceItem]:
    """根据已验证编号筛选证据，保留原始稳定编号。

    Args:
        evidence: 当前轮的原始证据注册表。
        assessment: 评估器的结构化判断。

    Returns:
        按注册顺序排列的批准证据。

    Raises:
        ConversationAgentError: 评估器引用了未知证据。
    """
    available = {item.number for item in evidence}
    selected = set(assessment.relevant_numbers)
    if not selected.issubset(available):
        raise ConversationAgentError("质量判断引用了未知证据")
    return [item for item in evidence if item.number in selected]

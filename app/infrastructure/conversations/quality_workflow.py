# -*- coding: utf-8 -*-
"""连接结构化质量判断与本轮状态，隔离模型调用和图拓扑。"""

import json

from app.application.conversations.quality import (
    ConversationQualityEvaluator,
    select_evidence,
)
from app.application.errors import ConversationAgentError
from app.domain.agent_quality import AnswerAssessment
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.execution import ExecutionBudget
from app.domain.execution import RetrievalAttempt
from app.infrastructure.conversations.state import AgentState


class AgentQualityWorkflow:
    """编排证据筛选、补检索决定与答案支持性判断。"""

    def __init__(
        self,
        evaluator: ConversationQualityEvaluator,
        max_retrieval_calls: int,
    ) -> None:
        """保存可替换评估器与共享检索预算。

        Args:
            evaluator: 无检索工具的在线质量判断器。
            max_retrieval_calls: 每轮允许的检索调用总数。
        """
        self._evaluator = evaluator
        self._max_retrieval_calls = max_retrieval_calls

    def assess_evidence(self, state: AgentState) -> AgentState:
        """判断累计证据，筛选来源并决定是否存在有效补查。

        Args:
            state: 含本轮原始证据注册表的图状态。

        Returns:
            质量结果、批准证据、判断计数和可选新查询。

        Raises:
            ConversationAgentError: 判断预算耗尽或评估协议无效。
        """
        count = state.get("evidence_assessments", 0)
        if count >= 3:
            raise ConversationAgentError("证据判断超出预算")
        evidence = self._items(state.get("evidence", []))
        if not evidence:
            assessment = EvidenceAssessment(
                sufficient=False, relevant_numbers=[],
                missing_information=["未检索到可用证据"], next_query=None,
            )
        else:
            assessment = EvidenceAssessment.model_validate(
                self._evaluator.assess_evidence(
                    state["current_question"], state["history"], evidence,
                )
            )
        approved = select_evidence(evidence, assessment)
        next_query = assessment.next_query
        if next_query is not None:
            attempt = RetrievalAttempt(
                query=next_query, kb_ids=tuple(state.get("kb_ids") or ()),
            )
            budget = ExecutionBudget.model_validate(state["execution_budget"])
            if any(previous.key == attempt.key for previous in budget.attempts):
                next_query = None
            if state["retrieval_calls"] >= self._max_retrieval_calls:
                next_query = None
        return {
            "evidence_assessments": count + 1,
            "evidence_assessment": assessment.model_dump(),
            "approved_evidence": [item.model_dump() for item in approved],
            "next_query": next_query,
            "quality_issues": assessment.missing_information,
        }

    def assess_answer(self, state: AgentState) -> AgentState:
        """仅用实际引用的批准证据验证回答。

        Args:
            state: 已通过引用编号校验的候选回答。

        Returns:
            判断计数与结果；不通过时清空待发布答案。

        Raises:
            ConversationAgentError: 判断预算耗尽或评估协议无效。
        """
        count = state.get("answer_assessments", 0)
        if count >= 2:
            raise ConversationAgentError("答案判断超出预算")
        positions = {
            int(item["position"]) for item in state["final_citations"]
        }
        evidence = self._items(state["approved_evidence"])
        cited = [item for item in evidence if item.number in positions]
        assessment = AnswerAssessment.model_validate(
            self._evaluator.assess_answer(
                state["current_question"], state["history"],
                state["draft_answer"], cited,
            )
        )
        # 证据已被判为不足时，回答本就只覆盖部分问题，覆盖度不再作为通过条件。
        passed = assessment.supported and (
            assessment.complete or bool(state.get("answer_partial"))
        )
        if not passed:
            return self._rejected(state, count, assessment)
        return {
            "answer_assessments": count + 1,
            "answer_assessment": assessment.model_dump(),
            "quality_issues": assessment.issues,
            "final_answer": state["draft_answer"],
            "final_citations": state["final_citations"],
        }

    def reject_answer(self, state: AgentState, issue: str) -> AgentState:
        """审核调用本身失败时，按审核未通过记录本次回答。

        Args:
            state: 已通过引用编号校验的候选回答。
            issue: 写入质量问题的说明。

        Returns:
            与审核未通过相同形状的状态更新。
        """
        assessment = AnswerAssessment(
            supported=False, complete=False, issues=[issue],
        )
        return self._rejected(state, state.get("answer_assessments", 0), assessment)

    @staticmethod
    def _rejected(
        state: AgentState, count: int, assessment: AnswerAssessment,
    ) -> AgentState:
        # 保留引用已校验的回答，审核预算用尽时可按部分依据发布而非拒答。
        return {
            "answer_assessments": count + 1,
            "answer_assessment": assessment.model_dump(),
            "quality_issues": assessment.issues,
            "rejected_answer": state["draft_answer"],
            "rejected_citations": state["final_citations"],
            "final_answer": "",
            "final_citations": [],
        }

    @staticmethod
    def approved_messages(state: AgentState) -> list[dict[str, object]]:
        """剔除旧工具消息中的未批准正文，保持调用 ID 配对。

        Args:
            state: 原始消息和当前批准证据注册表。

        Returns:
            仅向模型暴露批准证据的消息副本。
        """
        approved = {
            int(item["number"]) for item in state["approved_evidence"]
        }
        messages = []
        for original in state["messages"]:
            message = dict(original)
            if message.get("role") == "tool":
                payload = json.loads(str(message["content"]))
                if isinstance(payload, list):
                    payload = [
                        item for item in payload
                        if item["number"] in approved
                    ]
                    message["content"] = json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":"),
                    )
            elif message.get("tool_calls"):
                # 工具调用前的模型文字可能含未经证据支持的猜测。
                message["content"] = ""
            messages.append(message)
        return messages

    @staticmethod
    def _items(payloads: list[dict[str, object]]) -> list[EvidenceItem]:
        return [EvidenceItem.model_validate(item) for item in payloads]

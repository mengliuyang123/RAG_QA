# -*- coding: utf-8 -*-
"""统一Fast、自动扩查和Agent升级的确定性知识通道执行策略。"""

from collections.abc import Callable
from time import monotonic
from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict

from app.application.conversations.ports import ConversationAgent
from app.application.conversations.ports import StatusCallback
from app.application.conversations.progress import notify_status
from app.application.errors import ConversationAgentError
from app.application.errors import ConversationTimeoutError
from app.application.execution_control import check_cancellation
from app.application.tracing import traced_step
from app.application.knowledge.scopes import ScopePlan
from app.application.query.fast_lane import FastLane
from app.application.query.scope_guard import validate_handoff_scope
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.conversations import ConversationTurn
from app.domain.conversations import KNOWLEDGE_REFUSAL_ANSWER
from app.domain.errors import ExecutionDeadlineError
from app.domain.execution import EvidenceBundle
from app.domain.retrieval import RetrievalStrategy


class KnowledgeRunResult(BaseModel):
    """保存实际最终通道、升级与扩查状态，而不是复用检索策略字段。

    Attributes:
        result: 校验后的最终回答。
        lane: 实际完成执行的通道。
        lane_policy: 线上策略或禁止Agent升级的独立评估策略。
        scope_expanded: 是否在初次检索后扩大范围。
        escalated: 是否确实由Fast进入Agent。
        searched_kb_ids: 全轮实际查询范围的并集。
    """

    model_config = ConfigDict(frozen=True)

    result: AgentResult
    lane: Literal["fast", "agent"]
    lane_policy: Literal["online", "fast_only"]
    scope_expanded: bool
    escalated: bool
    searched_kb_ids: tuple[str, ...]


class KnowledgeRunner:
    """执行知识请求，不能绕过检索、重置预算或扩大显式范围。"""

    def __init__(
        self, fast: FastLane, agent: ConversationAgent,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """保存通道及相同基准的单调时钟。

        Args:
            fast: 检索、证据判断和稳定引用生成通道。
            agent: 多跳、补查与答案审核图。
            clock: 不在升级时重置的预算时钟。
        """
        self._fast = fast
        self._agent = agent
        self._clock = clock

    def run(
        self, conversation_id: UUID, question: str,
        history: list[ConversationTurn], *, scope: ScopePlan,
        bundle: EvidenceBundle, query: str | None = None,
        starting_lane: Literal["fast", "agent"] = "fast",
        strategy: RetrievalStrategy | str = RetrievalStrategy.HYBRID_RERANK,
        top_k: int = 5,
        lane_policy: Literal["online", "fast_only"] = "online",
        on_status: StatusCallback | None = None,
        sub_queries: tuple[str, ...] = (),
    ) -> KnowledgeRunResult:
        """按范围和策略执行一轮，全部通道共用输入中的预算快照。

        Args:
            conversation_id: 业务会话或无状态请求的执行ID。
            question: 当前原始问题。
            history: 预算内的正式历史。
            scope: 经目录验证的范围计划。
            bundle: 在理解阶段之前建立的预算及可复用证据。
            query: 理解层独立查询；不覆盖业务原问题。
            starting_lane: 执行起点，不保证最终停留在Fast。
            strategy: Fast与图工具共同使用的检索策略。
            top_k: 本轮共同的结果上限。
            lane_policy: fast_only仅用于独立评估，不允许升级到Agent。
            on_status: 立即通知扩查和升级的阶段回调。
            sub_queries: 理解层候选子查询，只提供规划提示。

        Returns:
            已校验结果与独立于检索策略的执行维度。
        """
        if starting_lane not in {"fast", "agent"}:
            raise ValueError("知识通道起点必须为fast或agent")
        if lane_policy not in {"online", "fast_only"}:
            raise ValueError("未知通道策略")
        if starting_lane == "agent" and lane_policy == "fast_only":
            raise ValueError("fast_only不允许Agent起点")
        selected_strategy = RetrievalStrategy(strategy)
        actual_query = query or question
        self._ensure_time(bundle)
        current_kbs = scope.initial_kb_ids
        expanded = False

        if starting_lane == "fast":
            history_payload = []
            for turn in history:
                history_payload.append({
                    "user_message": turn.user_message,
                    "assistant_message": turn.assistant_message,
                })
            validate_handoff_scope(bundle, current_kbs)
            for phase in ("initial", "expanded"):
                self._ensure_time(bundle)
                if bundle.budget.retrieval_calls >= 3:
                    break
                output = self._fast.run(
                    question, history_payload, query=actual_query,
                    bundle=bundle, kb_ids=current_kbs,
                    strategy=selected_strategy, top_k=top_k, on_status=on_status,
                )
                bundle = output.bundle
                if output.result is not None:
                    return self._finish(
                        output.result, "fast", lane_policy, expanded, False,
                        current_kbs,
                    )
                can_expand = (
                    phase == "initial"
                    and output.escalation_reason == "insufficient_evidence"
                    and scope.expansion_kb_ids is not None
                    and bundle.budget.retrieval_calls < 3
                    and bundle.budget.evidence_assessments < 3
                )
                if not can_expand:
                    break
                current_kbs = scope.expansion_kb_ids
                expanded = True
                with traced_step("scope_expansion", {
                    "kb_ids": list(current_kbs),
                }):
                    self._notify(on_status, "expanding", current_kbs)

            assessment = bundle.assessment
            sufficient = assessment is not None and assessment.sufficient
            if lane_policy == "fast_only" or (
                bundle.budget.retrieval_calls >= 3 and not sufficient
            ):
                return self._finish(
                    self._refusal(bundle), "fast", lane_policy, expanded,
                    False, current_kbs,
                )
            with traced_step("escalation", {"kb_ids": list(current_kbs)}):
                self._notify(on_status, "escalating", current_kbs)
            escalated = True
        else:
            # 多跳问题可能天然跨库，自动模式从全可用范围开始；显式仍不扩大。
            if scope.request.mode == "auto":
                current_kbs = scope.all_kb_ids
            escalated = False

        self._ensure_time(bundle)
        result = self._agent.run(
            conversation_id, question, history, on_status,
            seed=bundle, kb_ids=current_kbs, strategy=selected_strategy,
            top_k=top_k, initial_query=actual_query,
            **({"sub_queries": sub_queries} if sub_queries else {}),
        )
        return self._finish(
            result, "agent", lane_policy, expanded, escalated, current_kbs,
        )

    def _finish(
        self, result: AgentResult, lane: Literal["fast", "agent"],
        policy: Literal["online", "fast_only"], expanded: bool,
        escalated: bool, current_kbs: tuple[str, ...],
    ) -> KnowledgeRunResult:
        bundle = result.execution
        if bundle is None:
            raise ConversationAgentError("知识通道未返回可验证的执行快照")
        self._ensure_time(bundle)
        validate_handoff_scope(bundle, current_kbs)
        searched = set()
        for attempt in bundle.budget.attempts:
            searched.update(attempt.kb_ids)
        return KnowledgeRunResult(
            result=result, lane=lane, lane_policy=policy,
            scope_expanded=expanded, escalated=escalated,
            searched_kb_ids=tuple(sorted(searched)),
        )

    def _ensure_time(self, bundle: EvidenceBundle) -> None:
        check_cancellation()
        try:
            bundle.budget.ensure_time(self._clock())
        except ExecutionDeadlineError as exc:
            raise ConversationTimeoutError("知识通道整轮时间预算已耗尽") from exc

    @staticmethod
    def _refusal(bundle: EvidenceBundle) -> AgentResult:
        return AgentResult(
            answer=KNOWLEDGE_REFUSAL_ANSWER,
            answer_mode=AnswerMode.RETRIEVAL if bundle.budget.retrieval_calls
            else AnswerMode.DIRECT,
            retrieval_calls=bundle.budget.retrieval_calls,
            citations=[], execution=bundle,
        )

    @staticmethod
    def _notify(
        callback: StatusCallback | None, phase: str,
        kb_ids: tuple[str, ...] = (),
    ) -> None:
        notify_status(callback, phase, kb_ids=kb_ids)

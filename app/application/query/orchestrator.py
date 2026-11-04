# -*- coding: utf-8 -*-
"""两种HTTP入口共用的分层问题编排器，不负责正式历史提交。"""

from collections.abc import Callable
from contextlib import nullcontext
from time import monotonic
from uuid import UUID
from uuid import uuid4
from typing import Literal

from app.application.conversations.ports import StatusCallback
from app.application.errors import ConversationAgentError
from app.application.errors import ConversationTimeoutError
from app.application.errors import LLMError
from app.application.execution_control import check_cancellation
from app.application.knowledge.scopes import ScopeResolver
from app.application.knowledge.scopes import ScopePlan
from app.application.ports.knowledge_bases import KnowledgeBaseRepository
from app.application.ports.understanding import QuestionUnderstanding
from app.application.query.knowledge_runner import KnowledgeRunner
from app.application.query.routing import DeterministicRouter
from app.application.tracing import TraceService
from app.application.tracing import traced_step
from app.domain.answer_metadata import AnswerMetadata
from app.domain.answer_metadata import ClarificationInfo
from app.domain.answer_metadata import EvidenceStats
from app.domain.answer_metadata import RefusalInfo
from app.domain.answer_metadata import ScopeInfo
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.conversations import ConversationTurn
from app.domain.conversations import KNOWLEDGE_REFUSAL_ANSWER
from app.domain.errors import ExecutionDeadlineError
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.domain.retrieval import RetrievalStrategy
from app.domain.routing import KnowledgeScope
from app.domain.routing import QueryUnderstanding


class QueryOrchestrator:
    """统一理解、确定性路由与知识执行，不提供绕过审核的legacy分支。"""

    def __init__(
        self, understanding: QuestionUnderstanding, router: DeterministicRouter,
        catalog: KnowledgeBaseRepository, scope_resolver: ScopeResolver,
        knowledge_runner: Callable[[], KnowledgeRunner], *,
        strategy: RetrievalStrategy | str, top_k: int = 5,
        turn_timeout_seconds: float = 180.0,
        clock: Callable[[], float] = monotonic,
        traces: TraceService | None = None,
    ) -> None:
        """保存业务策略与由组合根注入的惰性知识执行器。

        Args:
            understanding: 规则与结构化理解端口。
            router: 确定性路由器。
            catalog: 后端目录。
            scope_resolver: 强制执行范围的应用策略。
            knowledge_runner: 组合根提供的惰性获取函数，不在应用层创建适配器。
            strategy: 未指定策略时的配置档。
            top_k: 默认结果上限。
            turn_timeout_seconds: 理解、Fast和图共用的时间预算。
            clock: 单调时钟。
            traces: 由组合根提供的追踪服务，独立单元测试可不启用。
        """
        if turn_timeout_seconds <= 0 or not 1 <= top_k <= 20:
            raise ValueError("执行器预算或结果上限无效")
        self._understanding = understanding
        self._router = router
        self._catalog = catalog
        self._scope_resolver = scope_resolver
        self._knowledge_runner = knowledge_runner
        self._strategy = RetrievalStrategy(strategy)
        self._top_k = top_k
        self._timeout = turn_timeout_seconds
        self._clock = clock
        self._traces = traces

    def run(
        self, conversation_id: UUID, question: str,
        history: list[ConversationTurn], on_status: StatusCallback | None = None,
        *, kb_scope: KnowledgeScope | None = None, requested_lane: str = "auto",
        strategy: RetrievalStrategy | str | None = None, top_k: int | None = None,
        debug: bool = False,
        execution_entry: Literal["query", "conversation", "evaluation"] = "conversation",
        user_id: UUID | None = None,
        lane_policy: Literal["online", "fast_only"] = "online",
    ) -> AgentResult:
        """执行同一安全链路，只返回可发布的答案和元数据。

        Args:
            conversation_id: 执行标识；是否保存为业务会话由调用方决定。
            question: 原始用户问题。
            history: 已按token预算选取的历史。
            on_status: 阶段回调。
            kb_scope: 用户范围，默认自动。
            requested_lane: 管理测试的起点，只接受auto/fast/agent。
            strategy: 管理测试可选的三种检索策略。
            top_k: 本轮结果上限。
            debug: 由入口投影召回诊断，不影响判断或发布规则。
            execution_entry: 后端入口类型，不是模型或HTTP用户可改写的通道。
            user_id: 后端验证的当前用户；离线评估可不提供。
            lane_policy: 仅评估入口可用fast_only，不暴露给HTTP请求。

        Returns:
            正式历史和两种HTTP外壳都能使用的已验证结果。
        """
        if lane_policy != "online":
            if execution_entry != "evaluation" or lane_policy != "fast_only":
                raise ValueError("非线上通道策略仅用于评估")
            if requested_lane != "fast":
                raise ValueError("fast_only评估必须从fast开始")
        if self._traces is None:
            return self._execute(
                conversation_id, question, history, on_status,
                kb_scope=kb_scope, requested_lane=requested_lane,
                strategy=strategy, top_k=top_k, debug=debug, lane_policy=lane_policy,
            )
        prepared = self._traces.claim_execution_trace(
            question, execution_id=conversation_id, entry=execution_entry,
            user_id=user_id, lane_policy=lane_policy,
        )
        if prepared is not None:
            trace_context = nullcontext(prepared)
        else:
            trace_context = self._traces.start(
                question,
                request_id=(
                    conversation_id if execution_entry == "query" else uuid4()
                ),
                entry=execution_entry,
                user_id=user_id,
                lane_policy=lane_policy,
                conversation_id=(
                    conversation_id if execution_entry == "conversation" else None
                ),
            )
        with trace_context as trace:
            result = self._execute(
                conversation_id, question, history, on_status,
                kb_scope=kb_scope, requested_lane=requested_lane,
                strategy=strategy, top_k=top_k, debug=debug, lane_policy=lane_policy,
            )
            metadata = result.metadata
            if metadata is None:
                raise ConversationAgentError("统一结果缺少路由元数据")
            trace.set_result(
                lane=metadata.lane, outcome=metadata.outcome,
                scope_expanded=metadata.scope_expanded,
                escalated=metadata.escalated,
            )
            return AgentResult.model_validate({
                **result.model_dump(), "metadata": {
                    **metadata.model_dump(), "trace_id": str(trace.id),
                },
            })

    def _execute(
        self, conversation_id: UUID, question: str,
        history: list[ConversationTurn], on_status: StatusCallback | None = None,
        *, kb_scope: KnowledgeScope | None = None, requested_lane: str = "auto",
        strategy: RetrievalStrategy | str | None = None, top_k: int | None = None,
        debug: bool = False,
        lane_policy: Literal["online", "fast_only"] = "online",
    ) -> AgentResult:
        original = question.strip()
        if not original:
            raise ValueError("问题不能为空")
        selected_strategy = self._strategy if strategy is None else RetrievalStrategy(strategy)
        selected_top_k = self._top_k if top_k is None else top_k
        if not 1 <= selected_top_k <= 20:
            raise ValueError("top_k必须在1到20之间")
        bundle = EvidenceBundle(budget=ExecutionBudget(
            deadline=self._clock() + self._timeout,
        ))
        scope_request = kb_scope or KnowledgeScope()
        plan = self._scope_resolver.resolve(scope_request)
        if on_status is not None:
            on_status("understanding", None)
        self._ensure_time(bundle)
        history_payload = [
            {"user_message": turn.user_message, "assistant_message": turn.assistant_message}
            for turn in history
        ]
        with traced_step("understanding", {
            "query": original, "history_turns": len(history_payload),
        }) as step:
            try:
                output = self._understanding.understand(
                    original, history_payload, self._catalog.list(),
                )
                payload = output.model_dump() if isinstance(output, QueryUnderstanding) else output
                understood = QueryUnderstanding.model_validate(payload)
            except (ValueError, LLMError, ConversationAgentError):
                understood = QueryUnderstanding.fallback(original)
            self._ensure_time(bundle)
            if self._router.fallback_required(understood):
                # 低置信度时不保留模型改写，避免用错误问题执行保守回退。
                understood = QueryUnderstanding.fallback(original)
            step.set_output({
                **understood.model_dump(mode="json"),
                "understanding_confidence": understood.confidence,
            })
        if scope_request.mode == "auto" and understood.preferred_kb_ids:
            try:
                # 固定首次目录快照，不在理解之后重新读取或信任模型生成的目录。
                plan = ScopePlan(
                    request=scope_request,
                    initial_kb_ids=understood.preferred_kb_ids,
                    all_kb_ids=plan.all_kb_ids,
                )
            except ValueError:
                pass
        with traced_step("routing", {
            "requested_lane": requested_lane, "strategy": selected_strategy.value,
        }) as step:
            decision = self._router.decide(understood, requested_lane)
            step.set_output({
                "lane": decision.lane, "reason": decision.reason,
                "target_kbs": list(plan.initial_kb_ids),
            })
        scope_info = ScopeInfo(
            mode=scope_request.mode, requested=scope_request.kb_ids, initial=(),
        )
        if decision.lane in {"direct", "clarify"}:
            return self._without_retrieval(understood, scope_info, bundle)
        outcome = self._knowledge_runner().run(
            conversation_id, original, history, bundle=bundle, scope=plan,
            query=understood.standalone_query, starting_lane=decision.lane,
            strategy=selected_strategy, top_k=selected_top_k,
            on_status=on_status,
            sub_queries=understood.sub_queries,
            **({"lane_policy": lane_policy} if lane_policy != "online" else {}),
        )
        result = outcome.result
        execution = result.execution
        if execution is None:
            raise ConversationAgentError("知识执行结果缺少执行快照")
        approved = len(execution.approved_numbers)
        refused = result.answer == KNOWLEDGE_REFUSAL_ANSWER
        refusal = None
        if refused:
            missing = result.quality_issues
            if not missing and execution.assessment is not None:
                missing = tuple(execution.assessment.missing_information)
            refusal = RefusalInfo(
                reason="insufficient_evidence" if approved else "no_evidence",
                found_summary=f"检索到{len(execution.evidence)}段资料，{approved}段相关。",
                missing=missing or ("缺少足以回答问题的可靠证据",),
                suggested_kb_id=self._suggested_kb(understood, plan),
            )
        metadata = AnswerMetadata(
            lane=outcome.lane, outcome="refused" if refused else "answered",
            # 拒答没有可发布的结论，依据一律为 ungrounded；
            # 是否找到过相关资料由 refusal.reason 和 evidence_stats 表达。
            # 证据不足仍作答或审核未通过的回答标为 partial，提示用户核对。
            grounding=(
                "ungrounded" if refused
                else "partial" if result.partial else "grounded"
            ),
            kb_scope=ScopeInfo(
                mode=scope_request.mode, requested=scope_request.kb_ids,
                searched=outcome.searched_kb_ids,
                initial=(
                    execution.budget.attempts[0].kb_ids
                    if execution.budget.attempts else ()
                ),
            ), scope_expanded=outcome.scope_expanded,
            escalated=outcome.escalated,
            evidence_stats=EvidenceStats(
                used=len(result.citations),
                excluded=len(execution.evidence) - approved,
                retrieved=len(execution.evidence),
                relevant=approved,
            ), refusal=refusal,
        )
        self._ensure_time(execution)
        return AgentResult.model_validate({**result.model_dump(), "metadata": metadata})

    @staticmethod
    def _suggested_kb(
        understood: QueryUnderstanding, plan: ScopePlan,
    ) -> str | None:
        if plan.request.mode != "explicit" or understood.intent != "knowledge":
            return None
        candidates = understood.preferred_kb_ids
        if not set(candidates).issubset(plan.all_kb_ids):
            return None
        # 只提示目录中可能相关的库，不把建议当作跨范围检索的授权。
        for kb_id in candidates:
            if kb_id not in plan.request.kb_ids:
                return kb_id
        return None

    @staticmethod
    def _without_retrieval(
        understood: QueryUnderstanding, scope: ScopeInfo, bundle: EvidenceBundle,
    ) -> AgentResult:
        clarification = None
        refusal = None
        lane = "direct"
        outcome = "answered"
        if understood.intent == "ambiguous":
            lane = "clarify"
            outcome = "clarification"
            answer = understood.clarification
            clarification = ClarificationInfo(question=answer)
        elif understood.intent == "out_of_scope":
            outcome = "refused"
            answer = "这个问题不属于当前知识库的服务范围，请补充相关业务背景。"
            refusal = RefusalInfo(
                reason="no_evidence", found_summary="未参考知识库。",
                missing=("与知识库相关的问题或更明确的业务背景",),
            )
        elif understood.intent == "meta":
            answer = "我是知识库问答助手，可以查找资料并提供带来源的答案；证据不足时会说明缺口。"
        else:
            answer = "你好！我可以帮你查找和解答知识库中的问题。"
        return AgentResult(
            answer=answer, answer_mode=AnswerMode.DIRECT,
            retrieval_calls=0, citations=[], execution=bundle,
            metadata=AnswerMetadata(
                lane=lane, outcome=outcome, grounding="ungrounded",
                kb_scope=scope, refusal=refusal, clarification=clarification,
            ),
        )

    def _ensure_time(self, bundle: EvidenceBundle) -> None:
        check_cancellation()
        try:
            bundle.budget.ensure_time(self._clock())
        except ExecutionDeadlineError as exc:
            raise ConversationTimeoutError("问答整轮时间预算已耗尽") from exc

    def validate_scope(self, scope: KnowledgeScope) -> None:
        """在流响应头发出前验证范围，不调用任何模型。

        Args:
            scope: 用户范围。

        Returns:
            None。
        """
        self._scope_resolver.resolve(scope)

# -*- coding: utf-8 -*-
"""执行确定性的检索、证据裁判、生成与引用校验快速通道。"""

from collections.abc import Callable
from time import monotonic

from pydantic import BaseModel
from pydantic import ConfigDict

from app.application.conversations.ports import StatusCallback
from app.application.conversations.progress import notify_status
from app.application.conversations.quality import ConversationQualityEvaluator
from app.application.conversations.quality import select_evidence
from app.application.errors import ConversationTimeoutError
from app.application.errors import ConversationAgentError
from app.application.errors import EmbeddingTimeoutError
from app.application.errors import LLMTimeoutError
from app.application.execution_control import check_cancellation
from app.application.execution_control import deadline_scope
from app.application.ports.generation import GroundedAnswerGenerator
from app.application.query.grounding import validate_grounded_citations
from app.application.query.service import RetrieverProtocol
from app.application.query.scope_guard import validate_handoff_scope
from app.application.tracing import traced_step
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.errors import ExecutionDeadlineError
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.domain.execution import RetrievalAttempt
from app.domain.retrieval import RetrievalStrategy
from app.domain.query_debug import RetrievalObservation


class FastLaneResult(BaseModel):
    """返回可发布结果或可无损升级的证据种子。

    Attributes:
        result: 引用校验通过的结果；需要升级时为空。
        bundle: 含实际已消费预算的证据快照。
        escalation_reason: 不足或引用非法的升级原因。
    """

    model_config = ConfigDict(frozen=True)

    result: AgentResult | None
    bundle: EvidenceBundle
    escalation_reason: str | None


class FastLane:
    """先裁判再生成，不在任何步骤流出未校验的答案。"""

    def __init__(
        self, retriever: RetrieverProtocol,
        quality_evaluator: ConversationQualityEvaluator,
        generator: GroundedAnswerGenerator,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """保存由组合根选择的检索配置档和共享模型端口。

        Args:
            retriever: 已装配向量、关键词及可选重排的检索服务。
            quality_evaluator: 与 Agent 共用协议的证据判断器。
            generator: 只接受批准证据的生成端口。
            strategy: 明确的检索策略，不因远端错误自动降级。
            clock: 与共享预算使用同一基准的单调时钟。
        """
        self._retriever = retriever
        self._quality = quality_evaluator
        self._generator = generator
        self._strategy = RetrievalStrategy(strategy)
        self._clock = clock

    def run(
        self, question: str, history: list[dict[str, str]], *,
        bundle: EvidenceBundle, query: str | None = None, top_k: int = 5,
        on_status: StatusCallback | None = None,
        kb_ids: tuple[str, ...] | None = None,
        strategy: RetrievalStrategy | str | None = None,
    ) -> FastLaneResult:
        """进行一次检索和裁判，不足时保留状态交给编排器。

        Args:
            question: 当前原始问题。
            history: 按预算选取的正式历史副本。
            bundle: 已有证据和整轮共享预算。
            query: 理解层提供的独立查询；为空时使用原问题。
            top_k: 本次候选返回上限。
            on_status: 立即发送阶段状态的回调。
            kb_ids: 本次召回前的知识库范围，不能由模型工具参数更改。
            strategy: 本轮统一策略；为空时使用装配配置档。

        Returns:
            校验后的回答或包含缺口、次数、截止时间的升级种子。
        """
        actual_query = (query or question).strip()
        budget = bundle.budget
        selected_strategy = RetrievalStrategy(strategy or self._strategy)
        validate_handoff_scope(bundle, kb_ids)
        if budget.evidence_assessments >= 3:
            return FastLaneResult(
                result=None, bundle=bundle,
                escalation_reason="judgement_budget_exhausted",
            )
        try:
            budget = budget.record_retrieval(
                RetrievalAttempt(query=actual_query, kb_ids=kb_ids or ()),
                now=self._clock(),
            )
            self._notify(
                on_status, "retrieving", budget.retrieval_calls, kb_ids or (),
            )
            self._ensure_time(budget)
            with (
                deadline_scope(budget.deadline, self._clock),
                traced_step("retrieval", {
                    "query": actual_query, "kb_ids": list(kb_ids or ()),
                    "strategy": selected_strategy.value, "top_k": top_k,
                }) as step,
            ):
                if kb_ids is None:
                    results = self._retriever.retrieve(
                        actual_query, strategy=selected_strategy, top_k=top_k,
                    )
                else:
                    results = self._retriever.retrieve(
                        actual_query, strategy=selected_strategy, top_k=top_k,
                        kb_ids=kb_ids,
                    )
                step.set_output({"results": [
                    {"chunk_id": item.chunk_id, "kb_id": item.kb_id,
                     "score": item.score, "rank": rank}
                    for rank, item in enumerate(results, start=1)
                ]})
            self._ensure_time(budget)
            if kb_ids is not None:
                for result in results:
                    if result.kb_id not in kb_ids:
                        raise ConversationAgentError("快速召回结果超出执行范围")
            evidence = list(bundle.evidence)
            known_chunks = {item.chunk_id for item in evidence}
            number = max((item.number for item in evidence), default=0)
            observations = list(bundle.retrieved)
            number_by_chunk = {item.chunk_id: item.number for item in evidence}
            for rank, result in enumerate(results[:top_k], start=1):
                position = number_by_chunk.get(result.chunk_id)
                if position is None:
                    position = number + 1
                observations.append(RetrievalObservation(
                    attempt=budget.retrieval_calls, rank=rank,
                    position=position, chunk_id=result.chunk_id,
                    score=result.score,
                ))
                if result.chunk_id in known_chunks:
                    continue
                number += 1
                known_chunks.add(result.chunk_id)
                number_by_chunk[result.chunk_id] = number
                evidence.append(EvidenceItem(
                    number=number, chunk_id=result.chunk_id,
                    doc_id=result.doc_id, filename=result.filename,
                    page=result.page, text=result.text,
                    kb_id=result.kb_id,
                    doc_updated_at=result.doc_updated_at,
                ))
            self._notify(on_status, "verifying")
            self._ensure_time(budget)
            with traced_step("evidence_judge", {
                "evidence_numbers": [item.number for item in evidence],
            }) as step:
                if evidence:
                    assessment = EvidenceAssessment.model_validate(
                        self._quality.assess_evidence(question, history, evidence),
                    )
                else:
                    assessment = EvidenceAssessment(
                        sufficient=False, relevant_numbers=[],
                        missing_information=["未检索到可用证据"], next_query=None,
                    )
                step.set_output(assessment.model_dump(mode="json"))
            self._ensure_time(budget)
            approved = select_evidence(evidence, assessment)
            budget = budget.model_copy(update={
                "evidence_assessments": budget.evidence_assessments + 1,
            })
            seed = EvidenceBundle(
                evidence=evidence,
                approved_numbers=[item.number for item in approved],
                assessment=assessment, budget=budget,
                retrieved=observations,
            )
            if not assessment.sufficient:
                return FastLaneResult(
                    result=None, bundle=seed,
                    escalation_reason="insufficient_evidence",
                )
            self._notify(on_status, "generating")
            self._ensure_time(budget)
            with traced_step("generation", {
                "approved_numbers": [item.number for item in approved],
            }) as step:
                answer = self._generator.generate_grounded(
                    question, history, approved,
                )
                step.set_output({"characters": len(answer)})
            self._ensure_time(budget)
            self._notify(on_status, "validating")
            try:
                with traced_step("citation_validation") as step:
                    citations = validate_grounded_citations(answer, approved)
                    step.set_output({
                        "positions": [citation.position for citation in citations],
                    })
            except ValueError:
                return FastLaneResult(
                    result=None, bundle=seed,
                    escalation_reason="invalid_citations",
                )
            return FastLaneResult(
                result=AgentResult(
                    answer=answer, answer_mode=AnswerMode.RETRIEVAL,
                    retrieval_calls=budget.retrieval_calls,
                    citations=citations, execution=seed,
                ), bundle=seed, escalation_reason=None,
            )
        except (
            ExecutionDeadlineError, EmbeddingTimeoutError, LLMTimeoutError,
        ) as exc:
            raise ConversationTimeoutError("快速通道整轮请求超时") from exc

    def _ensure_time(self, budget: ExecutionBudget) -> None:
        check_cancellation()
        budget.ensure_time(self._clock())

    @staticmethod
    def _notify(
        callback: StatusCallback | None, phase: str,
        attempt: int | None = None,
        kb_ids: tuple[str, ...] = (),
    ) -> None:
        notify_status(callback, phase, attempt, kb_ids)

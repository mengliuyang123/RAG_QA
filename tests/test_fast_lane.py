# -*- coding: utf-8 -*-
"""快速通道必须先判断证据，再生成并校验稳定编号。"""

import json
from collections.abc import Callable
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.errors import ConversationTimeoutError
from app.application.conversations.events import StatusConversationData
from app.application.query.fast_lane import FastLane
from app.application.query.generator import AnswerGenerator
from app.application.query.prompting import PromptBuilder
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import AnswerAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.domain.retrieval import RetrievalResult
from app.domain.retrieval import RetrievalStrategy
from app.infrastructure.conversations.langgraph_agent import (
    LangGraphConversationAgent,
)


class _Retriever:
    """只替代外部召回，保留快速通道的实际编排。"""

    def __init__(self) -> None:
        """初始化实际检索记录。"""
        self.calls: list[tuple[str, object, int]] = []

    def retrieve(
        self, question: str, strategy: RetrievalStrategy | str, top_k: int,
    ) -> list[RetrievalResult]:
        """按查询返回初查或补查的数据。

        Args:
            question: 实际检索文本。
            strategy: 服务端指定的策略。
            top_k: 候选上限。

        Returns:
            初查中的噪声和A，或补查中的B。
        """
        self.calls.append((question, strategy, top_k))
        if question == "B 存储方式":
            return [RetrievalResult(
                chunk_id="b", doc_id="b", filename="b.md",
                text="B 用 PostgreSQL", page=None, chunk_index=0, score=0.8,
            )]
        return [
            RetrievalResult(
                chunk_id="noise", doc_id="noise", filename="noise.md",
                text="UNRELATED_FAST 海洋生物", page=None,
                chunk_index=0, score=0.9,
            ),
            RetrievalResult(
                chunk_id="a", doc_id="a", filename="a.md",
                text="A 用 SQLite", page=None, chunk_index=1, score=0.8,
            ),
        ]


class _Quality:
    """初次判断可不足，补查后批准两份相关证据。"""

    def __init__(self, sufficient: bool = True) -> None:
        """配置初查是否充分。

        Args:
            sufficient: 初查预设结论。
        """
        self.sufficient = sufficient
        self.calls = 0

    def assess_evidence(
        self, question: str, history: list[dict[str, str]],
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment:
        """返回与实际累积来源一致的判断。

        Args:
            question: 原问题。
            history: 正式历史副本。
            evidence: 完整注册表。

        Returns:
            预设的初查或补查结论。
        """
        self.calls += 1
        if len(evidence) == 3:
            return EvidenceAssessment(
                sufficient=True, relevant_numbers=[2, 3],
                missing_information=[], next_query=None,
            )
        assert [item.number for item in evidence] == [1, 2]
        return EvidenceAssessment(
            sufficient=self.sufficient, relevant_numbers=[2],
            missing_information=[] if self.sufficient else ["缺少 B 存储"],
            next_query=None if self.sufficient else "B 存储方式",
        )

    def assess_answer(
        self, question: str, history: list[dict[str, str]],
        answer: str, evidence: list[EvidenceItem],
    ) -> AnswerAssessment:
        """批准测试中已确定受证据支持的完整答案。

        Args:
            question: 原问题。
            history: 正式历史。
            answer: 图生成的草稿。
            evidence: 实际引用的批准来源。

        Returns:
            两项均通过的判断。
        """
        return AnswerAssessment(supported=True, complete=True, issues=[])


class _Provider:
    """记录生成边界实际收到的消息。"""

    def __init__(self, answer: str = "A 使用 SQLite[2]") -> None:
        """保存预设草稿。

        Args:
            answer: 模型将返回的文本。
        """
        self.answer = answer
        self.messages: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        """记录批准证据构成的提示词。

        Args:
            messages: 实际生产提示词。

        Returns:
            预设草稿。
        """
        self.messages.append(messages)
        return self.answer


def _run(
    *, sufficient: bool = True, answer: str = "A 使用 SQLite[2]",
    clock: Callable[[], float] = lambda: 1.0,
):
    retriever = _Retriever()
    provider = _Provider(answer)
    statuses = []
    lane = FastLane(
        retriever=retriever, quality_evaluator=_Quality(sufficient),
        generator=AnswerGenerator(PromptBuilder(), provider), clock=clock,
    )
    result = lane.run(
        "它的存储是什么？", [{"user_message": "A", "assistant_message": "介绍"}],
        query="A 存储", bundle=EvidenceBundle(
            budget=ExecutionBudget(deadline=100.0),
        ), on_status=lambda phase, attempt: statuses.append((phase, attempt)),
    )
    return result, retriever, provider, statuses


def test_fast_generates_only_approved_chunks_with_original_numbers() -> None:
    """删掉第一条无关结果不能把第二条来源重新编号为1。"""
    output, retriever, provider, statuses = _run()
    assert output.result.answer == "A 使用 SQLite[2]"
    assert [item.position for item in output.result.citations] == [2]
    assert retriever.calls == [("A 存储", RetrievalStrategy.HYBRID_RERANK, 5)]
    assert "UNRELATED_FAST" not in json.dumps(provider.messages)
    assert "A 用 SQLite" in str(provider.messages)
    assert "它的存储是什么？" in str(provider.messages)
    assert output.bundle.budget.deadline == 100.0
    assert output.bundle.budget.retrieval_calls == 1
    assert output.bundle.budget.evidence_assessments == 1
    assert output.escalation_reason is None
    assert [phase for phase, _ in statuses] == [
        "retrieving", "verifying", "generating", "validating",
    ]


def test_insufficient_fast_returns_seed_without_generating() -> None:
    """不足时携带原证据和补查方向，不能先生成一个无依据回答。"""
    output, _, provider, _ = _run(sufficient=False)
    assert output.result is None
    assert provider.messages == []
    assert output.bundle.approved_numbers == (2,)
    assert output.bundle.assessment.next_query == "B 存储方式"
    assert output.escalation_reason == "insufficient_evidence"


def test_fast_status_callbacks_fit_shared_sse_contract() -> None:
    """接入会话外壳前，快速通道阶段必须能被公共事件模型接受。"""
    _, _, _, statuses = _run()
    payloads = []
    for phase, attempt in statuses:
        payloads.append(StatusConversationData(phase=phase, attempt=attempt))
    assert payloads[1].phase == "verifying"
    assert payloads[2].phase == "generating"


@pytest.mark.parametrize("answer", ["A 用 SQLite", "A 用 SQLite[1]", "A[9]"])
def test_invalid_fast_citation_is_never_published(answer: str) -> None:
    """缺少引用、引用噪声或虚构来源时只可升级，不能返回成功正文。

    Args:
        answer: 缺失或伪造引用的草稿。
    """
    output, _, _, _ = _run(answer=answer)
    assert output.result is None
    assert output.escalation_reason == "invalid_citations"
    assert output.bundle.assessment.sufficient is True


def test_fast_checks_deadline_after_retrieval_before_judging() -> None:
    """检索耗时越过截止时间后不得继续调用裁判或生成模型。"""
    ticks = iter([1.0, 1.0, 100.0])
    with pytest.raises(ConversationTimeoutError):
        _run(clock=lambda: next(ticks))


@pytest.mark.parametrize("phase", ["retrieving", "verifying", "generating"])
def test_status_callback_cannot_allow_an_external_call_after_deadline(
    phase: str,
) -> None:
    """阶段回调也可能阻塞，之后不得继续调用已经过期的外部操作。

    Args:
        phase: 模拟耗尽剩余时间的阶段。
    """
    now = [1.0]
    retriever = _Retriever()
    quality = _Quality()
    provider = _Provider()

    def on_status(current: str, attempt: int | None) -> None:
        if current == phase:
            now[0] = 100.0

    with pytest.raises(ConversationTimeoutError):
        FastLane(
            retriever, quality, AnswerGenerator(PromptBuilder(), provider),
            clock=lambda: now[0],
        ).run(
            "问题", [], bundle=EvidenceBundle(
                budget=ExecutionBudget(deadline=100.0),
            ), on_status=on_status,
        )
    if phase == "retrieving":
        assert retriever.calls == []
    if phase == "verifying":
        assert quality.calls == 0
    if phase == "generating":
        assert provider.messages == []


def test_real_fast_to_agent_handoff_only_searches_missing_information() -> None:
    """真实两个通道共用证据和预算，不重新调用首次查询。"""
    retriever = _Retriever()
    quality = _Quality(sufficient=False)
    fast = FastLane(
        retriever, quality, AnswerGenerator(PromptBuilder(), _Provider()),
        clock=lambda: 1.0,
    ).run(
        "比较 A/B", [], query="A 存储",
        bundle=EvidenceBundle(budget=ExecutionBudget(deadline=100.0)),
    )

    class Model:
        """只替代 Agent 的外部生成响应。"""

        def bind_tools(self, tools: object, *, strict: bool) -> "Model":
            """保持标准受控工具接口。

            Args:
                tools: 工具描述。
                strict: 参数要求。

            Returns:
                当前离线模型。
            """
            return self

        def invoke(self, messages: list[dict[str, object]]) -> AIMessage:
            """返回同时引用双方来源的完整答案。

            Args:
                messages: 图实际生成的消息。

            Returns:
                测试预设的完整模型消息。
            """
            assert "UNRELATED_FAST" not in str(messages)
            return AIMessage(content="A SQLite[2]，B PostgreSQL[3]")

    result = LangGraphConversationAgent(
        model=Model(), retrieval_service=retriever,
        checkpointer=InMemorySaver(), top_k=5,
        quality_evaluator=quality, clock=lambda: 2.0,
    ).run(uuid4(), "比较 A/B", [], seed=fast.bundle)
    assert result.answer == "A SQLite[2]，B PostgreSQL[3]"
    assert [call[0] for call in retriever.calls] == ["A 存储", "B 存储方式"]
    assert result.retrieval_calls == 2
    assert result.execution.budget.evidence_assessments == 2
    assert result.execution.budget.deadline == 100.0

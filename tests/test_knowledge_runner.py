# -*- coding: utf-8 -*-
"""用真实Fast与LangGraph验证范围扩大、升级及共享检索预算。"""

from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.conversations.events import StatusConversationData
from app.application.errors import ConversationTimeoutError
from app.application.knowledge.scopes import ScopePlan
from app.application.query.fast_lane import FastLane
from app.application.query.generator import AnswerGenerator
from app.application.query.knowledge_runner import KnowledgeRunner
from app.application.query.prompting import PromptBuilder
from app.application.retrieval.service import RetrievalService
from app.application.retrieval.service import VectorRetriever
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.domain.routing import KnowledgeScope
from app.domain.retrieval import RetrievalResult
from app.domain.retrieval import RetrievalStrategy
from app.infrastructure.conversations.langgraph_agent import (
    AGENT_REFUSAL_ANSWER,
    LangGraphConversationAgent,
)
from tests.test_fast_lane import _Provider
from tests.test_retriever import FakeQueryEmbedder
from tests.test_scoped_agent import _Model
from tests.test_scoped_agent import _Quality
from tests.test_scoped_retrieval import _stores


class _ScriptedQuality(_Quality):
    """仅替代远端裁判，实际执行两通道和本地双路召回。"""

    def __init__(self, verdicts: Sequence[EvidenceAssessment]) -> None:
        """保存判断脚本。

        Args:
            verdicts: 依次返回的证据结论。
        """
        super().__init__()
        self.verdicts = iter(verdicts)

    def assess_evidence(
        self, question: str, history: list[dict[str, str]],
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment:
        """返回脚本结论，不替代范围和编号逻辑。

        Args:
            question: 原问题。
            history: 正式历史。
            evidence: 实际累计证据。

        Returns:
            下一个确定性结论。
        """
        return next(self.verdicts)


class _RecordingRetriever:
    """记录真实检索服务收到的查询与范围。"""

    def __init__(self, service: RetrievalService) -> None:
        """保存真实服务。

        Args:
            service: 本地双召回实现。
        """
        self.service = service
        self.calls: list[tuple[str, tuple[str, ...] | None]] = []

    def retrieve(
        self, question: str, strategy: RetrievalStrategy | str, top_k: int,
        *, kb_ids: tuple[str, ...] | None = None,
    ) -> list[RetrievalResult]:
        """执行而非伪造本地检索。

        Args:
            question: 查询文本。
            strategy: 检索策略。
            top_k: 结果上限。
            kb_ids: 实际召回范围。

        Returns:
            本地真实索引命中。
        """
        self.calls.append((question, kb_ids))
        return self.service.retrieve(question, strategy, top_k, kb_ids=kb_ids)


def _missing(numbers: list[int], query: str | None) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=False, relevant_numbers=numbers,
        missing_information=["缺少另一份资料"], next_query=query,
    )


def _enough(numbers: list[int]) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=True, relevant_numbers=numbers,
        missing_information=[], next_query=None,
    )


def _plan(*, explicit=False, all_first=False) -> ScopePlan:
    return ScopePlan(
        request=KnowledgeScope(mode="explicit", kb_ids=("ops",))
        if explicit else KnowledgeScope(),
        initial_kb_ids=("dev", "ops") if all_first else ("ops",),
        all_kb_ids=("dev", "ops"),
    )


def _runner(tmp_path, verdicts, model_responses=(), clock=None):
    store, index = _stores(tmp_path)
    retriever = _RecordingRetriever(RetrievalService(
        VectorRetriever(store, FakeQueryEmbedder()), index,
    ))
    quality = _ScriptedQuality(verdicts)
    agent = LangGraphConversationAgent(
        _Model(model_responses), retriever, InMemorySaver(), top_k=1,
        quality_evaluator=quality, clock=clock or (lambda: 2.0),
    )
    fast = FastLane(
        retriever, quality, AnswerGenerator(PromptBuilder(), _Provider("整合[1][2]")),
        strategy="hybrid", clock=clock or (lambda: 1.0),
    )
    return KnowledgeRunner(fast, agent, clock=clock or (lambda: 2.0)), retriever


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(budget=ExecutionBudget(deadline=100.0))


def test_auto_expands_once_then_completes_in_fast_lane(tmp_path: Path) -> None:
    """扩查仍属于Fast，首次与全范围检索共用次数和截止时间。

    Args:
        tmp_path: 隔离索引目录。
    """
    runner, retriever = _runner(tmp_path, [
        _missing([1], "配置"), _enough([1, 2]),
    ])
    events = []
    output = runner.run(
        uuid4(), "整合资料", [], query="needle", bundle=_bundle(),
        scope=_plan(), strategy="hybrid", top_k=1,
        on_status=lambda phase, attempt: events.append(
            StatusConversationData(phase=phase, attempt=attempt),
        ),
    )
    assert output.lane == "fast"
    assert output.scope_expanded is True
    assert output.escalated is False
    assert output.result.retrieval_calls == 2
    assert output.result.execution.budget.deadline == 100.0
    assert retriever.calls == [("needle", ("ops",)), ("needle", ("dev", "ops"))]
    assert "expanding" in [event.phase for event in events]


def test_full_initial_scope_is_not_expanded_or_retrieved_again(tmp_path: Path) -> None:
    """已覆盖全部库时直接升级，Agent只补缺少的查询。

    Args:
        tmp_path: 隔离索引目录。
    """
    runner, retriever = _runner(
        tmp_path, [_missing([1], "配置"), _enough([1, 2])],
        [AIMessage(content="整合[1][2]")],
    )
    output = runner.run(
        uuid4(), "整合资料", [], query="needle", bundle=_bundle(),
        scope=_plan(all_first=True), strategy="hybrid", top_k=1,
    )
    assert output.lane == "agent"
    assert output.scope_expanded is False
    assert output.escalated is True
    assert retriever.calls == [("needle", ("dev", "ops")), ("配置", ("dev", "ops"))]


@pytest.mark.parametrize("sufficient", [True, False])
def test_expansion_and_agent_share_three_calls(
    tmp_path: Path, sufficient: bool,
) -> None:
    """第三次充分时完整回答；仍不足时按部分依据回答，且不能发起第四次查询。

    Args:
        tmp_path: 隔离索引目录。
        sufficient: 第三次检索后的充分性结论。
    """
    last = _enough([1, 2, 3]) if sufficient else _missing([1, 2, 3], "第四次")
    runner, retriever = _runner(tmp_path, [
        _missing([1], "配置"), _missing([1, 2], "unrelated"), last,
    ], [AIMessage(content="整合[1][2][3]")])
    output = runner.run(
        uuid4(), "整合资料", [], query="needle", bundle=_bundle(),
        scope=_plan(), strategy="hybrid", top_k=1,
    )
    assert output.result.retrieval_calls == 3
    assert output.scope_expanded and output.escalated
    assert output.result.answer == "整合[1][2][3]"
    assert output.result.partial is not sufficient
    assert retriever.calls == [
        ("needle", ("ops",)), ("needle", ("dev", "ops")),
        ("unrelated", ("dev", "ops")),
    ]


def test_explicit_scope_is_never_expanded_during_escalation(tmp_path: Path) -> None:
    """显式范围内证据不足只能补查同范围，之后按部分依据作答。

    Args:
        tmp_path: 隔离索引目录。
    """
    runner, retriever = _runner(tmp_path, [
        _missing([1], "研发资料"), _missing([1], None),
    ], [AIMessage(content="运维资料[1]")])
    output = runner.run(
        uuid4(), "整合资料", [], query="needle", bundle=_bundle(),
        scope=_plan(explicit=True), strategy="hybrid", top_k=1,
    )
    assert output.result.partial is True
    assert output.scope_expanded is False
    assert output.escalated is True
    assert retriever.calls == [("needle", ("ops",)), ("研发资料", ("ops",))]


def test_fast_only_is_distinct_from_online_fast_start(tmp_path: Path) -> None:
    """评估fast_only禁止升级，不得与线上lane=fast混为同一基线。

    Args:
        tmp_path: 隔离索引目录。
    """
    runner, retriever = _runner(tmp_path, [_missing([1], "研发资料")])
    output = runner.run(
        uuid4(), "整合资料", [], query="needle", bundle=_bundle(),
        scope=_plan(explicit=True), strategy="hybrid", top_k=1,
        lane_policy="fast_only",
    )
    assert output.result.answer == AGENT_REFUSAL_ANSWER
    assert output.lane == "fast"
    assert output.lane_policy == "fast_only"
    assert output.escalated is False
    assert len(retriever.calls) == 1


def test_bad_fast_citations_upgrade_without_retrieval_or_scope_expansion(
    tmp_path: Path,
) -> None:
    """证据已充分时引用非法只能升级审核，不能重复召回或扩大范围。

    Args:
        tmp_path: 隔离索引目录。
    """
    runner, retriever = _runner(
        tmp_path, [_enough([1])], [AIMessage(content="配置[1]")],
    )
    events = []
    output = runner.run(
        uuid4(), "资料", [], query="needle", scope=_plan(),
        bundle=_bundle(), strategy="hybrid", top_k=1,
        on_status=lambda phase, attempt: events.append(
            StatusConversationData(phase=phase, attempt=attempt),
        ),
    )
    assert output.result.answer == "配置[1]"
    assert output.escalated is True
    assert output.scope_expanded is False
    assert retriever.calls == [("needle", ("ops",))]
    assert "escalating" in [event.phase for event in events]


@pytest.mark.parametrize("explicit", [True, False])
def test_agent_start_keeps_explicit_scope_and_uses_understood_query(
    tmp_path: Path, explicit: bool,
) -> None:
    """复杂请求不先跑Fast，显式范围不变，自动模式从全可用范围开始。

    Args:
        tmp_path: 隔离索引目录。
        explicit: 是否由用户明确选定运维库。
    """
    runner, retriever = _runner(tmp_path, [_enough([1])], [
        AIMessage(content="资料[1]"),
    ])
    output = runner.run(
        uuid4(), "它的资料呢？", [], query="needle", bundle=_bundle(),
        scope=_plan(explicit=explicit), strategy="hybrid", top_k=1,
        starting_lane="agent",
    )
    assert output.lane == "agent"
    assert output.escalated is False
    assert output.scope_expanded is False
    assert retriever.calls == [
        ("needle", ("ops",) if explicit else ("dev", "ops")),
    ]


def test_deadline_is_not_reset_between_fast_and_agent(tmp_path: Path) -> None:
    """升级事件之后已经过期时，不能再进入补查或生成。

    Args:
        tmp_path: 隔离索引目录。
    """
    now = [1.0]
    runner, retriever = _runner(
        tmp_path, [_missing([1], "配置")], clock=lambda: now[0],
    )

    def on_status(phase: str, attempt: int | None) -> None:
        if phase == "escalating":
            now[0] = 100.0

    with pytest.raises(ConversationTimeoutError):
        runner.run(
            uuid4(), "资料", [], query="needle", bundle=_bundle(),
            scope=_plan(all_first=True), strategy="hybrid", top_k=1,
            on_status=on_status,
        )
    assert len(retriever.calls) == 1

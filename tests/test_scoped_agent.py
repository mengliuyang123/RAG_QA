# -*- coding: utf-8 -*-
"""范围在Fast、图工具、证据与升级预算之间必须一致。"""

from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from app.application.errors import ConversationAgentError
from app.application.query.fast_lane import FastLane
from app.application.query.generator import AnswerGenerator
from app.application.query.prompting import PromptBuilder
from app.application.retrieval.service import RetrievalService
from app.application.retrieval.service import VectorRetriever
from app.domain.agent_quality import AnswerAssessment
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.domain.execution import RetrievalAttempt
from app.domain.retrieval import RetrievalResult
from app.infrastructure.conversations.langgraph_agent import (
    LangGraphConversationAgent,
)
from tests.test_fast_lane import _Provider
from tests.test_retriever import FakeQueryEmbedder
from tests.test_scoped_retrieval import _stores


class _Model:
    """只替代图的外部聊天响应，图和检索服务仍实际运行。"""

    def __init__(self, responses: Sequence[AIMessage]) -> None:
        """保存响应脚本。

        Args:
            responses: 接下来应产生的完整模型消息。
        """
        self.responses = iter(responses)

    def bind_tools(self, tools: object, *, strict: bool) -> "_Model":
        """保留标准工具接口。

        Args:
            tools: 工具说明。
            strict: 严格参数要求。

        Returns:
            当前模型。
        """
        return self

    def invoke(self, messages: list[dict[str, object]]) -> AIMessage:
        """返回下一条响应。

        Args:
            messages: 图实际构造的输入。

        Returns:
            完整响应。
        """
        return next(self.responses)


class _Quality:
    """根据当前实际证据范围给出确定性结论。"""

    def __init__(self) -> None:
        """记录到达外部裁判边界的调用数。"""
        self.evidence_calls = 0

    def assess_evidence(
        self, question: str, history: list[dict[str, str]],
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment:
        """批准本次召回到的范围内证据。

        Args:
            question: 原问题。
            history: 正式历史。
            evidence: 实际累计证据。

        Returns:
            充分结论。
        """
        self.evidence_calls += 1
        return EvidenceAssessment(
            sufficient=True, relevant_numbers=[item.number for item in evidence],
            missing_information=[], next_query=None,
        )

    def assess_answer(
        self, question: str, history: list[dict[str, str]], answer: str,
        evidence: list[EvidenceItem],
    ) -> AnswerAssessment:
        """批准测试预先给定的引用答案。

        Args:
            question: 原问题。
            history: 正式历史。
            answer: 草稿。
            evidence: 获批且实际引用的证据。

        Returns:
            通过结论。
        """
        return AnswerAssessment(supported=True, complete=True, issues=[])


def _seed() -> EvidenceBundle:
    budget = ExecutionBudget(
        deadline=100.0, evidence_assessments=1,
    ).record_retrieval(
        RetrievalAttempt(query="needle", kb_ids=("ops",)), now=1.0,
    )
    return EvidenceBundle(
        evidence=(EvidenceItem(
            number=5, chunk_id="ops:0", doc_id="ops:0", filename="a.md",
            page=None, text="needle 与配置", kb_id="ops",
        ),), approved_numbers=(5,), budget=budget,
        assessment=EvidenceAssessment(
            sufficient=False, relevant_numbers=[5],
            missing_information=["缺少研发库内容"], next_query="needle",
        ),
    )


def test_fast_retrieval_records_scope_and_preserves_kb_identity(tmp_path: Path) -> None:
    """Fast不能只限制返回结果而把执行记录写成旧全局范围。

    Args:
        tmp_path: 隔离索引目录。
    """
    store, index = _stores(tmp_path)
    output = FastLane(
        RetrievalService(VectorRetriever(store, FakeQueryEmbedder()), index),
        _Quality(), AnswerGenerator(PromptBuilder(), _Provider("配置[1]")),
        strategy="hybrid", clock=lambda: 1.0,
    ).run(
        "needle", [], kb_ids=("ops",),
        bundle=EvidenceBundle(budget=ExecutionBudget(deadline=100.0)),
    )
    assert output.result.answer == "配置[1]"
    assert output.bundle.budget.attempts[0].kb_ids == ("ops",)
    assert [item.kb_id for item in output.bundle.evidence] == ["ops"]


def test_agent_tool_cannot_override_server_bound_scope(tmp_path: Path) -> None:
    """模型在工具参数夹带其他知识库也不能扩大服务端范围。

    Args:
        tmp_path: 隔离索引目录。
    """
    store, index = _stores(tmp_path)
    call = {
        "name": "search_knowledge_base", "id": "one", "type": "tool_call",
        "args": {"query": "needle", "kb_ids": ["dev"]},
    }
    result = LangGraphConversationAgent(
        model=_Model([
            AIMessage(content="", tool_calls=[call]),
            AIMessage(content="配置[1]"),
        ]), retrieval_service=RetrievalService(
            VectorRetriever(store, FakeQueryEmbedder()), index,
        ), checkpointer=InMemorySaver(), top_k=5, quality_evaluator=_Quality(),
        clock=lambda: 1.0,
    ).run(uuid4(), "needle", [], kb_ids=("ops",), strategy="hybrid")
    assert result.answer == "配置[1]"
    assert [item.kb_id for item in result.execution.evidence] == ["ops"]
    assert result.execution.budget.attempts[0].kb_ids == ("ops",)


def test_same_query_after_real_scope_expansion_is_not_a_duplicate(tmp_path: Path) -> None:
    """同问题从运维库扩大到全部库时必须允许第二次召回，不重编号。

    Args:
        tmp_path: 隔离索引目录。
    """
    store, index = _stores(tmp_path)
    result = LangGraphConversationAgent(
        model=_Model([AIMessage(content="配置[5]，补充[6]")]),
        retrieval_service=RetrievalService(
            VectorRetriever(store, FakeQueryEmbedder()), index,
        ), checkpointer=InMemorySaver(), top_k=5,
        quality_evaluator=_Quality(), clock=lambda: 2.0,
    ).run(
        uuid4(), "needle", [], seed=_seed(),
        kb_ids=("ops", "dev"), strategy="hybrid",
    )
    assert result.retrieval_calls == 2
    assert result.execution.budget.attempts[1].kb_ids == ("ops", "dev")
    assert result.execution.budget.deadline == 100.0
    assert [item.position for item in result.citations] == [5, 6]


def test_same_query_and_same_scope_seed_answers_without_duplicate_search() -> None:
    """只改变通道不算范围扩大，不得重查相同范围；用已有证据部分作答。"""
    result = LangGraphConversationAgent(
        model=_Model([AIMessage(content="needle 与配置[5]")]),
        retrieval_service=object(),
        checkpointer=InMemorySaver(), top_k=5,
        quality_evaluator=_Quality(), clock=lambda: 2.0,
    ).run(uuid4(), "needle", [], seed=_seed(), kb_ids=("ops",))
    assert result.answer == "needle 与配置[5]"
    assert result.partial is True
    assert result.retrieval_calls == 1


def test_handoff_cannot_narrow_to_scope_excluding_existing_evidence() -> None:
    """种子来自运维库时，不能带着旧证据冒充只查研发库。"""
    agent = LangGraphConversationAgent(
        model=_Model([]), retrieval_service=object(),
        checkpointer=InMemorySaver(), top_k=5, clock=lambda: 2.0,
    )
    with pytest.raises(ConversationAgentError):
        agent.run(uuid4(), "needle", [], seed=_seed(), kb_ids=("dev",))


def test_bundle_rejects_evidence_outside_recorded_retrieval_scopes() -> None:
    """范围内检索记录不能为范围外正文背书。"""
    payload = _seed().model_dump()
    payload["evidence"][0]["kb_id"] = "dev"
    with pytest.raises(ValidationError):
        EvidenceBundle.model_validate(payload)


@pytest.mark.parametrize("channel", ["fast", "agent"])
def test_out_of_scope_adapter_result_never_reaches_quality_model(
    channel: str,
) -> None:
    """即使适配器实现有误，范围外正文也不能被发送到裁判或生成模型。

    Args:
        channel: 两种消费检索端口的执行通道。
    """
    class BrokenRetriever:
        """模拟忽略scope参数的错误适配器。"""

        def retrieve(
            self, question: str, strategy: object, top_k: int, *,
            kb_ids: tuple[str, ...] | None = None,
        ) -> list[RetrievalResult]:
            """返回不属于ops的正文。

            Args:
                question: 检索问题。
                strategy: 检索策略。
                top_k: 返回上限。
                kb_ids: 被错误忽略的范围。

            Returns:
                范围外结果。
            """
            return [RetrievalResult(
                chunk_id="leaked", doc_id="leaked", kb_id="dev",
                filename="a.md", text="OUTSIDE_SCOPE", page=None,
                chunk_index=0, score=1.0,
            )]

    quality = _Quality()
    with pytest.raises(ConversationAgentError):
        if channel == "fast":
            FastLane(
                BrokenRetriever(), quality,
                AnswerGenerator(PromptBuilder(), _Provider("正文[1]")),
                clock=lambda: 1.0,
            ).run(
                "needle", [], kb_ids=("ops",),
                bundle=EvidenceBundle(budget=ExecutionBudget(deadline=100.0)),
            )
        else:
            LangGraphConversationAgent(
                model=_Model([
                    AIMessage(content=""), AIMessage(content="正文[1]"),
                ]), retrieval_service=BrokenRetriever(),
                checkpointer=InMemorySaver(), top_k=5,
                quality_evaluator=quality, clock=lambda: 1.0,
            ).run(uuid4(), "needle", [], kb_ids=("ops",))
    assert quality.evidence_calls == 0

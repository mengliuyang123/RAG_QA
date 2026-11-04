# -*- coding: utf-8 -*-
"""通过真实 LangGraph 验证证据判断、补查和答案修复。"""

import json
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.errors import ConversationAgentError
from app.application.errors import ConversationTimeoutError
from app.application.conversations.service import ConversationService
from app.application.conversations.history_window import HistoryWindowBuilder
from app.infrastructure.generation.token_counting import (
    Utf8TokenUpperBoundCounter,
)
from app.domain.agent_quality import AnswerAssessment
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.retrieval import RetrievalResult
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.domain.execution import RetrievalAttempt
from app.infrastructure.conversations.langgraph_agent import (
    AGENT_REFUSAL_ANSWER,
    LangGraphConversationAgent,
)
from app.infrastructure.conversations.sqlite_repository import (
    SQLiteConversationRepository,
)


def _call(query: str, call_id: str = "call-1") -> dict[str, object]:
    return {
        "name": "search_knowledge_base", "args": {"query": query},
        "id": call_id, "type": "tool_call",
    }


def _chunk(number: int, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{number}", doc_id=f"doc-{number}", text=text,
        filename=f"{number}.md", page=None, chunk_index=0, score=0.8,
    )


def _enough(numbers: list[int]) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=True, relevant_numbers=numbers,
        missing_information=[], next_query=None,
    )


def _missing(query: str | None) -> EvidenceAssessment:
    return EvidenceAssessment(
        sufficient=False, relevant_numbers=[1],
        missing_information=["缺少 B 的存储方式"], next_query=query,
    )


class _Model:
    """仅替代外部模型响应，实际执行图与质量状态流转。"""

    def __init__(self, responses: Sequence[AIMessage]) -> None:
        self.responses = iter(responses)
        self.messages: list[list[dict[str, object]]] = []

    def bind_tools(self, tools: object, *, strict: bool) -> "_Model":
        """绑定脚本模型的受控工具。

        Args:
            tools: 工具描述。
            strict: 是否启用严格模式。

        Returns:
            当前脚本模型。
        """
        return self

    def invoke(self, messages: list[dict[str, object]]) -> AIMessage:
        """返回下一条模型响应并记录实际上下文。

        Args:
            messages: 生产图构造的模型消息。

        Returns:
            下一条完整模型响应。
        """
        self.messages.append(messages)
        return next(self.responses)


class _Retriever:
    def __init__(self, results: dict[str, list[RetrievalResult]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def retrieve(
        self, question: str, strategy: object, top_k: int,
    ) -> list[RetrievalResult]:
        """返回按问题预设的知识库响应。

        Args:
            question: 实际检索查询。
            strategy: 图指定的检索策略。
            top_k: 候选数量上限。

        Returns:
            预设的检索结果。
        """
        self.queries.append(question)
        return self.results.get(question, [])


class _Evaluator:
    def __init__(
        self,
        evidence: Sequence[EvidenceAssessment | Exception],
        answers: Sequence[AnswerAssessment | Exception] = (),
    ) -> None:
        self.evidence = iter(evidence)
        self.answers = iter(answers)
        self.cited: list[list[EvidenceItem]] = []

    def assess_evidence(
        self, question: str, history: list[dict[str, str]],
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment:
        """返回预设证据判断，隔离模型的不确定性。

        Args:
            question: 本轮原始问题。
            history: 正式历史。
            evidence: 累计候选证据。

        Returns:
            当前脚本判断。
        """
        verdict = next(self.evidence)
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    def assess_answer(
        self, question: str, history: list[dict[str, str]], answer: str,
        evidence: list[EvidenceItem],
    ) -> AnswerAssessment:
        """返回预设答案判断并记录实际可用来源。

        Args:
            question: 本轮原始问题。
            history: 正式历史。
            answer: 候选答案。
            evidence: 实际引用来源。

        Returns:
            当前脚本判断。
        """
        self.cited.append(evidence)
        verdict = next(self.answers)
        if isinstance(verdict, Exception):
            raise verdict
        return verdict


def _passed() -> AnswerAssessment:
    return AnswerAssessment(supported=True, complete=True, issues=[])


def _incomplete() -> AnswerAssessment:
    return AnswerAssessment(supported=True, complete=False, issues=["缺少 B"])


def _agent(model, retriever, evaluator, saver=None):
    return LangGraphConversationAgent(
        model=model, retrieval_service=retriever,
        checkpointer=saver or InMemorySaver(), top_k=5,
        quality_evaluator=evaluator,
    )


def _seed(
    assessment: EvidenceAssessment | None, *, calls: int = 1,
    judgements: int = 1, repaired: bool = False,
) -> EvidenceBundle:
    budget = ExecutionBudget(
        deadline=100.0, evidence_assessments=judgements,
        repair_attempted=repaired,
    )
    for query in ("A", "A 背景", "B 背景")[:calls]:
        budget = budget.record_retrieval(
            RetrievalAttempt(query=query), now=1.0,
        )
    return EvidenceBundle(
        evidence=(
            EvidenceItem(
                number=2, chunk_id="old-2", doc_id="old", filename="old.md",
                page=None, text="UNRELATED_SEED 海洋生物",
            ),
            EvidenceItem(
                number=5, chunk_id="old-5", doc_id="A", filename="A.md",
                page=None, text="A 用 SQLite",
            ),
        ),
        approved_numbers=(5,) if assessment else (),
        assessment=assessment, budget=budget,
    )


def _seed_agent(model, retriever, evaluator, clock=lambda: 2.0):
    return LangGraphConversationAgent(
        model=model, retrieval_service=retriever,
        checkpointer=InMemorySaver(), top_k=5,
        quality_evaluator=evaluator, clock=clock,
    )


def test_assessed_seed_is_not_retrieved_or_judged_again() -> None:
    """升级复用裁判结论，生成仅能看到批准块且原编号不变。"""
    model = _Model([AIMessage(content="A 用 SQLite[5]")])
    retriever = _Retriever({})
    evaluator = _Evaluator([], [_passed()])
    result = _seed_agent(model, retriever, evaluator).run(
        uuid4(), "A 的存储", [], seed=_seed(_enough([5])),
    )
    assert result.retrieval_calls == 1
    assert result.answer == "A 用 SQLite[5]"
    assert [item.position for item in result.citations] == [5]
    assert retriever.queries == []
    assert "UNRELATED_SEED" not in str(model.messages[0])
    assert "A 用 SQLite" in str(model.messages[0])


def test_upgrade_uses_only_remaining_call_and_keeps_sparse_numbers() -> None:
    """已有两次检索后只补查一次，新块从最大编号之后追加。"""
    model = _Model([AIMessage(content="A SQLite[5]，B PostgreSQL[6]")])
    retriever = _Retriever({"B": [_chunk(3, "B 用 PostgreSQL")]})
    evaluator = _Evaluator([_enough([5, 6])], [_passed()])
    missing = EvidenceAssessment(
        sufficient=False, relevant_numbers=[5],
        missing_information=["缺少 B"], next_query="B",
    )
    result = _seed_agent(model, retriever, evaluator).run(
        uuid4(), "比较 A/B", [],
        seed=_seed(missing, calls=2, judgements=2),
    )
    assert result.retrieval_calls == 3
    assert retriever.queries == ["B"]
    assert [item.position for item in result.citations] == [5, 6]
    assert result.execution.budget.deadline == 100.0
    assert result.execution.budget.evidence_assessments == 3
    assert result.execution.budget.answer_assessments == 1


def test_upgrade_with_exhausted_budget_answers_partially_without_fourth_search() -> None:
    """升级不重置已用完的检索与判断次数，用已有相关证据作答并说明缺口。"""
    retriever = _Retriever({})
    model = _Model([AIMessage(content="A 用 SQLite[5]，资料未提及 B")])
    missing = EvidenceAssessment(
        sufficient=False, relevant_numbers=[5],
        missing_information=["缺少 B"], next_query="B",
    )
    result = _seed_agent(model, retriever, _Evaluator([], [_incomplete()])).run(
        uuid4(), "比较 A/B", [],
        seed=_seed(missing, calls=3, judgements=3),
    )
    assert result.answer == "A 用 SQLite[5]，资料未提及 B"
    assert result.partial is True
    assert result.retrieval_calls == 3
    assert retriever.queries == []
    assert "evidence_gaps" in str(model.messages[-1])
    assert "缺少 B" in str(model.messages[-1])


def test_upgrade_does_not_reset_expired_deadline() -> None:
    """截止时间来自 Fast，不能从进入 Agent 的时刻重新计时。"""
    with pytest.raises(ConversationTimeoutError):
        _seed_agent(
            _Model([]), _Retriever({}), _Evaluator([]), clock=lambda: 100.0,
        ).run(uuid4(), "A", [], seed=_seed(_enough([5])))


def test_duplicate_seed_retry_is_not_executed() -> None:
    """判断器改写大小写和空格不能绕过跨通道去重。"""
    missing = EvidenceAssessment(
        sufficient=False, relevant_numbers=[5],
        missing_information=["缺少 B"], next_query="  a  ",
    )
    retriever = _Retriever({})
    result = _seed_agent(
        _Model([AIMessage(content="A 用 SQLite[5]")]), retriever,
        _Evaluator([], [_incomplete()]),
    ).run(uuid4(), "比较 A/B", [], seed=_seed(missing))
    assert result.partial is True
    assert retriever.queries == []


def test_unjudged_seed_is_judged_once_without_retrieval() -> None:
    """只有候选证据的种子不能当成已经判断过。"""
    result = _seed_agent(
        _Model([AIMessage(content="A SQLite[5]")]), _Retriever({}),
        _Evaluator([_enough([5])], [_passed()]),
    ).run(uuid4(), "A", [], seed=_seed(None, judgements=0))
    assert result.answer == "A SQLite[5]"
    assert result.retrieval_calls == 1


def test_upgrade_cannot_restore_consumed_repair_attempt() -> None:
    """Fast 已修复过的答案不能在 Agent 再次消耗修复额度。"""
    evaluator = _Evaluator([], [AnswerAssessment(
        supported=False, complete=True, issues=["不支持"],
    )])
    result = _seed_agent(
        _Model([AIMessage(content="A PostgreSQL[5]")]),
        _Retriever({}), evaluator,
    ).run(
        uuid4(), "A", [], seed=_seed(_enough([5]), repaired=True),
    )
    # 不再修复也不再判断，引用已校验的回答按部分依据发布。
    assert result.answer == "A PostgreSQL[5]"
    assert result.partial is True
    assert result.execution.budget.answer_assessments == 1


def test_no_extra_search_when_seed_has_used_all_evidence_judgements() -> None:
    """没有剩余判断额度时不得继续产生无法审核的新检索结果。"""
    retriever = _Retriever({})
    result = _seed_agent(
        _Model([AIMessage(content="", tool_calls=[_call("B")])]),
        retriever, _Evaluator([]),
    ).run(
        uuid4(), "比较 A/B", [],
        seed=_seed(_enough([5]), judgements=3),
    )
    assert result.answer == AGENT_REFUSAL_ANSWER
    assert result.retrieval_calls == 1
    assert retriever.queries == []


def test_answer_judgement_budget_is_not_restored_on_upgrade() -> None:
    """第二次答案判断失败时不能再修复并发起第三次判断。"""
    seed = _seed(_enough([5]))
    payload = seed.model_dump()
    payload["budget"]["answer_assessments"] = 1
    result = _seed_agent(
        _Model([AIMessage(content="A PostgreSQL[5]")]), _Retriever({}),
        _Evaluator([], [AnswerAssessment(
            supported=False, complete=True, issues=["不支持"],
        )]),
    ).run(
        uuid4(), "A", [], seed=EvidenceBundle.model_validate(payload),
    )
    assert result.answer == "A PostgreSQL[5]"
    assert result.partial is True
    assert result.execution.budget.answer_assessments == 2


def test_seed_state_does_not_leak_to_next_turn_on_same_checkpoint() -> None:
    """后一轮没有种子时必须重建新预算，而不是继承上一轮来源。"""
    retriever = _Retriever({"B": [_chunk(3, "B PostgreSQL")]})
    agent = _seed_agent(_Model([
        AIMessage(content="A SQLite[5]"),
        AIMessage(content="", tool_calls=[_call("B")]),
        AIMessage(content="B PostgreSQL[1]"),
    ]), retriever, _Evaluator([_enough([1])], [_passed(), _passed()]))
    conversation_id = uuid4()
    first = agent.run(
        conversation_id, "A", [], seed=_seed(_enough([5])),
    )
    second = agent.run(conversation_id, "B", [])
    assert first.citations[0].position == 5
    assert second.citations[0].position == 1
    assert second.retrieval_calls == 1
    assert second.execution.budget.deadline == 182.0
    assert retriever.queries == ["B"]


@pytest.mark.parametrize("repair", [False, True])
def test_truncated_answer_is_not_published_even_with_valid_citations(
    repair: bool,
) -> None:
    """生成或修复的截断响应不能因编号有效而通过。

    Args:
        repair: 是否在唯一修复调用中产生截断响应。
    """
    responses = []
    if repair:
        responses.append(AIMessage(content="错误引用[99]"))
    responses.append(AIMessage(
        content="A SQLite[5]", response_metadata={"finish_reason": "length"},
    ))
    with pytest.raises(ConversationAgentError):
        _seed_agent(
            _Model(responses), _Retriever({}), _Evaluator([], [_passed()]),
        ).run(uuid4(), "A", [], seed=_seed(_enough([5])))


def test_agent_refusal_metadata_is_not_published_as_answer() -> None:
    """上游拒绝元数据不能被正常正文和有效编号掩盖。"""
    with pytest.raises(ConversationAgentError):
        _seed_agent(
            _Model([AIMessage(
                content="不应发布[5]", additional_kwargs={"refusal": "refused"},
            )]), _Retriever({}), _Evaluator([], [_passed()]),
        ).run(uuid4(), "A", [], seed=_seed(_enough([5])))


def test_missing_comparison_evidence_triggers_targeted_search() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 用 SQLite[1]，B 用 PostgreSQL[2]。"),
    ])
    retriever = _Retriever({
        "A": [_chunk(1, "A 用 SQLite")],
        "B 存储方式": [_chunk(2, "B 用 PostgreSQL")],
    })
    evaluator = _Evaluator(
        [_missing("B 存储方式"), _enough([1, 2])], [_passed()],
    )
    result = _agent(model, retriever, evaluator).run(uuid4(), "比较 A/B", [])
    assert result.retrieval_calls == 2
    assert retriever.queries == ["A", "B 存储方式"]
    assert [item.position for item in result.citations] == [1, 2]


def test_irrelevant_tool_text_is_removed_without_renumbering() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 使用 SQLite[2]。"),
    ])
    retriever = _Retriever({"A": [
        _chunk(1, "UNRELATED_SECRET 海洋生物"),
        _chunk(2, "A 使用 SQLite"),
    ]})
    evaluator = _Evaluator([_enough([2])], [_passed()])
    result = _agent(model, retriever, evaluator).run(uuid4(), "A 的存储", [])
    assert result.citations[0].position == 2
    assert result.citations[0].chunk_text == "A 使用 SQLite"
    assert "UNRELATED_SECRET" not in json.dumps(model.messages[1])
    assert [item.number for item in evaluator.cited[0]] == [2]


@pytest.mark.parametrize("supported,complete", [(False, True), (True, False)])
def test_unsupported_or_incomplete_answer_is_revised_and_rechecked(
    supported: bool, complete: bool,
) -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 用 PostgreSQL[1]"),
        AIMessage(content="A 用 SQLite[1]"),
    ])
    evaluator = _Evaluator([_enough([1])], [
        AnswerAssessment(
            supported=supported, complete=complete, issues=["修正存储结论"],
        ),
        _passed(),
    ])
    result = _agent(
        model, _Retriever({"A": [_chunk(1, "A 用 SQLite")]}), evaluator,
    ).run(uuid4(), "它的存储是什么？", [])
    assert result.answer == "A 用 SQLite[1]"
    assert len(evaluator.cited) == 2
    assert "它的存储是什么？" in str(model.messages[-1])
    assert "修正存储结论" in str(model.messages[-1])


def test_repair_budget_is_shared_with_citation_repair() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 用 PostgreSQL[9]"),
        AIMessage(content="A 用 PostgreSQL[1]"),
    ])
    evaluator = _Evaluator([_enough([1])], [
        AnswerAssessment(supported=False, complete=True, issues=["错误存储"]),
    ])
    result = _agent(
        model, _Retriever({"A": [_chunk(1, "A 用 SQLite")]}), evaluator,
    ).run(uuid4(), "A 的存储", [])
    # 修复额度已被引用修复用掉，审核不通过时发布修复后的回答并标为部分依据。
    assert result.answer == "A 用 PostgreSQL[1]"
    assert result.partial is True
    assert [item.position for item in result.citations] == [1]


def test_duplicate_retry_query_stops_without_extra_retrieval() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A sqlite")]),
        AIMessage(content="A 用 SQLite[1]，资料未提及 B"),
    ])
    retriever = _Retriever({"A sqlite": [_chunk(1, "A 用 SQLite")]})
    result = _agent(model, retriever, _Evaluator([
        _missing("  A   SQLITE  "),
    ], [_incomplete()])).run(uuid4(), "比较 A/B", [])
    assert result.partial is True
    assert result.retrieval_calls == 1
    assert retriever.queries == ["A sqlite"]


def test_insufficient_evidence_stops_at_three_searches() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A[1]，资料未提及 B 的存储"),
    ])
    retriever = _Retriever({
        "A": [_chunk(1, "A")], "B": [_chunk(2, "B 介绍")],
        "B 存储": [_chunk(3, "B 背景")],
    })
    result = _agent(model, retriever, _Evaluator([
        _missing("B"), _missing("B 存储"), _missing("更多"),
    ], [_incomplete()])).run(uuid4(), "比较 A/B", [])
    assert result.answer == "A[1]，资料未提及 B 的存储"
    assert result.partial is True
    assert result.retrieval_calls == 3


def test_insufficient_without_relevant_evidence_still_refuses() -> None:
    """证据不足且没有任何相关块时，不生成回答，仍然拒答。"""
    model = _Model([AIMessage(content="", tool_calls=[_call("A")])])
    result = _agent(model, _Retriever({"A": [_chunk(1, "无关内容")]}), _Evaluator([
        EvidenceAssessment(
            sufficient=False, relevant_numbers=[],
            missing_information=["缺少 A"], next_query=None,
        ),
    ])).run(uuid4(), "A 的存储", [])
    assert result.answer == AGENT_REFUSAL_ANSWER
    assert result.partial is False
    assert len(model.messages) == 1


def test_complete_answer_on_sufficient_evidence_is_not_partial() -> None:
    """完整通过审核的回答不带部分依据标记，生成时也不附带缺口说明。"""
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 用 SQLite[1]"),
    ])
    result = _agent(
        model, _Retriever({"A": [_chunk(1, "A 用 SQLite")]}),
        _Evaluator([_enough([1])], [_passed()]),
    ).run(uuid4(), "A 的存储", [])
    assert result.partial is False
    assert "evidence_gaps" not in str(model.messages[-1])


def test_partial_evidence_answer_must_still_be_supported() -> None:
    """缺口已说明时不再要求覆盖完整，但事实不受支持仍要修复。"""
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 用 PostgreSQL[1]"),
        AIMessage(content="A 用 SQLite[1]，资料未提及 B"),
    ])
    evaluator = _Evaluator([_missing(None)], [
        AnswerAssessment(supported=False, complete=False, issues=["存储错误"]),
        _incomplete(),
    ])
    result = _agent(
        model, _Retriever({"A": [_chunk(1, "A 用 SQLite")]}), evaluator,
    ).run(uuid4(), "比较 A/B", [])
    assert result.answer == "A 用 SQLite[1]，资料未提及 B"
    assert result.partial is True
    assert result.execution.budget.answer_assessments == 2


def test_answer_judge_failure_is_treated_as_failed_review() -> None:
    """审核模型输出无效不再让整轮报错：先修复，仍失败则按部分依据发布。"""
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 用 SQLite[1]"),
        AIMessage(content="A 使用 SQLite[1]"),
    ])
    evaluator = _Evaluator([_enough([1])], [
        ConversationAgentError("模型质量判断失败"),
        ConversationAgentError("模型质量判断失败"),
    ])
    result = _agent(
        model, _Retriever({"A": [_chunk(1, "A 用 SQLite")]}), evaluator,
    ).run(uuid4(), "A 的存储", [])
    assert result.answer == "A 使用 SQLite[1]"
    assert result.partial is True
    assert "答案审核未能完成" in result.quality_issues


def test_answer_judge_timeout_still_fails_the_turn() -> None:
    """超时不是审核结论，仍按超时报错，不降级发布。"""
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 用 SQLite[1]"),
    ])
    evaluator = _Evaluator([_enough([1])], [ConversationTimeoutError("超时")])
    with pytest.raises(ConversationTimeoutError):
        _agent(
            model, _Retriever({"A": [_chunk(1, "A 用 SQLite")]}), evaluator,
        ).run(uuid4(), "A 的存储", [])


def test_rejected_draft_is_published_when_repair_breaks_citations() -> None:
    """修复后引用无效时，发布此前引用有效但审核未通过的回答，而不是拒答。"""
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 用 SQLite[1]"),
        AIMessage(content="A 用 SQLite[7]"),
    ])
    evaluator = _Evaluator([_enough([1])], [
        AnswerAssessment(supported=True, complete=False, issues=["缺少版本"]),
    ])
    result = _agent(
        model, _Retriever({"A": [_chunk(1, "A 用 SQLite")]}), evaluator,
    ).run(uuid4(), "A 的存储和版本", [])
    assert result.answer == "A 用 SQLite[1]"
    assert result.partial is True
    assert [item.position for item in result.citations] == [1]


def test_empty_results_refuse_without_quality_model_call() -> None:
    result = _agent(
        _Model([AIMessage(content="", tool_calls=[_call("A")])]),
        _Retriever({}), _Evaluator([]),
    ).run(uuid4(), "A 的存储", [])
    assert result.answer == AGENT_REFUSAL_ANSWER
    assert result.retrieval_calls == 1


def test_quality_failure_does_not_turn_into_successful_refusal() -> None:
    with pytest.raises(ConversationAgentError):
        _agent(
            _Model([AIMessage(content="", tool_calls=[_call("A")])]),
            _Retriever({"A": [_chunk(1, "A")]}),
            _Evaluator([ConversationAgentError("评估服务失败")]),
        ).run(uuid4(), "A", [])


def test_same_thread_resets_query_and_quality_state() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 用 SQLite[1]"),
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="A 用 SQLite[1]"),
    ])
    retriever = _Retriever({"A": [_chunk(1, "A 用 SQLite")]})
    evaluator = _Evaluator([_enough([1]), _enough([1])], [
        _passed(), _passed(),
    ])
    agent = _agent(model, retriever, evaluator)
    conversation_id = uuid4()
    first = agent.run(conversation_id, "A 的存储", [])
    second = agent.run(conversation_id, "再查 A", [])
    assert first.retrieval_calls == second.retrieval_calls == 1
    assert retriever.queries == ["A", "A"]


def test_default_evaluator_uses_same_model_with_separate_json_calls() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content=json.dumps({
            "sufficient": True, "relevant_numbers": [1],
            "missing_information": [], "next_query": None,
        })),
        AIMessage(content="A 用 SQLite[1]"),
        AIMessage(content=json.dumps({
            "supported": True, "complete": True, "issues": [],
        })),
    ])
    agent = LangGraphConversationAgent(
        model=model, retrieval_service=_Retriever({
            "A": [_chunk(1, "A 用 SQLite")],
        }), checkpointer=InMemorySaver(), top_k=5,
    )
    result = agent.run(uuid4(), "A 的存储", [])
    assert result.answer == "A 用 SQLite[1]"
    assert result.citations[0].chunk_text == "A 用 SQLite"


def test_reassessment_can_restore_previously_excluded_evidence() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="", tool_calls=[_call("B")]),
        AIMessage(content="A SQLite[2]，B PostgreSQL[1][3]"),
    ])
    retriever = _Retriever({
        "A": [_chunk(1, "RESTORED_CONTEXT B 历史"), _chunk(2, "A SQLite")],
        "B": [_chunk(3, "B PostgreSQL")],
    })
    result = _agent(model, retriever, _Evaluator([
        _enough([2]), _enough([1, 2, 3]),
    ], [_passed()])).run(uuid4(), "比较 A/B", [])
    assert "RESTORED_CONTEXT" not in json.dumps(model.messages[1])
    assert "RESTORED_CONTEXT" in json.dumps(model.messages[2])
    assert len(result.citations) == 3


def test_longest_legal_path_finishes_with_default_recursion_budget() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="不完整回答[1]"),
        AIMessage(content="A SQLite[1]，B PostgreSQL[2][3]"),
    ])
    evaluator = _Evaluator([
        _missing("B"), _missing("B 存储"), _enough([1, 2, 3]),
    ], [
        AnswerAssessment(supported=True, complete=False, issues=["缺少 B"]),
        _passed(),
    ])
    result = _agent(model, _Retriever({
        "A": [_chunk(1, "A SQLite")],
        "B": [_chunk(2, "B PostgreSQL")],
        "B 存储": [_chunk(3, "B 更多说明")],
    }), evaluator).run(uuid4(), "比较 A/B", [])
    assert result.retrieval_calls == 3
    assert result.answer == "A SQLite[1]，B PostgreSQL[2][3]"


def test_all_irrelevant_evidence_refuses_before_answer_generation() -> None:
    result = _agent(
        _Model([AIMessage(content="", tool_calls=[_call("A")])]),
        _Retriever({"A": [_chunk(1, "海洋生物")]}),
        _Evaluator([EvidenceAssessment(
            sufficient=False, relevant_numbers=[],
            missing_information=["无相关内容"], next_query=None,
        )]),
    ).run(uuid4(), "A 存储", [])
    assert result.answer == AGENT_REFUSAL_ANSWER
    assert result.citations == []


def test_citation_to_rejected_evidence_is_not_accepted_after_repair() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="无关结论[1]"),
        AIMessage(content="仍是无关结论[1]"),
    ])
    result = _agent(model, _Retriever({"A": [
        _chunk(1, "海洋生物"), _chunk(2, "A SQLite"),
    ]}), _Evaluator([_enough([2])])).run(uuid4(), "A 存储", [])
    assert result.answer == AGENT_REFUSAL_ANSWER
    assert result.citations == []


def test_grouped_tools_deduplicate_queries_and_preserve_message_pairs() -> None:
    model = _Model([
        AIMessage(content="", tool_calls=[
            _call("A", "one"), _call("  a  ", "two"), _call("B", "three"),
        ]),
        AIMessage(content="A SQLite[1]，B PostgreSQL[2]"),
    ])
    retriever = _Retriever({
        "A": [_chunk(1, "A SQLite")], "B": [_chunk(2, "B PostgreSQL")],
    })
    result = _agent(model, retriever, _Evaluator([
        _enough([1, 2]),
    ], [_passed()])).run(uuid4(), "比较 A/B", [])
    assert result.retrieval_calls == 2
    assert retriever.queries == ["A", "B"]
    tool_ids = [
        item["tool_call_id"] for item in model.messages[1]
        if item.get("role") == "tool"
    ]
    assert tool_ids == ["one", "two", "three"]


@pytest.mark.parametrize("streaming", [False, True])
def test_quality_error_never_saves_a_successful_turn(
    tmp_path: Path, streaming: bool,
) -> None:
    repository = SQLiteConversationRepository(tmp_path / "history.db")
    conversation = repository.create()
    saver = InMemorySaver()
    model = _Model([
        AIMessage(content="", tool_calls=[_call("A")]),
        AIMessage(content="invalid quality JSON"),
    ])
    agent = LangGraphConversationAgent(
        model=model, retrieval_service=_Retriever({
            "A": [_chunk(1, "A SQLite")],
        }), checkpointer=saver, top_k=5,
    )
    service = ConversationService(
        repository, agent, saver,
        HistoryWindowBuilder(
            counter=Utf8TokenUpperBoundCounter(), token_budget=16384,
        ),
    )
    if streaming:
        events = list(service.stream_query(conversation.id, "A 存储"))
        assert events[-1].event == "error"
        assert not any(item.event == "complete" for item in events)
    else:
        with pytest.raises(ConversationAgentError):
            service.query(conversation.id, "A 存储")
    assert repository.get(conversation.id).turns_count == 0
    assert repository.get(conversation.id).turns == []

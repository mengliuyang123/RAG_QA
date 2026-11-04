# -*- coding: utf-8 -*-
"""受控 LangGraph Agent、检索工具和引用修复测试。"""

from collections.abc import Sequence
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.domain.agent_quality import AnswerAssessment
from app.domain.agent_quality import EvidenceAssessment
from app.domain.agent_quality import EvidenceItem
from app.domain.conversations import AnswerMode
from app.domain.conversations import ConversationTurn
from app.domain.retrieval import RetrievalResult
from app.domain.retrieval import RetrievalStrategy
from app.infrastructure.conversations.langgraph_agent import (
    AGENT_REFUSAL_ANSWER,
)
from app.infrastructure.conversations.langgraph_agent import (
    LangGraphConversationAgent,
)


def _tool_call(call_id: str, query: str) -> dict[str, object]:
    return {
        "name": "search_knowledge_base",
        "args": {"query": query},
        "id": call_id,
        "type": "tool_call",
    }


def _chunk(index: int, *, chunk_id: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or f"doc:{index}",
        doc_id=f"doc-{index}",
        text=f"证据 {index}",
        filename=f"doc-{index}.md",
        page=index,
        chunk_index=index,
        score=1.0 - index / 100,
    )


class _ScriptedModel:
    def __init__(self, responses: Sequence[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []
        self.bound_tools: list[dict[str, object]] = []
        self.strict: bool | None = None

    def bind_tools(
        self,
        tools: list[dict[str, object]],
        *,
        strict: bool,
    ) -> "_ScriptedModel":
        self.bound_tools = tools
        self.strict = strict
        return self

    def invoke(self, messages: list[dict[str, object]]) -> AIMessage:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("模型脚本响应已耗尽")
        return self._responses.pop(0)


class _FakeRetriever:
    def __init__(
        self,
        responses: dict[str, list[RetrievalResult]] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, RetrievalStrategy | str, int]] = []

    def retrieve(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
    ) -> list[RetrievalResult]:
        self.calls.append((question, strategy, top_k))
        return self.responses.get(question, [])[:top_k]


class _AcceptingQuality:
    """隔离旧图测试中的模型质量边界，语义判断另有专项测试。"""

    def assess_evidence(
        self, question: str, history: list[dict[str, str]],
        evidence: list[EvidenceItem],
    ) -> EvidenceAssessment:
        """批准本测试预先指定的有效证据。

        Args:
            question: 本轮问题。
            history: 正式历史。
            evidence: 确定性测试证据。

        Returns:
            全部证据相关且充分的判断。
        """
        return EvidenceAssessment(
            sufficient=True,
            relevant_numbers=[item.number for item in evidence],
            missing_information=[], next_query=None,
        )

    def assess_answer(
        self, question: str, history: list[dict[str, str]],
        answer: str, evidence: list[EvidenceItem],
    ) -> AnswerAssessment:
        """批准旧测试中预先指定的草稿。

        Args:
            question: 本轮问题。
            history: 正式历史。
            answer: 待检验引用的答案。
            evidence: 可信引用来源。

        Returns:
            支持且完整的判断。
        """
        return AnswerAssessment(supported=True, complete=True, issues=[])


def _agent(
    model: _ScriptedModel,
    retriever: _FakeRetriever,
) -> LangGraphConversationAgent:
    return LangGraphConversationAgent(
        model=model,
        retrieval_service=retriever,
        checkpointer=InMemorySaver(),
        top_k=5,
        max_retrieval_calls=3,
        quality_evaluator=_AcceptingQuality(),
    )


def test_agent_cannot_publish_model_direct_answer_without_retrieval() -> None:
    model = _ScriptedModel([
        AIMessage(content="无需检索，A 使用 PostgreSQL。"),
        AIMessage(content="A 使用 SQLite[1]。"),
    ])
    retriever = _FakeRetriever({"A 的存储": [_chunk(1)]})

    result = _agent(model, retriever).run(uuid4(), "A 的存储", [])

    assert result.answer_mode is AnswerMode.RETRIEVAL
    assert result.retrieval_calls == 1
    assert result.answer == "A 使用 SQLite[1]。"
    assert [item.position for item in result.citations] == [1]
    assert retriever.calls[0][0] == "A 的存储"
    assert "PostgreSQL" not in str(model.calls[1])
    tool = model.bound_tools[0]["function"]
    assert tool["name"] == "search_knowledge_base"
    assert set(tool["parameters"]["properties"]) == {"query"}
    assert model.strict is True


def test_agent_uses_history_aware_query_and_fixed_retrieval_policy() -> None:
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("call-1", "LangGraph 第三步含义")],
            ),
            AIMessage(content="第三步是生成查询或回答[1]。"),
        ]
    )
    retriever = _FakeRetriever(
        {"LangGraph 第三步含义": [_chunk(1)]}
    )
    history = [
        SimpleNamespace(
            user_message="LangGraph Agentic RAG 有哪些步骤？",
            assistant_message="它包含检索判断、检索和生成。",
        )
    ]

    result = _agent(model, retriever).run(
        uuid4(),
        "第三步是什么意思？",
        history,  # type: ignore[arg-type]
    )

    assert retriever.calls == [
        (
            "LangGraph 第三步含义",
            RetrievalStrategy.HYBRID_RERANK,
            5,
        )
    ]
    first_messages = model.calls[0]
    assert any(
        message.get("content") == "LangGraph Agentic RAG 有哪些步骤？"
        for message in first_messages
    )
    assert result.answer_mode is AnswerMode.RETRIEVAL
    assert result.citations[0].doc_id == "doc-1"


def test_multiple_retrievals_share_stable_deduplicated_evidence_numbers() -> None:
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("call-1", "query-a"),
                    _tool_call("call-2", "query-b"),
                ],
            ),
            AIMessage(content="综合结论[1][2][3]。"),
        ]
    )
    retriever = _FakeRetriever(
        {
            "query-a": [_chunk(1), _chunk(2, chunk_id="shared")],
            "query-b": [
                _chunk(2, chunk_id="shared"),
                _chunk(3),
            ],
        }
    )

    result = _agent(model, retriever).run(uuid4(), "综合问题", [])

    assert result.retrieval_calls == 2
    assert [citation.position for citation in result.citations] == [1, 2, 3]
    assert [citation.chunk_text for citation in result.citations] == [
        "证据 1",
        "证据 2",
        "证据 3",
    ]
    tool_messages = [
        message
        for message in model.calls[1]
        if message.get("role") == "tool"
    ]
    assert '"number":2' in tool_messages[0]["content"]
    assert '"number":2' in tool_messages[1]["content"]


def test_fourth_tool_call_is_not_executed_and_returns_safe_refusal() -> None:
    responses: list[AIMessage] = []
    for index in range(1, 5):
        responses.append(
            AIMessage(
                content="",
                tool_calls=[_tool_call(f"call-{index}", f"query-{index}")],
            )
        )
    model = _ScriptedModel(responses)
    retriever = _FakeRetriever(
        {f"query-{index}": [_chunk(index)] for index in range(1, 5)}
    )

    result = _agent(model, retriever).run(uuid4(), "循环问题", [])

    assert [call[0] for call in retriever.calls] == [
        "query-1",
        "query-2",
        "query-3",
    ]
    assert result.answer == AGENT_REFUSAL_ANSWER
    assert result.answer_mode is AnswerMode.RETRIEVAL
    assert result.retrieval_calls == 3
    assert result.citations == []


def test_invalid_citation_is_repaired_once_without_more_tools() -> None:
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("call-1", "query")],
            ),
            AIMessage(content="错误引用[9]。"),
            AIMessage(content="修复后的回答[1]。"),
        ]
    )
    retriever = _FakeRetriever({"query": [_chunk(1)]})

    result = _agent(model, retriever).run(uuid4(), "问题", [])

    assert result.answer == "修复后的回答[1]。"
    assert result.retrieval_calls == 1
    assert len(retriever.calls) == 1
    repair_messages = model.calls[-1]
    assert all("tools" not in message for message in repair_messages)
    assert any("错误引用[9]" in str(message["content"]) for message in repair_messages)


def test_failed_repair_returns_refusal_without_fake_citations() -> None:
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("call-1", "query")],
            ),
            AIMessage(content="无引用回答。"),
            AIMessage(content="仍然引用错误[8]。"),
        ]
    )
    retriever = _FakeRetriever({"query": [_chunk(1)]})

    result = _agent(model, retriever).run(uuid4(), "问题", [])

    assert result.answer == AGENT_REFUSAL_ANSWER
    assert result.citations == []
    assert result.retrieval_calls == 1


def test_status_callback_exposes_public_execution_phases_without_reasoning() -> None:
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("call-1", "query")],
            ),
            AIMessage(content="回答[1]。"),
        ]
    )
    retriever = _FakeRetriever({"query": [_chunk(1)]})
    statuses: list[tuple[str, int | None]] = []

    _agent(model, retriever).run(
        uuid4(),
        "问题",
        [],
        lambda phase, attempt: statuses.append((phase, attempt)),
    )

    assert statuses == [
        ("deciding", None), ("retrieving", 1), ("verifying", None),
        ("generating", None), ("validating", None), ("verifying", None),
    ]
    assert all("reasoning" not in phase for phase, _ in statuses)

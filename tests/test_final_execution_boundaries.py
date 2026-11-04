# -*- coding: utf-8 -*-
"""最终审查发现的批量工具预算和子阶段截止时间回归。"""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.errors import ConversationTimeoutError
from app.application.execution_control import check_cancellation
from app.application.query.fast_lane import FastLane
from app.application.query.generator import AnswerGenerator
from app.application.query.prompting import PromptBuilder
from app.application.retrieval.service import RetrievalService
from app.application.retrieval.service import VectorRetriever
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.infrastructure.conversations.langgraph_agent import (
    LangGraphConversationAgent,
)
from tests.test_fast_lane import _Provider
from tests.test_fast_lane import _Quality
from tests.test_fast_lane import _Retriever
from tests.test_retriever import FakeKeywordRetriever
from tests.test_retriever import FakeQueryEmbedder
from tests.test_retriever import FakeReranker
from tests.test_retriever import FakeVectorStore
from tests.test_retriever import make_search_result
from tests.test_scoped_agent import _Model


def test_extra_batched_tool_does_not_discard_third_retrieval() -> None:
    """第四次只被拒绝执行，不能阻止第三次充分证据进入审核和回答。

    Returns:
        None。
    """
    calls = []
    for query in ("C", "B 存储方式", "D"):
        calls.append({
            "name": "search_knowledge_base", "id": query,
            "args": {"query": query}, "type": "tool_call",
        })
    retriever = _Retriever()
    quality = _Quality()
    result = LangGraphConversationAgent(
        _Model([
            AIMessage(content="", tool_calls=calls),
            AIMessage(content="A使用SQLite[2]，B使用PostgreSQL[3]"),
        ]), retriever, InMemorySaver(), top_k=5, quality_evaluator=quality,
        clock=lambda: 1.0,
    ).run(uuid4(), "比较A和B", [], initial_query="A")
    assert [call[0] for call in retriever.calls] == ["A", "C", "B 存储方式"]
    assert quality.calls == 2
    assert result.execution.budget.retrieval_calls == 3
    assert result.execution.budget.evidence_assessments == 2
    assert result.answer == "A使用SQLite[2]，B使用PostgreSQL[3]"
    assert [citation.position for citation in result.citations] == [2, 3]


@pytest.mark.parametrize("lane", ["fast", "agent"])
@pytest.mark.parametrize("slow_stage", ["embedding", "vector", "bm25"])
def test_expired_retrieval_stops_next_stage(
    monkeypatch: pytest.MonkeyPatch, lane: str, slow_stage: str,
) -> None:
    """某子阶段越过整轮截止后，不得继续启动后面的召回或远程重排。

    Args:
        monkeypatch: 仅在外部端口返回时推进测试时钟。
        lane: 真实Fast或LangGraph调用方。
        slow_stage: 将耗尽时间预算的子阶段。

    Returns:
        None。
    """
    now = [1.0]
    seen = []
    embedder = FakeQueryEmbedder()
    vector = FakeVectorStore([make_search_result("doc:0", 0.1)])
    keyword = FakeKeywordRetriever()
    reranker = FakeReranker()
    for stage, adapter, method in (
        ("embedding", embedder, "embed_query"),
        ("vector", vector, "query_by_embedding"),
        ("bm25", keyword, "search"),
        ("rerank", reranker, "rerank"),
    ):
        original = getattr(adapter, method)

        def observe(*args, _stage=stage, _call=original, **kwargs):
            seen.append(_stage)
            result = _call(*args, **kwargs)
            if _stage == slow_stage:
                now[0] = 100.0
            return result

        monkeypatch.setattr(adapter, method, observe)
    retrieval = RetrievalService(
        VectorRetriever(vector, embedder), keyword, reranker,
    )
    seed = EvidenceBundle(budget=ExecutionBudget(deadline=100.0))
    with pytest.raises(ConversationTimeoutError):
        if lane == "fast":
            FastLane(
                retrieval, _Quality(),
                AnswerGenerator(PromptBuilder(), _Provider()),
                clock=lambda: now[0],
            ).run("问题", [], bundle=seed)
        else:
            LangGraphConversationAgent(
                _Model([]), retrieval, InMemorySaver(), top_k=5,
                quality_evaluator=_Quality(), clock=lambda: now[0],
            ).run(uuid4(), "问题", [], seed=seed, initial_query="问题")
    stages = ["embedding", "vector", "bm25", "rerank"]
    assert seen == stages[:stages.index(slow_stage) + 1]
    # 超时上下文退出后不能污染同线程后续请求。
    check_cancellation()

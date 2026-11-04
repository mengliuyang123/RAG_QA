# -*- coding: utf-8 -*-
"""取消必须在同一流水线内部的外部调用之间生效。"""

from pathlib import Path
from threading import Event
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.retrieval.service import RetrievalService
from app.application.retrieval.service import VectorRetriever
from app.application.streaming import EventSink
from app.application.streaming import LiveEventStream
from app.infrastructure.reranking.dashscope import RemoteReranker
from tests.test_retriever import FakeQueryEmbedder
from tests.test_scoped_retrieval import _stores
from tests.test_scoped_agent import _Model
from tests.test_scoped_agent import _Quality
from app.infrastructure.conversations.langgraph_agent import LangGraphConversationAgent
from app.domain.agent_quality import AnswerAssessment
from app.domain.agent_quality import EvidenceItem


def test_cancel_during_embedding_does_not_start_remote_reranking(
    tmp_path: Path,
) -> None:
    """仅在检索结束才检查取消会额外调用一次计费重排，必须拦在边界。

    Args:
        tmp_path: 真实本地索引的隔离路径。
    """
    requests = []
    stopped = Event()

    class CancelEmbedding(FakeQueryEmbedder):
        """模拟已发出的向量请求返回时客户端已断开。"""

        def embed_query(self, question: str) -> list[float]:
            """返回真实检索服务可使用的向量。

            Args:
                question: 实际检索问题。

            Returns:
                固定查询向量。
            """
            stream.close()
            return super().embed_query(question)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"results": [
            {"index": 0, "relevance_score": 0.9},
        ]})

    store, index = _stores(tmp_path)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        retriever = RetrievalService(
            VectorRetriever(store, CancelEmbedding()), index,
            RemoteReranker("https://rerank.example/v1", "offline", "rerank", client=client),
        )

        def produce(sink: EventSink[str]) -> None:
            try:
                retriever.retrieve("needle", "hybrid_rerank", 1, kb_ids=("ops",))
            finally:
                stopped.set()

        stream = LiveEventStream(produce)
        with pytest.raises(StopIteration):
            next(stream)
        assert stopped.wait(timeout=0.5)
    assert requests == []


def test_graph_stops_before_answer_judge_when_generation_returns_after_cancel(
    tmp_path: Path,
) -> None:
    """取消状态必须穿过LangGraph节点上下文，不能只依赖输出回调。

    Args:
        tmp_path: 隔离索引目录。
    """
    stopped = Event()
    answer_judgments = []

    class CancelModel(_Model):
        """仅替代已发出的生成请求。"""

        def invoke(self, messages: list[dict[str, object]]) -> AIMessage:
            """模拟生成返回前已断开。

            Args:
                messages: 真实图消息。

            Returns:
                有有效引用但不可继续审核或发布的草稿。
            """
            stream.close()
            return AIMessage(content="资料[1]")

    class Quality(_Quality):
        """记录不应在取消后启动的答案审核边界。"""

        def assess_answer(
            self, question: str, history: list[dict[str, str]],
            answer: str, evidence: list[EvidenceItem],
        ) -> AnswerAssessment:
            """保留实际图调用参数。

            Args:
                question: 原问题。
                history: 正式历史。
                answer: 草稿。
                evidence: 批准证据。

            Returns:
                固定审核结论。
            """
            answer_judgments.append(answer)
            return super().assess_answer(question, history, answer, evidence)

    store, index = _stores(tmp_path)
    graph = LangGraphConversationAgent(
        CancelModel([]),
        RetrievalService(VectorRetriever(store, FakeQueryEmbedder()), index),
        InMemorySaver(), top_k=1, quality_evaluator=Quality(),
    )

    def produce(sink: EventSink[str]) -> None:
        try:
            graph.run(
                uuid4(), "needle", [], kb_ids=("ops",),
                strategy="hybrid", initial_query="needle",
            )
        finally:
            stopped.set()

    stream = LiveEventStream(produce)
    with pytest.raises(StopIteration):
        next(stream)
    assert stopped.wait(timeout=0.5)
    assert answer_judgments == []

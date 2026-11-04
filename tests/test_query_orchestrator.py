# -*- coding: utf-8 -*-
"""同一编排器必须控制直答、澄清和真实Fast/Agent知识通道。"""

from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.knowledge.scopes import ScopeResolver
from app.application.query.fast_lane import FastLane
from app.application.query.generator import AnswerGenerator
from app.application.query.knowledge_runner import KnowledgeRunner
from app.application.query.orchestrator import QueryOrchestrator
from app.application.query.prompting import PromptBuilder
from app.application.query.routing import DeterministicRouter
from app.application.query.understanding import RuleUnderstandingService
from app.application.retrieval.service import RetrievalService
from app.application.retrieval.service import VectorRetriever
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.routing import QueryUnderstanding
from app.domain.routing import KnowledgeScope
from app.infrastructure.conversations.langgraph_agent import LangGraphConversationAgent
from app.infrastructure.persistence.knowledge_bases import SQLiteKnowledgeBaseRepository
from tests.test_fast_lane import _Provider
from tests.test_knowledge_runner import _RecordingRetriever
from tests.test_retriever import FakeQueryEmbedder
from tests.test_scoped_agent import _Model
from tests.test_scoped_agent import _Quality
from tests.test_scoped_retrieval import _stores


class _Understanding:
    """仅替代理解输出，后续路由与两个通道仍然真实执行。"""

    def __init__(self, output: object) -> None:
        """保存要验证或回退的输出。

        Args:
            output: 正常或非法的理解结果。
        """
        self.output = output

    def understand(self, question: str, history: object, catalog: object) -> object:
        """返回预设理解结果。

        Args:
            question: 原问题。
            history: 预算内历史。
            catalog: 后端目录快照。

        Returns:
            待严格校验的结果。
        """
        return self.output


def _build(tmp_path: Path, understanding=None, agent_model=None):
    catalog = SQLiteKnowledgeBaseRepository(tmp_path / "catalog.sqlite3")
    catalog.create(KnowledgeBase(id="ops", name="运维"))
    catalog.create(KnowledgeBase(id="dev", name="研发"))
    store, index = _stores(tmp_path / "chroma")
    retriever = _RecordingRetriever(RetrievalService(
        VectorRetriever(store, FakeQueryEmbedder()), index,
    ))
    quality = _Quality()
    graph = LangGraphConversationAgent(
        agent_model or _Model([AIMessage(content="资料[1]")]), retriever, InMemorySaver(),
        top_k=1, quality_evaluator=quality, clock=lambda: 2.0,
    )
    runner = KnowledgeRunner(FastLane(
        retriever, quality, AnswerGenerator(PromptBuilder(), _Provider("资料[1]")),
        clock=lambda: 2.0,
    ), graph, clock=lambda: 2.0)
    orchestrator = QueryOrchestrator(
        understanding=understanding or RuleUnderstandingService(),
        router=DeterministicRouter(), catalog=catalog,
        scope_resolver=ScopeResolver(catalog), knowledge_runner=lambda: runner,
        strategy="hybrid", top_k=1, turn_timeout_seconds=100.0,
        clock=lambda: 0.0,
    )
    return orchestrator, retriever


def test_chitchat_uses_direct_lane_without_retrieval(tmp_path: Path) -> None:
    """白名单闲聊不进入知识执行器。

    Args:
        tmp_path: 隔离目录。
    """
    orchestrator, retriever = _build(tmp_path)
    result = orchestrator.run(uuid4(), "你好", [])
    assert result.metadata.lane == "direct"
    assert result.metadata.outcome == "answered"
    assert result.metadata.grounding == "ungrounded"
    assert result.retrieval_calls == 0
    assert retriever.calls == []


def test_knowledge_uses_fast_judge_and_numbered_sources(tmp_path: Path) -> None:
    """普通知识问题必须查询并返回可复现的通道、范围和来源信息。

    Args:
        tmp_path: 隔离目录。
    """
    orchestrator, retriever = _build(tmp_path)
    result = orchestrator.run(uuid4(), "needle", [])
    assert result.answer == "资料[1]"
    assert result.metadata.lane == "fast"
    assert result.metadata.grounding == "grounded"
    assert result.metadata.kb_scope.searched == ("default", "dev", "ops")
    stats = result.metadata.evidence_stats
    assert stats.used == 1
    # 新口径必须自洽：检索=相关+排除，引用只能来自相关证据。
    assert stats.retrieved is not None and stats.relevant is not None
    assert stats.retrieved == stats.relevant + stats.excluded
    assert stats.used <= stats.relevant
    assert result.citations[0].position == 1
    assert result.citations[0].chunk_id == "dev:0"
    assert result.citations[0].kb_id == "dev"
    assert result.execution.budget.deadline == 100.0
    assert retriever.calls == [("needle", ("default", "dev", "ops"))]


def test_comparison_enters_agent_directly(tmp_path: Path) -> None:
    """复杂度只决定通道，不改变检索策略或免除首查。

    Args:
        tmp_path: 隔离目录。
    """
    orchestrator, _ = _build(tmp_path)
    result = orchestrator.run(uuid4(), "比较两种配置", [])
    assert result.metadata.lane == "agent"
    assert result.metadata.escalated is False
    assert result.retrieval_calls == 1


def test_low_confidence_rewrite_falls_back_to_original_question(tmp_path: Path) -> None:
    """低置信度不能保留模型幻觉出的独立查询。

    Args:
        tmp_path: 隔离目录。
    """
    orchestrator, retriever = _build(tmp_path, _Understanding(QueryUnderstanding(
        intent="chitchat", standalone_query="不应检索的改写", confidence=0.1,
    )))
    result = orchestrator.run(uuid4(), "原问题", [])
    assert result.metadata.lane == "fast"
    assert retriever.calls[0][0] == "原问题"


def test_clarification_does_not_call_knowledge_tools(tmp_path: Path) -> None:
    """明确的澄清问题按独立outcome返回，不伪装成已有证据的答案。

    Args:
        tmp_path: 隔离目录。
    """
    orchestrator, retriever = _build(tmp_path, _Understanding(QueryUnderstanding(
        intent="ambiguous", standalone_query="怎么配置", confidence=0.9,
        clarification="你指检索还是模型配置？",
    )))
    result = orchestrator.run(uuid4(), "怎么配置", [])
    assert result.metadata.lane == "clarify"
    assert result.metadata.outcome == "clarification"
    assert result.metadata.clarification.question == "你指检索还是模型配置？"
    assert result.retrieval_calls == 0
    assert retriever.calls == []


def test_understanding_scope_suggestion_is_applied_only_to_auto(tmp_path: Path) -> None:
    """理解层建议应缩小auto首查，但不能覆盖用户显式选择。

    Args:
        tmp_path: 隔离目录。
    """
    understood = _Understanding(QueryUnderstanding(
        standalone_query="needle", confidence=0.95,
        preferred_kb_ids=("dev",),
    ))
    orchestrator, retriever = _build(tmp_path, understood)
    result = orchestrator.run(uuid4(), "原问题", [])
    assert result.metadata.kb_scope.searched == ("dev",)
    assert retriever.calls == [("needle", ("dev",))]
    scoped = orchestrator.run(
        uuid4(), "原问题", [],
        kb_scope=KnowledgeScope(mode="explicit", kb_ids=("ops",)),
    )
    assert scoped.metadata.kb_scope.searched == ("ops",)
    assert scoped.metadata.scope_expanded is False


def test_unknown_model_scope_uses_entire_original_catalog(tmp_path: Path) -> None:
    """未知模型选库不能误当成空集合或创建库。

    Args:
        tmp_path: 隔离目录。
    """
    orchestrator, retriever = _build(tmp_path, _Understanding(QueryUnderstanding(
        standalone_query="needle", confidence=0.95,
        preferred_kb_ids=("hallucinated",),
    )))
    orchestrator.run(uuid4(), "原问题", [])
    assert retriever.calls == [("needle", ("default", "dev", "ops"))]


def test_subquery_hints_reach_real_graph_without_forcing_extra_searches(
    tmp_path: Path,
) -> None:
    """子问题供Agent覆盖任务使用，不应预先耗尽三次检索预算。

    Args:
        tmp_path: 隔离目录。
    """
    messages = []

    class HintModel(_Model):
        """记录外部模型实际收到的提示词边界。"""

        def invoke(self, payload: list[dict[str, object]]) -> AIMessage:
            """返回已预设的引用草稿。

            Args:
                payload: 图构造的模型消息。

            Returns:
                带合法引用的草稿。
            """
            messages.extend(payload)
            return super().invoke(payload)

    orchestrator, retriever = _build(
        tmp_path, _Understanding(QueryUnderstanding(
            standalone_query="needle", complexity="comparison",
            sub_queries=("SQLite保留期", "Chroma保留期"), confidence=0.95,
        )), HintModel([AIMessage(content="资料[1]")]),
    )
    result = orchestrator.run(uuid4(), "比较保留期", [])
    assert result.metadata.lane == "agent"
    assert result.retrieval_calls == 1
    assert retriever.calls == [("needle", ("default", "dev", "ops"))]
    planning = [message for message in messages if "candidate_sub_queries" in str(message)]
    assert len(planning) == 1
    assert "SQLite保留期" in str(planning[0])
    assert "Chroma保留期" in str(planning[0])

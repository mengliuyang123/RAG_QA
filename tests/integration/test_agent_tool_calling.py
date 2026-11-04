# -*- coding: utf-8 -*-
"""使用当前 .env 模型验证标准 tool calling 与真实多轮图执行。"""

import os
from pathlib import Path

import pytest

from app.application.conversations.service import ConversationService
from app.application.conversations.history_window import HistoryWindowBuilder
from app.bootstrap.factories import _create_conversation_agent
from app.config import Settings
from app.domain.conversations import AnswerMode
from app.domain.retrieval import RetrievalResult
from app.domain.retrieval import RetrievalStrategy
from app.infrastructure.conversations.checkpoint import SQLiteCheckpointStore
from app.infrastructure.conversations.sqlite_repository import (
    SQLiteConversationRepository,
)
from app.infrastructure.generation.token_counting import (
    Utf8TokenUpperBoundCounter,
)


pytestmark = pytest.mark.integration


class _RecordingKnowledgeBase:
    """避免额外 Embedding/Reranker 计费的确定性工具边界。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
    ) -> list[RetrievalResult]:
        assert strategy is RetrievalStrategy.HYBRID
        self.queries.append(question)
        return [
            RetrievalResult(
                chunk_id="langgraph:persistence",
                doc_id="langgraph",
                text=(
                    "LangGraph checkpoint 按 thread_id 组织状态快照，"
                    "可用于跨调用恢复图状态。"
                ),
                filename="langgraph_persistence.md",
                page=None,
                chunk_index=0,
                score=1.0,
            )
        ][:top_k]


def test_configured_model_runs_real_history_aware_tool_loop(
    tmp_path: Path,
) -> None:
    if os.getenv("RUN_AGENT_INTEGRATION") != "1":
        pytest.skip("设置 RUN_AGENT_INTEGRATION=1 才会产生真实模型请求")
    settings = Settings()
    if settings.resolved_llm_api_key is None:
        pytest.skip("当前 .env 未配置可用于 LLM_BASE_URL 的 LLM_API_KEY")

    knowledge_base = _RecordingKnowledgeBase()
    checkpoint_store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    agent = _create_conversation_agent(
        settings.resolved_llm_api_key,
        settings.llm_base_url,
        settings.llm_model,
        settings.agent_temperature,
        settings.llm_reasoning_effort,
        knowledge_base,  # type: ignore[arg-type]
        checkpoint_store.checkpointer,
        settings.top_k,
        settings.agent_max_retrieval_calls,
    )
    service = ConversationService(
        repository=SQLiteConversationRepository(tmp_path / "history.db"),
        agent=agent,
        checkpoint_store=checkpoint_store,
        history_turns=settings.conversation_history_turns,
        history_window=HistoryWindowBuilder(
            counter=Utf8TokenUpperBoundCounter(),
            token_budget=(
                settings.conversation_context_tokens
                - settings.conversation_reserved_tokens
            ),
            assistant_token_limit=settings.conversation_assistant_token_limit,
        ),
    )
    conversation = service.create()
    try:
        first = service.query(
            conversation.id,
            "请先查询知识库，再说明 LangGraph checkpoint 如何保存状态。",
        )
        second = service.query(
            conversation.id,
            "请再次查询知识库：它和 thread_id 是什么关系？",
        )
    finally:
        agent.close()
        checkpoint_store.close()

    assert first.answer_mode is AnswerMode.RETRIEVAL
    assert second.answer_mode is AnswerMode.RETRIEVAL
    assert first.citations and second.citations
    assert len(knowledge_base.queries) >= 2
    assert service.get(conversation.id).turns_count == 2

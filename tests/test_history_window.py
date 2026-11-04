# -*- coding: utf-8 -*-
"""以输入预算选择历史，不修改存储原文或截断用户问题。"""

from datetime import UTC
from datetime import datetime
from collections.abc import Iterator
from pathlib import Path
import sqlite3
from uuid import uuid4

import httpx
import pytest

from app.application.conversations.history_window import HistoryWindowBuilder
from app.application.conversations.service import ConversationService
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.conversations import ConversationTurn
from app.domain.errors import HistoryContextTooLargeError
from app.infrastructure.conversations.sqlite_repository import (
    SQLiteConversationRepository,
)
from app.infrastructure.generation.token_counting import (
    Utf8TokenUpperBoundCounter,
)
from app.interfaces.http.dependencies import get_conversation_service
from tests.http_auth import create_test_app as create_app
from tests.test_conversation_service import _FakeAgent
from tests.test_conversation_service import _FakeCheckpointStore


class _CharacterCounter:
    def count_text(self, text: str) -> int:
        """以字符作为确定性测试 token。

        Args:
            text: 输入文本。

        Returns:
            可手算的计数。
        """
        return len(text)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """按同一测试口径取前缀。

        Args:
            text: 输入文本。
            max_tokens: 前缀长度。

        Returns:
            预算内的前缀。
        """
        return text[:max_tokens]


def _turn(user: str, assistant: str) -> ConversationTurn:
    return ConversationTurn(
        id=uuid4(), conversation_id=uuid4(), user_message=user,
        assistant_message=assistant, answer_mode=AnswerMode.DIRECT,
        retrieval_calls=0, citations=[], latency_ms=0,
        created_at=datetime.now(UTC),
    )


def test_window_keeps_more_than_six_short_turns() -> None:
    """足够预算下不能再用固定六轮裁掉有效历史。"""
    turns = [_turn(f"u{i}", "a") for i in range(10)]
    window = HistoryWindowBuilder(
        counter=_CharacterCounter(), token_budget=100,
        assistant_token_limit=20, message_overhead_tokens=1,
    )
    result = window.build("q", reversed(turns))
    assert [turn.user_message for turn in result] == [f"u{i}" for i in range(10)]


def test_window_preserves_user_full_text_and_truncates_assistant_copy() -> None:
    """只裁剪模型看到的助手回答，调用者持有的原文不变。"""
    turn = _turn("完整用户问题", "abcdefghij")
    window = HistoryWindowBuilder(
        counter=_CharacterCounter(), token_budget=20,
        assistant_token_limit=5, message_overhead_tokens=1,
    )
    result = window.build("追问", iter([turn]))
    assert result[0].user_message == "完整用户问题"
    assert result[0].assistant_message == "ab..."
    assert turn.assistant_message == "abcdefghij"


def test_window_drops_oldest_whole_turn_and_stops_loading() -> None:
    """用户消息放不下时移除完整旧轮次，不截断或跳过缺口继续读。"""
    newest = _turn("new", "answer")
    too_large = _turn("old" * 100, "answer")

    def history() -> Iterator[ConversationTurn]:
        yield newest
        yield too_large
        raise AssertionError("窗口已满，不应继续扫描全部历史")

    window = HistoryWindowBuilder(
        counter=_CharacterCounter(), token_budget=30,
        assistant_token_limit=20, message_overhead_tokens=1,
    )
    assert window.build("q", history()) == [newest]


def test_current_question_cannot_be_silently_truncated() -> None:
    """当前问题超过预留后的预算时应明确拒绝，而非悄悄丢信息。"""
    window = HistoryWindowBuilder(
        counter=_CharacterCounter(), token_budget=4,
        assistant_token_limit=3, message_overhead_tokens=1,
    )
    with pytest.raises(HistoryContextTooLargeError):
        window.build("1234", iter([]))


def test_utf8_upper_bound_truncation_does_not_split_unicode() -> None:
    """通用保守计数不能假冒 provider usage，也不能切坏 UTF-8 字符。"""
    counter = Utf8TokenUpperBoundCounter()
    assert counter.count_text("你好A") == 7
    assert counter.truncate_text("你好A", 4) == "你"
    assert counter.truncate_text("你好A", 0) == ""


def test_repository_reverse_paging_and_window_do_not_change_stored_text(
    tmp_path: Path,
) -> None:
    """历史可以跨分页取足预算；截断仅作用于输入副本。

    Args:
        tmp_path: 隔离会话数据库。
    """
    repository = SQLiteConversationRepository(tmp_path / "history.db")
    conversation = repository.create()
    for index in range(9):
        repository.save_turn(
            conversation.id, f"u{index}",
            AgentResult(
                answer="abcdefghij", answer_mode=AnswerMode.DIRECT,
                retrieval_calls=0, citations=[],
            ),
            1.0,
        )
    # 相同时间戳必须用 rowid 继续分页，否则会跳过整页历史。
    with sqlite3.connect(tmp_path / "history.db") as connection:
        connection.execute(
            "UPDATE conversation_turns SET created_at = ?",
            ("2026-10-23T00:00:00+00:00",),
        )
    reverse = list(repository.iter_turns_reverse(conversation.id, batch_size=2))
    assert [turn.user_message for turn in reverse] == [
        f"u{i}" for i in range(8, -1, -1)
    ]
    window = HistoryWindowBuilder(
        counter=_CharacterCounter(), token_budget=100,
        assistant_token_limit=5, message_overhead_tokens=1,
    )
    selected = window.build("q", iter(reverse))
    assert len(selected) == 9
    assert all(turn.assistant_message == "ab..." for turn in selected)
    assert all(
        turn.assistant_message == "abcdefghij"
        for turn in repository.get(conversation.id).turns
    )


@pytest.fixture
def anyio_backend() -> str:
    """固定无需额外依赖的异步后端。

    Returns:
        asyncio 后端名称。
    """
    return "asyncio"


@pytest.mark.anyio
async def test_query_and_stream_reject_oversized_context_before_model_call(
    tmp_path: Path,
) -> None:
    """当前问题不被静默截断，流式应在发响应头前明确返回 422。

    Args:
        tmp_path: 正式历史的隔离目录。
    """
    agent = _FakeAgent()
    service = ConversationService(
        repository=SQLiteConversationRepository(tmp_path / "history.db"),
        agent=agent, checkpoint_store=_FakeCheckpointStore(),
        history_window=HistoryWindowBuilder(
            counter=_CharacterCounter(), token_budget=4,
            assistant_token_limit=3, message_overhead_tokens=1,
        ),
    )
    conversation = service.create()
    app = create_app()
    app.dependency_overrides[get_conversation_service] = lambda: service
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            for suffix in ("query", "query/stream"):
                response = await client.post(
                    f"/api/conversations/{conversation.id}/{suffix}",
                    json={"question": "1234"},
                )
                assert response.status_code == 422
        assert agent.questions == []
        assert service.get(conversation.id).turns == []
    finally:
        app.state.container.close()

# -*- coding: utf-8 -*-
"""多轮会话应用服务、并发与流式提交边界测试。"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from uuid import UUID

import pytest

from app.application.conversations.events import CompleteConversationEvent
from app.application.conversations.events import ErrorConversationEvent
from app.application.conversations.events import StatusConversationEvent
from app.application.conversations.events import TokenConversationEvent
from app.application.conversations.service import ConversationService
from app.application.conversations.history_window import HistoryWindowBuilder
from app.application.errors import ConversationAgentError
from app.application.errors import ConversationBusyError
from app.application.errors import ConversationDeleteError
from app.application.errors import ConversationDeletingError
from app.application.errors import ConversationStorageError
from app.domain.conversations import AgentResult
from app.domain.conversations import AnswerMode
from app.domain.conversations import ConversationStatus
from app.domain.conversations import ConversationTurn
from app.infrastructure.conversations.sqlite_repository import (
    SQLiteConversationRepository,
)
from app.infrastructure.generation.token_counting import Utf8TokenUpperBoundCounter


class _FakeCheckpointStore:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.deleted: list[UUID] = []

    def delete_thread(self, conversation_id: UUID) -> None:
        self.deleted.append(conversation_id)
        if self.error is not None:
            raise self.error


class _FakeAgent:
    def __init__(
        self,
        result: AgentResult | None = None,
        error: Exception | None = None,
        entered: Event | None = None,
        release: Event | None = None,
    ) -> None:
        self.result = result or AgentResult(
            answer="直接回答",
            answer_mode=AnswerMode.DIRECT,
            retrieval_calls=0,
            citations=[],
        )
        self.error = error
        self.entered = entered
        self.release = release
        self.histories: list[list[ConversationTurn]] = []
        self.questions: list[str] = []

    def run(
        self,
        conversation_id: UUID,
        question: str,
        history: list[ConversationTurn],
        on_status: Callable[[str, int | None], None] | None = None,
    ) -> AgentResult:
        del conversation_id
        self.questions.append(question)
        self.histories.append(history)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=2)
        if self.error is not None:
            raise self.error
        if on_status is not None:
            on_status("retrieving", 1)
            on_status("validating", None)
        return self.result


def _service(
    tmp_path: Path,
    agent: _FakeAgent,
    checkpoint: _FakeCheckpointStore | None = None,
) -> ConversationService:
    return ConversationService(
        repository=SQLiteConversationRepository(
            tmp_path / "conversations.db"
        ),
        agent=agent,
        checkpoint_store=checkpoint or _FakeCheckpointStore(),
        history_turns=6,
        history_window=HistoryWindowBuilder(
            counter=Utf8TokenUpperBoundCounter(), token_budget=16384,
        ),
    )


def test_stream_history_failure_is_a_terminal_error_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首状态已发出后，读取历史失败必须发送 error 而非断流。"""
    service = _service(tmp_path, _FakeAgent())
    conversation = service.create()

    def fail_history(
        conversation_id: UUID, batch_size: int = 32,
    ) -> list[ConversationTurn]:
        raise ConversationStorageError("private database detail")

    monkeypatch.setattr(service._repository, "iter_turns_reverse", fail_history)
    events = list(service.stream_query(conversation.id, "问题"))
    assert [event.event for event in events] == ["status", "error"]
    assert "private" not in events[-1].model_dump_json()


def test_status_check_does_not_load_entire_history_and_releases_lock_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一次查询的状态检查只读摘要，结束后释放锁字典条目。"""
    service = _service(tmp_path, _FakeAgent())
    conversation = service.create()

    def fail_full_history(*args: object) -> None:
        raise AssertionError("状态检查不应读取整个会话")

    monkeypatch.setattr(service._repository, "get", fail_full_history)
    result = service.query(conversation.id, "问题")
    assert result.answer == "直接回答"
    assert service._conversation_locks == {}


def test_query_uses_budget_instead_of_fixed_six_turns_and_saves_result(
    tmp_path: Path,
) -> None:
    agent = _FakeAgent()
    service = _service(tmp_path, agent)
    conversation = service.create()
    for index in range(8):
        service.query(conversation.id, f"问题 {index}")

    result = service.query(conversation.id, "  指代性追问  ")

    assert agent.questions[-1] == "指代性追问"
    assert len(agent.histories[-1]) == 8
    assert agent.histories[-1][0].user_message == "问题 0"
    loaded = service.get(conversation.id)
    assert loaded.turns[-1].id == result.turn_id
    assert loaded.turns[-1].assistant_message == result.answer
    assert result.citations == []


def test_agent_failure_does_not_create_half_turn(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        _FakeAgent(error=ConversationAgentError("model failed")),
    )
    conversation = service.create()

    with pytest.raises(ConversationAgentError):
        service.query(conversation.id, "问题")

    assert service.get(conversation.id).turns == []


def test_same_conversation_conflicts_while_different_one_can_run(
    tmp_path: Path,
) -> None:
    entered = Event()
    release = Event()
    service = _service(
        tmp_path,
        _FakeAgent(entered=entered, release=release),
    )
    first = service.create()
    second = service.create()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(service.query, first.id, "阻塞问题")
        assert entered.wait(timeout=2)
        with pytest.raises(ConversationBusyError):
            service.query(first.id, "同会话")
        release.set()
        future.result(timeout=2)

    assert service.query(second.id, "不同会话").conversation_id == second.id


def test_deleting_conversation_rejects_new_query(tmp_path: Path) -> None:
    service = _service(tmp_path, _FakeAgent())
    conversation = service.create()
    service._repository.mark_deleting(conversation.id)

    with pytest.raises(ConversationDeletingError):
        service.query(conversation.id, "问题")


def test_delete_failure_preserves_retryable_deleting_state(
    tmp_path: Path,
) -> None:
    checkpoint = _FakeCheckpointStore(
        error=ConversationDeleteError("checkpoint failed")
    )
    service = _service(tmp_path, _FakeAgent(), checkpoint)
    conversation = service.create()

    with pytest.raises(ConversationDeleteError):
        service.delete(conversation.id)

    assert service.get(conversation.id).status is ConversationStatus.DELETING
    assert checkpoint.deleted == [conversation.id]


def test_stream_emits_safe_order_and_saves_immediately_before_complete(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _FakeAgent())
    conversation = service.create()

    events = list(service.stream_query(conversation.id, "问题"))

    assert [event.event for event in events] == [
        "status",
        "status",
        "status",
        "token",
        "complete",
    ]
    assert [
        event.data.phase
        for event in events
        if isinstance(event, StatusConversationEvent)
    ] == ["deciding", "retrieving", "validating"]
    tokens = [
        event.data.token
        for event in events
        if isinstance(event, TokenConversationEvent)
    ]
    complete = events[-1]
    assert isinstance(complete, CompleteConversationEvent)
    assert "".join(tokens) == complete.data.answer
    assert service.get(conversation.id).turns_count == 1


def test_closing_stream_before_complete_does_not_save_turn(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _FakeAgent())
    conversation = service.create()
    iterator = service.stream_query(conversation.id, "问题")

    while True:
        event = next(iterator)
        if isinstance(event, TokenConversationEvent):
            break
    iterator.close()

    assert service.get(conversation.id).turns == []


def test_stream_converts_agent_failure_to_stable_error(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        _FakeAgent(error=ConversationAgentError("private")),
    )
    conversation = service.create()

    events = list(service.stream_query(conversation.id, "问题"))

    assert events[0].event == "status"
    assert isinstance(events[-1], ErrorConversationEvent)
    assert events[-1].data.model_dump() == {
        "code": "agent_error",
        "message": "会话生成服务暂时不可用",
    }
    assert "private" not in events[-1].model_dump_json()


def test_unknown_save_failure_is_one_safe_terminal_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最后提交阶段的未知异常也不能直接断流或泄露内部文案。

    Args:
        tmp_path: 隔离SQLite目录。
        monkeypatch: 注入持久化边界的未知故障。
    """
    service = _service(tmp_path, _FakeAgent())
    conversation = service.create()

    def fail_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError("private storage detail")

    monkeypatch.setattr(service._repository, "save_turn", fail_save)
    events = list(service.stream_query(conversation.id, "问题"))
    terminals = [event for event in events if event.event in {"error", "complete"}]
    assert len(terminals) == 1
    assert terminals[0].event == "error"
    assert terminals[0].data.code == "internal_error"
    assert "private" not in terminals[0].model_dump_json()
    assert service.get(conversation.id).turns == []
    assert not service._conversation_locks

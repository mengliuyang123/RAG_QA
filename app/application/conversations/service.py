# -*- coding: utf-8 -*-
"""编排多轮历史、Agent 执行、并发控制与正式轮次提交。"""

from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from threading import RLock
from time import perf_counter
from uuid import UUID

from app.application.conversations.events import CompleteConversationEvent
from app.application.conversations.events import ConversationStreamEvent
from app.application.conversations.events import ErrorConversationData
from app.application.conversations.events import ErrorConversationEvent
from app.application.conversations.events import StatusConversationEvent
from app.application.conversations.events import TokenConversationData
from app.application.conversations.events import TokenConversationEvent
from app.application.conversations.history_window import HistoryWindowBuilder
from app.application.conversations.ports import ConversationCheckpointStore
from app.application.conversations.ports import ConversationRepository
from app.application.conversations.progress import ProgressReporter
from app.application.conversations.progress import notify_queued
from app.application.admission import QuestionAdmissionController
from app.application.tracing.admission import admitted_question
from app.application.errors import QuestionRateLimitError
from app.application.errors import QuestionQueueFullError
from app.application.errors import QuestionQueueTimeoutError
from app.application.errors import ConversationAgentError
from app.application.errors import ConversationBusyError
from app.application.errors import ConversationDeletingError
from app.application.errors import ConversationNotFoundError
from app.application.errors import ConversationStorageError
from app.application.errors import ConversationTimeoutError
from app.application.errors import EmbeddingError
from app.application.errors import EmbeddingTimeoutError
from app.application.errors import KnowledgeBaseStorageError
from app.application.errors import LLMError
from app.application.errors import LLMTimeoutError
from app.application.errors import RerankerUnavailableError
from app.application.errors import VectorStoreError
from app.application.errors import TraceStorageError
from app.application.ports.query_execution import QuestionExecutor
from app.application.streaming import EventSink
from app.application.streaming import LiveEventStream
from app.domain.conversations import AgentResult
from app.domain.conversations import Conversation
from app.domain.conversations import ConversationQueryResult
from app.domain.conversations import ConversationStatus
from app.domain.conversations import ConversationSummary
from app.domain.conversations import ConversationTurn
from app.domain.conversation_pages import ConversationPage
from app.domain.conversation_pages import ConversationDetailPage
from app.domain.errors import InvalidConversationCursorError
from app.domain.errors import PromptContextTooLargeError
from app.domain.routing import KnowledgeScope
from app.domain.tracing import CommitStatus
from app.application.tracing import TraceService
from app.application.execution_control import StreamCancelled
from app.application.execution_control import check_cancellation


class ConversationService:
    """执行会话 CRUD、Agentic RAG 查询和安全流式输出。"""

    def __init__(
        self,
        repository: ConversationRepository,
        agent: QuestionExecutor,
        checkpoint_store: ConversationCheckpointStore,
        history_window: HistoryWindowBuilder,
        history_turns: int = 6,
        clock: Callable[[], float] = perf_counter,
        token_chunk_size: int = 32,
        scope_validator: Callable[[KnowledgeScope], None] | None = None,
        traces: TraceService | None = None,
        admission: QuestionAdmissionController | None = None,
    ) -> None:
        """保存正式历史、执行器及由组合根注入的历史预算策略。

        Args:
            repository: 正式会话存储。
            agent: 统一高层执行器；参数名称保留旧调用兼容。
            checkpoint_store: 图状态清理端口。
            history_window: 当前问题与历史共享的预算选择器。
            history_turns: 兼容旧调用的分页大小，不再限制历史轮数。
            clock: 统计延迟的时钟。
            token_chunk_size: 已校验答案的输出分片大小。
            scope_validator: 生产编排器提供的范围预检，不调用模型。
            traces: 正式提交结果的追踪记录，不参与业务库事务。
            admission: 与单轮入口共享的线上准入，不影响可信离线评估。

        Returns:
            None。
        """
        if history_turns < 1:
            raise ValueError("history_turns 必须大于零")
        if token_chunk_size < 1:
            raise ValueError("token_chunk_size 必须大于零")
        self._repository = repository
        self._agent = agent
        self._checkpoint_store = checkpoint_store
        self._history_turns = history_turns
        self._history_window = history_window
        self._clock = clock
        self._token_chunk_size = token_chunk_size
        self._scope_validator = scope_validator
        self._traces = traces
        self._admission = admission
        self._locks_guard = RLock()
        self._conversation_locks: dict[UUID, Lock] = {}

    def create(
        self, kb_scope: KnowledgeScope | None = None, *,
        owner_id: UUID | None = None,
    ) -> Conversation:
        """创建空会话，显式范围须先通过目录检查。

        Args:
            kb_scope: 会话默认范围，未指定时自动选库。
            owner_id: 来自已验证会话，不能取自HTTP请求体。

        Returns:
            已持久化的空会话。
        """
        if kb_scope is not None and self._scope_validator is not None:
            self._scope_validator(kb_scope)
        options: dict[str, object] = {}
        if kb_scope is not None:
            options["kb_scope"] = kb_scope
        if owner_id is not None:
            options["owner_id"] = owner_id
        return self._repository.create(**options)

    def update(
        self,
        conversation_id: UUID,
        *,
        title: str | None = None,
        kb_scope: KnowledgeScope | None = None,
        owner_id: UUID | None = None,
    ) -> ConversationSummary:
        """在会话锁内修改标题或默认范围。

        Args:
            conversation_id: 待修改会话。
            title: 可选新标题，去除首尾空白后长度须为 1–100。
            kb_scope: 可选新默认范围，不修改历史轮次。
            owner_id: 已验证的当前用户。

        Returns:
            修改后的会话摘要。
        """
        if title is None and kb_scope is None:
            raise ValueError("至少提供标题或知识库范围")
        if title is not None:
            title = title.strip()
            if not title or len(title) > 100:
                raise ValueError("标题长度必须为 1–100")
        with self._acquire(conversation_id, owner_id=owner_id):
            if kb_scope is not None and self._scope_validator is not None:
                self._scope_validator(kb_scope)
            return self._repository.update(
                conversation_id, title=title, kb_scope=kb_scope,
            )

    def list(self, *, owner_id: UUID | None = None) -> list[ConversationSummary]:
        """返回当前用户的会话摘要。

        Args:
            owner_id: 已验证用户，None仅供可信内部调用。

        Returns:
            最近更新在前的摘要。
        """
        if owner_id is None:
            return self._repository.list_active()
        return self._repository.list_active(owner_id=owner_id)

    def get(
        self, conversation_id: UUID, *, owner_id: UUID | None = None,
    ) -> Conversation:
        """检查归属后才加载完整历史。

        Args:
            conversation_id: 会话标识。
            owner_id: 已验证用户。

        Returns:
            当前用户的会话与完整正式轮次。
        """
        self._check_owner(conversation_id, owner_id)
        return self._repository.get(conversation_id)

    def list_page(
        self, *, owner_id: UUID, limit: int = 20,
        cursor: str | None = None,
    ) -> ConversationPage:
        """列出当前用户的有限摘要页，不加载任何正文。

        Args:
            owner_id: 已认证身份。
            limit: 每页1–100条。
            cursor: 同用户下一页位置。

        Returns:
            当前页与全部本人active会话数。
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InvalidConversationCursorError("分页大小必须为整数")
        if limit < 1 or limit > 100:
            raise InvalidConversationCursorError("分页大小必须为1–100")
        return self._repository.list_active_page(owner_id, limit, cursor)

    def get_page(
        self, conversation_id: UUID, *, owner_id: UUID,
        limit: int = 20, cursor: str | None = None,
    ) -> ConversationDetailPage:
        """返回有限正式历史页，归属校验与正文读取共用快照。

        Args:
            conversation_id: 目标会话。
            owner_id: 已认证身份，无管理员特例。
            limit: 每页1–100轮。
            cursor: 更早一页的位置。

        Returns:
            页内正序历史与保留完整计数的会话摘要。
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InvalidConversationCursorError("分页大小必须为整数")
        if limit < 1 or limit > 100:
            raise InvalidConversationCursorError("分页大小必须为1–100")
        return self._repository.get_page(
            conversation_id, owner_id, limit, cursor,
        )

    def delete(
        self, conversation_id: UUID, *, owner_id: UUID | None = None,
    ) -> None:
        """只清理当前用户会话，不能借删除操作探测其他会话。

        Args:
            conversation_id: 会话标识。
            owner_id: 已验证用户。

        Returns:
            None。
        """
        with self._acquire(
            conversation_id, allow_deleting=True, owner_id=owner_id,
        ):
            self._repository.mark_deleting(conversation_id)
            self._checkpoint_store.delete_thread(conversation_id)
            self._repository.delete(conversation_id)

    def query(
        self,
        conversation_id: UUID,
        question: str,
        *, kb_scope: KnowledgeScope | None = None,
        owner_id: UUID | None = None,
    ) -> ConversationQueryResult:
        """运行统一编排器并在成功后原子保存一个正式轮次。

        Args:
            conversation_id: 正式会话标识。
            question: 本轮原始问题。
            kb_scope: 本轮范围覆盖，不改会话默认值。
            owner_id: 已验证身份；None仅供可信内部调用。

        Returns:
            已提交的回答与引用。
        """
        normalized_question = self._normalize_question(question)
        self._history_window.validate_question(normalized_question)
        with self._acquire(conversation_id, owner_id=owner_id) as conversation:
            started_at = self._clock()
            options = self._execution_options(conversation, kb_scope, owner_id)
            if self._scope_validator is not None:
                self._scope_validator(options.get("kb_scope", KnowledgeScope()))
            with admitted_question(
                self._admission, self._traces, user_id=owner_id,
                question=normalized_question, execution_id=conversation_id,
                entry="conversation",
            ):
                history = self._history_window.build(
                    normalized_question,
                    self._repository.iter_turns_reverse(
                        conversation_id, batch_size=self._history_turns,
                    ),
                )
                result = self._agent.run(
                    conversation_id, normalized_question, history, **options,
                )
            latency_ms = self._elapsed_ms(started_at)
            turn = self._save_turn(
                conversation_id,
                normalized_question,
                result,
                latency_ms,
            )
            return self._query_result(turn)

    def stream_query(
        self,
        conversation_id: UUID,
        question: str,
        *, kb_scope: KnowledgeScope | None = None,
        owner_id: UUID | None = None,
    ) -> LiveEventStream[ConversationStreamEvent]:
        """实时交付阶段，完整校验后交付正文，complete前提交。

        Args:
            conversation_id: 正式会话。
            question: 本轮原始问题。
            kb_scope: 单轮范围覆盖，不修改默认值。
            owner_id: 后端验证的用户身份。

        Returns:
            可取消的单槽背压事件流。
        """
        normalized_question = self._normalize_question(question)
        self._history_window.validate_question(normalized_question)
        return LiveEventStream(lambda sink: self._produce_stream(
            sink, conversation_id, normalized_question, kb_scope, owner_id,
        ))

    def _produce_stream(
        self, sink: EventSink[ConversationStreamEvent], conversation_id: UUID,
        normalized_question: str, kb_scope: KnowledgeScope | None,
        owner_id: UUID | None,
    ) -> None:
        with self._acquire(conversation_id, owner_id=owner_id) as conversation:
            started_at = self._clock()
            scope_options = self._execution_options(conversation, kb_scope, owner_id)
            if self._scope_validator is not None:
                self._scope_validator(
                    scope_options.get("kb_scope", KnowledgeScope()),
                )
            capture_status = ProgressReporter(
                lambda data: sink.emit(StatusConversationEvent(data=data)),
            )
            try:
                with admitted_question(
                    self._admission, self._traces, user_id=owner_id,
                    question=normalized_question, execution_id=conversation_id,
                    entry="conversation",
                    on_queued=lambda position: notify_queued(capture_status, position),
                ):
                    capture_status("deciding")
                    history = self._history_window.build(
                        normalized_question,
                        self._repository.iter_turns_reverse(
                            conversation_id, batch_size=self._history_turns,
                        ),
                    )
                    result = self._agent.run(
                        conversation_id, normalized_question, history,
                        capture_status, **scope_options,
                    )
            except QuestionRateLimitError:
                raise
            except QuestionQueueFullError as error:
                if not error.was_queued:
                    raise
                sink.emit(ErrorConversationEvent(data=ErrorConversationData(
                    code="queue_full", message="问答暂时不可用，请稍后重试",
                )))
                return
            except QuestionQueueTimeoutError:
                sink.emit(ErrorConversationEvent(data=ErrorConversationData(
                    code="queue_timeout", message="排队等待超时，请稍后重试",
                )))
                return
            except PromptContextTooLargeError:
                sink.emit(ErrorConversationEvent(
                    data=ErrorConversationData(
                        code="context_too_large",
                        message="系统、历史与检索证据超过模型输入预算",
                    ),
                ))
                return
            except (ConversationTimeoutError, EmbeddingTimeoutError, LLMTimeoutError):
                sink.emit(ErrorConversationEvent(
                    data=ErrorConversationData(
                        code="timeout",
                        message="会话处理超时，请稍后重试",
                    )
                ))
                return
            except (EmbeddingError, VectorStoreError, RerankerUnavailableError):
                sink.emit(ErrorConversationEvent(data=ErrorConversationData(
                    code="retrieval_error", message="检索服务暂时不可用",
                )))
                return
            except LLMError:
                sink.emit(ErrorConversationEvent(data=ErrorConversationData(
                    code="generation_error", message="问答生成服务暂时不可用",
                )))
                return
            except KnowledgeBaseStorageError:
                sink.emit(ErrorConversationEvent(data=ErrorConversationData(
                    code="storage_error", message="知识库目录暂时不可用",
                )))
                return
            except TraceStorageError:
                sink.emit(ErrorConversationEvent(data=ErrorConversationData(
                    code="storage_error", message="追踪存储暂时不可用",
                )))
                return
            except ConversationAgentError:
                sink.emit(ErrorConversationEvent(
                    data=ErrorConversationData(
                        code="agent_error",
                        message="会话生成服务暂时不可用",
                    )
                ))
                return
            except ConversationStorageError:
                sink.emit(self._storage_stream_error())
                return
            except Exception:
                sink.emit(ErrorConversationEvent(data=ErrorConversationData(
                    code="internal_error", message="会话处理失败",
                )))
                return
            try:
                for token in self._chunks(result.answer):
                    sink.emit(TokenConversationEvent(
                        data=TokenConversationData(token=token)
                    ))
                latency_ms = self._elapsed_ms(started_at)
                sink.check()
                turn = self._save_turn(
                    conversation_id,
                    normalized_question,
                    result,
                    latency_ms,
                )
            except StreamCancelled:
                self._record_commit(result, "cancelled")
                raise
            except ConversationStorageError:
                sink.emit(self._storage_stream_error())
                return
            except Exception:
                sink.emit(ErrorConversationEvent(data=ErrorConversationData(
                    code="internal_error", message="会话提交失败",
                )))
                return
            sink.emit(CompleteConversationEvent(data=self._query_result(turn)))

    def _save_turn(
        self, conversation_id: UUID, question: str,
        result: AgentResult, latency_ms: float,
    ) -> ConversationTurn:
        try:
            check_cancellation()
            turn = self._repository.save_turn(
                conversation_id, question, result, latency_ms,
            )
        except BaseException as error:
            status = "cancelled" if isinstance(error, StreamCancelled) else "failed"
            self._record_commit(result, status)
            raise
        self._record_commit(result, "committed", turn.id)
        return turn

    def _record_commit(
        self, result: AgentResult, status: CommitStatus,
        turn_id: UUID | None = None,
    ) -> None:
        if (
            self._traces is not None and result.metadata is not None
            and result.metadata.trace_id is not None
        ):
            self._traces.record_commit(
                UUID(result.metadata.trace_id), status, turn_id,
            )

    @contextmanager
    def _acquire(
        self,
        conversation_id: UUID,
        *,
        allow_deleting: bool = False,
        owner_id: UUID | None = None,
    ) -> Iterator[ConversationSummary]:
        # 先查归属再查忙碌状态，避免409泄漏他人会话存在和活动情况。
        self._check_owner(conversation_id, owner_id)
        with self._locks_guard:
            lock = self._conversation_locks.setdefault(
                conversation_id,
                Lock(),
            )
            if not lock.acquire(blocking=False):
                raise ConversationBusyError("该会话已有请求正在运行")
        try:
            conversation = self._repository.get_summary(conversation_id)
            if owner_id is not None and conversation.owner_id != owner_id:
                raise ConversationNotFoundError("会话不存在")
            if (
                conversation.status is ConversationStatus.DELETING
                and not allow_deleting
            ):
                raise ConversationDeletingError("会话正在删除")
            yield conversation
        finally:
            # 获取、释放与移除共享同一保护锁，避免新请求绑定到旧锁。
            with self._locks_guard:
                lock.release()
                self._conversation_locks.pop(conversation_id, None)

    def _check_owner(
        self, conversation_id: UUID, owner_id: UUID | None,
    ) -> None:
        if owner_id is not None:
            summary = self._repository.get_summary(conversation_id)
            if summary.owner_id != owner_id:
                raise ConversationNotFoundError("会话不存在")

    @staticmethod
    def _execution_options(
        conversation: ConversationSummary,
        override: KnowledgeScope | None,
        owner_id: UUID | None,
    ) -> dict[str, object]:
        options: dict[str, object] = {}
        if owner_id is not None:
            options["user_id"] = owner_id
        if override is not None:
            options["kb_scope"] = override
        elif conversation.kb_scope.mode == "explicit":
            options["kb_scope"] = conversation.kb_scope
        return options

    def _elapsed_ms(self, started_at: float) -> float:
        return max(0.0, round((self._clock() - started_at) * 1_000, 3))

    def _chunks(self, answer: str) -> Iterator[str]:
        for offset in range(0, len(answer), self._token_chunk_size):
            token = answer[offset : offset + self._token_chunk_size]
            if token:
                yield token

    @staticmethod
    def _normalize_question(question: str) -> str:
        normalized = question.strip()
        if not normalized:
            raise ValueError("问题不能为空")
        return normalized

    @staticmethod
    def _query_result(turn: ConversationTurn) -> ConversationQueryResult:
        return ConversationQueryResult(
            conversation_id=turn.conversation_id,
            turn_id=turn.id,
            answer=turn.assistant_message,
            answer_mode=turn.answer_mode,
            retrieval_calls=turn.retrieval_calls,
            citations=turn.citations,
            latency_ms=turn.latency_ms,
            metadata=turn.metadata, created_at=turn.created_at,
        )

    @staticmethod
    def _storage_stream_error() -> ErrorConversationEvent:
        return ErrorConversationEvent(
            data=ErrorConversationData(
                code="storage_error",
                message="会话存储暂时不可用",
            )
        )

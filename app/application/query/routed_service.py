# -*- coding: utf-8 -*-
"""把统一编排器适配为无正式会话历史的查询入口。"""

from collections.abc import Callable
from datetime import datetime
from datetime import timezone
from time import perf_counter
from uuid import UUID
from uuid import uuid4

from app.application.conversations.ports import StatusCallback
from app.application.conversations.progress import ProgressReporter
from app.application.conversations.progress import notify_queued
from app.application.admission import QuestionAdmissionController
from app.application.tracing.admission import admitted_question
from app.application.tracing.service import TraceService
from app.application.errors import QuestionRateLimitError
from app.application.errors import QuestionQueueFullError
from app.application.errors import QuestionQueueTimeoutError
from app.application.errors import ConversationAgentError
from app.application.errors import ConversationTimeoutError
from app.application.errors import EmbeddingError
from app.application.errors import EmbeddingTimeoutError
from app.application.errors import LLMError
from app.application.errors import LLMTimeoutError
from app.application.errors import RerankerUnavailableError
from app.application.errors import VectorStoreError
from app.application.errors import KnowledgeBaseStorageError
from app.application.errors import TraceStorageError
from app.application.query.events import CompleteStreamData
from app.application.query.events import CompleteStreamEvent
from app.application.query.events import ErrorStreamData
from app.application.query.events import ErrorStreamEvent
from app.application.query.events import StatusStreamEvent
from app.application.query.events import StreamEvent
from app.application.query.events import TokenStreamData
from app.application.query.events import TokenStreamEvent
from app.application.query.orchestrator import QueryOrchestrator
from app.application.query.debug import retrieval_debug
from app.application.streaming import EventSink
from app.application.streaming import LiveEventStream
from app.domain.answers import Citation
from app.domain.answers import QueryResult
from app.domain.errors import InvalidKnowledgeScopeError
from app.domain.errors import PromptContextTooLargeError
from app.domain.retrieval import RetrievalStrategy
from app.domain.routing import KnowledgeScope


class RoutedQueryService:
    """无状态入口只增加外壳，不再自行检索或直接调用生成器。"""

    def __init__(
        self, orchestrator: QueryOrchestrator,
        cleanup: Callable[[UUID], None],
        clock: Callable[[], float] = perf_counter,
        *, admission: QuestionAdmissionController | None = None,
        traces: TraceService | None = None,
    ) -> None:
        """保存共享编排器与精确清理临时checkpoint的回调。

        Args:
            orchestrator: 与正式会话共用的同一实例。
            cleanup: 只删除本入口创建的临时执行ID。
            clock: 延迟计时器。
            admission: 两入口共享的线上准入与个人窗口。
            traces: 与编排器共用的追踪服务，覆盖准入等待。
        """
        self._orchestrator = orchestrator
        self._cleanup = cleanup
        self._clock = clock
        self._admission = admission
        self._traces = traces

    def query(
        self, question: str,
        strategy: RetrievalStrategy | str = RetrievalStrategy.VECTOR,
        top_k: int = 5, *, kb_scope: KnowledgeScope | None = None,
        lane: str = "auto", debug: bool = False,
        on_status: StatusCallback | None = None,
        user_id: UUID | None = None,
    ) -> QueryResult:
        """运行统一链路，不保存为正式会话轮次。

        Args:
            question: 原始问题。
            strategy: 保留管理检索接口原有的三策略参数。
            top_k: 结果上限。
            kb_scope: 可选显式范围。
            lane: 测试起点，不影响是否允许升级。
            debug: 调试不绕过任何审核。
            on_status: 阶段回调。
            user_id: 由HTTP认证确定的当前用户。

        Returns:
            保留旧字段并携带统一元数据的结果。
        """
        self._orchestrator.validate_scope(kb_scope or KnowledgeScope())
        return self._execute_question(
            question, strategy, top_k, kb_scope=kb_scope, lane=lane,
            debug=debug, on_status=on_status, user_id=user_id,
        )

    def _execute_question(
        self, question: str, strategy: RetrievalStrategy | str, top_k: int, *,
        kb_scope: KnowledgeScope | None, lane: str, debug: bool,
        on_status: StatusCallback | None, user_id: UUID | None,
    ) -> QueryResult:
        request_id = uuid4()
        started = self._clock()
        try:
            with admitted_question(
                self._admission, self._traces, user_id=user_id,
                question=question, execution_id=request_id, entry="query",
                on_queued=lambda position: notify_queued(on_status, position),
            ):
                if on_status is not None:
                    on_status("deciding", None)
                result = self._orchestrator.run(
                    request_id, question, [], on_status,
                    kb_scope=kb_scope, requested_lane=lane,
                    strategy=strategy, top_k=top_k, debug=debug,
                    execution_entry="query", user_id=user_id,
                )
            return QueryResult(
                answer=result.answer,
                citations=[Citation.model_validate(item.model_dump()) for item in result.citations],
                latency_ms=max(0.0, (self._clock() - started) * 1000),
                retrieval_strategy=RetrievalStrategy(strategy),
                answer_mode=result.answer_mode,
                retrieval_calls=result.retrieval_calls,
                metadata=result.metadata, created_at=datetime.now(timezone.utc),
                retrieved=retrieval_debug(result) if debug else None,
            )
        finally:
            self._cleanup(request_id)

    def stream_query(
        self, question: str,
        strategy: RetrievalStrategy | str = RetrievalStrategy.VECTOR,
        top_k: int = 5, *, kb_scope: KnowledgeScope | None = None,
        lane: str = "auto", debug: bool = False,
        user_id: UUID | None = None,
    ) -> LiveEventStream[StreamEvent]:
        """实时交付阶段，仅在校验成功后分片正文。

        Args:
            question: 原问题。
            strategy: 检索策略。
            top_k: 结果上限。
            kb_scope: 请求范围。
            lane: 执行起点。
            debug: 在complete中附加实际召回诊断，不改变审核规则。
            user_id: 由HTTP认证确定的当前用户。

        Returns:
            可取消且单槽背压的阶段、正文及唯一终止事件流。
        """
        return LiveEventStream(lambda sink: self._produce_stream(
            sink, question, strategy, top_k,
            kb_scope=kb_scope, lane=lane, debug=debug,
            user_id=user_id,
        ))

    def _produce_stream(
        self, sink: EventSink[StreamEvent], question: str,
        strategy: RetrievalStrategy | str, top_k: int, *,
        kb_scope: KnowledgeScope | None, lane: str, debug: bool,
        user_id: UUID | None,
    ) -> None:
        self._orchestrator.validate_scope(kb_scope or KnowledgeScope())
        on_status = ProgressReporter(
            lambda data: sink.emit(StatusStreamEvent(data=data)),
        )
        try:
            result = self._execute_question(
                question, strategy, top_k, kb_scope=kb_scope,
                lane=lane, debug=debug, on_status=on_status,
                user_id=user_id,
            )
        except QuestionRateLimitError:
            raise
        except QuestionQueueFullError as error:
            if not error.was_queued:
                raise
            sink.emit(self._error("queue_full", "问答暂时不可用，请稍后重试"))
            return
        except QuestionQueueTimeoutError:
            sink.emit(self._error("queue_timeout", "排队等待超时，请稍后重试"))
            return
        except (ConversationTimeoutError, EmbeddingTimeoutError, LLMTimeoutError):
            sink.emit(self._error("timeout", "问答处理超时，请稍后重试"))
            return
        except (PromptContextTooLargeError, InvalidKnowledgeScopeError):
            sink.emit(self._error("context_error", "输入上下文或知识库范围无效"))
            return
        except (EmbeddingError, VectorStoreError, RerankerUnavailableError):
            sink.emit(self._error("retrieval_error", "检索服务暂时不可用"))
            return
        except (LLMError, ConversationAgentError):
            sink.emit(self._error("generation_error", "问答生成服务暂时不可用"))
            return
        except KnowledgeBaseStorageError:
            sink.emit(self._error("storage_error", "知识库目录暂时不可用"))
            return
        except TraceStorageError:
            sink.emit(self._error("storage_error", "追踪存储暂时不可用"))
            return
        except Exception:
            sink.emit(self._error("internal_error", "问答处理失败"))
            return
        for offset in range(0, len(result.answer), 32):
            sink.emit(TokenStreamEvent(data=TokenStreamData(
                token=result.answer[offset:offset + 32],
            )))
        sink.emit(CompleteStreamEvent(
            data=CompleteStreamData.model_validate(result.model_dump()),
        ))

    @staticmethod
    def _error(code: str, message: str) -> ErrorStreamEvent:
        return ErrorStreamEvent(data=ErrorStreamData(code=code, message=message))

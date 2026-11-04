# -*- coding: utf-8 -*-
"""连接准入与执行追踪，保持两入口一致且不改变内部离线调用。"""

from collections.abc import Callable
from collections.abc import Iterator
from contextlib import ExitStack
from contextlib import contextmanager
from typing import Literal
from uuid import UUID

from app.application.admission import QuestionAdmissionController
from app.application.admission import question_slot
from app.application.tracing.service import TraceService
from app.domain.admission import AdmissionDecision


@contextmanager
def admitted_question(
    admission: QuestionAdmissionController | None,
    traces: TraceService | None, *, question: str, execution_id: UUID,
    entry: Literal["query", "conversation"], user_id: UUID | None,
    on_queued: Callable[[int], None] | None = None,
) -> Iterator[None]:
    """只为真实线上准入建立早期追踪，许可覆盖实际执行而非发送字节。

    Args:
        admission: 两入口共享的准入组件。
        traces: 组合根同一追踪服务。
        question: 本次问题，不在准入元数据中复制正文。
        execution_id: 实际编排的执行标识。
        entry: 真实入口。
        user_id: 后端身份，None保留可信内部兼容。
        on_queued: 实际位置通知。

    Yields:
        入场后的同步用例区间。
    """
    if traces is None or admission is None or user_id is None:
        with question_slot(admission, user_id, on_queued=on_queued):
            yield
        return
    with traces.start_execution(
        question, execution_id=execution_id, entry=entry, user_id=user_id,
    ) as trace:
        # ExitStack使admission步骤结束后仍持有许可，执行完毕才释放。
        with ExitStack() as permits:
            with trace.step("admission") as capture:
                def queued(position: int) -> None:
                    trace.mark_queued()
                    if on_queued is not None:
                        on_queued(position)

                def decided(decision: AdmissionDecision) -> None:
                    capture.set_output(decision.model_dump(mode="json"))
                    trace.set_admission(decision)

                permits.enter_context(question_slot(
                    admission, user_id, on_queued=queued, on_decision=decided,
                ))
            with trace.step("execution"):
                yield

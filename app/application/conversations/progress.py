# -*- coding: utf-8 -*-
"""传播实际检索范围，兼容原有两个参数的阶段回调。"""

from collections.abc import Callable
from typing import Protocol
from typing import runtime_checkable

from app.application.conversations.events import StatusConversationData
from app.application.conversations.ports import StatusCallback


_LABELS = {
    "deciding": "正在理解问题",
    "understanding": "正在理解问题",
    "retrieving": "正在查找资料",
    "verifying": "正在核对",
    "validating": "正在核对",
    "generating": "正在整理回答",
    "expanding": "正在扩大查找范围",
    "escalating": "正在深入查找资料",
}


@runtime_checkable
class _ScopedCallback(Protocol):
    def report(
        self, phase: str, attempt: int | None, kb_ids: tuple[str, ...],
    ) -> None:
        """接收实际阶段范围。

        Args:
            phase: 阶段名。
            attempt: 可选累计次数。
            kb_ids: 实际执行范围。

        Returns:
            None。
        """


@runtime_checkable
class _QueueCallback(Protocol):
    def queued(self, position: int) -> None:
        """接收与模型推理无关的实际等待位置。

        Args:
            position: 前面的等待人数。

        Returns:
            None。
        """


def notify_queued(callback: StatusCallback | None, position: int) -> None:
    """向增强回调传递位置，兼容旧的两个参数回调。

    Args:
        callback: 当前请求的公开阶段回调。
        position: 前面等待者数量，首位为0。

    Returns:
        None。
    """
    if isinstance(callback, _QueueCallback):
        callback.queued(position)
    elif callback is not None:
        callback("queued", None)


def notify_status(
    callback: StatusCallback | None,
    phase: str,
    attempt: int | None = None,
    kb_ids: tuple[str, ...] = (),
) -> None:
    """对增强回调传递真实范围，不破坏旧的两参数函数调用方。

    Args:
        callback: 阶段回调或未启用状态。
        phase: 可公开的阶段名。
        attempt: 可选累计检索次数。
        kb_ids: 当前阶段实际使用的知识库集合。

    Returns:
        None。
    """
    if isinstance(callback, _ScopedCallback):
        callback.report(phase, attempt, kb_ids)
    elif callback is not None:
        callback(phase, attempt)


class ProgressReporter:
    """用固定安全文案映射公开阶段，不传播模型推理或目录描述。"""

    def __init__(
        self, publish: Callable[[StatusConversationData], None],
    ) -> None:
        """保存当前请求的类型化发布边界。

        Args:
            publish: 事件通道的发布函数。

        Returns:
            None。
        """
        self._publish = publish

    def __call__(self, phase: str, attempt: int | None = None) -> None:
        """兼容尚未提供范围信息的旧阶段回调。

        Args:
            phase: 阶段名。
            attempt: 可选累计检索次数。

        Returns:
            None。
        """
        self.report(phase, attempt, ())

    def report(
        self, phase: str, attempt: int | None, kb_ids: tuple[str, ...],
    ) -> None:
        """校验阶段并发布，不从用户输入猜测实际范围。

        Args:
            phase: 受限的公开阶段。
            attempt: 可选累计次数。
            kb_ids: 执行层给出的实际范围。

        Returns:
            None。
        """
        self._publish(StatusConversationData(
            phase=phase, attempt=attempt,
            kb_ids=kb_ids, label=_LABELS.get(phase),
        ))

    def queued(self, position: int) -> None:
        """发布等待人数，不伪造检索次数或尚未确定的知识范围。

        Args:
            position: 前面等待者数量，不包括已执行请求。

        Returns:
            None。
        """
        label = "正在等待执行名额"
        if position > 0:
            label = f"当前提问人数较多，前面还有 {position} 人"
        self._publish(StatusConversationData(
            phase="queued", position=position, label=label,
        ))

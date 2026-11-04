# -*- coding: utf-8 -*-
"""定义与具体数据库无关的追踪快照存储边界。"""

from typing import Protocol
from uuid import UUID

from app.domain.tracing import TraceSnapshot
from app.domain.tracing import CommitStatus
from app.domain.auditing import AdminAccessAudit


class TraceRepository(Protocol):
    """追踪权威存储，不由HTTP或模型创建数据库适配器。"""

    def record_admin_access(self, audit: AdminAccessAudit) -> None:
        """记录后台数据访问，不保存内容正文。

        Args:
            audit: 后端验证的访问元数据。

        Returns:
            None，失败必须阻止未审计的敏感数据访问。
        """

    def record_commit(
        self, trace_id: UUID, status: CommitStatus, turn_id: UUID | None,
    ) -> None:
        """记录业务提交结果，不覆盖已保存的模型调用和执行步骤。

        Args:
            trace_id: 当前追踪。
            status: 业务提交状态。
            turn_id: 仅提交成功时的正式轮次。

        Returns:
            None。
        """

    def save(self, snapshot: TraceSnapshot) -> None:
        """原子保存当前快照。

        Args:
            snapshot: 已过滤、已验证的快照。

        Returns:
            None。
        """

    def get(self, trace_id: UUID) -> TraceSnapshot:
        """返回未过期追踪。

        Args:
            trace_id: 精确追踪标识。

        Returns:
            完整不可变快照。
        """

    def prune_expired(self) -> int:
        """显式维护操作，只清理已过期的追踪。

        Returns:
            实际删除记录数。
        """

# -*- coding: utf-8 -*-
"""后台敏感内容访问必须先落审计，故障时不返回正文或执行分类。"""

from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from uuid import UUID

from app.application.ports.tracing import TraceRepository
from app.domain.auditing import AdminAccessAudit


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AdminAuditService:
    """只记录元数据，不把隐私内容复制到另一个永久日志。"""

    def __init__(
        self, repository: TraceRepository, retention_days: int = 30,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """保存审计存储和明确期限。

        Args:
            repository: 与当前运行隔离的追踪／审计存储。
            retention_days: 审计保留期限。
            clock: UTC时钟。

        Returns:
            None。
        """
        self._repository = repository
        self._retention_days = retention_days
        self._clock = clock

    def record(
        self, actor_id: UUID, action: str, kind: str,
        resource_ids: tuple[UUID, ...], *, knowledge_base_id: str | None = None,
    ) -> None:
        """写入访问或分类尝试，不吞掉存储错误。

        Args:
            actor_id: 已认证的后台操作者。
            action: 读取或分类尝试。
            kind: 资源类别。
            resource_ids: 本次访问的资源，不含正文。
            knowledge_base_id: 非UUID知识库资源的稳定标识。

        Returns:
            None，存储失败由HTTP映射为503。
        """
        now = self._clock()
        self._repository.record_admin_access(AdminAccessAudit(
            actor_id=actor_id, action=action, resource_kind=kind,
            resource_ids=resource_ids, created_at=now,
            knowledge_base_id=knowledge_base_id,
            expires_at=now + timedelta(days=self._retention_days),
        ))

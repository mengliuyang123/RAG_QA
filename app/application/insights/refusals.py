# -*- coding: utf-8 -*-
"""管理员读取拒答原问题时复用强制审计和筛选绑定游标。"""

import base64
import binascii
import hashlib
import json
from uuid import UUID

from app.application.ports.insights import RefusalReader
from app.application.tracing.auditing import AdminAuditService
from app.domain.insights import RefusalPage
from app.domain.insights import RefusalPosition


class RefusalInsightService:
    """只处理明确拒答事实，不从错误状态或相似文案猜测拒答。"""

    def __init__(
        self, reader: RefusalReader, audit: AdminAuditService,
    ) -> None:
        """保存读取与审计端口，不访问模型或普通会话服务。

        Args:
            reader: 有限单轮投影端口。
            audit: 返回原文之前必须持久化的访问审计。

        Returns:
            None。
        """
        self._reader = reader
        self._audit = audit

    def list(
        self, actor_id: UUID, *, limit: int = 20,
        kb_id: str | None = None, cursor: str | None = None,
    ) -> RefusalPage:
        """管理员授权后分页读取，审计失败时绝不返回正文。

        Args:
            actor_id: 已验证的管理员，不能来自查询参数。
            limit: 本页数量，1–100。
            kb_id: 实际检索范围中的知识库。
            cursor: 绑定筛选条件的上页边界。

        Returns:
            明确拒答列表及续页游标。
        """
        if not 1 <= limit <= 100:
            raise ValueError("拒答分页上限无效")
        fingerprint = hashlib.sha256(json.dumps(
            {"kb_id": kb_id}, sort_keys=True,
        ).encode()).hexdigest()
        before = None
        if cursor is not None:
            try:
                position = RefusalPosition.model_validate_json(
                    base64.b64decode(cursor, altchars=b"-_", validate=True),
                )
                if position.fingerprint != fingerprint:
                    raise ValueError("游标筛选条件不一致")
                before = (position.created_at, position.turn_id)
            except (ValueError, binascii.Error) as error:
                raise ValueError("拒答游标无效") from error
        items = self._reader.list(limit + 1, kb_id, before)
        page = items[:limit]
        self._audit.record(
            actor_id, "read", "refusals", tuple(item.turn_id for item in page),
        )
        next_cursor = None
        if len(items) > limit:
            position = RefusalPosition(
                created_at=page[-1].created_at, turn_id=page[-1].turn_id,
                fingerprint=fingerprint,
            )
            next_cursor = base64.urlsafe_b64encode(
                position.model_dump_json().encode(),
            ).decode()
        return RefusalPage(refusals=page, next_cursor=next_cursor)

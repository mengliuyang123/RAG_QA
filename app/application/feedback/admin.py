# -*- coding: utf-8 -*-
"""后台反馈只返回有审计的单轮投影，分类不改变用户原始反馈。"""

import base64
import binascii
import hashlib
import json
from uuid import UUID

from app.application.errors import ConversationStorageError
from app.application.errors import InsightStorageError
from app.application.ports.feedback_admin import FeedbackAdminRepository
from app.application.tracing.auditing import AdminAuditService
from app.domain.feedback import AdminFeedbackItem
from app.domain.feedback import AdminFeedbackPage
from app.domain.feedback import FeedbackPosition
from app.domain.feedback import FeedbackTriage
from app.domain.feedback import FeedbackTriageFilter


class FeedbackAdminService:
    """敏感内容读取先审计，处理尝试先留痕，再由业务事务核对版本。"""

    def __init__(self, repository: FeedbackAdminRepository, audit: AdminAuditService) -> None:
        """保存后台专用端口，不获取普通会话服务或绕过其归属。

        Args:
            repository: 只提供点踩投影和分类的存储。
            audit: 无正文审计。

        Returns:
            None。
        """
        self._repository = repository
        self._audit = audit

    def list(
        self, actor_id: UUID, *, limit: int = 20, kb_id: str | None = None,
        triage: FeedbackTriageFilter | None = None, cursor: str | None = None,
    ) -> AdminFeedbackPage:
        """过滤后分页返回已审计点踩记录。

        Args:
            actor_id: 已授权后台操作者。
            limit: 每页1–100条。
            kb_id: 知识库过滤。
            triage: 处理状态过滤。
            cursor: 绑定过滤条件的游标。

        Returns:
            单轮投影和下一页游标，不含用户身份。
        """
        if not 1 <= limit <= 100:
            raise ValueError("反馈分页上限无效")
        fingerprint = hashlib.sha256(json.dumps(
            {"kb_id": kb_id, "triage": triage}, sort_keys=True,
        ).encode()).hexdigest()
        before = None
        if cursor is not None:
            try:
                position = FeedbackPosition.model_validate_json(
                    base64.b64decode(cursor, altchars=b"-_", validate=True),
                )
                if position.fingerprint != fingerprint:
                    raise ValueError("游标条件不一致")
                before = (position.sort_time, position.feedback_id)
            except (ValueError, binascii.Error) as error:
                raise ValueError("反馈游标无效") from error
        try:
            items = self._repository.list(limit + 1, kb_id, triage, before)
        except ConversationStorageError as error:
            raise InsightStorageError("后台反馈读取失败") from error
        page = items[:limit]
        self._audit.record(actor_id, "read", "feedback", tuple(item.feedback_id for item in page))
        next_cursor = None
        if len(items) > limit:
            position = FeedbackPosition(
                sort_time=page[-1].sort_time, feedback_id=page[-1].feedback_id,
                fingerprint=fingerprint,
            )
            next_cursor = base64.urlsafe_b64encode(position.model_dump_json().encode()).decode()
        return AdminFeedbackPage(feedback=page, next_cursor=next_cursor)

    def triage(
        self, feedback_id: UUID, actor_id: UUID, value: FeedbackTriage | None,
        expected_revision: int | None = None,
    ) -> AdminFeedbackItem:
        """先记录分类尝试，版本不一致时不修改当前反馈。

        Args:
            feedback_id: 规范反馈资源标识。
            actor_id: 已验证后台操作者。
            value: 分类或清空。
            expected_revision: 可选客户端看到的用户反馈版本。

        Returns:
            保存后的分类状态。
        """
        self._audit.record(actor_id, "triage_attempt", "feedback", (feedback_id,))
        try:
            return self._repository.triage(feedback_id, actor_id, value, expected_revision)
        except ConversationStorageError as error:
            raise InsightStorageError("后台反馈分类失败") from error

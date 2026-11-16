# -*- coding: utf-8 -*-
"""知识库访问控制、文档时效与标准答案用例。

权限规则：
- 管理员可见、可管理全部知识库。
- 普通成员可见全部公开库，以及自己是成员的受限库；只能管理自己是 editor 的库。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from datetime import date
from datetime import datetime
from uuid import UUID
from uuid import uuid4

from app.application.errors import UserNotFoundError
from app.application.ports.ingestion_jobs import DocumentPublication
from app.application.ports.knowledge_access import DocumentReviewRepository
from app.application.ports.knowledge_access import FaqRepository
from app.application.ports.knowledge_access import KnowledgeAccessRepository
from app.domain.auth import User
from app.domain.knowledge_access import DocumentReview
from app.domain.knowledge_access import FaqEntry
from app.domain.knowledge_access import KbAccess
from app.domain.knowledge_access import KbMember
from app.domain.knowledge_access import KbPermissions


FAQ_FILENAME = "常见问题（标准答案）.md"
UNRESTRICTED = KbPermissions(unrestricted=True)
NOTHING = KbPermissions(deny_all=True)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class FaqNotFoundError(LookupError):
    """标准答案不存在。"""


class KnowledgeAccessService:
    """计算用户的知识库权限，并维护访问设置。"""

    def __init__(self, repository: KnowledgeAccessRepository) -> None:
        self._repository = repository

    def permissions(self, user: User) -> KbPermissions:
        """按已验证身份计算权限。"""
        if user.role == "admin":
            return UNRESTRICTED
        return self._member_permissions(user.id)

    def permissions_for_user_id(self, user_id: UUID | None) -> KbPermissions:
        """问答链路只拿到用户ID时使用。

        无用户（离线评估）不受限；已禁用的账号什么都看不到。没有账号记录的ID按不受限处理：
        HTTP 会话依赖账号记录（删除账号会级联删除会话，认证时也联表检查），
        所以这种ID只来自评估或测试注入的身份，不会来自真实请求。
        """
        if user_id is None:
            return UNRESTRICTED
        role = self._repository.user_role(user_id)
        if role == "admin" or role is None:
            return UNRESTRICTED
        if role == "disabled":
            return NOTHING
        return self._member_permissions(user_id)

    def _member_permissions(self, user_id: UUID) -> KbPermissions:
        memberships = self._repository.memberships(user_id)
        return KbPermissions(
            restricted=self._repository.restricted_kb_ids(),
            member_of=frozenset(memberships),
            editable=frozenset(kb for kb, role in memberships.items() if role == "editor"),
        )

    def get(self, kb_id: str) -> KbAccess:
        """读取访问设置。"""
        return self._repository.get(kb_id)

    def save(self, kb_id: str, *, restricted: bool, members: tuple[KbMember, ...]) -> KbAccess:
        """整体替换访问设置；同一用户只能出现一次。"""
        seen: set[UUID] = set()
        for member in members:
            if member.user_id in seen:
                raise ValueError("同一用户只能出现一次")
            seen.add(member.user_id)
        return self._repository.save(KbAccess(kb_id=kb_id, restricted=restricted, members=members))


class DocumentReviewService:
    """复审日期与已过时标记。"""

    def __init__(
        self, repository: DocumentReviewRepository, *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._repository = repository
        self._clock = clock

    def list(self, kb_id: str) -> list[DocumentReview]:
        """列出一个知识库的复审记录。"""
        return self._repository.list(kb_id)

    def save(
        self, actor_id: UUID, kb_id: str, doc_id: str, *,
        review_due_on: date | None, outdated: bool, note: str | None,
    ) -> DocumentReview:
        """保存复审信息；标记已过时后该文档立即停止参与检索。"""
        return self._repository.save(DocumentReview(
            doc_id=doc_id, kb_id=kb_id, review_due_on=review_due_on, outdated=outdated,
            note=(note or "").strip() or None, updated_at=self._clock(), updated_by=actor_id,
        ))


class OutdatedAwarePublication:
    """把已过时文档并入检索前的排除集合，复用检索链路已有的未发布文档排除机制。"""

    def __init__(self, publication: DocumentPublication, reviews: DocumentReviewRepository) -> None:
        self._publication = publication
        self._reviews = reviews

    def unpublished_document_ids(self) -> tuple[UUID, ...]:
        identities = list(self._publication.unpublished_document_ids())
        seen = set(identities)
        for doc_id in self._reviews.outdated_doc_ids():
            try:
                identity = UUID(doc_id)
            except ValueError:
                continue
            if identity not in seen:
                seen.add(identity)
                identities.append(identity)
        return tuple(identities)

    def __getattr__(self, name: str) -> object:
        return getattr(self._publication, name)


class FaqService:
    """标准答案的增删改；发布由调用方把 render() 的结果作为文档上传。"""

    def __init__(
        self, repository: FaqRepository, *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._repository = repository
        self._clock = clock

    def list(self, kb_id: str) -> list[FaqEntry]:
        """列出一个知识库的标准答案。"""
        return self._repository.list(kb_id)

    def save(
        self, actor_id: UUID, kb_id: str, *, entry_id: str | None,
        question: str, answer: str,
    ) -> FaqEntry:
        """新建或修改一条标准答案。"""
        now = self._clock()
        created_at = now
        if entry_id is not None:
            existing = next((item for item in self._repository.list(kb_id) if item.id == entry_id), None)
            if existing is None:
                raise FaqNotFoundError("标准答案不存在")
            created_at = existing.created_at
        return self._repository.save(FaqEntry(
            id=entry_id or uuid4().hex, kb_id=kb_id, question=question.strip(),
            answer=answer.strip(), created_at=created_at, updated_at=now, updated_by=actor_id,
        ))

    def delete(self, kb_id: str, entry_id: str) -> None:
        """删除一条标准答案。"""
        if not self._repository.delete(kb_id, entry_id):
            raise FaqNotFoundError("标准答案不存在")

    @staticmethod
    def render(entries: list[FaqEntry]) -> str:
        """渲染为一篇 Markdown 文档，每条一个小节，便于分块后整条被检索到。"""
        sections = ["# 常见问题（标准答案）", "", "以下是管理员确认的标准答案。", ""]
        for entry in entries:
            sections += [f"## {entry.question}", "", entry.answer, ""]
        return "\n".join(sections)


__all__ = [
    "FAQ_FILENAME", "DocumentReviewService", "FaqNotFoundError", "FaqService",
    "KnowledgeAccessService", "OutdatedAwarePublication", "UserNotFoundError",
]

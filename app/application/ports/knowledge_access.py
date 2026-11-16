# -*- coding: utf-8 -*-
"""知识库访问控制、文档复审与标准答案的存储端口。"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.domain.knowledge_access import DocumentReview
from app.domain.knowledge_access import FaqEntry
from app.domain.knowledge_access import KbAccess
from app.domain.knowledge_access import KbMemberRole


class KnowledgeAccessRepository(Protocol):
    """访问设置的事实来源。"""

    def user_role(self, user_id: UUID) -> str | None:
        """读取账号当前角色；禁用返回"disabled"，不存在返回None。"""

    def restricted_kb_ids(self) -> frozenset[str]:
        """全部受限知识库。"""

    def memberships(self, user_id: UUID) -> dict[str, KbMemberRole]:
        """用户所在的知识库及角色。"""

    def get(self, kb_id: str) -> KbAccess:
        """读取一个知识库的访问设置，未设置时为公开、无成员。"""

    def save(self, access: KbAccess) -> KbAccess:
        """整体替换一个知识库的访问设置。"""


class DocumentReviewRepository(Protocol):
    """文档复审信息。"""

    def list(self, kb_id: str) -> list[DocumentReview]:
        """列出一个知识库的复审记录。"""

    def save(self, review: DocumentReview) -> DocumentReview:
        """插入或替换。"""

    def outdated_doc_ids(self) -> tuple[str, ...]:
        """全部已过时文档。"""


class FaqRepository(Protocol):
    """标准答案。"""

    def list(self, kb_id: str) -> list[FaqEntry]:
        """按创建时间列出。"""

    def save(self, entry: FaqEntry) -> FaqEntry:
        """插入或替换。"""

    def delete(self, kb_id: str, entry_id: str) -> bool:
        """删除，返回是否存在。"""

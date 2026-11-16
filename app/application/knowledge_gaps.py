# -*- coding: utf-8 -*-
"""知识缺口工作台：只记录处理状态，复用现有洞察、上传与问答接口。"""

from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from uuid import UUID

from app.application.ports.knowledge_gaps import KnowledgeGapRepository
from app.domain.knowledge_gaps import GapCheck
from app.domain.knowledge_gaps import GapSource
from app.domain.knowledge_gaps import GapStatus
from app.domain.knowledge_gaps import KnowledgeGap


class GapVerificationError(ValueError):
    """标记为已验证时缺少全部回答成功的复测结果。"""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class KnowledgeGapService:
    """状态规则：verified 必须附带一次全部问题都已回答的复测。"""

    def __init__(
        self, repository: KnowledgeGapRepository, *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._repository = repository
        self._clock = clock

    def list(self, limit: int = 200) -> list[KnowledgeGap]:
        """列出已跟踪的缺口。

        Args:
            limit: 上限。

        Returns:
            按最近修改倒序的缺口。
        """
        return self._repository.list(limit)

    def save(
        self, actor_id: UUID, gap_key: str, *, source: GapSource, title: str,
        questions: tuple[str, ...], kb_id: str | None, status: GapStatus,
        doc_id: str | None, last_check: GapCheck | None,
    ) -> KnowledgeGap:
        """插入或更新一个缺口的处理状态。

        Args:
            actor_id: 当前管理员。
            gap_key: 来源稳定键。
            source: 来源类型。
            title: 代表问题。
            questions: 复测问题。
            kb_id: 目标知识库。
            status: 新状态。
            doc_id: 补充的文档。
            last_check: 最近复测。

        Returns:
            已保存的缺口。
        """
        if status == "verified" and (last_check is None or not last_check.all_answered):
            raise GapVerificationError("只有全部样例问题复测通过后才能标记为已验证")
        gap = KnowledgeGap(
            gap_key=gap_key, source=source, title=title.strip(), questions=questions,
            kb_id=kb_id, status=status, doc_id=doc_id, last_check=last_check,
            updated_at=self._clock(), updated_by=actor_id,
        )
        return self._repository.save(gap)

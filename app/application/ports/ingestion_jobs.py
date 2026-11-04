# -*- coding: utf-8 -*-
"""持久化入队、执行权及原子发布的应用端口。"""

from typing import Protocol
from uuid import UUID

from app.domain.ingestion_jobs import IngestionFailureCode
from app.domain.ingestion_jobs import IngestionJob
from app.domain.ingestion_jobs import IngestionJobClaim
from app.domain.ingestion_jobs import IngestionJobSubmission
from app.domain.ingestion_jobs import IngestionStage
from app.domain.ingestion_jobs import DocumentJobInfo


class DocumentPublication(Protocol):
    """只读发布边界，检索和浏览不拥有任务状态变更能力。"""

    def unpublished_document_ids(self) -> tuple[UUID, ...]:
        """返回召回前必须排除的身份，故障不得返回假空集。

        Returns:
            稳定的未发布文档身份。
        """


class DocumentJobReader(Protocol):
    """只读管理关联，不把任务提交、执行权或内部恢复数据交给列表用例。"""

    def list_knowledge_base_jobs(self, kb_id: str) -> tuple[DocumentJobInfo, ...]:
        """读取指定库的实际任务身份，包含没有索引的失败源文件。

        Args:
            kb_id: 当前受控知识库。

        Returns:
            经完整一致性校验的任务管理元数据。
        """

    def get_document_jobs(self, doc_ids: tuple[UUID, ...]) -> dict[UUID, DocumentJobInfo]:
        """同一快照分批关联文档身份，不逐文档查询任务。

        Args:
            doc_ids: 当前实际索引中的文档身份。

        Returns:
            有真实任务登记的文档元数据，旧文档不补造任务。
        """


class IngestionJobRepository(DocumentPublication, DocumentJobReader, Protocol):
    """任务不是线程池Future；进度与发布状态必须可跨重启恢复。"""

    def enqueue(
        self, submission: IngestionJobSubmission, *, capacity: int,
    ) -> IngestionJob:
        """在同一事务核对知识库、身份及非终态队列容量。

        Args:
            submission: 已持久保存源文件的服务端提交资料。
            capacity: queued和running合计上限。

        Returns:
            已持久入队的公开任务。
        """

    def get(self, job_id: str) -> IngestionJob:
        """按稳定标识查询公开任务。

        Args:
            job_id: 不透明任务标识。

        Returns:
            实际持久状态，不包含源路径和租约。
        """

    def claim_next(self, *, lease_seconds: int) -> IngestionJobClaim | None:
        """原子认领队首或过期任务，过期接管必须先补偿旧部分索引。

        Args:
            lease_seconds: 此次认领的有限有效期。

        Returns:
            带新栅栏的执行权，没有可认领任务时为None。
        """

    def advance(
        self, claim: IngestionJobClaim, *, stage: IngestionStage,
        completed_batches: int = 0, total_batches: int = 0,
    ) -> IngestionJob:
        """只接受有效租约的实际阶段及批次进度。

        Args:
            claim: 当前执行权。
            stage: 实际阶段，不允许退回。
            completed_batches: 已完成的向量化批次数。
            total_batches: 本次执行真实计算的总批次数。

        Returns:
            已提交的新进度。
        """

    def renew(self, claim: IngestionJobClaim, *, lease_seconds: int) -> None:
        """仅延长有效执行权，不改变完成进度。

        Args:
            claim: 当前执行权。
            lease_seconds: 从当前时刻延长的秒数。

        Returns:
            None。
        """

    def succeed(
        self, claim: IngestionJobClaim, *, chunks_count: int,
    ) -> IngestionJob:
        """完整向量与BM25发布之后，原子提交成功和文档可见性。

        Args:
            claim: 当前执行权。
            chunks_count: 实际完成的正文本块数。

        Returns:
            成功任务；调用方不能提前提交此状态。
        """

    def fail(
        self, claim: IngestionJobClaim, *, code: IngestionFailureCode,
    ) -> IngestionJob:
        """持久记录分类失败，失败文档仍未发布。

        Args:
            claim: 当前执行权。
            code: 安全分类码，不接受原始异常信息。

        Returns:
            释放容量但不发布内容的失败任务。
        """

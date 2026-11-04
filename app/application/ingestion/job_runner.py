# -*- coding: utf-8 -*-
"""执行单个持久任务，心跳、恢复补偿与发布共用同一执行权。"""

import logging
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import AbstractContextManager
from contextlib import contextmanager
from threading import Event
from threading import RLock
from threading import Thread

from app.application.errors import DocumentParsingError
from app.application.errors import EmbeddingError
from app.application.errors import EmptyDocumentError
from app.application.errors import IngestionJobConflictError
from app.application.errors import IngestionJobStorageError
from app.application.errors import IngestionSourceError
from app.application.ports.ingestion import DocumentDeletion
from app.application.ports.ingestion import IngestionExecutionGuard
from app.application.ports.ingestion import ProgressiveDocumentIngestion
from app.application.ports.ingestion_jobs import IngestionJobRepository
from app.application.ports.ingestion_sources import IngestionSources
from app.domain.ingestion_jobs import IngestionFailureCode
from app.domain.ingestion_jobs import IngestionJobClaim
from app.domain.ingestion_jobs import IngestionProgress
from app.domain.ingestion_jobs import IngestionStage


logger = logging.getLogger(__name__)


class _Interrupted(RuntimeError):
    """服务停止时保留未完成任务供下一次启动接管。"""


class IngestionJobRunner:
    """有界执行一个任务；不创建无限Future队列或偷偷重试终态失败。"""

    def __init__(
        self, jobs: IngestionJobRepository, sources: IngestionSources,
        pipeline_loader: Callable[[], ProgressiveDocumentIngestion],
        deletion_loader: Callable[[], DocumentDeletion],
        execution_guard: IngestionExecutionGuard, *,
        lease_seconds: int = 60, heartbeat_seconds: float = 10,
        mutation_lock: AbstractContextManager[object] | None = None,
    ) -> None:
        """注入持久化与惰性索引能力，不在构造时打开模型连接。

        Args:
            jobs: 带写入栅栏的持久队列。
            sources: 恢复前验证完整性的源文件端口。
            pipeline_loader: 只有取得任务且来源有效时才装配的流水线。
            deletion_loader: 旧部分索引的补偿删除能力。
            execution_guard: 同一物理索引的跨进程排他上下文。
            lease_seconds: 每次认领的有效秒数。
            heartbeat_seconds: 小于租约的心跳间隔。
            mutation_lock: 与本进程旧上传／删除共用的变更锁。

        Returns:
            None。
        """
        if lease_seconds <= 0 or not 0 < heartbeat_seconds < lease_seconds:
            raise ValueError("心跳间隔必须为正且小于租约")
        self._jobs = jobs
        self._sources = sources
        self._pipeline_loader = pipeline_loader
        self._deletion_loader = deletion_loader
        self._execution_guard = execution_guard
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._mutation_lock = mutation_lock or RLock()

    def run_next(self, stop: Event) -> bool:
        """在完整排他区间内处理至多一个任务，普通失败只记录一次。

        Args:
            stop: 生命周期发出的协作停止信号。

        Returns:
            是否认领过任务；False表示队列为空或开始前已停止。
        """
        if stop.is_set():
            return False
        with self._execution_guard.hold():
            claim = self._jobs.claim_next(lease_seconds=self._lease_seconds)
            if claim is None:
                return False
            stage: IngestionStage = "parsing"
            try:
                with self._keep_lease(claim, stop) as check:
                    source = self._sources.verified_snapshot(claim.submission)
                    with source as path:
                        check()
                        with self._mutation_lock:
                            if claim.recovered:
                                self._deletion_loader().delete_document(
                                    str(claim.job.doc_id),
                                )
                            check()

                            def progress(value: IngestionProgress) -> None:
                                nonlocal stage
                                check()
                                self._jobs.advance(claim, **value.model_dump())
                                stage = value.stage

                            count = self._pipeline_loader().ingest_job(
                                path, claim.submission, progress,
                                check_active=check,
                            )
                            check()
                            self._jobs.succeed(claim, chunks_count=count)
            except _Interrupted:
                # 不伪报失败／成功，也不在关闭过程中重新开始下一批。
                logger.info("入库任务等待恢复", extra={"job_id": claim.job.job_id})
            except Exception as error:
                self._record_failure(claim, self._failure_code(error, stage))
            return True

    @contextmanager
    def _keep_lease(
        self, claim: IngestionJobClaim, stop: Event,
    ) -> Iterator[Callable[[], None]]:
        finished = Event()
        failed = Event()

        def heartbeat() -> None:
            while not finished.wait(self._heartbeat_seconds):
                try:
                    self._jobs.renew(claim, lease_seconds=self._lease_seconds)
                except Exception:
                    failed.set()
                    return

        def check() -> None:
            if stop.is_set():
                raise _Interrupted("入库执行已停止")
            if failed.is_set():
                raise IngestionJobConflictError("任务执行权无法继续维持")

        thread = Thread(target=heartbeat, name="rag-ingestion-heartbeat")
        thread.start()
        try:
            yield check
        finally:
            finished.set()
            thread.join()

    def _record_failure(
        self, claim: IngestionJobClaim, code: IngestionFailureCode,
    ) -> None:
        try:
            self._jobs.fail(claim, code=code)
        except (IngestionJobConflictError, IngestionJobStorageError):
            # 不能在失去栅栏或提交结果未知时覆盖真实成功或新worker状态。
            logger.warning("入库终态登记未确认", extra={"job_id": claim.job.job_id})

    @staticmethod
    def _failure_code(
        error: Exception, stage: IngestionStage,
    ) -> IngestionFailureCode:
        if isinstance(error, IngestionSourceError):
            return "source_unavailable"
        if isinstance(error, EmptyDocumentError):
            return "empty_document"
        if isinstance(error, EmbeddingError):
            return "embedding_failed"
        if isinstance(error, DocumentParsingError):
            return "parse_failed"
        if stage in {"parsing", "chunking"}:
            return "parse_failed"
        if stage == "embedding":
            return "embedding_failed"
        return "indexing_failed"

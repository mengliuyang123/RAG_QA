# -*- coding: utf-8 -*-
"""SQLite有界任务队列；租约栅栏与最终发布状态在同一事务提交。"""

import sqlite3
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import closing
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from uuid import UUID
from uuid import uuid4

from app.application.errors import IngestionJobConflictError
from app.application.errors import IngestionJobNotFoundError
from app.application.errors import IngestionJobStorageError
from app.application.errors import IngestionQueueFullError
from app.application.errors import KnowledgeBaseNotFoundError
from app.domain.ingestion_jobs import FAILURE_MESSAGES
from app.domain.ingestion_jobs import IngestionFailureCode
from app.domain.ingestion_jobs import IngestionJob
from app.domain.ingestion_jobs import IngestionJobClaim
from app.domain.ingestion_jobs import IngestionJobFailure
from app.domain.ingestion_jobs import IngestionJobSubmission
from app.domain.ingestion_jobs import IngestionStage
from app.domain.ingestion_jobs import DocumentJobInfo


STAGES = ("parsing", "chunking", "embedding", "indexing")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _positive_integer(value: int, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label}必须为正整数")


def has_pending_ingestion_jobs(path: Path) -> bool:
    """只读探测既有任务库；空闲健康检查不能隐式创建表或调用模型。

    Args:
        path: 组合根派生的本运行时目录数据库。

    Returns:
        是否有queued或running任务；缺库／尚无任务表时为False。
    """
    if not path.exists():
        return False
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='ingestion_jobs'",
            ).fetchone()
            if table is None:
                return False
            pending = connection.execute(
                "SELECT 1 FROM ingestion_jobs "
                "WHERE status IN ('queued', 'running') LIMIT 1",
            ).fetchone()
            return pending is not None
    except (OSError, sqlite3.Error) as error:
        raise IngestionJobStorageError("入库队列探测暂时不可用") from error


class SQLiteIngestionJobRepository:
    """不持有长连接，BEGIN IMMEDIATE序列化容量、认领与状态转换。"""

    def __init__(
        self, path: Path, *, clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """创建任务表，不修改目录、源文件或模型索引。

        Args:
            path: 已有知识库目录数据库的隔离路径。
            clock: 用于持久UTC时间与租约校验的服务器时钟。

        Returns:
            None。
        """
        self._path = path
        self._clock = clock
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection() as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_jobs (
                        job_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL UNIQUE,
                        kb_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
                        uploaded_by TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN (
                            'queued', 'running', 'succeeded', 'failed'
                        )),
                        stage TEXT NOT NULL CHECK(stage IN (
                            'parsing', 'chunking', 'embedding', 'indexing'
                        )),
                        progress REAL NOT NULL CHECK(
                            progress >= 0 AND progress <= 1
                        ),
                        completed_batches INTEGER NOT NULL DEFAULT 0,
                        total_batches INTEGER NOT NULL DEFAULT 0,
                        chunks_count INTEGER,
                        error_code TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 0,
                        lease_token TEXT,
                        lease_expires_at TEXT,
                        FOREIGN KEY(kb_id) REFERENCES knowledge_bases(id)
                            ON DELETE RESTRICT
                    )
                """)
                connection.execute("""
                    CREATE INDEX IF NOT EXISTS ingestion_jobs_pending
                    ON ingestion_jobs(status, lease_expires_at, created_at)
                """)
        except OSError as error:
            raise IngestionJobStorageError("任务登记初始化失败") from error

    def list_knowledge_base_jobs(self, kb_id: str) -> tuple[DocumentJobInfo, ...]:
        """读取同库全部真实任务，失败源也不能在强制删除时遗失。

        Args:
            kb_id: 已由用例确定的知识库。

        Returns:
            不含租约和摘要的完整管理身份。
        """
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE kb_id = ? ORDER BY doc_id",
                (kb_id,),
            ).fetchall()
        result = []
        for row in rows:
            job = self._job(row)
            submission = self._submission(row)
            result.append(DocumentJobInfo(
                doc_id=job.doc_id, kb_id=job.kb_id, filename=job.filename,
                uploaded_by=submission.uploaded_by, status=job.status,
            ))
        return tuple(result)

    def get_document_jobs(self, doc_ids: tuple[UUID, ...]) -> dict[UUID, DocumentJobInfo]:
        """单一只读快照按100个身份关联任务，复用完整状态与提交校验。

        Args:
            doc_ids: 已存在索引的文档身份。

        Returns:
            不含恢复凭据和SHA的管理元数据。
        """
        if not doc_ids:
            return {}
        identifiers = tuple(dict.fromkeys(doc_ids))
        result = {}
        connection = None
        try:
            connection = sqlite3.connect(
                self._path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN")
            for offset in range(0, len(identifiers), 100):
                batch = identifiers[offset:offset + 100]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT * FROM ingestion_jobs WHERE doc_id IN (" + placeholders + ")",
                    tuple(str(identifier) for identifier in batch),
                ).fetchall()
                for row in rows:
                    job = self._job(row)
                    submission = self._submission(row)
                    result[job.doc_id] = DocumentJobInfo(
                        doc_id=job.doc_id, kb_id=job.kb_id, filename=job.filename,
                        uploaded_by=submission.uploaded_by, status=job.status,
                    )
            return result
        except (sqlite3.Error, OSError, ValueError) as error:
            raise IngestionJobStorageError("文档任务元数据暂时不可用") from error
        finally:
            if connection is not None:
                connection.close()

    def enqueue(
        self, submission: IngestionJobSubmission, *, capacity: int,
    ) -> IngestionJob:
        """事务内核对目录、重复身份和非终态容量，然后持久入队。

        Args:
            submission: 服务端已保存文件的身份和完整性资料。
            capacity: 排队和运行任务合计上限。

        Returns:
            持久入队的公开任务。
        """
        _positive_integer(capacity, "队列容量")
        submission = IngestionJobSubmission.model_validate(
            submission.model_dump(),
        )
        now = self._now().isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM knowledge_bases WHERE id = ? AND lifecycle = 'active'",
                (submission.kb_id,),
            ).fetchone()
            if exists is None:
                raise KnowledgeBaseNotFoundError("知识库不存在")
            duplicate = connection.execute(
                "SELECT 1 FROM ingestion_jobs WHERE job_id = ? OR doc_id = ?",
                (submission.job_id, str(submission.doc_id)),
            ).fetchone()
            if duplicate is not None:
                raise IngestionJobConflictError("入库任务身份已经存在")
            active_count = connection.execute(
                "SELECT COUNT(*) FROM ingestion_jobs "
                "WHERE status IN ('queued', 'running')",
            ).fetchone()[0]
            if active_count >= capacity:
                raise IngestionQueueFullError("入库队列已满，请稍后重试")
            connection.execute("""
                INSERT INTO ingestion_jobs (
                    job_id, doc_id, kb_id, filename, size_bytes, uploaded_by,
                    content_sha256, status, stage, progress,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 'parsing', 0, ?, ?)
            """, (
                submission.job_id, str(submission.doc_id), submission.kb_id,
                submission.filename, submission.size_bytes,
                str(submission.uploaded_by), submission.content_sha256,
                now, now,
            ))
            return self._job(self._row(connection, submission.job_id))

    def get(self, job_id: str) -> IngestionJob:
        """读取实际进度，不把不存在或损坏记录转换成虚假成功。

        Args:
            job_id: 稳定任务标识。

        Returns:
            不包含内部资料的当前任务。
        """
        with self._connection() as connection:
            return self._job(self._row(connection, job_id))

    def claim_next(self, *, lease_seconds: int) -> IngestionJobClaim | None:
        """按入队顺序认领，接管过期任务时更换栅栏并重置本次进度。

        Args:
            lease_seconds: 新执行权有效秒数。

        Returns:
            新执行权；没有排队或过期任务则为None。
        """
        _positive_integer(lease_seconds, "租约秒数")
        now = self._now()
        expires = now + timedelta(seconds=lease_seconds)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""
                SELECT * FROM ingestion_jobs
                WHERE status = 'queued' OR (
                    status = 'running' AND lease_expires_at <= ?
                ) ORDER BY created_at, job_id LIMIT 1
            """, (now.isoformat(),)).fetchone()
            if row is None:
                return None
            self._job(row)
            submission = self._submission(row)
            token = uuid4()
            attempt = row["attempt"] + 1
            recovered = row["status"] == "running"
            connection.execute("""
                UPDATE ingestion_jobs SET status='running', stage='parsing',
                    progress=0, completed_batches=0, total_batches=0,
                    chunks_count=NULL, error_code=NULL, updated_at=?,
                    attempt=?, lease_token=?, lease_expires_at=?
                WHERE job_id=?
            """, (
                now.isoformat(), attempt, str(token), expires.isoformat(),
                submission.job_id,
            ))
            return IngestionJobClaim(
                job=self._job(self._row(connection, submission.job_id)),
                submission=submission, lease_token=token, attempt=attempt,
                recovered=recovered,
            )

    def advance(
        self, claim: IngestionJobClaim, *, stage: IngestionStage,
        completed_batches: int = 0, total_batches: int = 0,
    ) -> IngestionJob:
        """仅接受实际向前的阶段和稳定总批次，不根据耗时猜进度。

        Args:
            claim: 此次执行权。
            stage: 当前实际阶段。
            completed_batches: 已完整提交的向量批次数。
            total_batches: 实际准备完成的总批次数。

        Returns:
            已持久化的单调进度。
        """
        self._validate_batches(stage, completed_batches, total_batches)
        now = self._now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._active(connection, claim, now)
            if STAGES.index(stage) < STAGES.index(row["stage"]):
                raise IngestionJobConflictError("入库阶段不能回退")
            if stage == "embedding" and row["stage"] == "embedding":
                if total_batches != row["total_batches"]:
                    raise IngestionJobConflictError("总批次数不能中途改变")
                if completed_batches < row["completed_batches"]:
                    raise IngestionJobConflictError("已完成批次数不能回退")
            if stage == "indexing":
                if row["stage"] not in {"embedding", "indexing"}:
                    raise IngestionJobConflictError("向量化尚未完成")
                if row["completed_batches"] != row["total_batches"]:
                    raise IngestionJobConflictError("向量批次尚未全部完成")
                completed_batches = row["completed_batches"]
                total_batches = row["total_batches"]
            progress = self._progress(stage, completed_batches, total_batches)
            connection.execute("""
                UPDATE ingestion_jobs SET stage=?, progress=?,
                    completed_batches=?, total_batches=?, updated_at=?
                WHERE job_id=?
            """, (
                stage, progress, completed_batches, total_batches,
                now.isoformat(), claim.job.job_id,
            ))
            return self._job(self._row(connection, claim.job.job_id))

    def renew(self, claim: IngestionJobClaim, *, lease_seconds: int) -> None:
        """只延长仍有效的执行权，不伪造阶段完成或状态更新时间。

        Args:
            claim: 此次执行权。
            lease_seconds: 从当前时刻延长的有效秒数。

        Returns:
            None。
        """
        _positive_integer(lease_seconds, "租约秒数")
        now = self._now()
        expires = now + timedelta(seconds=lease_seconds)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._active(connection, claim, now)
            current_expiry = datetime.fromisoformat(row["lease_expires_at"])
            expires = max(expires, current_expiry)
            connection.execute(
                "UPDATE ingestion_jobs SET lease_expires_at=? WHERE job_id=?",
                (expires.isoformat(), claim.job.job_id),
            )

    def succeed(
        self, claim: IngestionJobClaim, *, chunks_count: int,
    ) -> IngestionJob:
        """完整索引发布后提交唯一成功标记，先前所有块才允许可见。

        Args:
            claim: 此次执行权。
            chunks_count: 实际完成的文本块数，必须大于零。

        Returns:
            持久成功任务。
        """
        _positive_integer(chunks_count, "块数")
        now = self._now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._active(connection, claim, now)
            if row["stage"] != "indexing" or row["total_batches"] <= 0:
                raise IngestionJobConflictError("任务尚未进入完整索引发布阶段")
            if row["completed_batches"] != row["total_batches"]:
                raise IngestionJobConflictError("任务批次尚未完成")
            connection.execute("""
                UPDATE ingestion_jobs SET status='succeeded', progress=1,
                    chunks_count=?, updated_at=?, lease_token=NULL,
                    lease_expires_at=NULL WHERE job_id=?
            """, (chunks_count, now.isoformat(), claim.job.job_id))
            return self._job(self._row(connection, claim.job.job_id))

    def fail(
        self, claim: IngestionJobClaim, *, code: IngestionFailureCode,
    ) -> IngestionJob:
        """释放队列容量，但保留未发布事实以屏蔽可能的补偿残留。

        Args:
            claim: 此次执行权。
            code: 预定义安全分类，不接受异常文本。

        Returns:
            持久失败任务。
        """
        if code not in FAILURE_MESSAGES:
            raise ValueError("入库失败分类无效")
        now = self._now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._active(connection, claim, now)
            connection.execute("""
                UPDATE ingestion_jobs SET status='failed', error_code=?,
                    updated_at=?, lease_token=NULL, lease_expires_at=NULL
                WHERE job_id=?
            """, (code, now.isoformat(), claim.job.job_id))
            return self._job(self._row(connection, claim.job.job_id))

    def unpublished_document_ids(self) -> tuple[UUID, ...]:
        """读取所有未发布身份；故障或损坏数据均抛错而不是放行。

        Returns:
            按UUID稳定排序的文档身份。
        """
        with self._connection() as connection:
            # 不能先按status排除全部成功行：损坏的成功标记也必须被校验，
            # 否则“未发布列表为空”会错误地放行未完成内容。
            rows = connection.execute("""
                SELECT * FROM ingestion_jobs WHERE status != 'succeeded'
                    OR stage != 'indexing' OR progress != 1
                    OR chunks_count IS NULL OR chunks_count <= 0
                    OR total_batches <= 0 OR completed_batches != total_batches
                    OR error_code IS NOT NULL OR lease_token IS NOT NULL
                    OR lease_expires_at IS NOT NULL
                ORDER BY doc_id
            """).fetchall()
        identifiers = []
        for row in rows:
            job = self._job(row)
            identifiers.append(job.doc_id)
        return tuple(identifiers)

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("任务时钟必须包含时区")
        return now.astimezone(UTC)

    @staticmethod
    def _validate_batches(
        stage: IngestionStage, completed: int, total: int,
    ) -> None:
        if stage not in STAGES:
            raise ValueError("入库阶段无效")
        if type(completed) is not int or type(total) is not int:
            raise ValueError("批次数必须为整数")
        if completed < 0 or total < 0 or completed > total:
            raise ValueError("入库批次数无效")
        if stage == "embedding":
            _positive_integer(total, "总批次数")
        elif completed != 0 or total != 0:
            raise ValueError("只有向量化阶段接受批次进度")

    @staticmethod
    def _progress(stage: IngestionStage, completed: int, total: int) -> float:
        if stage == "parsing":
            return 0.0
        if stage == "chunking":
            return 0.05
        if stage == "embedding":
            return 0.1 + 0.8 * completed / total
        return 0.95

    @staticmethod
    def _row(connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM ingestion_jobs WHERE job_id=?", (job_id,),
        ).fetchone()
        if row is None:
            raise IngestionJobNotFoundError("入库任务不存在")
        return row

    def _active(
        self, connection: sqlite3.Connection, claim: IngestionJobClaim,
        now: datetime,
    ) -> sqlite3.Row:
        row = self._row(connection, claim.job.job_id)
        self._job(row)
        if row["status"] != "running":
            raise IngestionJobConflictError("任务已不处于执行状态")
        if row["lease_token"] != str(claim.lease_token):
            raise IngestionJobConflictError("任务执行权已被接管")
        if row["attempt"] != claim.attempt:
            raise IngestionJobConflictError("任务执行次数不匹配")
        if row["doc_id"] != str(claim.job.doc_id):
            raise IngestionJobConflictError("任务文档身份不匹配")
        if row["lease_expires_at"] is None:
            raise IngestionJobStorageError("执行任务缺少租约")
        if row["lease_expires_at"] <= now.isoformat():
            raise IngestionJobConflictError("任务执行权已经过期")
        return row

    @staticmethod
    def _submission(row: sqlite3.Row) -> IngestionJobSubmission:
        try:
            return IngestionJobSubmission(
                job_id=row["job_id"], doc_id=UUID(row["doc_id"]),
                kb_id=row["kb_id"], filename=row["filename"],
                size_bytes=row["size_bytes"],
                uploaded_by=UUID(row["uploaded_by"]),
                content_sha256=row["content_sha256"],
            )
        except (ValueError, TypeError) as error:
            raise IngestionJobStorageError("任务提交资料无效") from error

    @staticmethod
    def _job(row: sqlite3.Row) -> IngestionJob:
        try:
            completed = row["completed_batches"]
            total = row["total_batches"]
            if completed < 0 or total < completed:
                raise ValueError("持久化批次数无效")
            if row["stage"] in {"embedding", "indexing"} and total <= 0:
                raise ValueError("当前阶段缺少总批次数")
            if row["stage"] == "indexing" and completed != total:
                raise ValueError("索引阶段仍有未完成批次")
            if row["status"] == "running":
                UUID(row["lease_token"])
                expiry = datetime.fromisoformat(row["lease_expires_at"])
                if expiry.tzinfo is None or expiry.utcoffset() is None:
                    raise ValueError("持久化租约缺少时区")
                if row["attempt"] < 1:
                    raise ValueError("执行任务缺少认领次数")
            elif (
                row["lease_token"] is not None
                or row["lease_expires_at"] is not None
            ):
                raise ValueError("非执行任务不应持有租约")
            failure = None
            code = row["error_code"]
            if code is not None:
                failure = IngestionJobFailure(
                    code=code, message=FAILURE_MESSAGES[code],
                )
            return IngestionJob(
                job_id=row["job_id"], doc_id=UUID(row["doc_id"]),
                kb_id=row["kb_id"], filename=row["filename"],
                status=row["status"], stage=row["stage"],
                progress=row["progress"], chunks_count=row["chunks_count"],
                error=failure,
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
        except (ValueError, TypeError, KeyError) as error:
            raise IngestionJobStorageError("任务进度记录无效") from error

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            database = sqlite3.connect(self._path, timeout=5.0)
            with closing(database) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                with connection:
                    yield connection
        except sqlite3.Error as error:
            raise IngestionJobStorageError("入库任务登记暂时不可用") from error

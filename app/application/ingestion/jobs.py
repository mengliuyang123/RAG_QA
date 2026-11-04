# -*- coding: utf-8 -*-
"""先持久源文件再事务入队，前台不执行解析或向量化。"""

from hashlib import sha256
from pathlib import PurePath
from pathlib import PureWindowsPath
from uuid import UUID
from uuid import uuid4

from app.application.errors import EmptyDocumentError
from app.application.errors import FileTooLargeError
from app.application.errors import IngestionJobConflictError
from app.application.errors import IngestionQueueFullError
from app.application.errors import KnowledgeBaseNotFoundError
from app.application.errors import MissingFilenameError
from app.application.ports.ingestion_jobs import IngestionJobRepository
from app.application.ports.ingestion_sources import IngestionSources
from app.application.ports.knowledge_bases import KnowledgeBaseRepository
from app.application.ports.parsing import ParserResolver
from app.domain.ingestion_jobs import IngestionJob
from app.domain.ingestion_jobs import IngestionJobSubmission


class IngestionJobService:
    """入队与任务读取不装配Embedding或LLM，旧upload契约不变。"""

    def __init__(
        self, catalog: KnowledgeBaseRepository, jobs: IngestionJobRepository,
        sources: IngestionSources, parsers: ParserResolver, *,
        max_upload_size: int, capacity: int,
    ) -> None:
        """保存目录、持久队列和受控文件端口。

        Args:
            catalog: 权威知识库目录。
            jobs: 事务容量约束的任务仓储。
            sources: 可恢复源文件端口。
            parsers: 只用于检查支持类型，不读取正文。
            max_upload_size: 单文件实际字节上限。
            capacity: queued和running合计上限。

        Returns:
            None。
        """
        if max_upload_size <= 0 or capacity <= 0:
            raise ValueError("入库容量和文件上限必须为正数")
        self.max_upload_size = max_upload_size
        self._catalog = catalog
        self._jobs = jobs
        self._sources = sources
        self._parsers = parsers
        self._capacity = capacity

    def submit(
        self, kb_id: str, filename: str | None, content: bytes,
        uploaded_by: UUID,
    ) -> IngestionJob:
        """验证元数据和类型，持久文件后提交queued，不调用模型。

        Args:
            kb_id: 请求路径中的目标知识库。
            filename: 客户端文件名，可能带浏览器路径。
            content: 已受HTTP读取上限保护的字节。
            uploaded_by: 服务端认证身份，不能来自上传表单。

        Returns:
            已持久入队的公开任务。
        """
        self._catalog.get(kb_id)
        if not filename or not filename.strip():
            raise MissingFilenameError("上传文件缺少文件名")
        safe_name = PurePath(PureWindowsPath(filename).name).name
        if not content:
            raise EmptyDocumentError("上传文件内容为空")
        if len(content) > self.max_upload_size:
            raise FileTooLargeError("上传文件超过大小限制")
        self._parsers.resolve(safe_name)
        try:
            submission = IngestionJobSubmission(
                job_id=f"job_{uuid4().hex}", doc_id=uuid4(), kb_id=kb_id,
                filename=safe_name, size_bytes=len(content),
                uploaded_by=uploaded_by,
                content_sha256=sha256(content).hexdigest(),
            )
            if len(f"{submission.doc_id}_{safe_name}".encode("utf-8")) > 255:
                raise ValueError("源文件名超出受控存储上限")
        except ValueError as error:
            raise MissingFilenameError("上传文件名无效或过长") from error
        self._sources.persist(submission, content)
        try:
            return self._jobs.enqueue(submission, capacity=self._capacity)
        except (
            IngestionQueueFullError, IngestionJobConflictError,
            KnowledgeBaseNotFoundError,
        ):
            # 只有已确认回滚的拒绝才能删源文件；存储异常可能意味着提交响应丢失。
            self._sources.discard(submission)
            raise

    def get(self, job_id: str) -> IngestionJob:
        """仅读持久任务进度，不加载模型或正文。

        Args:
            job_id: 请求中的不透明标识。

        Returns:
            当前公开任务。
        """
        return self._jobs.get(job_id)

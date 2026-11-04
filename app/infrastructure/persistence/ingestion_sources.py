# -*- coding: utf-8 -*-
"""排他持久源文件与校验后私有快照，不读取请求指定的任意路径。"""

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from app.application.errors import IngestionSourceError
from app.domain.ingestion_jobs import IngestionJobSubmission


class LocalIngestionSources:
    """源文件沿用旧文档管理命名方式，恢复只读取受控目录中的身份。"""

    def __init__(self, docs_path: Path) -> None:
        """保存组合根指定的目录，构造时不创建文件或目录。

        Args:
            docs_path: 本运行时上传文件受控目录。

        Returns:
            None。
        """
        self._docs_path = docs_path

    def persist(
        self, submission: IngestionJobSubmission, content: bytes,
    ) -> None:
        """排他创建，文件及目录同步成功后才允许入队。

        Args:
            submission: 已校验的服务端身份。
            content: 上传字节，必须与登记摘要和大小一致。

        Returns:
            None。
        """
        path = self._path(submission)
        if len(content) != submission.size_bytes:
            raise IngestionSourceError("源文件大小不一致")
        if sha256(content).hexdigest() != submission.content_sha256:
            raise IngestionSourceError("源文件摘要不一致")
        created = False
        try:
            self._docs_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            created = True
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(content)
                destination.flush()
                os.fsync(destination.fileno())
            self._sync_directory()
        except OSError as error:
            if created:
                path.unlink(missing_ok=True)
            raise IngestionSourceError("源文件持久保存失败") from error

    def discard(self, submission: IngestionJobSubmission) -> None:
        """只删除这次提交持有的精确文件，不递归、不匹配通配符。

        Args:
            submission: 确定未被队列接受的身份。

        Returns:
            None。
        """
        try:
            self._path(submission).unlink(missing_ok=True)
            self._sync_directory()
        except OSError as error:
            raise IngestionSourceError("未入队源文件清理失败") from error

    @contextmanager
    def verified_snapshot(
        self, submission: IngestionJobSubmission,
    ) -> Iterator[Path]:
        """拒绝链接，流式复制并核对实际字节，校验后才交给解析器。

        Args:
            submission: 任务表保存的完整性资料。

        Yields:
            已验证的独立解析快照路径。
        """
        with ExitStack() as resources:
            try:
                directory = resources.enter_context(TemporaryDirectory(
                    prefix="rag-source-", dir=self._docs_path,
                ))
                snapshot = Path(directory) / submission.filename
                self._verified_copy(submission, snapshot)
            except OSError as error:
                raise IngestionSourceError("任务源文件暂时不可用") from error
            # yield不在源读取的except内，不能把解析器抛出的OSError误报成来源损坏。
            yield snapshot

    def _verified_copy(
        self, submission: IngestionJobSubmission, snapshot: Path,
    ) -> None:
        path = self._path(submission)
        if path.is_symlink():
            raise IngestionSourceError("任务源文件不能是符号链接")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        # 先以非阻塞方式打开，再核对fstat；否则FIFO会在校验类型前永久阻塞。
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise IngestionSourceError("任务源文件类型无效")
            if info.st_size != submission.size_bytes:
                raise IngestionSourceError("任务源文件大小已变化")
            digest = sha256()
            size = 0
            with snapshot.open("xb") as destination:
                while True:
                    part = source.read(65536)
                    if not part:
                        break
                    size += len(part)
                    if size > submission.size_bytes:
                        raise IngestionSourceError("任务源文件大小已变化")
                    digest.update(part)
                    destination.write(part)
            if size != submission.size_bytes:
                raise IngestionSourceError("任务源文件读取不完整")
            if digest.hexdigest() != submission.content_sha256:
                raise IngestionSourceError("任务源文件摘要不一致")

    def _path(self, submission: IngestionJobSubmission) -> Path:
        validated = IngestionJobSubmission.model_validate(
            submission.model_dump(),
        )
        name = f"{validated.doc_id}_{validated.filename}"
        if len(os.fsencode(name)) > 255:
            raise IngestionSourceError("任务源文件名过长")
        return self._docs_path / name

    def _sync_directory(self) -> None:
        descriptor = os.open(self._docs_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

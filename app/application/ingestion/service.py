# -*- coding: utf-8 -*-
"""安全保存上传内容并提交文档入库。"""

from pathlib import Path, PurePath, PureWindowsPath
from uuid import uuid4

from app.application.errors import (
    EmptyDocumentError,
    FileTooLargeError,
    MissingFilenameError,
)
from app.application.ports.ingestion import DocumentIngestion
from app.domain.documents import UploadResult


class UploadService:
    """安全保存上传内容，并执行完整向量入库流程。"""

    def __init__(
        self,
        docs_path: Path,
        max_upload_size: int,
        ingestion_pipeline: DocumentIngestion,
    ) -> None:
        """创建上传服务并保存目录、大小限制及入库流水线。

        Args:
            docs_path: 上传源文件的受控保存目录。
            max_upload_size: 单个上传内容允许的最大字节数。
            ingestion_pipeline: 校验类型并提交索引写入的应用端口。
        """
        self.docs_path = docs_path
        self.max_upload_size = max_upload_size
        self._ingestion_pipeline = ingestion_pipeline

    def save_and_parse(
        self,
        filename: str | None,
        content: bytes,
    ) -> UploadResult:
        """校验、保存并向量化上传文档，失败时回滚源文件。

        Args:
            filename: 客户端提供的文件名，可能包含路径或为空。
            content: 已受 HTTP 层读取上限保护的上传内容。

        Returns:
            不依赖 HTTP 模型的稳定领域上传结果。

        Raises:
            MissingFilenameError: 客户端没有提供安全文件名。
            EmptyDocumentError: 内容为空或无法产生文本块。
            FileTooLargeError: 内容超过配置的字节上限。
            UnsupportedFileTypeError: 文件类型没有可用解析器。
            Exception: 解析、向量化或存储失败时保留原异常。
        """
        safe_filename = self._sanitize_filename(filename)
        self._validate_content(content)

        # 类型检查必须发生在 mkdir/write 之前，避免无效上传留下痕迹。
        self._ingestion_pipeline.validate_file_type(safe_filename)
        doc_id = str(uuid4())
        saved_path = self.docs_path / f"{doc_id}_{safe_filename}"

        self.docs_path.mkdir(parents=True, exist_ok=True)
        saved_path.write_bytes(content)

        try:
            chunks_count = self._ingestion_pipeline.ingest(
                saved_path,
                doc_id,
                safe_filename,
            )
            if chunks_count == 0:
                raise EmptyDocumentError("文档中没有可提取的文本")
        except Exception:
            saved_path.unlink(missing_ok=True)
            raise

        return UploadResult(
            doc_id=doc_id,
            filename=safe_filename,
            chunks_count=chunks_count,
        )

    @staticmethod
    def _sanitize_filename(filename: str | None) -> str:
        if not filename or not filename.strip():
            raise MissingFilenameError("上传文件缺少文件名")

        # 浏览器可能提交 Windows 或 POSIX 路径，两种分隔符都需剥离。
        safe_filename = PurePath(PureWindowsPath(filename).name).name
        if safe_filename in {"", ".", ".."}:
            raise MissingFilenameError("上传文件缺少有效文件名")
        return safe_filename

    def _validate_content(self, content: bytes) -> None:
        if not content:
            raise EmptyDocumentError("上传文件内容为空")
        if len(content) > self.max_upload_size:
            raise FileTooLargeError(
                f"上传文件超过大小限制：{self.max_upload_size} 字节"
            )

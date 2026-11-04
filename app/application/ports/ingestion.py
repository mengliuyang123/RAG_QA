# -*- coding: utf-8 -*-
"""声明上传和文档管理用例所需的入库操作端口。"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from app.domain.ingestion_jobs import IngestionJobSubmission
from app.domain.ingestion_jobs import IngestionProgress


class DocumentIngestion(Protocol):
    """定义上传服务校验类型和提交文档入库的能力。"""

    def validate_file_type(self, filename: str) -> None:
        """确认文件名对应受支持的解析器。

        Args:
            filename: 已清理且不含客户端目录的文件名。

        Returns:
            None。
        """

    def ingest(
        self,
        saved_path: Path,
        doc_id: str,
        original_filename: str,
    ) -> int:
        """解析、分块并持久化一份已保存文档。

        Args:
            saved_path: 上传服务已经持久化的源文件路径。
            doc_id: 新文档的稳定唯一标识。
            original_filename: 清理后的原始文件名。

        Returns:
            成功写入的文本块数量。
        """


class DocumentDeletion(Protocol):
    """定义文档管理服务提交索引删除的能力。"""

    def delete_document(self, doc_id: str) -> None:
        """删除指定文档的向量与关键词索引。

        Args:
            doc_id: 要删除的文档标识。

        Returns:
            None。
        """


class ProgressiveDocumentIngestion(Protocol):
    """带实际阶段回调的后台任务入库能力。"""

    def ingest_job(
        self, saved_path: Path, submission: IngestionJobSubmission,
        on_progress: Callable[[IngestionProgress], None],
        *, check_active: Callable[[], None] | None = None,
    ) -> int:
        """处理已验证快照并生成实际阶段进度。

        Args:
            saved_path: 私有源快照。
            submission: 持久任务资料。
            on_progress: 同步持久阶段回调，失败即停止。
            check_active: 在实际分段、分块和批次边界检查执行权。

        Returns:
            实际完成的正文本块数。
        """


class IngestionExecutionGuard(Protocol):
    """协调同一物理索引的执行器，不把数据库租约当作进程写锁。"""

    def hold(self) -> AbstractContextManager[None]:
        """取得不可并发执行的上下文，进程退出由操作系统释放。

        Returns:
            包含认领、恢复、写索引及最终提交的完整执行上下文。
        """

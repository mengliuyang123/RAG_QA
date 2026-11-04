# -*- coding: utf-8 -*-
"""文档维护共用的精确源文件身份与普通文件检查。"""

from datetime import datetime
from datetime import timezone
from pathlib import Path
from pathlib import PurePath
from pathlib import PureWindowsPath
from stat import S_ISREG
from uuid import UUID

from app.application.errors import DocumentStorageCorruptionError
from app.domain.documents import DocumentFileStatus


def safe_source_path(docs_path: Path, doc_id: UUID, filename: str) -> Path:
    """只接受受控目录内该文档的普通文件，绝不返回符号链接目标。

    Args:
        docs_path: 组合根确定的源目录。
        doc_id: 已验证的文档身份。
        filename: 原始单个文件名，不接受路径。

    Returns:
        精确身份的路径；不存在的普通文件位置也可返回。
    """
    if filename in {"", ".", ".."}:
        raise DocumentStorageCorruptionError("文档来源文件名无效")
    if PurePath(filename).name != filename or PureWindowsPath(filename).name != filename:
        raise DocumentStorageCorruptionError("文档来源文件名不安全")
    try:
        root = docs_path.resolve(strict=False)
        candidate = root / f"{doc_id}_{filename}"
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise DocumentStorageCorruptionError("文档源文件路径越界")
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            return candidate
        if not S_ISREG(info.st_mode):
            raise DocumentStorageCorruptionError("文档源文件必须是普通文件")
        # 即使文件本身随后被替换为链接，unlink也不跟随它删除其他文件。
        return candidate
    except (OSError, RuntimeError, ValueError) as error:
        raise DocumentStorageCorruptionError("文档源文件路径不可读取") from error


def source_details(path: Path) -> tuple[DocumentFileStatus, datetime | None, int | None]:
    """读取当前文件大小和mtime，未知保持空值，不读取内容或跟随链接。

    Args:
        path: 已确认的精确源身份路径。

    Returns:
        当前可用状态、UTC修改时间、实际字节数。
    """
    try:
        info = path.lstat()
    except FileNotFoundError:
        return DocumentFileStatus.MISSING, None, None
    except OSError as error:
        raise DocumentStorageCorruptionError("文档源文件状态不可读取") from error
    if not S_ISREG(info.st_mode):
        raise DocumentStorageCorruptionError("文档源文件必须是普通文件")
    return (
        DocumentFileStatus.AVAILABLE,
        datetime.fromtimestamp(info.st_mtime, tz=timezone.utc), info.st_size,
    )

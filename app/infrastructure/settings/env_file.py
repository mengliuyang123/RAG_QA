# -*- coding: utf-8 -*-
"""原地更新 .env 文件中的指定键，保留注释、顺序和其他配置。"""

import errno
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

_NEEDS_QUOTES = re.compile(r"[\s#'\"\\=]")


def format_env_value(value: str) -> str:
    """把配置值写成 python-dotenv 能原样读回的形式。

    Args:
        value: 待写入的原始字符串。

    Returns:
        普通值原样返回；含空白、引号、``#`` 等字符时用双引号包裹并转义。
    """
    if value and not _NEEDS_QUOTES.search(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _line_key(line: str) -> str | None:
    match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
    return match.group(1).upper() if match else None


# rename 无法替换挂载点文件时的错误码。
_UNREPLACEABLE_ERRNOS = frozenset({errno.EBUSY, errno.EXDEV})


class EnvFile:
    """对 .env 做最小改动：只替换、追加或删除指定键所在的行。"""

    def __init__(self, path: Path) -> None:
        """记录目标文件路径，不在构造时读写磁盘。

        Args:
            path: .env 文件路径；不存在时首次写入会创建。

        Returns:
            None。
        """
        self.path = path

    def update(self, values: Mapping[str, str | None]) -> None:
        """原子地写入一组键值。

        Args:
            values: 键名（大小写不敏感）到新值的映射；值为 None 时删除该键，
                让配置回到代码默认值。

        Returns:
            None。

        Raises:
            OSError: 读取或写入 .env 失败。
        """
        pending = {key.upper(): value for key, value in values.items()}
        original = ""
        if self.path.exists():
            # newline="" 保留原始行尾；read_text 会把 \r\n 转成 \n，导致 CRLF 文件被整体改写。
            with open(self.path, encoding="utf-8", newline="") as handle:
                original = handle.read()
        newline = "\r\n" if "\r\n" in original else "\n"
        # 原文件末尾没有换行时保持原样，只改动目标行。
        trailing = newline if not original or original.endswith(("\n", "\r")) else ""
        lines = original.splitlines()

        written: set[str] = set()
        result: list[str] = []
        for line in lines:
            key = _line_key(line)
            if key is None or key not in pending:
                result.append(line)
                continue
            # 同一个键重复出现时只保留第一处，避免读取时被后面的旧值覆盖。
            if key in written:
                continue
            written.add(key)
            value = pending[key]
            if value is not None:
                result.append(f"{key}={format_env_value(value)}")

        for key, value in pending.items():
            if key not in written and value is not None:
                result.append(f"{key}={format_env_value(value)}")

        content = newline.join(result) + (trailing if result else "")
        self._atomic_write(content)

    def _atomic_write(self, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".env.", suffix=".tmp", dir=self.path.parent,
            )
        except PermissionError:
            # 容器以非 root 运行时目录不可写，只能原地写入已挂载的文件。
            self._write_in_place(content)
            return
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            if self.path.exists():
                shutil.copymode(self.path, temporary)
            os.replace(temporary, self.path)
        except OSError as exc:
            Path(temporary).unlink(missing_ok=True)
            # Docker 单文件挂载点不能被 rename 替换（EBUSY/EXDEV），退回原地写入。
            if exc.errno not in _UNREPLACEABLE_ERRNOS or not self.path.exists():
                raise
            self._write_in_place(content)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def _write_in_place(self, content: str) -> None:
        # 非原子：先完整编码再一次写入，缩短文件处于截断状态的时间。
        data = content.encode("utf-8")
        with open(self.path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

# -*- coding: utf-8 -*-
"""原地更新 .env 文件中的指定键，保留注释、顺序和其他配置。"""

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
        original = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        newline = "\r\n" if "\r\n" in original else "\n"
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

        content = newline.join(result) + (newline if result else "")
        self._atomic_write(content)

    def _atomic_write(self, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".env.", suffix=".tmp", dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            if self.path.exists():
                shutil.copymode(self.path, temporary)
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

# -*- coding: utf-8 -*-
"""扫描 Git 交付范围并只输出包含疑似凭据的相对文件名。"""

import os
import re
import subprocess
from pathlib import Path


SECRET_VARIABLES = (
    "OPENAI_API_KEY",
    "LLM_API_KEY",
    "UNDERSTANDING_API_KEY",
    "EMBEDDING_API_KEY",
    "RERANKER_API_KEY",
    "EVALUATION_JUDGE_API_KEY",
    "DASHSCOPE_API_KEY",
)
ASSIGNMENT_PATTERN = re.compile(
    rf"(?im)^[ \t]*(?:{'|'.join(SECRET_VARIABLES)})"
    r"[ \t]*=[ \t]*([^\r\n]*)$"
)
TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}")
TEMPLATE_PATTERN = re.compile(
    r"(?:<[^>\r\n]+>|your-[A-Za-z0-9_-]+|replace-me)",
    re.IGNORECASE,
)


def main() -> int:
    """扫描当前 Git 仓库并返回稳定退出码。

    Returns:
        没有疑似凭据时为零，存在匹配时为一。
    """
    matched_files = []
    for relative_path in _delivery_files():
        if _contains_probable_secret(relative_path):
            matched_files.append(relative_path)
    if not matched_files:
        print("No probable secrets found.")
        return 0
    print("Probable secrets found in:")
    for relative_path in sorted(matched_files, key=os.fspath):
        print(os.fspath(relative_path))
    return 1


def _delivery_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    paths = []
    for raw_path in result.stdout.split(b"\x00"):
        if not raw_path:
            continue
        paths.append(Path(os.fsdecode(raw_path)))
    return paths


def _contains_probable_secret(relative_path: Path) -> bool:
    try:
        content = relative_path.read_bytes()
    except OSError:
        return False
    if b"\x00" in content:
        return False
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    for match in ASSIGNMENT_PATTERN.finditer(text):
        value = _unquote(match.group(1).strip())
        if not _is_safe_template(value):
            return True
    return TOKEN_PATTERN.search(text) is not None


def _unquote(value: str) -> str:
    if len(value) < 2:
        return value
    matching_quotes = (
        value.startswith('"') and value.endswith('"')
    ) or (
        value.startswith("'") and value.endswith("'")
    )
    if matching_quotes:
        return value[1:-1].strip()
    return value


def _is_safe_template(value: str) -> bool:
    if not value:
        return True
    # 项目测试使用短占位值；真实服务密钥通常显著更长，且通用
    # sk- 规则仍会独立拦截长 token。
    if len(value) < 20:
        return True
    return TEMPLATE_PATTERN.fullmatch(value) is not None


if __name__ == "__main__":
    raise SystemExit(main())

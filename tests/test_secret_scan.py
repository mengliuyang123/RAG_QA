# -*- coding: utf-8 -*-
"""交付范围凭据扫描器的文件名输出和忽略规则测试。"""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCANNER = PROJECT_ROOT / "scripts" / "check_secrets.py"
FAKE_SECRET = "sk-" + "a" * 32


def _initialize_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


def _run_scan(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCANNER)],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )


def test_scanner_flags_tracked_secret_without_printing_value(
    tmp_path: Path,
) -> None:
    """已跟踪文件中的疑似凭据只应输出相对文件名。"""
    _initialize_repository(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text(f"LLM_API_KEY={FAKE_SECRET}\n", encoding="utf-8")
    subprocess.run(["git", "add", "notes.txt"], cwd=tmp_path, check=True)

    result = _run_scan(tmp_path)

    assert result.returncode == 1
    assert "notes.txt" in result.stdout
    assert FAKE_SECRET not in result.stdout


def test_scanner_flags_untracked_secret(tmp_path: Path) -> None:
    """未忽略且未跟踪的交付文件也必须进入扫描范围。"""
    _initialize_repository(tmp_path)
    (tmp_path / "config.txt").write_text(
        f"token={FAKE_SECRET}\n",
        encoding="utf-8",
    )

    result = _run_scan(tmp_path)

    assert result.returncode == 1
    assert "config.txt" in result.stdout
    assert FAKE_SECRET not in result.stdout


def test_scanner_flags_understanding_credentials_without_sk_prefix(
    tmp_path: Path,
) -> None:
    """新理解模型凭据也需要扫描，不应依赖供应商固定前缀。

    Args:
        tmp_path: 隔离Git仓库。
    """
    _initialize_repository(tmp_path)
    fake_key = "offline" * 5
    (tmp_path / "config.txt").write_text(
        f"UNDERSTANDING_API_KEY={fake_key}\n", encoding="utf-8",
    )
    result = _run_scan(tmp_path)
    assert result.returncode == 1
    assert "config.txt" in result.stdout
    assert fake_key not in result.stdout


def test_scanner_ignores_env_and_accepts_templates_utf8_and_binary(
    tmp_path: Path,
) -> None:
    """被忽略 `.env`、模板值、中文和二进制文件不应误报。"""
    _initialize_repository(tmp_path)
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        f"OPENAI_API_KEY={FAKE_SECRET}\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.example").write_text(
        "OPENAI_API_KEY=<your-openai-key>\n"
        "LLM_API_KEY=your-generator-key\n"
        "EMBEDDING_API_KEY=replace-me\n"
        "RERANKER_API_KEY=\n"
        "EVALUATION_JUDGE_API_KEY=test-key\n",
        encoding="utf-8",
    )
    (tmp_path / "说明.txt").write_text("安全模板文件\n", encoding="utf-8")
    (tmp_path / "asset.bin").write_bytes(
        b"\x00" + FAKE_SECRET.encode("ascii")
    )
    subprocess.run(
        ["git", "add", ".gitignore", ".env.example", "说明.txt", "asset.bin"],
        cwd=tmp_path,
        check=True,
    )

    result = _run_scan(tmp_path)

    assert result.returncode == 0
    assert result.stdout.strip() == "No probable secrets found."

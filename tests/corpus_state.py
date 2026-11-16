# -*- coding: utf-8 -*-
"""固定评估语料完整性探测，语料缺失时相关数据集测试显式跳过。"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_MISSING_REASON = (
    "evaluation/corpus 语料不完整（部分文件未随仓库迁移），"
    "从原机器拷贝齐全部 manifest 文件后本测试自动恢复"
)


def evaluation_corpus_complete() -> bool:
    """检查 evaluation/corpus 是否包含 manifest 声明的全部文件。

    Returns:
        全部文件就位时为 True。
    """
    manifest_path = PROJECT_ROOT / "evaluation" / "corpus_manifest.json"
    corpus_root = PROJECT_ROOT / "evaluation" / "corpus"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        documents = payload["documents"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return all(
        (corpus_root / entry["filename"]).is_file() for entry in documents
    )

# -*- coding: utf-8 -*-
"""固定评估数据集的离线校验测试。"""

import hashlib
import json
from pathlib import Path

import pytest


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _corpus_entry(filename: str, digest: str) -> dict[str, object]:
    return {
        "filename": filename,
        "title": "测试文档",
        "source_url": "https://example.test/docs/source.md",
        "source_revision": "a" * 40,
        "source_path": "docs/source.md",
        "source_sha256": digest,
        "license": "MIT",
        "sha256": digest,
    }


def _qa_case(
    filename: str,
    page: int | None = None,
) -> dict[str, object]:
    return {
        "id": "local-001",
        "question": "什么是测试？",
        "reference_answer": "测试用于验证行为。",
        "expected_sources": [
            {
                "filename": filename,
                "page": page,
            }
        ],
        "tags": ["definition", "single-document"],
    }


def _rgb_manifest() -> dict[str, object]:
    return {
        "repository_url": "https://github.com/chen700564/RGB",
        "commit": "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615",
        "license": "CC BY-NC-SA 4.0",
        "noncommercial": True,
        "passage_num": 5,
        "passage_seed": 2333,
        "noise_rates": [0.0, 0.4, 0.8, 1.0],
        "files": {
            "zh_refine": {
                "path": "data/zh_refine.json",
                "sha256": (
                    "7a5c7547dcfa1b83429b53a41644453508f820f5"
                    "02d3817312cbf8051afb4563"
                ),
                "records_count": 300,
            },
            "zh_int": {
                "path": "data/zh_int.json",
                "sha256": (
                    "e2ddb0461e2130f4b006996a574a89d5898989ff"
                    "8b15f51d900c17162c7d1b17"
                ),
                "records_count": 100,
            },
            "instruction": {
                "path": "config/instruction.yaml",
                "sha256": (
                    "0232a62c325544f7527d0c35bf672169f4ec14f0"
                    "4af722b22c2ea381070390e3"
                ),
                "records_count": None,
            },
        },
        "zh_refine_ids": list(range(30)),
        "zh_int_ids": list(range(30, 60)),
    }


def test_corpus_loader_reads_verified_documents(tmp_path: Path) -> None:
    """有效 manifest 应返回经过哈希校验的文档正文。"""
    from evaluation.datasets import load_corpus

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    content = "# 测试\n\n正文。".encode()
    (corpus_root / "source.md").write_bytes(content)
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_json(
        manifest_path,
        {"documents": [_corpus_entry("source.md", _sha256(content))]},
    )

    corpus = load_corpus(corpus_root, manifest_path, expected_count=1)

    assert [entry.filename for entry in corpus.entries] == ["source.md"]
    assert corpus.documents == {"source.md": content.decode()}


def test_corpus_loader_rejects_hash_mismatch(tmp_path: Path) -> None:
    """语料字节变化后不得静默复用 manifest。"""
    from evaluation.datasets import DatasetValidationError, load_corpus

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "source.md").write_text("changed", encoding="utf-8")
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_json(
        manifest_path,
        {"documents": [_corpus_entry("source.md", "a" * 64)]},
    )

    with pytest.raises(DatasetValidationError, match="SHA-256 不匹配"):
        load_corpus(corpus_root, manifest_path, expected_count=1)


def test_corpus_loader_rejects_duplicate_filename(tmp_path: Path) -> None:
    """两个 manifest 条目不得声明同一目标文件名。"""
    from evaluation.datasets import DatasetValidationError, load_corpus

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    content = b"content"
    (corpus_root / "source.md").write_bytes(content)
    entry = _corpus_entry("source.md", _sha256(content))
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_json(manifest_path, {"documents": [entry, entry]})

    with pytest.raises(DatasetValidationError, match="文件名重复"):
        load_corpus(corpus_root, manifest_path, expected_count=2)


def test_corpus_loader_rejects_missing_file(tmp_path: Path) -> None:
    """manifest 声明的语料不存在时必须立即失败。"""
    from evaluation.datasets import DatasetValidationError, load_corpus

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_json(
        manifest_path,
        {"documents": [_corpus_entry("missing.md", "a" * 64)]},
    )

    with pytest.raises(DatasetValidationError, match="corpus 文件不存在"):
        load_corpus(corpus_root, manifest_path, expected_count=1)


def test_corpus_loader_rejects_unknown_manifest_field(tmp_path: Path) -> None:
    """未知 manifest 字段不得被静默忽略。"""
    from evaluation.datasets import DatasetValidationError, load_corpus

    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    content = b"content"
    (corpus_root / "source.md").write_bytes(content)
    entry = _corpus_entry("source.md", _sha256(content))
    entry["unexpected"] = True
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_json(manifest_path, {"documents": [entry]})

    with pytest.raises(DatasetValidationError, match="格式无效"):
        load_corpus(corpus_root, manifest_path, expected_count=1)


def test_qa_loader_rejects_unknown_source(tmp_path: Path) -> None:
    """QA 引用未入库文件时必须拒绝整份数据。"""
    from evaluation.datasets import (
        CorpusDataset,
        DatasetValidationError,
        load_qa_dataset,
    )

    corpus = CorpusDataset(entries=[], documents={})
    path = tmp_path / "qa.json"
    _write_json(path, [_qa_case("missing.md")])

    with pytest.raises(DatasetValidationError, match="不在 corpus manifest"):
        load_qa_dataset(path, corpus, expected_count=1)


def test_qa_loader_rejects_non_null_markdown_page(tmp_path: Path) -> None:
    """Markdown 来源不得伪造页码标注。"""
    from evaluation.datasets import (
        CorpusDataset,
        CorpusEntry,
        DatasetValidationError,
        load_qa_dataset,
    )

    entry = CorpusEntry.model_validate(
        _corpus_entry("source.md", "a" * 64)
    )
    corpus = CorpusDataset(entries=[entry], documents={"source.md": "text"})
    path = tmp_path / "qa.json"
    _write_json(path, [_qa_case("source.md", page=1)])

    with pytest.raises(DatasetValidationError, match="Markdown 页码必须为空"):
        load_qa_dataset(path, corpus, expected_count=1)


def test_qa_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    """重复 QA ID 会破坏检查点键，必须拒绝。"""
    from evaluation.datasets import (
        CorpusDataset,
        CorpusEntry,
        DatasetValidationError,
        load_qa_dataset,
    )

    entry = CorpusEntry.model_validate(
        _corpus_entry("source.md", "a" * 64)
    )
    corpus = CorpusDataset(entries=[entry], documents={"source.md": "text"})
    path = tmp_path / "qa.json"
    case = _qa_case("source.md")
    _write_json(path, [case, case])

    with pytest.raises(DatasetValidationError, match="QA ID 重复"):
        load_qa_dataset(path, corpus, expected_count=2)


@pytest.mark.parametrize("field", ["question", "reference_answer"])
def test_qa_loader_rejects_blank_text(tmp_path: Path, field: str) -> None:
    """问题或参考答案只有空白字符时不得进入评估。"""
    from evaluation.datasets import (
        CorpusDataset,
        CorpusEntry,
        DatasetValidationError,
        load_qa_dataset,
    )

    entry = CorpusEntry.model_validate(
        _corpus_entry("source.md", "a" * 64)
    )
    corpus = CorpusDataset(entries=[entry], documents={"source.md": "text"})
    case = _qa_case("source.md")
    case[field] = "   "
    path = tmp_path / "qa.json"
    _write_json(path, [case])

    with pytest.raises(DatasetValidationError, match="QA 数据格式无效"):
        load_qa_dataset(path, corpus, expected_count=1)


def test_rgb_manifest_accepts_complete_fixed_subset(tmp_path: Path) -> None:
    """完整 RGB 清单应保留固定数量和协议参数。"""
    from evaluation.datasets import load_rgb_manifest

    path = tmp_path / "rgb_manifest.json"
    _write_json(path, _rgb_manifest())

    manifest = load_rgb_manifest(path)

    assert manifest.passage_num == 5
    assert manifest.passage_seed == 2333
    assert manifest.noise_rates == [0.0, 0.4, 0.8, 1.0]
    assert len(manifest.zh_refine_ids) == 30
    assert len(manifest.zh_int_ids) == 30


def test_rgb_manifest_rejects_duplicate_fixed_ids(tmp_path: Path) -> None:
    """RGB 固定 ID 重复会减少实际 case 数，必须拒绝。"""
    from evaluation.datasets import DatasetValidationError, load_rgb_manifest

    payload = _rgb_manifest()
    payload["zh_refine_ids"] = [0] * 30
    path = tmp_path / "rgb_manifest.json"
    _write_json(path, payload)

    with pytest.raises(DatasetValidationError, match="zh_refine ID 必须唯一"):
        load_rgb_manifest(path)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("commit",), "f" * 40),
        (("files", "zh_refine", "sha256"), "f" * 64),
        (("files", "zh_int", "sha256"), "f" * 64),
        (("files", "instruction", "sha256"), "f" * 64),
    ],
)
def test_rgb_manifest_rejects_changed_upstream_identity(
    tmp_path: Path,
    field_path: tuple[str, ...],
    replacement: str,
) -> None:
    """RGB 上游 revision 或文件内容漂移时必须拒绝运行。"""
    from evaluation.datasets import DatasetValidationError, load_rgb_manifest

    payload = _rgb_manifest()
    target = payload
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[field_path[-1]] = replacement  # type: ignore[index]
    path = tmp_path / "rgb_manifest.json"
    _write_json(path, payload)

    with pytest.raises(DatasetValidationError, match="RGB manifest 格式无效"):
        load_rgb_manifest(path)


def test_committed_evaluation_datasets_are_valid() -> None:
    """仓库交付的语料、QA 和 RGB 清单必须满足生产固定数量。"""
    from evaluation.datasets import (
        load_corpus,
        load_qa_dataset,
        load_rgb_manifest,
    )

    project_root = Path(__file__).resolve().parents[2]
    evaluation_root = project_root / "evaluation"
    corpus = load_corpus(
        evaluation_root / "corpus",
        evaluation_root / "corpus_manifest.json",
    )
    qa_cases = load_qa_dataset(
        evaluation_root / "qa_dataset.json",
        corpus,
    )
    rgb_manifest = load_rgb_manifest(
        evaluation_root / "rgb_manifest.json"
    )

    assert len(corpus.entries) == 15
    assert len(qa_cases) == 50
    assert len(rgb_manifest.zh_refine_ids) == 30
    assert len(rgb_manifest.zh_int_ids) == 30

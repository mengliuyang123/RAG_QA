# -*- coding: utf-8 -*-
"""加载并验证本地语料、QA 与 RGB 固定清单。"""

import hashlib
import json
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from evaluation.models import (
    CorpusDataset,
    CorpusEntry,
    QACase,
    RGBManifest,
)


class DatasetValidationError(ValueError):
    """评估数据缺失、损坏或不符合固定协议时抛出。"""


class _CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    documents: list[CorpusEntry]


def sha256_file(path: Path) -> str:
    """以有界二进制块计算一个文件的 SHA-256。

    Args:
        path: 待读取的文件。

    Returns:
        小写十六进制 SHA-256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus(
    root: Path,
    manifest_path: Path,
    expected_count: int = 15,
) -> CorpusDataset:
    """加载经过来源、数量和哈希校验的固定语料。

    Args:
        root: corpus 文件目录。
        manifest_path: corpus manifest JSON 路径。
        expected_count: 调用方要求的精确文档数量。

    Returns:
        包含有序条目和 UTF-8 正文的固定语料。

    Raises:
        DatasetValidationError: manifest、文件或哈希不符合约定。
    """
    payload = _read_json(manifest_path, "corpus manifest")
    try:
        manifest = _CorpusManifest.model_validate(payload)
    except ValidationError as exc:
        raise DatasetValidationError("corpus manifest 格式无效") from exc
    if len(manifest.documents) != expected_count:
        raise DatasetValidationError(
            f"corpus 文档数量必须为 {expected_count}"
        )

    seen_filenames: set[str] = set()
    documents: dict[str, str] = {}
    resolved_root = root.resolve()
    for entry in manifest.documents:
        _validate_plain_filename(entry.filename)
        if entry.filename in seen_filenames:
            raise DatasetValidationError("corpus manifest 文件名重复")
        seen_filenames.add(entry.filename)

        document_path = (root / entry.filename).resolve()
        if not document_path.is_relative_to(resolved_root):
            raise DatasetValidationError("corpus 文件路径越界")
        if not document_path.is_file():
            raise DatasetValidationError(
                f"corpus 文件不存在：{entry.filename}"
            )
        if sha256_file(document_path) != entry.sha256:
            raise DatasetValidationError(
                f"corpus 文件 SHA-256 不匹配：{entry.filename}"
            )
        try:
            documents[entry.filename] = document_path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError as exc:
            raise DatasetValidationError(
                f"corpus 文件不是 UTF-8：{entry.filename}"
            ) from exc

    return CorpusDataset(entries=manifest.documents, documents=documents)


def load_qa_dataset(
    path: Path,
    corpus: CorpusDataset,
    expected_count: int = 50,
) -> list[QACase]:
    """加载与固定 corpus 一致的人工 QA 标注。

    Args:
        path: QA JSON 数组路径。
        corpus: 已完成完整性校验的语料。
        expected_count: 调用方要求的精确问题数量。

    Returns:
        按文件顺序保留的严格 QA 记录。

    Raises:
        DatasetValidationError: 数量、ID、字段或来源标注无效。
    """
    payload = _read_json(path, "QA dataset")
    try:
        cases = TypeAdapter(list[QACase]).validate_python(payload)
    except ValidationError as exc:
        raise DatasetValidationError("QA 数据格式无效") from exc
    if len(cases) != expected_count:
        raise DatasetValidationError(f"QA 数量必须为 {expected_count}")

    known_sources = {entry.filename for entry in corpus.entries}
    seen_ids: set[str] = set()
    for case in cases:
        if case.id in seen_ids:
            raise DatasetValidationError("QA ID 重复")
        seen_ids.add(case.id)
        for source in case.expected_sources:
            if source.filename not in known_sources:
                raise DatasetValidationError(
                    f"QA 来源不在 corpus manifest：{source.filename}"
                )
            if source.filename.lower().endswith(".md"):
                if source.page is not None:
                    raise DatasetValidationError("Markdown 页码必须为空")
    return cases


def load_rgb_manifest(path: Path) -> RGBManifest:
    """加载 RGB 固定 commit、文件哈希和样本 ID 清单。

    Args:
        path: RGB manifest JSON 路径。

    Returns:
        通过固定协议校验的 RGB manifest。

    Raises:
        DatasetValidationError: 字段、协议参数或固定 ID 无效。
    """
    payload = _read_json(path, "RGB manifest")
    try:
        manifest = RGBManifest.model_validate(payload)
    except ValidationError as exc:
        raise DatasetValidationError("RGB manifest 格式无效") from exc
    if len(set(manifest.zh_refine_ids)) != len(manifest.zh_refine_ids):
        raise DatasetValidationError("zh_refine ID 必须唯一")
    if len(set(manifest.zh_int_ids)) != len(manifest.zh_int_ids):
        raise DatasetValidationError("zh_int ID 必须唯一")
    return manifest


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"{label} 不存在") from exc
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"{label} 不是 UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"{label} 不是有效 JSON") from exc


def _validate_plain_filename(filename: str) -> None:
    posix_name = PurePath(filename).name
    windows_name = PureWindowsPath(filename).name
    if posix_name != filename or windows_name != filename:
        raise DatasetValidationError("corpus filename 必须为纯文件名")

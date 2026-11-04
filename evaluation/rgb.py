# -*- coding: utf-8 -*-
"""RGB 固定中文子集缓存、passage 组装与官方规则评分。"""

import hashlib
import json
import math
import os
import random
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from evaluation.models import RGBFile, RGBManifest


AnswerComponent = str | list[str]
RGBCapability = Literal[
    "noise_robustness",
    "rejection",
    "multi_document_integration",
]


class RGBDatasetError(ValueError):
    """RGB 下载、哈希、JSONL 或固定样本结构无效时抛出。"""


class RGBRefineRecord(BaseModel):
    """RGB `zh_refine` 的一条上游记录。

    Attributes:
        id: 上游稳定整数 ID。
        query: 中文问题。
        answer: 一个或多个答案组成部分。
        positive: 包含答案的候选 passage。
        negative: 不包含答案的噪声 passage。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=0)
    query: str = Field(min_length=1)
    answer: list[AnswerComponent] = Field(min_length=1)
    positive: list[str] = Field(min_length=1)
    negative: list[str] = Field(min_length=1)


class RGBIntegrationRecord(BaseModel):
    """RGB `zh_int` 的一条多文档信息整合记录。

    Attributes:
        id: 上游稳定整数 ID。
        query: 中文问题。
        answer: 必须全部命中的整合答案组成部分。
        asnwer1: 上游保留的第一部分答案原始拼写。
        answer2: 上游可为空的第二部分辅助答案。
        positive: 每个答案部分对应的一组 passage。
        negative: 不包含完整答案的噪声 passage。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=0)
    query: str = Field(min_length=1)
    answer: list[AnswerComponent] = Field(min_length=1)
    asnwer1: list[AnswerComponent] = Field(min_length=1)
    answer2: list[AnswerComponent]
    positive: list[list[str]] = Field(min_length=1)
    negative: list[str]


class RGBDataset(BaseModel):
    """已完成哈希与 JSONL 校验的 RGB 文件内容。

    Attributes:
        refine: 按 ID 索引的噪声与拒答记录。
        integration: 按 ID 索引的信息整合记录。
        system_prompt: 固定中文 system prompt。
        instruction_template: 包含 DOCS 与 QUERY 占位符的模板。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    refine: dict[int, RGBRefineRecord]
    integration: dict[int, RGBIntegrationRecord]
    system_prompt: str = Field(min_length=1)
    instruction_template: str = Field(min_length=1)


class RGBCase(BaseModel):
    """一次确定性 RGB 中文生成调用。

    Attributes:
        key: 检查点唯一键。
        source: `zh_refine` 或 `zh_int`。
        record_id: 上游固定 ID。
        query: 中文问题。
        answer: 官方评分所需答案组成部分。
        passages: 仅运行时使用、序列化时排除的有序正文。
        passage_hashes: 与 passages 同序的 SHA-256。
        noise_rate: 本次 passage 噪声率。
        capability: 噪声、拒答或信息整合能力。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    source: Literal["zh_refine", "zh_int"]
    record_id: int = Field(ge=0)
    query: str = Field(min_length=1)
    answer: list[AnswerComponent] = Field(min_length=1)
    passages: list[str] = Field(min_length=1, exclude=True, repr=False)
    passage_hashes: list[str] = Field(min_length=1)
    noise_rate: float = Field(ge=0.0, le=1.0)
    capability: RGBCapability


class RGBDatasetCache:
    """在隔离目录缓存固定 commit 的 RGB 原始文件。"""

    def __init__(
        self,
        cache_root: Path,
        downloader: Callable[[str], bytes] | None = None,
    ) -> None:
        """保存缓存目录与可替换下载函数。

        Args:
            cache_root: RGB 固定文件缓存目录。
            downloader: 接收 raw URL 并返回完整字节的函数。
        """
        self._cache_root = cache_root
        self._downloader = downloader or _download_bytes

    def prepare(self, manifest: RGBManifest) -> RGBDataset:
        """下载缺失文件、验证哈希并解析固定 RGB 内容。

        Args:
            manifest: 固定 commit、文件哈希与记录数清单。

        Returns:
            可用于构建 150 个 case 的严格数据集。

        Raises:
            RGBDatasetError: 下载、哈希、JSONL、数量或 Prompt 无效。
        """
        self._cache_root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for name, file_info in manifest.files.items():
            paths[name] = self._prepare_file(manifest, name, file_info)

        refine_records = self._parse_jsonl(
            paths["zh_refine"],
            RGBRefineRecord,
            manifest.files["zh_refine"].records_count,
        )
        integration_records = self._parse_jsonl(
            paths["zh_int"],
            RGBIntegrationRecord,
            manifest.files["zh_int"].records_count,
        )
        prompt = self._parse_prompt(paths["instruction"])
        return RGBDataset(
            refine=_index_records(refine_records, "zh_refine"),
            integration=_index_records(integration_records, "zh_int"),
            system_prompt=prompt["system"],
            instruction_template=prompt["instruction"],
        )

    def _prepare_file(
        self,
        manifest: RGBManifest,
        name: str,
        file_info: RGBFile,
    ) -> Path:
        destination = self._cache_root / name
        if destination.is_file():
            if _sha256_bytes(destination.read_bytes()) == file_info.sha256:
                return destination

        repository_path = manifest.repository_url.removeprefix(
            "https://github.com/"
        ).strip("/")
        url = (
            "https://raw.githubusercontent.com/"
            f"{repository_path}/{manifest.commit}/{file_info.path}"
        )
        content = self._downloader(url)
        if _sha256_bytes(content) != file_info.sha256:
            raise RGBDatasetError(f"RGB 文件 SHA-256 不匹配：{name}")
        temporary_path = self._cache_root / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary_path.open("xb") as file_handle:
                file_handle.write(content)
                file_handle.flush()
                os.fsync(file_handle.fileno())
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        return destination

    @staticmethod
    def _parse_jsonl(
        path: Path,
        record_type: type[BaseModel],
        expected_count: int | None,
    ) -> list[BaseModel]:
        records: list[BaseModel] = []
        adapter = TypeAdapter(record_type)
        try:
            with path.open("r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    if not line.strip():
                        continue
                    records.append(adapter.validate_json(line))
        except (UnicodeDecodeError, ValidationError) as exc:
            raise RGBDatasetError(f"RGB JSONL 格式无效：{path.name}") from exc
        if expected_count is not None and len(records) != expected_count:
            raise RGBDatasetError(f"RGB JSONL 记录数量不匹配：{path.name}")
        return records

    @staticmethod
    def _parse_prompt(path: Path) -> dict[str, str]:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            chinese = payload["zh"]
            system_prompt = chinese["system"]
            instruction = chinese["instruction"]
        except (
            UnicodeDecodeError,
            yaml.YAMLError,
            KeyError,
            TypeError,
        ) as exc:
            raise RGBDatasetError("RGB instruction Prompt 格式无效") from exc
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise RGBDatasetError("RGB system Prompt 为空")
        if not isinstance(instruction, str):
            raise RGBDatasetError("RGB instruction Prompt 为空")
        if "{DOCS}" not in instruction or "{QUERY}" not in instruction:
            raise RGBDatasetError("RGB instruction Prompt 缺少固定占位符")
        return {"system": system_prompt, "instruction": instruction}


def build_refine_case(
    record: RGBRefineRecord,
    noise_rate: float,
    passage_num: int = 5,
    seed: int = 2333,
) -> RGBCase:
    """按官方头部取值与逐 case seed 构建 refine case。

    Args:
        record: `zh_refine` 上游记录。
        noise_rate: 零到一之间的固定噪声率。
        passage_num: 最终 passage 数量。
        seed: 每道题重置的 shuffle seed。

    Returns:
        有序 passage 正文与哈希均已固定的 case。
    """
    negative_count = math.ceil(passage_num * noise_rate)
    positive_count = passage_num - negative_count
    if noise_rate == 1.0:
        negative_count = passage_num
        positive_count = 0
    if negative_count > len(record.negative):
        negative_count = len(record.negative)
        positive_count = passage_num - negative_count
    if positive_count > len(record.positive):
        positive_count = len(record.positive)
        negative_count = passage_num - positive_count
    passages = (
        record.positive[:positive_count]
        + record.negative[:negative_count]
    )
    if len(passages) != passage_num:
        raise RGBDatasetError("RGB refine passage 数量不足")
    random.Random(seed).shuffle(passages)
    capability: RGBCapability = "noise_robustness"
    if noise_rate == 1.0:
        capability = "rejection"
    return _make_case(
        source="zh_refine",
        record_id=record.id,
        query=record.query,
        answer=record.answer,
        passages=passages,
        noise_rate=noise_rate,
        capability=capability,
    )


def build_integration_case(
    record: RGBIntegrationRecord,
    passage_num: int = 5,
    seed: int = 2333,
) -> RGBCase:
    """先覆盖每个 positive 分组，再按官方顺序补足 passage。

    Args:
        record: `zh_int` 上游记录。
        passage_num: 最终 passage 数量。
        seed: 每道题重置的 shuffle seed。

    Returns:
        有序 passage 正文与哈希均已固定的信息整合 case。
    """
    generator = random.Random(seed)
    groups = []
    for group in record.positive:
        copied_group = list(group)
        generator.shuffle(copied_group)
        groups.append(copied_group)
    passages = []
    for group in groups:
        if group:
            passages.append(group[0])
    if len(passages) < passage_num:
        max_group_length = max(len(group) for group in groups)
        for group_index in range(1, max_group_length):
            for group in groups:
                if len(group) > group_index:
                    passages.append(group[group_index])
                if len(passages) == passage_num:
                    break
            if len(passages) == passage_num:
                break
    if len(passages) < passage_num:
        missing = passage_num - len(passages)
        passages.extend(record.negative[:missing])
    passages = passages[:passage_num]
    if len(passages) != passage_num:
        raise RGBDatasetError("RGB integration passage 数量不足")
    generator.shuffle(passages)
    return _make_case(
        source="zh_int",
        record_id=record.id,
        query=record.query,
        answer=record.answer,
        passages=passages,
        noise_rate=0.0,
        capability="multi_document_integration",
    )


def build_rgb_cases(
    dataset: RGBDataset,
    manifest: RGBManifest,
) -> list[RGBCase]:
    """从固定 ID 和四档协议构建精确 150 个 RGB case。

    Args:
        dataset: 已完成哈希与 JSONL 校验的上游数据。
        manifest: 固定 ID、passage 数量和 seed 清单。

    Returns:
        90 个噪声、30 个拒答和 30 个信息整合 case。

    Raises:
        RGBDatasetError: 固定 ID 缺失或最终数量不为 150。
    """
    cases: list[RGBCase] = []
    for record_id in manifest.zh_refine_ids:
        record = dataset.refine.get(record_id)
        if record is None:
            raise RGBDatasetError(f"RGB zh_refine 固定 ID 缺失：{record_id}")
        for noise_rate in manifest.noise_rates:
            cases.append(
                build_refine_case(
                    record,
                    noise_rate,
                    manifest.passage_num,
                    manifest.passage_seed,
                )
            )
    for record_id in manifest.zh_int_ids:
        record = dataset.integration.get(record_id)
        if record is None:
            raise RGBDatasetError(f"RGB zh_int 固定 ID 缺失：{record_id}")
        cases.append(
            build_integration_case(
                record,
                manifest.passage_num,
                manifest.passage_seed,
            )
        )
    if len(cases) != 150:
        raise RGBDatasetError("RGB 固定 case 数量必须为 150")
    return cases


def score_rgb_official(case: RGBCase, response: str) -> bool:
    """按 RGB 官方全组成部分规则计算确定性布尔分数。

    Args:
        case: 固定 RGB 生成 case。
        response: 模型生成的中文回答。

    Returns:
        普通 case 是否命中所有答案组成部分，或纯噪声是否拒答。
    """
    normalized_response = response.replace(" ", "").lower()
    if case.capability == "rejection":
        return "信息不足" in normalized_response
    for component in case.answer:
        alternatives = [component]
        if isinstance(component, list):
            alternatives = component
        matched = False
        for alternative in alternatives:
            normalized_answer = alternative.replace(" ", "").lower()
            if normalized_answer in normalized_response:
                matched = True
                break
        if not matched:
            return False
    return True


def _make_case(
    source: Literal["zh_refine", "zh_int"],
    record_id: int,
    query: str,
    answer: list[AnswerComponent],
    passages: list[str],
    noise_rate: float,
    capability: RGBCapability,
) -> RGBCase:
    passage_hashes = []
    for passage in passages:
        passage_hashes.append(_sha256_bytes(passage.encode("utf-8")))
    key = f"{source}:{record_id}:{noise_rate:g}"
    return RGBCase(
        key=key,
        source=source,
        record_id=record_id,
        query=query,
        answer=answer,
        passages=passages,
        passage_hashes=passage_hashes,
        noise_rate=noise_rate,
        capability=capability,
    )


def _index_records(
    records: list[BaseModel],
    label: str,
) -> dict[int, BaseModel]:
    indexed: dict[int, BaseModel] = {}
    for record in records:
        record_id = getattr(record, "id")
        if record_id in indexed:
            raise RGBDatasetError(f"RGB {label} ID 重复：{record_id}")
        indexed[record_id] = record
    return indexed


def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

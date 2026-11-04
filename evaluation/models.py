# -*- coding: utf-8 -*-
"""评估数据集与清单使用的严格领域模型。"""

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

RGB_REPOSITORY_URL = "https://github.com/chen700564/RGB"
RGB_COMMIT = "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615"
RGB_FILE_IDENTITIES = {
    "zh_refine": (
        "data/zh_refine.json",
        "7a5c7547dcfa1b83429b53a41644453508f820f502d3817312cbf8051afb4563",
        300,
    ),
    "zh_int": (
        "data/zh_int.json",
        "e2ddb0461e2130f4b006996a574a89d5898989ff8b15f51d900c17162c7d1b17",
        100,
    ),
    "instruction": (
        "config/instruction.yaml",
        "0232a62c325544f7527d0c35bf672169f4ec14f04af722b22c2ea381070390e3",
        None,
    ),
}


class CorpusEntry(BaseModel):
    """固定语料中一篇文档的来源与完整性信息。

    Attributes:
        filename: `evaluation/corpus` 下的纯文件名。
        title: 供报告展示的文档标题。
        source_url: 上游或本仓库来源链接。
        source_revision: 固定来源 Git revision。
        source_path: revision 内的来源路径。
        source_sha256: 来源文件的 SHA-256。
        license: 来源内容许可证。
        sha256: corpus 快照文件的 SHA-256。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: NonBlankString
    title: NonBlankString
    source_url: NonBlankString
    source_revision: GitRevision
    source_path: NonBlankString
    source_sha256: Sha256Digest
    license: NonBlankString
    sha256: Sha256Digest


class CorpusDataset(BaseModel):
    """已完成哈希校验的固定语料。

    Attributes:
        entries: manifest 中的有序文档条目。
        documents: 按文件名索引的 UTF-8 正文。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: list[CorpusEntry]
    documents: dict[str, str]


class ExpectedSource(BaseModel):
    """一道 QA 的人工标注来源。

    Attributes:
        filename: corpus 中的文件名。
        page: PDF 正整数页码；Markdown 必须为空。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: NonBlankString
    page: int | None = Field(default=None, ge=1)


class QACase(BaseModel):
    """本地端到端评估的一道人工校验问题。

    Attributes:
        id: 稳定检查点标识。
        question: 发送给 RAG Pipeline 的问题。
        reference_answer: 人工复核的参考答案。
        expected_sources: 一个或多个期望来源。
        tags: 问题类型与技术范围标签。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^local-\d{3}$")
    question: NonBlankString
    reference_answer: NonBlankString
    expected_sources: list[ExpectedSource] = Field(min_length=1)
    tags: list[NonBlankString] = Field(min_length=1)


class RGBFile(BaseModel):
    """固定 RGB 上游文件的下载与校验信息。

    Attributes:
        path: 上游 commit 内的相对路径。
        sha256: 固定文件 SHA-256。
        records_count: JSONL 预期记录数；Prompt 文件为空。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: NonBlankString
    sha256: Sha256Digest
    records_count: int | None = Field(default=None, ge=1)


class RGBManifest(BaseModel):
    """RGB 中文固定子集的完整可复现协议。

    Attributes:
        repository_url: RGB 官方仓库地址。
        commit: 固定上游 commit。
        license: 数据集许可证。
        noncommercial: 是否只允许非商业使用。
        passage_num: 每个 case 提供的 passage 数量。
        passage_seed: 每个 case 重置的随机种子。
        noise_rates: `zh_refine` 的固定噪声比例。
        files: 两份 JSONL 与 Prompt 文件描述。
        zh_refine_ids: 噪声和拒答固定样本 ID。
        zh_int_ids: 信息整合固定样本 ID。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository_url: NonBlankString
    commit: GitRevision
    license: NonBlankString
    noncommercial: bool
    passage_num: int = Field(ge=1)
    passage_seed: int
    noise_rates: list[float]
    files: dict[str, RGBFile]
    zh_refine_ids: list[int] = Field(min_length=30, max_length=30)
    zh_int_ids: list[int] = Field(min_length=30, max_length=30)

    @model_validator(mode="after")
    def validate_protocol_constants(self) -> "RGBManifest":
        """拒绝会改变固定 RGB case 集的协议参数。

        Returns:
            已通过校验的当前模型。

        Raises:
            ValueError: 文件集合或协议参数偏离固定设计。
        """
        expected_files = {"zh_refine", "zh_int", "instruction"}
        if set(self.files) != expected_files:
            raise ValueError("RGB files 必须包含固定的三个文件")
        if self.repository_url != RGB_REPOSITORY_URL:
            raise ValueError("RGB repository_url 与固定来源不一致")
        if self.commit != RGB_COMMIT:
            raise ValueError("RGB commit 与固定版本不一致")
        for name, expected_identity in RGB_FILE_IDENTITIES.items():
            rgb_file = self.files[name]
            actual_identity = (
                rgb_file.path,
                rgb_file.sha256,
                rgb_file.records_count,
            )
            if actual_identity != expected_identity:
                raise ValueError(f"RGB 文件身份不一致：{name}")
        if self.passage_num != 5:
            raise ValueError("RGB passage_num 必须为 5")
        if self.passage_seed != 2333:
            raise ValueError("RGB passage_seed 必须为 2333")
        if self.noise_rates != [0.0, 0.4, 0.8, 1.0]:
            raise ValueError("RGB noise_rates 与固定协议不一致")
        return self

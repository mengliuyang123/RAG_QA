# -*- coding: utf-8 -*-
"""严格评估报告模型、Markdown 渲染与原子发布。"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from evaluation.checkpoints import StableErrorCategory


Sha256Digest = str
RunStatus = Literal["complete", "failed", "preview"]
RecordStatus = Literal["success", "failed"]
LocalStrategy = Literal["vector", "hybrid", "hybrid_rerank"]
RGBCapability = Literal[
    "noise_robustness",
    "rejection",
    "multi_document_integration",
]
README_RESULTS_START = "<!-- EVALUATION_RESULTS_START -->"
README_RESULTS_END = "<!-- EVALUATION_RESULTS_END -->"


class DatasetHashes(BaseModel):
    """本次评估使用的数据与运行清单哈希。

    Attributes:
        corpus_manifest: 固定语料 manifest SHA-256。
        qa_dataset: 本地 QA 数据 SHA-256。
        rgb_manifest: RGB 固定清单 SHA-256。
        run_manifest: 完整非敏感运行清单 SHA-256。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_manifest: str = Field(pattern=r"^[0-9a-f]{64}$")
    qa_dataset: str = Field(pattern=r"^[0-9a-f]{64}$")
    rgb_manifest: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest: str = Field(pattern=r"^[0-9a-f]{64}$")


class ModelInventory(BaseModel):
    """报告允许公开的模型名与规范服务 origin。

    Attributes:
        generator_model: 回答生成模型名。
        generator_origin: 回答服务 origin。
        embedding_model: 向量模型名。
        embedding_origin: 向量服务 origin。
        reranker_model: 重排模型名。
        reranker_origin: 重排服务 origin。
        judge_model: 独立裁判模型名。
        judge_origin: 独立裁判服务 origin。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    generator_model: str = Field(min_length=1)
    generator_origin: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_origin: str = Field(min_length=1)
    reranker_model: str = Field(min_length=1)
    reranker_origin: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    judge_origin: str = Field(min_length=1)

    @field_validator(
        "generator_origin",
        "embedding_origin",
        "reranker_origin",
        "judge_origin",
    )
    @classmethod
    def validate_origin(cls, value: str) -> str:
        """拒绝包含路径、查询或凭据的完整服务 URL。

        Args:
            value: 待验证服务 origin。

        Returns:
            去除末尾斜杠的规范 origin。

        Raises:
            ValueError: 输入不是安全 origin。
        """
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        has_extra_parts = bool(
            parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        )
        if not parsed.scheme or parsed.hostname is None or has_extra_parts:
            raise ValueError("报告模型地址必须是安全 origin")
        return normalized


class RetryPolicySummary(BaseModel):
    """报告中的非敏感重试参数。

    Attributes:
        max_attempts: 包含首次调用的最大次数。
        initial_delay_seconds: 第一次重试前等待秒数。
        multiplier: 指数退避倍率。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(ge=1)
    initial_delay_seconds: float = Field(ge=0.0)
    multiplier: float = Field(ge=1.0)


class EvaluationConfiguration(BaseModel):
    """影响评估结果的固定运行参数。

    Attributes:
        chunk_size: 文本块最大字符数。
        chunk_overlap: 相邻文本块重叠字符数。
        embedding_batch_size: 文档向量批次上限。
        top_k: 每题最终检索结果数。
        top_k_rerank: 进入重排的候选数。
        local_temperature: 本地 QA 生成温度。
        rgb_temperature: RGB 生成温度。
        judge_temperature: 独立裁判温度。
        llm_reasoning_effort: 回答生成模型的可选推理强度。
        judge_reasoning_effort: 独立裁判模型的可选推理强度。
        retry: 远程调用重试参数。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    embedding_batch_size: int = Field(gt=0)
    top_k: int = Field(gt=0)
    top_k_rerank: int = Field(gt=0)
    local_temperature: float = Field(ge=0.0)
    rgb_temperature: float = Field(ge=0.0)
    judge_temperature: float = Field(ge=0.0)
    llm_reasoning_effort: str | None = None
    judge_reasoning_effort: str | None = None
    retry: RetryPolicySummary


class RetrievedChunkReport(BaseModel):
    """不含正文的检索文本块证据。

    Attributes:
        rank: 检索名次。
        doc_id: 来源文档标识。
        filename: 来源文件名。
        page: 可选来源页码。
        chunk_index: 文档内文本块序号。
        score: 最终检索分数。
        text_sha256: 文本块正文哈希。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)
    score: float
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceReport(BaseModel):
    """不含 chunk 正文的引用或期望来源。

    Attributes:
        filename: 来源文件名。
        page: 可选来源页码。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


class LocalCaseReport(BaseModel):
    """一道本地题目在一种检索策略下的公开结果。

    Attributes:
        key: 检查点唯一键。
        case_id: QA 稳定 ID。
        strategy: 实际检索策略。
        status: 该记录是否成功。
        answer: 可选生成答案，不包含上下文正文。
        expected_sources: 人工期望来源。
        cited_sources: 回答实际引用来源。
        retrieved: 不含正文的检索追踪。
        hit: 前五条是否命中来源。
        first_relevant_rank: 首个命中名次。
        citation_precision: 当前题引用精确率。
        citation_recall: 当前题引用召回率。
        judge_semantic_correctness: 独立裁判语义正确性。
        answer_correctness: RAGAS 答案正确性。
        faithfulness: RAGAS 忠实度。
        answer_relevancy: RAGAS 回答相关性。
        retrieval_ms: 检索耗时。
        generation_ms: 生成耗时。
        latency_ms: 端到端耗时。
        error_category: 失败时的稳定错误类别。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    strategy: LocalStrategy
    status: RecordStatus
    answer: str | None = None
    expected_sources: list[SourceReport] = Field(default_factory=list)
    cited_sources: list[SourceReport] = Field(default_factory=list)
    retrieved: list[RetrievedChunkReport] = Field(default_factory=list)
    hit: bool | None = None
    first_relevant_rank: int | None = Field(default=None, ge=1)
    citation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    judge_semantic_correctness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    answer_correctness: float | None = Field(default=None, ge=0.0, le=1.0)
    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = None
    retrieval_ms: float | None = Field(default=None, ge=0.0)
    generation_ms: float | None = Field(default=None, ge=0.0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    error_category: StableErrorCategory | None = None


class RGBCaseReport(BaseModel):
    """一道 RGB 生成 case 的公开结果。

    Attributes:
        key: 检查点唯一键。
        case_id: 上游固定记录 ID。
        capability: 噪声、拒答或多文档整合类别。
        noise_rate: 该 case 使用的噪声率。
        status: 该记录是否成功。
        response: 可选生成回答，不包含 passage 正文。
        passage_hashes: 最终有序 passage 文本哈希。
        official_correct: RGB 官方规则布尔结果。
        judge_semantic_correctness: 独立裁判语义分数。
        latency_ms: 生成耗时。
        error_category: 失败时的稳定错误类别。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    case_id: int = Field(ge=0)
    capability: RGBCapability
    noise_rate: float = Field(ge=0.0, le=1.0)
    status: RecordStatus
    response: str | None = None
    passage_hashes: list[str] = Field(min_length=5, max_length=5)
    official_correct: bool | None = None
    judge_semantic_correctness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    latency_ms: float | None = Field(default=None, ge=0.0)
    error_category: StableErrorCategory | None = None

    @field_validator("passage_hashes")
    @classmethod
    def validate_passage_hashes(cls, values: list[str]) -> list[str]:
        """校验 passage 追踪只包含 SHA-256。

        Args:
            values: 最终有序 passage 哈希。

        Returns:
            已验证的原始顺序哈希。

        Raises:
            ValueError: 任一值不是小写 SHA-256。
        """
        for value in values:
            if len(value) != 64:
                raise ValueError("RGB passage hash 必须是 SHA-256")
            if any(character not in "0123456789abcdef" for character in value):
                raise ValueError("RGB passage hash 必须是 SHA-256")
        return values


class EvaluationReport(BaseModel):
    """可公开、可追溯且禁止上下文正文的完整评估报告。

    Attributes:
        status: 完整成功、失败或不可发布的小样本预览。
        run_id: UTC 时间与代码摘要构成的运行标识。
        started_at: UTC 开始时间。
        finished_at: UTC 完成时间。
        git_commit: 运行时 Git revision。
        dirty_worktree: 运行时工作树是否有未提交变化。
        dataset_hashes: 固定输入及运行清单哈希。
        models: 非敏感模型名和服务 origin。
        configuration: 影响结果的检索、生成与重试参数。
        local_records: 50 题乘三种策略的结果。
        rgb_records: 固定 RGB 中文子集 150 条结果。
        local_aggregates: 按策略汇总的本地指标。
        rgb_aggregates: 按能力汇总的 RGB 指标。
        rgb_license: RGB 数据许可证。
        rgb_noncommercial: RGB 是否仅限非商业使用。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: RunStatus
    run_id: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty_worktree: bool
    dataset_hashes: DatasetHashes
    models: ModelInventory
    configuration: EvaluationConfiguration
    local_records: list[LocalCaseReport]
    rgb_records: list[RGBCaseReport]
    local_aggregates: dict[str, dict[str, float]]
    rgb_aggregates: dict[str, dict[str, float]]
    rgb_license: str = Field(min_length=1)
    rgb_noncommercial: bool

    @model_validator(mode="after")
    def validate_complete_run(self) -> "EvaluationReport":
        """完整状态只允许精确且全部成功的固定记录集。

        Returns:
            已通过完整性校验的报告。

        Raises:
            ValueError: complete 记录数量、状态或 key 不符合协议。
        """
        if self.status != "complete":
            return self
        if len(self.local_records) != 150:
            raise ValueError("完整报告必须包含 150 条本地记录")
        if len(self.rgb_records) != 150:
            raise ValueError("完整报告必须包含 150 条 RGB 记录")
        if not all(record.status == "success" for record in self.local_records):
            raise ValueError("完整报告的本地记录必须全部成功")
        if not all(record.status == "success" for record in self.rgb_records):
            raise ValueError("完整报告的 RGB 记录必须全部成功")
        local_keys = {record.key for record in self.local_records}
        rgb_keys = {record.key for record in self.rgb_records}
        if len(local_keys) != 150 or len(rgb_keys) != 150:
            raise ValueError("完整报告记录 key 必须唯一")
        return self


def publish_report(report: EvaluationReport, results_root: Path) -> None:
    """校验完整报告并原子发布归档与 latest 文件。

    Args:
        report: 待发布的严格评估报告。
        results_root: `latest` 与 `runs` 所在目录。

    Returns:
        None。

    Raises:
        ValueError: 报告失败、不完整或被绕过模型校验。
    """
    try:
        validated = EvaluationReport.model_validate(
            report.model_dump(mode="json")
        )
    except ValidationError as exc:
        raise ValueError("报告校验失败，未更新 latest") from exc
    if validated.status != "complete":
        raise ValueError("只有完整成功报告可以更新 latest")

    results_root.mkdir(parents=True, exist_ok=True)
    runs_root = results_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    json_content = validated.model_dump_json(indent=2) + "\n"
    markdown_content = render_markdown_report(validated)
    targets = [
        (runs_root / f"{validated.run_id}.json", json_content),
        (runs_root / f"{validated.run_id}.md", markdown_content),
        (results_root / "latest.json", json_content),
        (results_root / "latest.md", markdown_content),
    ]
    for target, content in targets:
        _atomic_write_text(target, content)


def render_markdown_report(report: EvaluationReport) -> str:
    """把完整报告渲染为简洁 Markdown 证据页。

    Args:
        report: 已通过完整性校验的报告。

    Returns:
        UTF-8 Markdown 文本。
    """
    lines = [
        "# RAG 评估报告",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Git commit: `{report.git_commit}`",
        f"- Dirty worktree: `{str(report.dirty_worktree).lower()}`",
        f"- 本地记录: `{len(report.local_records)}`",
        f"- RGB 记录: `{len(report.rgb_records)}`",
        "",
        "## 模型",
        "",
        f"- Generator: `{report.models.generator_model}`",
        f"- Embedding: `{report.models.embedding_model}`",
        f"- Reranker: `{report.models.reranker_model}`",
        f"- Independent judge: `{report.models.judge_model}`",
        "",
        "## 本地指标",
        "",
    ]
    for strategy, metrics in sorted(report.local_aggregates.items()):
        lines.append(f"### {strategy}")
        lines.append("")
        for name, value in sorted(metrics.items()):
            lines.append(f"- {name}: `{value:.6f}`")
        lines.append("")
    lines.extend(
        [
            "## RGB 中文子集生成评估",
            "",
            f"许可证：{report.rgb_license}；仅限非商业用途。",
            "",
        ]
    )
    for capability, metrics in sorted(report.rgb_aggregates.items()):
        lines.append(f"### {capability}")
        lines.append("")
        for name, value in sorted(metrics.items()):
            lines.append(f"- {name}: `{value:.6f}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def update_readme_metrics(
    readme_path: Path,
    report: EvaluationReport,
) -> None:
    """使用完整报告原子替换 README 的受控指标块。

    Args:
        readme_path: 包含唯一顺序标记对的 UTF-8 README。
        report: 已完成 150+150 记录的严格报告。

    Returns:
        None。

    Raises:
        ValueError: 报告不完整或 README 标记缺失、重复、反向。
    """
    try:
        validated = EvaluationReport.model_validate(
            report.model_dump(mode="json")
        )
    except ValidationError as exc:
        raise ValueError("报告校验失败，未更新 README") from exc
    if validated.status != "complete":
        raise ValueError("只有完整成功报告可以更新 README")
    try:
        content = readme_path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as exc:
        raise ValueError("README 指标标记文件不可读取") from exc
    start_count = content.count(README_RESULTS_START)
    end_count = content.count(README_RESULTS_END)
    start_index = content.find(README_RESULTS_START)
    end_index = content.find(README_RESULTS_END)
    markers_valid = (
        start_count == 1
        and end_count == 1
        and start_index < end_index
    )
    if not markers_valid:
        raise ValueError("README 指标标记必须唯一且顺序正确")

    before = content[:start_index]
    after_index = end_index + len(README_RESULTS_END)
    after = content[after_index:]
    block = _render_readme_metrics(validated)
    updated = before + block + after
    _atomic_write_text(readme_path, updated)


def _render_readme_metrics(report: EvaluationReport) -> str:
    lines = [
        README_RESULTS_START,
        "",
        "### 最新完整评估",
        "",
        f"- Run ID：`{report.run_id}`",
        f"- 完成时间（UTC）：`{report.finished_at.isoformat()}`",
        f"- Git dirty：`{str(report.dirty_worktree).lower()}`",
        (
            "- 模型：Generator `{}`；Embedding `{}`；"
            "Reranker `{}`；Judge `{}`"
        ).format(
            report.models.generator_model,
            report.models.embedding_model,
            report.models.reranker_model,
            report.models.judge_model,
        ),
        (
            "- 报告：[JSON](evaluation/results/latest.json) / "
            "[Markdown](evaluation/results/latest.md)"
        ),
        "",
        "| 目标阈值 | 实际结果 |",
        "| --- | --- |",
        "| 本项目未预设发布阈值 | 以下均来自 latest.json |",
        "",
        (
            "| 策略 | Hit Rate@5 | MRR | Citation Precision | "
            "Citation Recall | Answer Correctness | Faithfulness | "
            "Answer Relevancy | Mean ms | P95 ms |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metrics in sorted(report.local_aggregates.items()):
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | "
            "{:.4f} | {:.4f} | {:.2f} | {:.2f} |".format(
                strategy,
                metrics.get("hit_rate", 0.0),
                metrics.get("mrr", 0.0),
                metrics.get("citation_precision", 0.0),
                metrics.get("citation_recall", 0.0),
                metrics.get("answer_correctness", 0.0),
                metrics.get("faithfulness", 0.0),
                metrics.get("answer_relevancy", 0.0),
                metrics.get("latency_mean_ms", 0.0),
                metrics.get("latency_p95_ms", 0.0),
            )
        )
    lines.extend(
        [
            "",
            (
                "| RGB 能力/噪声 | 官方准确率 | Judge 语义正确性 | "
                "Mean ms | P95 ms |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for capability, metrics in sorted(report.rgb_aggregates.items()):
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.2f} | {:.2f} |".format(
                capability,
                metrics.get("official_accuracy", 0.0),
                metrics.get("judge_semantic_correctness", 0.0),
                metrics.get("latency_mean_ms", 0.0),
                metrics.get("latency_p95_ms", 0.0),
            )
        )
    lines.extend(["", README_RESULTS_END])
    return "\n".join(lines)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_path.open("x", encoding="utf-8") as file_handle:
            file_handle.write(content)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

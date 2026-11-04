# -*- coding: utf-8 -*-
"""Day14 固定评估的命令行编排、检查点恢复与报告发布。"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from app.application.ports.generation import LLMProvider
from app.bootstrap.factories import AdapterFactories
from app.config import Settings
from app.domain.retrieval import RetrievalStrategy
from evaluation.checkpoints import (
    CheckpointChannel,
    EvaluationRunLock,
    RetryPolicy,
    RunCheckpoint,
    StableErrorCategory,
    classify_remote_error,
    run_with_retry,
)
from evaluation.datasets import (
    load_corpus,
    load_qa_dataset,
    load_rgb_manifest,
    sha256_file,
)
from evaluation.judge import JudgeClient
from evaluation.metrics import (
    CaseTrace,
    compute_citation_metrics,
    compute_retrieval_metrics,
    summarize_latency,
)
from evaluation.models import CorpusDataset, QACase, RGBManifest
from evaluation.ragas_eval import RagasCase, RagasEvaluator
from evaluation.reporting import (
    DatasetHashes,
    EvaluationConfiguration,
    EvaluationReport,
    LocalCaseReport,
    ModelInventory,
    RGBCaseReport,
    RetrievedChunkReport,
    RetryPolicySummary,
    SourceReport,
    publish_report,
)
from evaluation.rgb import (
    RGBCase,
    RGBDataset,
    RGBDatasetCache,
    build_rgb_cases,
    score_rgb_official,
)
from evaluation.runtime import build_evaluation_runtime


EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_STRATEGIES = (
    RetrievalStrategy.VECTOR,
    RetrievalStrategy.HYBRID,
    RetrievalStrategy.HYBRID_RERANK,
)


class EvaluationOptions(BaseModel):
    """一次评估运行的生产参数与显式测试缩减项。

    Attributes:
        full: 是否执行完整评估入口。
        resume_run_id: 要恢复的稳定运行 ID。
        rebuild_index: 是否重建当前指纹的隔离索引。
        work_root: 检查点和运行清单目录。
        results_root: 正式报告归档和 latest 目录。
        rgb_cache_root: RGB 固定上游文件缓存目录。
        corpus_root: 固定语料正文目录。
        corpus_manifest_path: 固定语料清单路径。
        qa_dataset_path: 固定 50 题 QA 路径。
        rgb_manifest_path: RGB 固定子集清单路径。
        local_case_limit: 仅供测试缩减本地题目数。
        strategies: 本次执行的检索策略。
        rgb_case_limit: 仅供测试缩减 RGB case 数。
        publish: 是否允许完整报告更新 latest。
        run_id: 仅供测试固定新运行 ID。
        top_k: 最终检索上下文数。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    full: bool = True
    resume_run_id: str | None = None
    rebuild_index: bool = False
    work_root: Path = EVALUATION_ROOT / "work"
    results_root: Path = EVALUATION_ROOT / "results"
    rgb_cache_root: Path = EVALUATION_ROOT / "cache" / "rgb"
    corpus_root: Path = EVALUATION_ROOT / "corpus"
    corpus_manifest_path: Path = EVALUATION_ROOT / "corpus_manifest.json"
    qa_dataset_path: Path = EVALUATION_ROOT / "qa_dataset.json"
    rgb_manifest_path: Path = EVALUATION_ROOT / "rgb_manifest.json"
    local_case_limit: int | None = Field(default=None, ge=1)
    strategies: tuple[RetrievalStrategy, ...] = DEFAULT_STRATEGIES
    rgb_case_limit: int | None = Field(default=None, ge=1)
    publish: bool = True
    run_id: str | None = None
    top_k: int = Field(default=5, ge=1)


class EvaluationDependencies(BaseModel):
    """评估编排中可由离线测试替换的依赖工厂。

    Attributes:
        settings_loader: 加载项目环境配置的函数。
        corpus_loader: 加载固定语料的函数。
        qa_loader: 加载固定 QA 的函数。
        rgb_manifest_loader: 加载 RGB 清单的函数。
        runtime_builder: 构建隔离 RAG 运行时的函数。
        judge_builder: 构建独立裁判的函数。
        ragas_builder: 构建 RAGAS 适配器的函数。
        rgb_preparer: 校验或下载 RGB 固定数据的函数。
        rgb_cases_builder: 构建确定性 RGB cases 的函数。
        rgb_answerer_builder: 构建 RGB 专用生成器的函数。
        git_state_loader: 返回 commit 和 dirty 状态的函数。
        now: 返回当前 UTC 时间的函数。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    settings_loader: Callable[[], Settings] | None = None
    corpus_loader: Callable[..., CorpusDataset] | None = None
    qa_loader: Callable[..., list[QACase]] | None = None
    rgb_manifest_loader: Callable[..., RGBManifest] | None = None
    runtime_builder: Callable[..., Any] | None = None
    judge_builder: Callable[..., Any] | None = None
    ragas_builder: Callable[..., Any] | None = None
    rgb_preparer: Callable[..., Any] | None = None
    rgb_cases_builder: Callable[..., list[RGBCase]] | None = None
    rgb_answerer_builder: Callable[..., Any] | None = None
    git_state_loader: Callable[[], tuple[str, bool]] | None = None
    now: Callable[[], datetime] | None = None


class RGBAnswerer:
    """使用 RGB 官方中文 Prompt 直接生成回答。

    Attributes:
        _provider: 本回答器负责释放的生成 provider。
        _dataset: 经过哈希校验的 RGB Prompt 数据。
        _closed: 是否已经尝试释放 provider。
    """

    def __init__(
        self,
        provider: LLMProvider,
        dataset: RGBDataset,
    ) -> None:
        """保存生成客户端和经过哈希校验的 Prompt。

        Args:
            provider: 固定模型和温度的兼容生成客户端。
            dataset: 包含 RGB 中文 system 与 instruction Prompt 的
                数据。
        """
        self._provider = provider
        self._dataset = dataset
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        dataset: RGBDataset,
        factories: AdapterFactories | None = None,
    ) -> "RGBAnswerer":
        """从评估配置构建 RGB 专用生成器。

        Args:
            settings: 生成模型、地址、凭据和 RGB 温度。
            dataset: 已验证 RGB 数据与 Prompt。
            factories: 可选的共享基础设施 factory 集合。

        Returns:
            不经过 RetrievalService 的 RGB 回答生成器。
        """
        adapter_factories = factories or AdapterFactories()
        provider = adapter_factories.llm_provider(
            settings.resolved_llm_api_key,
            settings.llm_base_url,
            settings.llm_model,
            settings.evaluation_rgb_temperature,
            settings.llm_reasoning_effort,
        )
        return cls(provider=provider, dataset=dataset)

    def generate(self, case: RGBCase) -> str:
        """把固定顺序 passages 和问题送入官方中文 Prompt。

        Args:
            case: 一条确定性 RGB 生成 case。

        Returns:
            生成模型的中文回答。
        """
        documents = "\n".join(case.passages)
        instruction = self._dataset.instruction_template.format(
            QUERY=case.query,
            DOCS=documents,
        )
        messages = [
            {"role": "system", "content": self._dataset.system_prompt},
            {"role": "user", "content": instruction},
        ]
        return self._provider.generate(messages)

    def close(self) -> None:
        """幂等释放本回答器持有的生成 provider。

        Returns:
            None。
        """
        if self._closed:
            return
        self._closed = True
        close = getattr(self._provider, "close", None)
        if callable(close):
            close()


class _LocalGenerationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    status: Literal["success"] = "success"
    case_id: str
    strategy: RetrievalStrategy
    answer: str
    contexts: list[str]
    expected_sources: list[SourceReport]
    cited_sources: list[SourceReport]
    retrieved: list[RetrievedChunkReport]
    hit: bool
    first_relevant_rank: int | None
    citation_precision: float
    citation_recall: float
    retrieval_ms: float
    generation_ms: float
    latency_ms: float


class _LocalScoreRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    status: Literal["success"] = "success"
    report: LocalCaseReport


class _RGBGenerationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    status: Literal["success"] = "success"
    case_key: str
    response: str
    official_correct: bool
    latency_ms: float


class _RGBScoreRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    status: Literal["success"] = "success"
    report: RGBCaseReport


class _FailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    status: Literal["failed"] = "failed"
    error_category: StableErrorCategory


def build_parser() -> argparse.ArgumentParser:
    """构建只暴露正式运行、恢复和重建开关的 CLI parser。

    Returns:
        配置完成的命令行参数解析器。
    """
    parser = argparse.ArgumentParser(
        description="运行固定 Day14 RAG 与 RGB 中文评估",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        required=True,
        help="执行固定 15 文档、50 QA、三策略和 150 RGB case",
    )
    parser.add_argument("--resume", metavar="RUN_ID")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def run_evaluation(
    options: EvaluationOptions,
    dependencies: EvaluationDependencies | None = None,
) -> EvaluationReport:
    """执行一次可恢复评估，并只发布严格完整的正式报告。

    Args:
        options: 正式参数或测试显式缩减参数。
        dependencies: 可替换的离线测试工厂。

    Returns:
        完整正式报告或不可发布的小样本预览报告。

    Raises:
        Exception: 预检、数据、远程调用或报告校验失败。
    """
    if not options.full:
        raise ValueError("评估必须显式使用 full 模式")
    resolved = dependencies or EvaluationDependencies()
    settings = _load_settings(resolved)
    settings.validate_evaluation_settings(require_judge=True)
    _validate_remote_settings(settings)

    corpus = _load_corpus(options, resolved)
    qa_cases = _load_qa(options, resolved, corpus)
    rgb_manifest = _load_rgb_manifest(options, resolved)
    selected_qa = _limit_values(qa_cases, options.local_case_limit)
    git_commit, dirty_worktree = _load_git_state(resolved)
    base_manifest = _base_manifest(
        options,
        settings,
        git_commit,
        corpus,
        selected_qa,
        rgb_manifest,
    )
    lock_hash = _sha256_json(base_manifest)
    run_id = _select_run_id(options, resolved, git_commit)
    resume = options.resume_run_id is not None
    lock_path = options.work_root / "evaluation.lock"

    with EvaluationRunLock.acquire(
        lock_path,
        resume=resume,
        run_id=run_id,
        manifest_hash=lock_hash,
    ):
        rgb_dataset = _prepare_rgb(options, resolved, rgb_manifest)
        rgb_cases = _build_rgb_cases(resolved, rgb_dataset, rgb_manifest)
        selected_rgb = _limit_values(rgb_cases, options.rgb_case_limit)
        manifest = dict(base_manifest)
        manifest["rgb_cases"] = _rgb_case_manifest(selected_rgb)
        checkpoint = _open_checkpoint(
            options,
            run_id,
            manifest,
        )
        completed = _load_completed_report(checkpoint)
        if completed is not None:
            return completed
        return _execute_evaluation(
            options=options,
            dependencies=resolved,
            settings=settings,
            corpus=corpus,
            qa_cases=selected_qa,
            rgb_manifest=rgb_manifest,
            rgb_dataset=rgb_dataset,
            rgb_cases=selected_rgb,
            checkpoint=checkpoint,
            git_commit=git_commit,
            dirty_worktree=dirty_worktree,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行并执行完整评估。

    Args:
        argv: 可选参数序列；为空时读取进程命令行。

    Returns:
        成功时返回零。
    """
    arguments = build_parser().parse_args(argv)
    options = EvaluationOptions(
        full=arguments.full,
        resume_run_id=arguments.resume,
        rebuild_index=arguments.rebuild_index,
    )
    dependencies = EvaluationDependencies()
    run_id = options.resume_run_id
    if run_id is None:
        git_commit, _dirty_worktree = _load_git_state(dependencies)
        run_id = _select_run_id(options, dependencies, git_commit)
        options = options.model_copy(update={"run_id": run_id})
    print(f"Run ID: {run_id}", flush=True)
    print(
        "Resume: uv run --group eval python -m evaluation.run_eval "
        f"--full --resume {run_id}",
        flush=True,
    )
    report = run_evaluation(options, dependencies)
    print(
        f"evaluation {report.status}: {report.run_id} "
        f"({len(report.local_records)} local, "
        f"{len(report.rgb_records)} RGB)"
    )
    return 0


def _execute_evaluation(
    options: EvaluationOptions,
    dependencies: EvaluationDependencies,
    settings: Settings,
    corpus: CorpusDataset,
    qa_cases: list[QACase],
    rgb_manifest: RGBManifest,
    rgb_dataset: Any,
    rgb_cases: list[RGBCase],
    checkpoint: RunCheckpoint,
    git_commit: str,
    dirty_worktree: bool,
) -> EvaluationReport:
    started_at = _now(dependencies)
    retry_policy = RetryPolicy()
    runtime_builder = dependencies.runtime_builder or build_evaluation_runtime
    runtime = run_with_retry(
        lambda: runtime_builder(
            settings,
            corpus,
            options.rebuild_index,
        ),
        retry_policy,
    )
    try:
        generations = _generate_local_cases(
            options,
            qa_cases,
            runtime.query_service,
            checkpoint,
            retry_policy,
        )
        local_records = _score_local_cases(
            qa_cases,
            options.strategies,
            generations,
            settings,
            dependencies,
            checkpoint,
            retry_policy,
        )
    finally:
        runtime.close()

    rgb_records = _run_rgb_cases(
        rgb_cases,
        rgb_dataset,
        settings,
        dependencies,
        checkpoint,
        retry_policy,
    )
    report = _build_report(
        options=options,
        settings=settings,
        rgb_manifest=rgb_manifest,
        checkpoint=checkpoint,
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
        started_at=started_at,
        finished_at=_now(dependencies),
        local_records=local_records,
        rgb_records=rgb_records,
    )
    _write_completed_report(checkpoint, report)
    if options.publish:
        publish_report(report, options.results_root)
    return report


def _generate_local_cases(
    options: EvaluationOptions,
    qa_cases: list[QACase],
    query_service: Any,
    checkpoint: RunCheckpoint,
    retry_policy: RetryPolicy,
) -> dict[str, _LocalGenerationRecord]:
    existing = checkpoint.records(CheckpointChannel.LOCAL)
    scored = checkpoint.records(CheckpointChannel.RAGAS)
    generated: dict[str, _LocalGenerationRecord] = {}
    for case in qa_cases:
        for strategy in options.strategies:
            generation_key = _local_key(case.id, strategy, "generation")
            score_key = _local_key(case.id, strategy, "ragas")
            if score_key in scored:
                continue
            saved = existing.get(generation_key)
            if saved is not None:
                generated[generation_key] = (
                    _LocalGenerationRecord.model_validate(saved)
                )
                continue
            try:
                result = run_with_retry(
                    lambda case=case, strategy=strategy: (
                        query_service.evaluate_query(
                            case.question,
                            strategy,
                            options.top_k,
                        )
                    ),
                    retry_policy,
                )
                record = _make_local_generation(
                    generation_key,
                    case,
                    strategy,
                    result,
                    options.top_k,
                )
                checkpoint.append(CheckpointChannel.LOCAL, record)
                generated[generation_key] = record
            except Exception as exc:
                _append_failure(
                    checkpoint,
                    CheckpointChannel.LOCAL,
                    generation_key,
                    exc,
                )
                raise
    return generated


def _score_local_cases(
    qa_cases: list[QACase],
    strategies: tuple[RetrievalStrategy, ...],
    generations: dict[str, _LocalGenerationRecord],
    settings: Settings,
    dependencies: EvaluationDependencies,
    checkpoint: RunCheckpoint,
    retry_policy: RetryPolicy,
) -> list[LocalCaseReport]:
    existing = checkpoint.records(CheckpointChannel.RAGAS)
    judge_builder = dependencies.judge_builder or JudgeClient.from_settings
    ragas_builder = dependencies.ragas_builder or RagasEvaluator.from_settings
    judge = judge_builder(settings)
    ragas = ragas_builder(settings)
    case_by_id = {case.id: case for case in qa_cases}
    reports: list[LocalCaseReport] = []
    for case in qa_cases:
        for strategy in strategies:
            score_key = _local_key(case.id, strategy, "ragas")
            saved = existing.get(score_key)
            if saved is not None:
                score_record = _LocalScoreRecord.model_validate(saved)
                reports.append(score_record.report)
                continue
            generation_key = _local_key(case.id, strategy, "generation")
            generation = generations[generation_key]
            current_case = case_by_id[generation.case_id]
            try:
                judge_score = run_with_retry(
                    lambda: judge.score_semantic_correctness(
                        current_case.question,
                        current_case.reference_answer,
                        generation.answer,
                    ),
                    retry_policy,
                )
                ragas_scores = run_with_retry(
                    lambda: ragas.score(
                        RagasCase(
                            user_input=current_case.question,
                            response=generation.answer,
                            retrieved_contexts=generation.contexts,
                            reference=current_case.reference_answer,
                        )
                    ),
                    retry_policy,
                )
                report = _local_report(
                    generation,
                    judge_score.score,
                    ragas_scores,
                )
                score_record = _LocalScoreRecord(
                    key=score_key,
                    report=report,
                )
                checkpoint.append(CheckpointChannel.RAGAS, score_record)
                reports.append(report)
            except Exception as exc:
                _append_failure(
                    checkpoint,
                    CheckpointChannel.RAGAS,
                    score_key,
                    exc,
                )
                raise
    return reports


def _run_rgb_cases(
    cases: list[RGBCase],
    dataset: Any,
    settings: Settings,
    dependencies: EvaluationDependencies,
    checkpoint: RunCheckpoint,
    retry_policy: RetryPolicy,
) -> list[RGBCaseReport]:
    existing = checkpoint.records(CheckpointChannel.RGB)
    answerer_builder = (
        dependencies.rgb_answerer_builder or RGBAnswerer.from_settings
    )
    judge_builder = dependencies.judge_builder or JudgeClient.from_settings
    answerer = answerer_builder(settings, dataset)
    try:
        judge = judge_builder(settings)
        return _evaluate_rgb_cases(
            cases,
            answerer,
            judge,
            checkpoint,
            retry_policy,
            existing,
        )
    finally:
        close = getattr(answerer, "close", None)
        if callable(close):
            close()


def _evaluate_rgb_cases(
    cases: list[RGBCase],
    answerer: Any,
    judge: Any,
    checkpoint: RunCheckpoint,
    retry_policy: RetryPolicy,
    existing: dict[str, dict[str, Any]],
) -> list[RGBCaseReport]:
    reports: list[RGBCaseReport] = []
    for case in cases:
        generation_key = _rgb_key(case, "generation")
        score_key = _rgb_key(case, "judge")
        saved_score = existing.get(score_key)
        if saved_score is not None:
            score_record = _RGBScoreRecord.model_validate(saved_score)
            reports.append(score_record.report)
            continue
        saved_generation = existing.get(generation_key)
        if saved_generation is None:
            generation = _generate_rgb_case(
                case,
                generation_key,
                answerer,
                checkpoint,
                retry_policy,
            )
        else:
            generation = _RGBGenerationRecord.model_validate(
                saved_generation
            )
        try:
            reference = _rgb_reference(case)
            judge_score = run_with_retry(
                lambda: judge.score_semantic_correctness(
                    case.query,
                    reference,
                    generation.response,
                ),
                retry_policy,
            )
            report = RGBCaseReport(
                key=_rgb_key(case, None),
                case_id=case.record_id,
                capability=case.capability,
                noise_rate=case.noise_rate,
                status="success",
                response=generation.response,
                passage_hashes=case.passage_hashes,
                official_correct=generation.official_correct,
                judge_semantic_correctness=judge_score.score,
                latency_ms=generation.latency_ms,
            )
            checkpoint.append(
                CheckpointChannel.RGB,
                _RGBScoreRecord(key=score_key, report=report),
            )
            reports.append(report)
        except Exception as exc:
            _append_failure(
                checkpoint,
                CheckpointChannel.RGB,
                score_key,
                exc,
            )
            raise
    return reports


def _generate_rgb_case(
    case: RGBCase,
    key: str,
    answerer: Any,
    checkpoint: RunCheckpoint,
    retry_policy: RetryPolicy,
) -> _RGBGenerationRecord:
    started_at = perf_counter()
    try:
        response = run_with_retry(
            lambda: answerer.generate(case),
            retry_policy,
        )
        latency_ms = max(0.0, (perf_counter() - started_at) * 1000)
        record = _RGBGenerationRecord(
            key=key,
            case_key=case.key,
            response=response,
            official_correct=score_rgb_official(case, response),
            latency_ms=latency_ms,
        )
        checkpoint.append(CheckpointChannel.RGB, record)
        return record
    except Exception as exc:
        _append_failure(
            checkpoint,
            CheckpointChannel.RGB,
            key,
            exc,
        )
        raise


def _make_local_generation(
    key: str,
    case: QACase,
    strategy: RetrievalStrategy,
    result: Any,
    top_k: int,
) -> _LocalGenerationRecord:
    trace = CaseTrace(
        expected_sources=case.expected_sources,
        retrieved=result.retrieved,
        citations=result.citations,
    )
    retrieval_metrics = compute_retrieval_metrics([trace], top_k)
    citation_metrics = compute_citation_metrics([trace])
    first_relevant_rank = None
    if retrieval_metrics.mrr > 0:
        first_relevant_rank = round(1.0 / retrieval_metrics.mrr)
    expected_sources = []
    for source in case.expected_sources:
        expected_sources.append(
            SourceReport(filename=source.filename, page=source.page)
        )
    cited_sources = []
    for citation in result.citations:
        cited_sources.append(
            SourceReport(filename=citation.filename, page=citation.page)
        )
    retrieved = []
    contexts = []
    for item in result.retrieved:
        contexts.append(item.text)
        retrieved.append(
            RetrievedChunkReport(
                rank=item.rank,
                doc_id=item.doc_id,
                filename=item.filename,
                page=item.page,
                chunk_index=item.chunk_index,
                score=item.score,
                text_sha256=item.text_sha256,
            )
        )
    return _LocalGenerationRecord(
        key=key,
        case_id=case.id,
        strategy=strategy,
        answer=result.answer,
        contexts=contexts,
        expected_sources=expected_sources,
        cited_sources=cited_sources,
        retrieved=retrieved,
        hit=retrieval_metrics.hit_rate == 1.0,
        first_relevant_rank=first_relevant_rank,
        citation_precision=citation_metrics.precision,
        citation_recall=citation_metrics.recall,
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
        latency_ms=result.latency_ms,
    )


def _local_report(
    generation: _LocalGenerationRecord,
    judge_score: float,
    ragas_scores: Any,
) -> LocalCaseReport:
    return LocalCaseReport(
        key=_local_key(
            generation.case_id,
            generation.strategy,
            None,
        ),
        case_id=generation.case_id,
        strategy=generation.strategy.value,
        status="success",
        answer=generation.answer,
        expected_sources=generation.expected_sources,
        cited_sources=generation.cited_sources,
        retrieved=generation.retrieved,
        hit=generation.hit,
        first_relevant_rank=generation.first_relevant_rank,
        citation_precision=generation.citation_precision,
        citation_recall=generation.citation_recall,
        judge_semantic_correctness=judge_score,
        answer_correctness=ragas_scores.answer_correctness,
        faithfulness=ragas_scores.faithfulness,
        answer_relevancy=ragas_scores.answer_relevancy,
        retrieval_ms=generation.retrieval_ms,
        generation_ms=generation.generation_ms,
        latency_ms=generation.latency_ms,
    )


def _build_report(
    options: EvaluationOptions,
    settings: Settings,
    rgb_manifest: RGBManifest,
    checkpoint: RunCheckpoint,
    git_commit: str,
    dirty_worktree: bool,
    started_at: datetime,
    finished_at: datetime,
    local_records: list[LocalCaseReport],
    rgb_records: list[RGBCaseReport],
) -> EvaluationReport:
    complete = (
        len(local_records) == 150
        and len(rgb_records) == 150
        and options.local_case_limit is None
        and options.rgb_case_limit is None
        and options.strategies == DEFAULT_STRATEGIES
        and options.top_k == 5
    )
    status: Literal["complete", "preview"] = "preview"
    if complete:
        status = "complete"
    return EvaluationReport(
        status=status,
        run_id=checkpoint.run_id,
        started_at=started_at,
        finished_at=finished_at,
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
        dataset_hashes=DatasetHashes(
            corpus_manifest=sha256_file(options.corpus_manifest_path),
            qa_dataset=sha256_file(options.qa_dataset_path),
            rgb_manifest=sha256_file(options.rgb_manifest_path),
            run_manifest=checkpoint.manifest_hash,
        ),
        models=_model_inventory(settings),
        configuration=_configuration(settings, options.top_k),
        local_records=local_records,
        rgb_records=rgb_records,
        local_aggregates=_local_aggregates(local_records),
        rgb_aggregates=_rgb_aggregates(rgb_records),
        rgb_license=rgb_manifest.license,
        rgb_noncommercial=rgb_manifest.noncommercial,
    )


def _local_aggregates(
    records: list[LocalCaseReport],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[LocalCaseReport]] = {}
    for record in records:
        grouped.setdefault(record.strategy, []).append(record)
    aggregates = {}
    for strategy, values in grouped.items():
        latencies = [record.latency_ms or 0.0 for record in values]
        latency = summarize_latency(latencies)
        aggregates[strategy] = {
            "hit_rate": _mean_bool(values, "hit"),
            "mrr": _mean_reciprocal_rank(values),
            "citation_precision": _mean_optional(
                values,
                "citation_precision",
            ),
            "citation_recall": _mean_optional(
                values,
                "citation_recall",
            ),
            "judge_semantic_correctness": _mean_optional(
                values,
                "judge_semantic_correctness",
            ),
            "answer_correctness": _mean_optional(
                values,
                "answer_correctness",
            ),
            "faithfulness": _mean_optional(values, "faithfulness"),
            "answer_relevancy": _mean_optional(
                values,
                "answer_relevancy",
            ),
            "latency_mean_ms": latency.mean,
            "latency_p50_ms": latency.p50,
            "latency_p95_ms": latency.p95,
        }
    return aggregates


def _rgb_aggregates(
    records: list[RGBCaseReport],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RGBCaseReport]] = {}
    for record in records:
        grouped.setdefault(record.capability, []).append(record)
        if record.capability == "noise_robustness":
            noise_key = f"noise_robustness:noise_{record.noise_rate:g}"
            grouped.setdefault(noise_key, []).append(record)
    aggregates = {}
    for capability, values in grouped.items():
        latencies = [record.latency_ms or 0.0 for record in values]
        latency = summarize_latency(latencies)
        aggregates[capability] = {
            "official_accuracy": _mean_bool(
                values,
                "official_correct",
            ),
            "judge_semantic_correctness": _mean_optional(
                values,
                "judge_semantic_correctness",
            ),
            "latency_mean_ms": latency.mean,
            "latency_p50_ms": latency.p50,
            "latency_p95_ms": latency.p95,
        }
    return aggregates


def _mean_optional(records: list[Any], attribute: str) -> float:
    values = []
    for record in records:
        value = getattr(record, attribute)
        if value is not None:
            values.append(float(value))
    if not values:
        return 0.0
    return statistics.fmean(values)


def _mean_bool(records: list[Any], attribute: str) -> float:
    values = []
    for record in records:
        value = getattr(record, attribute)
        values.append(1.0 if value else 0.0)
    return statistics.fmean(values)


def _mean_reciprocal_rank(records: list[LocalCaseReport]) -> float:
    values = []
    for record in records:
        if record.first_relevant_rank is None:
            values.append(0.0)
        else:
            values.append(1.0 / record.first_relevant_rank)
    return statistics.fmean(values)


def _load_settings(dependencies: EvaluationDependencies) -> Settings:
    loader = dependencies.settings_loader or Settings
    return loader()


def _load_corpus(
    options: EvaluationOptions,
    dependencies: EvaluationDependencies,
) -> CorpusDataset:
    loader = dependencies.corpus_loader or load_corpus
    return loader(
        options.corpus_root,
        options.corpus_manifest_path,
        15,
    )


def _load_qa(
    options: EvaluationOptions,
    dependencies: EvaluationDependencies,
    corpus: CorpusDataset,
) -> list[QACase]:
    loader = dependencies.qa_loader or load_qa_dataset
    return loader(options.qa_dataset_path, corpus, 50)


def _load_rgb_manifest(
    options: EvaluationOptions,
    dependencies: EvaluationDependencies,
) -> RGBManifest:
    loader = dependencies.rgb_manifest_loader or load_rgb_manifest
    return loader(options.rgb_manifest_path)


def _prepare_rgb(
    options: EvaluationOptions,
    dependencies: EvaluationDependencies,
    manifest: RGBManifest,
) -> Any:
    if dependencies.rgb_preparer is not None:
        return dependencies.rgb_preparer(options.rgb_cache_root, manifest)
    return RGBDatasetCache(options.rgb_cache_root).prepare(manifest)


def _build_rgb_cases(
    dependencies: EvaluationDependencies,
    dataset: Any,
    manifest: RGBManifest,
) -> list[RGBCase]:
    builder = dependencies.rgb_cases_builder or build_rgb_cases
    return builder(dataset, manifest)


def _load_git_state(
    dependencies: EvaluationDependencies,
) -> tuple[str, bool]:
    if dependencies.git_state_loader is not None:
        return dependencies.git_state_loader()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _base_manifest(
    options: EvaluationOptions,
    settings: Settings,
    git_commit: str,
    corpus: CorpusDataset,
    qa_cases: list[QACase],
    rgb_manifest: RGBManifest,
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "git_commit": git_commit,
        "dataset_hashes": {
            "corpus_manifest": sha256_file(
                options.corpus_manifest_path
            ),
            "qa_dataset": sha256_file(options.qa_dataset_path),
            "rgb_manifest": sha256_file(options.rgb_manifest_path),
        },
        "corpus_files": sorted(corpus.documents),
        "qa_ids": [case.id for case in qa_cases],
        "rgb_commit": rgb_manifest.commit,
        "models": _model_inventory(settings).model_dump(mode="json"),
        "configuration": _configuration(
            settings,
            options.top_k,
        ).model_dump(mode="json"),
        "strategies": [strategy.value for strategy in options.strategies],
    }


def _rgb_case_manifest(cases: list[RGBCase]) -> list[dict[str, Any]]:
    values = []
    for case in cases:
        values.append(
            {
                "key": case.key,
                "passage_hashes": case.passage_hashes,
            }
        )
    return values


def _model_inventory(settings: Settings) -> ModelInventory:
    reranker_url = settings.reranker_base_url
    if reranker_url is None:
        raise ValueError("Reranker 服务地址未配置")
    judge_url = settings.evaluation_judge_base_url
    judge_model = settings.evaluation_judge_model
    if judge_url is None or judge_model is None:
        raise ValueError("独立裁判配置不完整")
    return ModelInventory(
        generator_model=settings.llm_model,
        generator_origin=_origin(settings.llm_base_url),
        embedding_model=settings.embedding_model,
        embedding_origin=_origin(settings.embedding_base_url),
        reranker_model=settings.reranker_model,
        reranker_origin=_origin(reranker_url),
        judge_model=judge_model,
        judge_origin=_origin(judge_url),
    )


def _configuration(
    settings: Settings,
    top_k: int,
) -> EvaluationConfiguration:
    policy = RetryPolicy()
    return EvaluationConfiguration(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_batch_size=settings.embedding_batch_size,
        top_k=top_k,
        top_k_rerank=settings.top_k_rerank,
        local_temperature=settings.evaluation_local_temperature,
        rgb_temperature=settings.evaluation_rgb_temperature,
        judge_temperature=settings.evaluation_judge_temperature,
        llm_reasoning_effort=settings.llm_reasoning_effort,
        judge_reasoning_effort=(
            settings.evaluation_judge_reasoning_effort
        ),
        retry=RetryPolicySummary(
            max_attempts=policy.max_attempts,
            initial_delay_seconds=policy.initial_delay_seconds,
            multiplier=policy.multiplier,
        ),
    )


def _validate_remote_settings(settings: Settings) -> None:
    missing = []
    if not settings.resolved_llm_api_key:
        missing.append("LLM_API_KEY")
    if not settings.resolved_embedding_api_key:
        missing.append("EMBEDDING_API_KEY")
    if not settings.resolved_reranker_api_key:
        missing.append("RERANKER_API_KEY")
    if not settings.reranker_base_url:
        missing.append("RERANKER_BASE_URL")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"评估远程配置缺失：{joined}")


def _origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    if not parsed.scheme or parsed.hostname is None:
        raise ValueError("模型服务地址必须包含 scheme 和 host")
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    port = parsed.port
    is_default = (
        scheme == "https" and port == 443
    ) or (
        scheme == "http" and port == 80
    )
    if port is None or is_default:
        return f"{scheme}://{hostname}"
    return f"{scheme}://{hostname}:{port}"


def _open_checkpoint(
    options: EvaluationOptions,
    run_id: str,
    manifest: dict[str, Any],
) -> RunCheckpoint:
    if options.resume_run_id is not None:
        return RunCheckpoint.resume(options.work_root, run_id, manifest)
    return RunCheckpoint.create(options.work_root, run_id, manifest)


def _select_run_id(
    options: EvaluationOptions,
    dependencies: EvaluationDependencies,
    git_commit: str,
) -> str:
    if options.resume_run_id is not None:
        return options.resume_run_id
    if options.run_id is not None:
        return options.run_id
    timestamp = _now(dependencies).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{git_commit[:8]}"


def _now(dependencies: EvaluationDependencies) -> datetime:
    provider = dependencies.now
    value = datetime.now(timezone.utc)
    if provider is not None:
        value = provider()
    if value.tzinfo is None:
        raise ValueError("评估时间必须包含时区")
    return value.astimezone(timezone.utc)


def _limit_values(values: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return list(values)
    return list(values[:limit])


def _local_key(
    case_id: str,
    strategy: RetrievalStrategy,
    stage: str | None,
) -> str:
    base = f"local:{case_id}:{strategy.value}"
    if stage is None:
        return base
    return f"{base}:{stage}"


def _rgb_key(case: RGBCase, stage: str | None) -> str:
    base = (
        f"rgb:{case.source}:{case.record_id}:{case.noise_rate:g}"
    )
    if stage is None:
        return base
    return f"{base}:{stage}"


def _rgb_reference(case: RGBCase) -> str:
    components = []
    for component in case.answer:
        if isinstance(component, list):
            components.append(" / ".join(component))
        else:
            components.append(component)
    return "；".join(components)


def _append_failure(
    checkpoint: RunCheckpoint,
    channel: CheckpointChannel,
    key: str,
    error: Exception,
) -> None:
    checkpoint.append(
        channel,
        _FailureRecord(
            key=key,
            error_category=classify_remote_error(error),
        ),
    )


def _sha256_json(payload: dict[str, Any]) -> str:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _load_completed_report(
    checkpoint: RunCheckpoint,
) -> EvaluationReport | None:
    path = checkpoint.run_path / "report.json"
    if not path.exists():
        return None
    try:
        return EvaluationReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("已完成运行报告损坏") from exc


def _write_completed_report(
    checkpoint: RunCheckpoint,
    report: EvaluationReport,
) -> None:
    path = checkpoint.run_path / "report.json"
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_path.open("x", encoding="utf-8") as file_handle:
            file_handle.write(report.model_dump_json(indent=2) + "\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

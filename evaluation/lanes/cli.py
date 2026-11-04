# -*- coding: utf-8 -*-
"""读取.env的一键小样本入口，显式区分生成、评分与不计费预检。"""

import argparse
import json
from pathlib import Path
from typing import cast

from app.application.query.context_budget import PromptBudgetGuard
from app.application.tracing import TraceService
from app.bootstrap.container import ApplicationContainer
from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from app.bootstrap.factories import configured_adapter_factories
from app.bootstrap.factories import create_evaluation_judge_provider
from app.config import Settings
from evaluation.checkpoints import EvaluationRunLock
from evaluation.checkpoints import classify_remote_error
from evaluation.crud_rag import build_crud_sample
from evaluation.crud_rag import fetch_crud_split
from evaluation.datasets import load_corpus
from evaluation.lanes.artifacts import write_lane_report
from evaluation.lanes.checkpoints import LaneRunManifest
from evaluation.lanes.checkpoints import LaneRunStore
from evaluation.lanes.dataset import LaneDataset
from evaluation.lanes.dataset import load_lane_dataset
from evaluation.lanes.judge import LaneJudge
from evaluation.lanes.manifest import build_lane_manifest
from evaluation.lanes.run import EvaluationPhase
from evaluation.lanes.run import execute_lane_runs
from evaluation.lanes.runtime import LaneEpisodeRunner
from evaluation.models import CorpusDataset
from evaluation.runtime import build_evaluation_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None, *, settings: Settings | None = None) -> int:
    """默认执行13轮烟雾；可直接被Python入口调用，不要求手工拼参数。

    Args:
        argv: 可选CLI参数，None读取当前进程参数。
        settings: Python调用可传冻结配置副本；None正常读取项目.env。

    Returns:
        完整成功0，不完整1，配置／执行入口失败2，用户中断130。
    """
    parser = argparse.ArgumentParser(description="分通道RAG小样本与可恢复独立评分")
    parser.add_argument("--dry-run", action="store_true", help="只预检，不计费、不写入")
    parser.add_argument("--full", action="store_true", help="全部固定能力片段，非150条旧评估")
    parser.add_argument("--phase", choices=("all", "generate", "score"), default="all")
    parser.add_argument("--resume", action="store_true", help="允许恢复确认已失效的进程锁")
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT / "evaluation/work")
    parser.add_argument(
        "--benchmark", choices=("builtin", "crud-rag"), default="builtin",
        help="builtin=固定能力片段；crud-rag=CRUD-RAG问答分层小批量（首次需下载约27MB）",
    )
    parser.add_argument("--crud-per-tier", type=int, default=15, help="单篇/两篇/三篇每档题数")
    parser.add_argument("--crud-distractors", type=int, default=200, help="额外放入索引的干扰新闻篇数")
    parser.add_argument(
        "--policies", nargs="+", choices=("auto", "fast_only", "agent"),
        help="只评估这些策略，默认三种都跑；大样本可只跑 auto",
    )
    parser.add_argument(
        "--crud-cache", type=Path, default=PROJECT_ROOT / "evaluation/cache/crud_rag",
    )
    options = parser.parse_args(argv)
    try:
        config = settings if settings is not None else Settings()
        if options.benchmark == "crud-rag":
            # 抽样本身就是本次的完整批次，按能力评估而非连通性烟雾统计。
            corpus, dataset = build_crud_sample(
                fetch_crud_split(options.crud_cache),
                per_tier=options.crud_per_tier, distractors=options.crud_distractors,
                seed=config.evaluation_random_seed,
            )
            smoke = False
        else:
            corpus = load_corpus(
                PROJECT_ROOT / "evaluation/corpus",
                PROJECT_ROOT / "evaluation/corpus_manifest.json",
            )
            dataset = load_lane_dataset(PROJECT_ROOT / "evaluation/lane_dataset.json", corpus)
            smoke = not options.full
        manifest = build_lane_manifest(
            config, dataset, corpus, PROJECT_ROOT, smoke=smoke,
            policies=tuple(dict.fromkeys(options.policies)) if options.policies else None,
        )
        if config.evaluation_judge_max_output_tokens >= config.conversation_context_tokens:
            raise ValueError("独立裁判输出预留必须小于配置的完整上下文预算")
        catalog = {episode.id: episode for episode in dataset.episodes}
        planned_turns = sum(
            len(catalog[episode_id].turns)
            for episode_id, _ in manifest.selections
        )
        print(json.dumps({
            "benchmark": options.benchmark,
            "purpose": "connectivity_smoke" if smoke else "capability_evaluation",
            "corpus_documents": len(corpus.entries),
            "strategy": manifest.strategy.value,
            "planned_episodes": len(manifest.selections), "planned_turns": planned_turns,
            "fingerprint": manifest.fingerprint(),
        }, ensure_ascii=False), flush=True)
        if options.dry_run:
            return 0
        run_id = f"lanes-{manifest.fingerprint()[:24]}"
        destination = options.work_root / "lanes" / run_id
        # 与旧评估入口共用work根锁，防止默认配置同时写同一个评估索引。
        with EvaluationRunLock.acquire(
            options.work_root / "evaluation.lock", resume=options.resume,
            run_id=run_id, manifest_hash=manifest.fingerprint(),
        ):
            store = LaneRunStore(destination / "records.sqlite3", manifest)
            return _execute(
                config, corpus, dataset, manifest, store, destination,
                phase=cast(EvaluationPhase, options.phase), smoke=smoke,
            )
    except KeyboardInterrupt:
        print("已中断；已持久化答案和分数保留，使用同一配置续跑。", flush=True)
        return 130
    except Exception as error:
        # CLI是公开输出边界；SDK异常可能含地址、请求正文或响应正文。
        print(f"评估未启动或未完成：{classify_remote_error(error).value}", flush=True)
        return 2


def _execute(
    settings: Settings, corpus: CorpusDataset, dataset: LaneDataset,
    manifest: LaneRunManifest, store: LaneRunStore, destination: Path,
    *, phase: EvaluationPhase, smoke: bool,
) -> int:
    catalog = {episode.id: episode for episode in dataset.episodes}
    requires_generation = False
    if phase != "score":
        for episode_id, policy in manifest.selections:
            if len(store.load(episode_id, policy)) < len(catalog[episode_id].turns):
                requires_generation = True
                break
    if requires_generation:
        _progress("准备或复用隔离固定语料索引；不会重建匹配指纹的Embedding。")
        runtime = build_evaluation_runtime(settings, corpus)
        root = cast(ApplicationContainer, runtime.container)
    else:
        # 评分／完全复用不需要打开Chroma，更不能为了评分重新向量化语料。
        root = build_application_container(settings, RuntimeOptions(
            chroma_path=settings.evaluation_chroma_path,
            collection_name=(
                f"{settings.evaluation_collection_prefix}_"
                f"{manifest.index_fingerprint[:12]}"
            ),
            llm_temperature=settings.evaluation_local_temperature,
        ))
    provider = None
    try:
        judge = None
        if phase != "generate":
            provider = create_evaluation_judge_provider(settings)
            traces = TraceService(
                root.trace_repository,
                capture_queries=settings.trace_capture_queries,
                retention_days=settings.trace_retention_days,
                prices=settings.trace_model_prices,
                configuration_version=f"evaluation-judge-v1:{manifest.judge_configuration_sha256}",
            )
            guard = PromptBudgetGuard(
                configured_adapter_factories(settings).token_counter(),
                settings.conversation_context_tokens - settings.evaluation_judge_max_output_tokens,
            )
            judge = LaneJudge(provider, traces, root.trace_repository, guard)
        runner = LaneEpisodeRunner(
            root.query_orchestrator, root.history_window,
            root.trace_repository, root.cleanup_temporary_execution,
        )
        report = execute_lane_runs(
            manifest, dataset, runner, judge, store,
            smoke=smoke, phase=phase, on_progress=_progress,
        )
        write_lane_report(report, destination)
        _progress(
            f"{report.status}：计划{report.expected_turns}，生成{report.generated_turns}，"
            f"评分{report.scored_turns}。报告：{destination / 'report.json'}",
        )
        return 0 if report.status == "complete" else 1
    finally:
        try:
            if provider is not None:
                provider.close()
        finally:
            root.close()


def _progress(message: str) -> None:
    print(message, flush=True)

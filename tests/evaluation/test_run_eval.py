# -*- coding: utf-8 -*-
"""Day14 评估 CLI 编排、恢复与发布边界测试。"""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.domain.answers import Citation
from app.domain.answers import EvaluationQueryResult
from app.domain.answers import RetrievedChunkTrace
from app.domain.retrieval import RetrievalStrategy
from evaluation.judge import JudgeScore
from evaluation.ragas_eval import RagasScores
from evaluation.rgb import RGBCase


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def evaluate_query(
        self,
        question: str,
        strategy: RetrievalStrategy | str,
        top_k: int,
    ) -> EvaluationQueryResult:
        """返回可计算本地指标的固定查询追踪。"""
        selected = RetrievalStrategy(strategy)
        self.calls.append((question, selected.value, top_k))
        text = f"context for {question}"
        filename = "langchain_quickstart.md"
        citation = Citation(
            doc_id="doc-1",
            filename=filename,
            page=None,
            chunk_text=text,
        )
        retrieved = RetrievedChunkTrace(
            rank=1,
            chunk_id="chunk-1",
            doc_id="doc-1",
            text=text,
            text_sha256="a" * 64,
            filename=filename,
            page=None,
            chunk_index=0,
            score=0.9,
        )
        return EvaluationQueryResult(
            answer="固定回答 [1]",
            citations=[citation],
            retrieved=[retrieved],
            retrieval_ms=10.0,
            generation_ms=20.0,
            latency_ms=30.0,
            retrieval_strategy=selected,
        )


class _FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def score_semantic_correctness(
        self,
        question: str,
        reference: str,
        response: str,
    ) -> JudgeScore:
        """记录调用并返回固定独立裁判分数。"""
        del question
        del reference
        del response
        self.calls += 1
        return JudgeScore(score=0.9, reason="ok")


class _FakeRagas:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, case: object) -> RagasScores:
        """记录调用并返回三项固定 RAGAS 分数。"""
        del case
        self.calls += 1
        return RagasScores(
            faithfulness=0.8,
            answer_relevancy_raw=0.7,
            answer_relevancy=0.7,
            answer_correctness=0.75,
        )


class _FakeRGBAnswerer:
    def __init__(self) -> None:
        self.calls = 0
        self.close_calls = 0

    def generate(self, case: RGBCase) -> str:
        """返回能通过固定官方规则的回答。"""
        self.calls += 1
        if case.capability == "rejection":
            return "信息不足"
        return "答案"

    def close(self) -> None:
        """记录编排器释放本次运行创建的回答器。

        Returns:
            None。
        """
        self.close_calls += 1


def _settings(tmp_path: Path, isolated: bool = True) -> Settings:
    chroma_path = tmp_path / "production"
    evaluation_chroma_path = tmp_path / "evaluation"
    if not isolated:
        evaluation_chroma_path = chroma_path
    return Settings(
        _env_file=None,
        llm_api_key="generator-key",
        llm_base_url="https://generator.example/v1",
        llm_model="generator-model",
        llm_reasoning_effort="none",
        embedding_api_key="embedding-key",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
        reranker_api_key="reranker-key",
        reranker_base_url="https://reranker.example/v1",
        reranker_model="reranker-model",
        evaluation_judge_api_key="judge-key",
        evaluation_judge_base_url="https://judge.example/v1",
        evaluation_judge_model="judge-model",
        evaluation_judge_reasoning_effort="none",
        chroma_path=chroma_path,
        evaluation_chroma_path=evaluation_chroma_path,
    )


def _rgb_cases() -> list[RGBCase]:
    cases = []
    capabilities = [
        "noise_robustness",
        "noise_robustness",
        "rejection",
        "multi_document_integration",
    ]
    for index, capability in enumerate(capabilities):
        source = "zh_int"
        if capability != "multi_document_integration":
            source = "zh_refine"
        noise_rate = 1.0 if capability == "rejection" else 0.4
        passages = [f"passage-{index}-{part}" for part in range(5)]
        passage_hashes = [f"{index + part:064x}" for part in range(5)]
        cases.append(
            RGBCase(
                key=f"{source}:{index}:{noise_rate:g}",
                source=source,
                record_id=index,
                query=f"rgb question {index}",
                answer=["答案"],
                passages=passages,
                passage_hashes=passage_hashes,
                noise_rate=noise_rate,
                capability=capability,
            )
        )
    return cases


def test_rgb_answerer_forwards_generator_reasoning_effort(
    tmp_path: Path,
) -> None:
    """RGB 生成器必须使用与本地评估相同的显式推理强度。

    Args:
        tmp_path: 构造隔离评估配置使用的临时目录。
    """
    from app.bootstrap.factories import AdapterFactories
    from evaluation import run_eval

    configurations: list[dict[str, object]] = []

    def fake_provider(
        api_key: str | None,
        base_url: str,
        model: str,
        temperature: float | None,
        reasoning_effort: str | None,
    ) -> object:
        configurations.append(
            {
                "api_key": api_key,
                "base_url": base_url,
                "model": model,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            }
        )
        return object()

    factories = AdapterFactories(llm_provider=fake_provider)
    settings = _settings(tmp_path).model_copy(
        update={"llm_reasoning_effort": "none"}
    )

    run_eval.RGBAnswerer.from_settings(settings, object(), factories)

    assert configurations == [
        {
            "api_key": "generator-key",
            "base_url": "https://generator.example/v1",
            "model": "generator-model",
            "temperature": 0.2,
            "reasoning_effort": "none",
        }
    ]


def _options(tmp_path: Path):
    from evaluation.run_eval import EvaluationOptions

    return EvaluationOptions(
        work_root=tmp_path / "work",
        results_root=tmp_path / "results",
        rgb_cache_root=tmp_path / "rgb-cache",
        local_case_limit=2,
        strategies=(
            RetrievalStrategy.VECTOR,
            RetrievalStrategy.HYBRID,
        ),
        rgb_case_limit=4,
        publish=False,
        run_id="20260820T120000Z-deadbeef",
    )


def test_preflight_fails_before_datasets_or_remote_factories(
    tmp_path: Path,
) -> None:
    """路径隔离无效时不得读取数据或构造远程客户端。"""
    from evaluation.run_eval import EvaluationDependencies, run_evaluation

    calls: list[str] = []

    def unexpected(*args: object, **kwargs: object) -> object:
        calls.append("unexpected")
        raise AssertionError("预检失败后不应继续")

    dependencies = EvaluationDependencies(
        settings_loader=lambda: _settings(tmp_path, isolated=False),
        corpus_loader=unexpected,
        qa_loader=unexpected,
        rgb_manifest_loader=unexpected,
        runtime_builder=unexpected,
        judge_builder=unexpected,
        ragas_builder=unexpected,
        rgb_preparer=unexpected,
        rgb_cases_builder=unexpected,
        rgb_answerer_builder=unexpected,
    )

    with pytest.raises(ValueError, match="路径必须与生产路径隔离"):
        run_evaluation(_options(tmp_path), dependencies)

    assert calls == []


def test_small_offline_run_checkpoints_and_resume_skip_successes(
    tmp_path: Path,
) -> None:
    """小样本运行恢复时不得重复已经成功的调用。"""
    from evaluation.run_eval import EvaluationDependencies, run_evaluation

    pipeline = _FakePipeline()
    runtime = SimpleNamespace(query_service=pipeline, close=lambda: None)
    judge = _FakeJudge()
    ragas = _FakeRagas()
    rgb_answerer = _FakeRGBAnswerer()
    dependencies = EvaluationDependencies(
        settings_loader=lambda: _settings(tmp_path),
        runtime_builder=lambda settings, corpus, rebuild: runtime,
        judge_builder=lambda settings: judge,
        ragas_builder=lambda settings: ragas,
        rgb_preparer=lambda cache_root, manifest: object(),
        rgb_cases_builder=lambda dataset, manifest: _rgb_cases(),
        rgb_answerer_builder=lambda settings, dataset: rgb_answerer,
        git_state_loader=lambda: ("a" * 40, True),
        now=lambda: datetime(2026, 8, 20, 12, tzinfo=timezone.utc),
    )

    report = run_evaluation(_options(tmp_path), dependencies)

    assert report.status == "preview"
    assert len(report.local_records) == 4
    assert len(report.rgb_records) == 4
    assert len(pipeline.calls) == 4
    assert judge.calls == 8
    assert ragas.calls == 4
    assert rgb_answerer.calls == 4
    assert rgb_answerer.close_calls == 1
    assert all(call[2] == 5 for call in pipeline.calls)
    assert report.local_records[0].judge_semantic_correctness == 0.9
    assert report.local_records[0].answer_correctness == 0.75
    assert report.configuration.llm_reasoning_effort == "none"
    assert report.configuration.judge_reasoning_effort == "none"

    run_path = _options(tmp_path).work_root / report.run_id
    local_lines = (run_path / "local.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    ragas_lines = (run_path / "ragas.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    rgb_lines = (run_path / "rgb.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(local_lines) == 4
    assert len(ragas_lines) == 4
    assert len(rgb_lines) == 8
    assert json.loads(local_lines[0])["key"].endswith(":generation")
    assert json.loads(ragas_lines[0])["key"].endswith(":ragas")
    assert json.loads(rgb_lines[0])["key"].endswith(":generation")
    assert json.loads(rgb_lines[1])["key"].endswith(":judge")

    resumed_options = _options(tmp_path)
    resumed_options = resumed_options.model_copy(
        update={"resume_run_id": report.run_id}
    )
    resumed = run_evaluation(resumed_options, dependencies)

    assert resumed.model_dump() == report.model_dump()
    assert len(pipeline.calls) == 4
    assert judge.calls == 8
    assert ragas.calls == 4
    assert rgb_answerer.calls == 4
    # 完成报告可直接恢复，不会重新创建或重复关闭回答器。
    assert rgb_answerer.close_calls == 1


def test_rgb_answerer_closes_owned_provider_once(tmp_path: Path) -> None:
    """RGB 回答器应幂等关闭由 factory 创建的生成 provider。

    Args:
        tmp_path: 构造隔离评估配置使用的临时目录。

    Returns:
        None。
    """
    from app.bootstrap.factories import AdapterFactories
    from evaluation import run_eval

    provider = SimpleNamespace(close_calls=0)

    def close() -> None:
        provider.close_calls += 1

    provider.close = close
    factories = AdapterFactories(
        llm_provider=lambda *_args: provider,
    )
    answerer = run_eval.RGBAnswerer.from_settings(
        _settings(tmp_path),
        object(),
        factories,
    )

    answerer.close()
    answerer.close()

    assert provider.close_calls == 1


def test_rgb_answerer_closes_after_generation_failure(
    tmp_path: Path,
) -> None:
    """RGB 生成失败时编排器仍应释放本次运行的回答器。

    Args:
        tmp_path: 构造隔离评估运行使用的临时目录。

    Returns:
        None。
    """
    from evaluation.run_eval import EvaluationDependencies
    from evaluation.run_eval import run_evaluation

    class FailingRGBAnswerer(_FakeRGBAnswerer):
        """首次生成时抛出异常的回答器。"""

        def generate(self, case: RGBCase) -> str:
            """模拟 RGB 上游生成失败。

            Args:
                case: 当前 RGB case。

            Returns:
                永远不会返回。
            """
            del case
            raise RuntimeError("rgb generation failed")

    answerer = FailingRGBAnswerer()
    runtime = SimpleNamespace(
        query_service=_FakePipeline(),
        close=lambda: None,
    )
    dependencies = EvaluationDependencies(
        settings_loader=lambda: _settings(tmp_path),
        runtime_builder=lambda settings, corpus, rebuild: runtime,
        judge_builder=lambda settings: _FakeJudge(),
        ragas_builder=lambda settings: _FakeRagas(),
        rgb_preparer=lambda cache_root, manifest: object(),
        rgb_cases_builder=lambda dataset, manifest: _rgb_cases(),
        rgb_answerer_builder=lambda settings, dataset: answerer,
        git_state_loader=lambda: ("a" * 40, True),
    )

    with pytest.raises(RuntimeError, match="rgb generation failed"):
        run_evaluation(_options(tmp_path), dependencies)

    assert answerer.close_calls == 1


def test_rgb_answerer_closes_when_judge_construction_fails(
    tmp_path: Path,
) -> None:
    """裁判构造失败时仍应释放已经创建的 RGB 回答器。

    Args:
        tmp_path: 构造隔离评估运行使用的临时目录。

    Returns:
        None。
    """
    from evaluation.run_eval import EvaluationDependencies
    from evaluation.run_eval import run_evaluation

    answerer = _FakeRGBAnswerer()
    runtime = SimpleNamespace(
        query_service=_FakePipeline(),
        close=lambda: None,
    )
    judge_builder_calls = 0

    def failing_judge_builder(settings: Settings) -> object:
        """先服务本地评分，再模拟 RGB 裁判构造失败。

        Args:
            settings: 当前评估配置。

        Returns:
            永远不会返回。
        """
        nonlocal judge_builder_calls
        del settings
        judge_builder_calls += 1
        if judge_builder_calls == 1:
            return _FakeJudge()
        raise RuntimeError("judge construction failed")

    dependencies = EvaluationDependencies(
        settings_loader=lambda: _settings(tmp_path),
        runtime_builder=lambda settings, corpus, rebuild: runtime,
        judge_builder=failing_judge_builder,
        ragas_builder=lambda settings: _FakeRagas(),
        rgb_preparer=lambda cache_root, manifest: object(),
        rgb_cases_builder=lambda dataset, manifest: _rgb_cases(),
        rgb_answerer_builder=lambda settings, dataset: answerer,
        git_state_loader=lambda: ("a" * 40, True),
    )

    with pytest.raises(RuntimeError, match="judge construction failed"):
        run_evaluation(_options(tmp_path), dependencies)

    assert answerer.close_calls == 1


def test_failed_run_preserves_existing_latest(tmp_path: Path) -> None:
    """运行失败必须抛出且完全保留上次正式报告。"""
    from evaluation.run_eval import EvaluationDependencies, run_evaluation

    results_root = tmp_path / "results"
    results_root.mkdir()
    old_json = b'{"old":true}\n'
    old_markdown = b"# old\n"
    (results_root / "latest.json").write_bytes(old_json)
    (results_root / "latest.md").write_bytes(old_markdown)

    class FailingPipeline(_FakePipeline):
        def evaluate_query(
            self,
            question: str,
            strategy: RetrievalStrategy | str,
            top_k: int,
        ) -> EvaluationQueryResult:
            """模拟首次生成失败。"""
            del question
            del strategy
            del top_k
            raise RuntimeError("private upstream payload")

    runtime = SimpleNamespace(
        query_service=FailingPipeline(),
        close=lambda: None,
    )
    dependencies = EvaluationDependencies(
        settings_loader=lambda: _settings(tmp_path),
        runtime_builder=lambda settings, corpus, rebuild: runtime,
        judge_builder=lambda settings: _FakeJudge(),
        ragas_builder=lambda settings: _FakeRagas(),
        rgb_preparer=lambda cache_root, manifest: object(),
        rgb_cases_builder=lambda dataset, manifest: _rgb_cases(),
        rgb_answerer_builder=lambda settings, dataset: _FakeRGBAnswerer(),
        git_state_loader=lambda: ("a" * 40, True),
    )
    options = _options(tmp_path).model_copy(
        update={"results_root": results_root}
    )

    with pytest.raises(RuntimeError, match="private upstream payload"):
        run_evaluation(options, dependencies)

    assert (results_root / "latest.json").read_bytes() == old_json
    assert (results_root / "latest.md").read_bytes() == old_markdown


def test_cli_exposes_only_full_resume_and_rebuild_flags() -> None:
    """生产 CLI 不得暴露缩减正式样本数量的隐藏参数。"""
    from evaluation.run_eval import build_parser

    parser = build_parser()
    options = parser.parse_args(["--full", "--resume", "run-1"])

    assert options.full is True
    assert options.resume == "run-1"
    with pytest.raises(SystemExit):
        parser.parse_args(["--full", "--local-case-limit", "2"])


def test_cli_prints_run_id_and_resume_command_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """长任务开始前必须给出异常退出后可复制的恢复命令。

    Args:
        monkeypatch: pytest 属性替换工具。
        capsys: pytest 标准输出捕获工具。
    """
    from evaluation import run_eval

    received = []

    def fake_run(options, dependencies=None):
        received.append(options)
        return SimpleNamespace(
            status="preview",
            run_id=options.run_id,
            local_records=[],
            rgb_records=[],
        )

    monkeypatch.setattr(run_eval, "run_evaluation", fake_run)
    monkeypatch.setattr(
        run_eval,
        "_load_git_state",
        lambda dependencies: ("a" * 40, True),
    )

    exit_code = run_eval.main(["--full"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert received[0].run_id is not None
    assert f"Run ID: {received[0].run_id}" in output
    assert f"--resume {received[0].run_id}" in output

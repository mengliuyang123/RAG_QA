# -*- coding: utf-8 -*-
"""完整评估报告校验与原子发布测试。"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _report(status: str, local_count: int, rgb_count: int):
    from evaluation.reporting import (
        DatasetHashes,
        EvaluationConfiguration,
        EvaluationReport,
        LocalCaseReport,
        ModelInventory,
        RGBCaseReport,
        RetryPolicySummary,
    )

    local_records = []
    for index in range(local_count):
        local_records.append(
            LocalCaseReport(
                key=f"local-{index:03d}:vector",
                case_id=f"local-{index:03d}",
                strategy="vector",
                status="success",
                hit=True,
                first_relevant_rank=1,
                citation_precision=1.0,
                citation_recall=1.0,
                judge_semantic_correctness=0.95,
                answer_correctness=0.9,
                faithfulness=0.8,
                answer_relevancy=0.7,
                retrieval_ms=10.0,
                generation_ms=20.0,
                latency_ms=30.0,
            )
        )
    rgb_records = []
    for index in range(rgb_count):
        rgb_records.append(
            RGBCaseReport(
                key=f"rgb-{index:03d}",
                case_id=index,
                capability="noise_robustness",
                noise_rate=0.4,
                status="success",
                passage_hashes=["a" * 64] * 5,
                official_correct=True,
                judge_semantic_correctness=0.8,
                latency_ms=25.0,
            )
        )
    return EvaluationReport(
        status=status,
        run_id="20260820T120000Z-deadbeef",
        started_at=datetime(2026, 8, 20, 12, tzinfo=timezone.utc),
        finished_at=datetime(2026, 8, 20, 13, tzinfo=timezone.utc),
        git_commit="a" * 40,
        dirty_worktree=True,
        dataset_hashes=DatasetHashes(
            corpus_manifest="a" * 64,
            qa_dataset="b" * 64,
            rgb_manifest="c" * 64,
            run_manifest="d" * 64,
        ),
        models=ModelInventory(
            generator_model="generator-model",
            generator_origin="https://generator.example",
            embedding_model="embedding-model",
            embedding_origin="https://embedding.example",
            reranker_model="reranker-model",
            reranker_origin="https://reranker.example",
            judge_model="judge-model",
            judge_origin="https://judge.example",
        ),
        configuration=EvaluationConfiguration(
            chunk_size=500,
            chunk_overlap=50,
            embedding_batch_size=100,
            top_k=5,
            top_k_rerank=20,
            local_temperature=0.0,
            rgb_temperature=0.2,
            judge_temperature=0.0,
            retry=RetryPolicySummary(
                max_attempts=3,
                initial_delay_seconds=1.0,
                multiplier=2.0,
            ),
        ),
        local_records=local_records,
        rgb_records=rgb_records,
        local_aggregates={"vector": {"hit_rate": 1.0}},
        rgb_aggregates={"noise_robustness": {"official_accuracy": 1.0}},
        rgb_license="CC BY-NC-SA 4.0",
        rgb_noncommercial=True,
    )


def _seed_latest(results_root: Path) -> tuple[bytes, bytes]:
    results_root.mkdir(parents=True, exist_ok=True)
    old_json = b'{"status":"old"}\n'
    old_markdown = b"# old report\n"
    (results_root / "latest.json").write_bytes(old_json)
    (results_root / "latest.md").write_bytes(old_markdown)
    return old_json, old_markdown


def test_failed_report_does_not_replace_latest(tmp_path: Path) -> None:
    """失败运行必须完整保留上一次成功证据。"""
    from evaluation.reporting import publish_report

    old_json, old_markdown = _seed_latest(tmp_path)
    report = _report("failed", 1, 1)

    with pytest.raises(ValueError, match="只有完整成功报告"):
        publish_report(report, tmp_path)

    assert (tmp_path / "latest.json").read_bytes() == old_json
    assert (tmp_path / "latest.md").read_bytes() == old_markdown


def test_preview_report_preserves_separate_judge_score_and_cannot_publish(
    tmp_path: Path,
) -> None:
    """小样本预览应保留两类正确性分数且不得更新 latest。"""
    from evaluation.reporting import publish_report

    old_json, old_markdown = _seed_latest(tmp_path)
    report = _report("preview", 1, 1)

    assert report.local_records[0].judge_semantic_correctness == 0.95
    assert report.local_records[0].answer_correctness == 0.9
    with pytest.raises(ValueError, match="只有完整成功报告"):
        publish_report(report, tmp_path)

    assert (tmp_path / "latest.json").read_bytes() == old_json
    assert (tmp_path / "latest.md").read_bytes() == old_markdown


def test_incomplete_complete_report_does_not_replace_latest(
    tmp_path: Path,
) -> None:
    """绕过构造校验的不完整 complete 对象也不得发布。"""
    from evaluation.reporting import EvaluationReport, publish_report

    old_json, old_markdown = _seed_latest(tmp_path)
    failed = _report("failed", 1, 1)
    payload = failed.model_dump()
    payload["status"] = "complete"
    invalid = EvaluationReport.model_construct(
        **payload,
    )

    with pytest.raises(ValueError, match="报告校验失败"):
        publish_report(invalid, tmp_path)

    assert (tmp_path / "latest.json").read_bytes() == old_json
    assert (tmp_path / "latest.md").read_bytes() == old_markdown


def test_complete_report_archives_and_atomically_replaces_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整报告应归档，并从目标目录内的临时文件原子替换。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest 属性替换工具。
    """
    from evaluation import reporting

    _seed_latest(tmp_path)
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        replacements.append((source_path, target_path))
        assert source_path.parent == target_path.parent
        real_replace(source_path, target_path)

    monkeypatch.setattr(reporting.os, "replace", recording_replace)

    reporting.publish_report(_report("complete", 150, 150), tmp_path)

    latest_json = (tmp_path / "latest.json").read_text(encoding="utf-8")
    latest_markdown = (tmp_path / "latest.md").read_text(encoding="utf-8")
    assert '"status": "complete"' in latest_json
    assert "# RAG 评估报告" in latest_markdown
    assert len(replacements) == 4
    assert (tmp_path / "runs" / "20260820T120000Z-deadbeef.json").is_file()
    assert (tmp_path / "runs" / "20260820T120000Z-deadbeef.md").is_file()


@pytest.mark.parametrize(
    "content",
    [
        "# README\n",
        (
            "<!-- EVALUATION_RESULTS_START -->\n"
            "<!-- EVALUATION_RESULTS_START -->\n"
            "<!-- EVALUATION_RESULTS_END -->\n"
        ),
        (
            "<!-- EVALUATION_RESULTS_END -->\n"
            "<!-- EVALUATION_RESULTS_START -->\n"
        ),
    ],
)
def test_readme_metrics_rejects_missing_duplicate_or_reversed_markers(
    tmp_path: Path,
    content: str,
) -> None:
    """指标 updater 只接受唯一且顺序正确的标记对。"""
    from evaluation.reporting import update_readme_metrics

    readme_path = tmp_path / "README.md"
    readme_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="README 指标标记"):
        update_readme_metrics(
            readme_path,
            _report("complete", 150, 150),
        )


def test_readme_metrics_rejects_failed_report_and_preserves_file(
    tmp_path: Path,
) -> None:
    """失败报告不得改变已有 README 指标块。"""
    from evaluation.reporting import update_readme_metrics

    readme_path = tmp_path / "README.md"
    original = (
        "before\n"
        "<!-- EVALUATION_RESULTS_START -->\n"
        "old metrics\n"
        "<!-- EVALUATION_RESULTS_END -->\n"
        "after\n"
    )
    readme_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="完整成功报告"):
        update_readme_metrics(readme_path, _report("failed", 1, 1))

    assert readme_path.read_text(encoding="utf-8") == original


def test_readme_metrics_atomically_updates_only_controlled_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整报告应原子更新指标块，保持外部文本且重复运行幂等。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest 属性替换工具。
    """
    from evaluation import reporting

    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "before\n"
        "<!-- EVALUATION_RESULTS_START -->\n"
        "old metrics\n"
        "<!-- EVALUATION_RESULTS_END -->\n"
        "after\n",
        encoding="utf-8",
    )
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        replacements.append((source_path, target_path))
        assert source_path.parent == target_path.parent
        real_replace(source_path, target_path)

    monkeypatch.setattr(reporting.os, "replace", recording_replace)
    report = _report("complete", 150, 150)

    reporting.update_readme_metrics(readme_path, report)
    first = readme_path.read_text(encoding="utf-8")
    reporting.update_readme_metrics(readme_path, report)
    second = readme_path.read_text(encoding="utf-8")

    assert first == second
    assert first.startswith("before\n")
    assert first.endswith("after\n")
    assert "| 策略 | Hit Rate@5 | MRR |" in first
    assert "| 目标阈值 | 实际结果 |" in first
    assert "evaluation/results/latest.json" in first
    assert len(replacements) == 2

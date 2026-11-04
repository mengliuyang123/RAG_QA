# -*- coding: utf-8 -*-
"""分通道记录须持久化复用，配置变化不得误用成功缓存。"""

from pathlib import Path
from uuid import uuid4

import pytest

from app.domain.retrieval import RetrievalStrategy
from evaluation.lanes.models import LaneEpisode
from evaluation.lanes.models import LaneTurn
from evaluation.lanes.models import LaneJudgement
from evaluation.lanes.models import LaneScoreRecord
from evaluation.lanes.runtime import LaneEpisodeRunner
from tests.test_unified_query_api import _app


def test_reopened_store_restores_answers_without_internal_queries(
    tmp_path: Path,
) -> None:
    """只存在内存缓存、数据库重开丢结果或复制执行种子都会破坏断点复用。

    Args:
        tmp_path: 隔离运行目录。
    """
    from evaluation.lanes.checkpoints import LaneRunManifest, LaneRunStore

    app, _, _ = _app(tmp_path)
    root = app.state.container
    try:
        episode = LaneEpisode(id="checkpoint", tags=("single_hop",), turns=(
            LaneTurn(question="needle", reference="SQLite", expected_lane="fast"),
        ))
        record = LaneEpisodeRunner(
            root.query_orchestrator, root.history_window,
            root.trace_repository, root.cleanup_temporary_execution,
        ).run(episode, "auto", RetrievalStrategy.HYBRID)[0]
        manifest = LaneRunManifest(
            dataset_sha256="1" * 64, index_fingerprint="2" * 64,
            configuration_version="configuration-v1", code_sha256="3" * 64,
            judge_configuration_sha256="4" * 64,
            selections=(("checkpoint", "auto"),),
            strategy=RetrievalStrategy.HYBRID,
        )
        path = tmp_path / "run" / "lanes.sqlite3"
        LaneRunStore(path, manifest).save(record)
        reopened = LaneRunStore(path, manifest)
        cached = reopened.load("checkpoint", "auto")
        assert len(cached) == 1
        assert cached[0].result.answer == "资料[1]"
        assert cached[0].trace.trace_id == record.trace.trace_id
        assert cached[0].result.execution is None
        assert cached[0].query_hashes == record.query_hashes
        reopened.save(record)
        assert len(reopened.load("checkpoint", "auto")) == 1
        with root.trace_service.start(
            "needle", request_id=uuid4(), entry="evaluation",
        ) as scoring:
            pass
        score = LaneScoreRecord(
            episode_id="checkpoint", turn_index=0, policy="auto",
            strategy=RetrievalStrategy.HYBRID,
            source_trace_id=record.trace.trace_id,
            judgement=LaneJudgement(correctness=0.4, supported=True,
                                    complete=False, reason="回答不完整"),
            trace=root.trace_repository.get(scoring.id),
        )
        assert reopened.load_score("checkpoint", "auto", 0) is None
        reopened.save_score(score)
        reopened.save_score(score)
        restored_score = LaneRunStore(path, manifest).load_score("checkpoint", "auto", 0)
        assert restored_score.source_trace_id == record.trace.trace_id
        assert restored_score.judgement.correctness == 0.4
        with pytest.raises(ValueError, match="对应"):
            reopened.save_score(score.model_copy(update={"source_trace_id": uuid4()}))
        with pytest.raises(ValueError, match="冲突"):
            reopened.save_score(score.model_copy(update={
                "judgement": score.judgement.model_copy(update={"correctness": 0.9}),
            }))
        with pytest.raises(ValueError, match="冲突"):
            reopened.save(record.model_copy(update={"latency_ms": record.latency_ms + 1}))
        for key, value in (
            ("dataset_sha256", "5" * 64),
            ("configuration_version", "new-model-config"),
            ("index_fingerprint", "6" * 64),
            ("code_sha256", "7" * 64),
            ("judge_configuration_sha256", "8" * 64),
        ):
            changed = manifest.model_copy(update={key: value})
            with pytest.raises(ValueError, match="指纹"):
                LaneRunStore(path, changed)
        assert reopened.load("checkpoint", "agent") == ()
        expired = LaneRunStore(
            path, manifest, clock=lambda: record.trace.expires_at,
        )
        assert expired.load("checkpoint", "auto") == ()
        assert expired.load_score("checkpoint", "auto", 0) is None
        assert reopened.load("checkpoint", "auto") == ()
        with pytest.raises(ValueError, match="过期"):
            expired.save(record)
        with pytest.raises(ValueError, match="对应"):
            expired.save_score(score)
    finally:
        root.close()


def test_store_rejects_unselected_record_before_insertion(tmp_path: Path) -> None:
    """别的案例或检索策略不得落到同一运行指纹中。

    Args:
        tmp_path: 隔离运行目录。
    """
    from evaluation.lanes.checkpoints import LaneRunManifest, LaneRunStore

    app, _, _ = _app(tmp_path)
    root = app.state.container
    try:
        episode = LaneEpisode(id="allowed", tags=("single_hop",), turns=(
            LaneTurn(question="needle", reference="SQLite", expected_lane="fast"),
        ))
        record = LaneEpisodeRunner(
            root.query_orchestrator, root.history_window,
            root.trace_repository, root.cleanup_temporary_execution,
        ).run(episode, "auto", RetrievalStrategy.HYBRID)[0]
        manifest = LaneRunManifest(
            dataset_sha256="1" * 64, index_fingerprint="2" * 64,
            configuration_version="config", code_sha256="3" * 64,
            judge_configuration_sha256="4" * 64,
            selections=(("allowed", "auto"),), strategy=RetrievalStrategy.VECTOR,
        )
        store = LaneRunStore(tmp_path / "lanes.sqlite3", manifest)
        with pytest.raises(ValueError, match="策略"):
            store.save(record)
        assert store.load("allowed", "auto") == ()
    finally:
        root.close()

# -*- coding: utf-8 -*-
"""评分失败后的实际续跑必须复用生成结果，不重新消费检索与生成。"""

from pathlib import Path

from app.application.errors import LLMTimeoutError
from app.application.ports.structured_output import ResponseModel
from app.application.query.context_budget import PromptBudgetGuard
from app.domain.retrieval import RetrievalStrategy
from app.infrastructure.generation.token_counting import Utf8TokenUpperBoundCounter
from evaluation.lanes.checkpoints import LaneRunManifest
from evaluation.lanes.checkpoints import LaneRunStore
from evaluation.lanes.dataset import LaneDataset
from evaluation.lanes.judge import LaneJudge
from evaluation.lanes.models import LaneEpisode
from evaluation.lanes.models import LaneTurn
from evaluation.lanes.runtime import LaneEpisodeRunner
from tests.test_unified_query_api import _app


def test_failed_judging_resumes_without_regenerating_answers(tmp_path: Path) -> None:
    """若评分异常重跑答案、缓存分数未复用或缺失分数当作零，此测试失败。

    Args:
        tmp_path: 隔离数据库与索引。
    """
    from evaluation.lanes.run import execute_lane_runs

    class TimeoutOnce:
        """只替代外部裁判响应，生产片段和存储保持真实。"""

        def __init__(self) -> None:
            """记录外部评分请求次数。

            Returns:
                None。
            """
            self.calls = 0

        def generate(self, messages: list[dict[str, object]],
                     response_model: type[ResponseModel]) -> ResponseModel:
            """首次超时，之后返回可解析的独立评分。

            Args:
                messages: 生产评分提示词。
                response_model: 评分schema。

            Returns:
                通过schema校验的结论。
            """
            self.calls += 1
            if self.calls == 1:
                raise LLMTimeoutError("不可出现在报告的私有上游说明")
            return response_model.model_validate({
                "correctness": 0.5, "supported": True,
                "complete": False, "reason": "答案过于简略",
            })

    app, _, quality = _app(tmp_path)
    root = app.state.container
    try:
        episode = LaneEpisode(id="resume", tags=("multi_turn",), turns=(
            LaneTurn(question="needle", reference="SQLite", expected_lane="fast"),
            LaneTurn(question="needle怎么保存", reference="SQLite", expected_lane="fast"),
        ))
        dataset = LaneDataset(version="test", smoke_ids=("resume",), episodes=(episode,))
        manifest = LaneRunManifest(
            dataset_sha256="1" * 64, index_fingerprint="2" * 64,
            configuration_version="config", code_sha256="3" * 64,
            judge_configuration_sha256="4" * 64,
            selections=(("resume", "auto"),), strategy=RetrievalStrategy.HYBRID,
        )
        store = LaneRunStore(tmp_path / "run.sqlite3", manifest)
        runner = LaneEpisodeRunner(
            root.query_orchestrator, root.history_window,
            root.trace_repository, root.cleanup_temporary_execution,
        )
        provider = TimeoutOnce()
        judge = LaneJudge(provider, root.trace_service, root.trace_repository,
                          PromptBudgetGuard(Utf8TokenUpperBoundCounter(), 16000))
        first = execute_lane_runs(manifest, dataset, runner, judge, store, smoke=True)
        assert first.status == "incomplete"
        assert first.generated_turns == 2
        assert first.scored_turns == 1
        assert first.issues[0].stage == "scoring"
        assert first.issues[0].error_category == "timeout"
        assert "私有上游说明" not in first.model_dump_json()
        assert quality.evidence_calls == 2
        resumed = execute_lane_runs(manifest, dataset, runner, judge, store, smoke=True)
        assert resumed.status == "complete"
        assert resumed.generated_turns == resumed.scored_turns == 2
        assert resumed.lanes["auto"].metrics["correctness"].value == 0.5
        assert resumed.lanes["auto"].metrics["correctness"].n == 2
        assert quality.evidence_calls == 2
        assert provider.calls == 3
        again = execute_lane_runs(manifest, dataset, runner, judge, store, smoke=True)
        assert again.status == "complete"
        assert quality.evidence_calls == 2
        assert provider.calls == 3
    finally:
        root.close()

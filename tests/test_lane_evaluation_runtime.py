# -*- coding: utf-8 -*-
"""分通道评估必须执行真实编排器、真实历史窗口和隔离追踪。"""

import json
from pathlib import Path

import pytest

from app.application.ports.structured_output import ResponseModel
from app.domain.retrieval import RetrievalStrategy
from tests.test_unified_query_api import _app
from tests.understanding import OfflineUnderstandingProvider


@pytest.mark.parametrize("policy,lane,trace_policy", [
    ("auto", "fast", "online"),
    ("fast_only", "fast", "fast_only"),
    ("agent", "agent", "online"),
])
def test_episode_executes_production_lane_and_keeps_strategy_separate(
    tmp_path: Path, policy: str, lane: str, trace_policy: str,
) -> None:
    """改成原始QueryService或只给答案改名会失去真实通道与追踪。

    Args:
        tmp_path: 隔离数据目录。
        policy: 三种评估策略。
        lane: 手工推导的最终通道。
        trace_policy: 后端实际执行约束。
    """
    from evaluation.lanes.models import LaneEpisode, LaneTurn
    from evaluation.lanes.runtime import LaneEpisodeRunner

    app, _, _ = _app(tmp_path)
    root = app.state.container
    try:
        episode = LaneEpisode(id="single", tags=("single_hop",), turns=(
            LaneTurn(question="needle", reference="使用SQLite保存状态",
                     expected_lane="fast", expected_sources=("guide.md",)),
        ))
        runner = LaneEpisodeRunner(
            root.query_orchestrator, root.history_window,
            root.trace_repository, root.cleanup_temporary_execution,
        )
        records = runner.run(episode, policy, RetrievalStrategy.HYBRID)
        assert len(records) == 1
        record = records[0]
        assert record.policy == policy
        assert record.strategy == RetrievalStrategy.HYBRID
        assert record.result.metadata.lane == lane
        assert record.result.retrieval_calls == 1
        assert record.result.answer == "资料[1]"
        assert record.retrieval_compliant is True
        assert record.trace.entry == "evaluation"
        assert record.trace.commit_status == "not_applicable"
        assert record.trace.lane_policy == trace_policy
        assert record.trace.question is None
        assert len(record.query_hashes) == 1
        assert "needle" not in json.dumps(record.query_hashes)
        assert not (tmp_path / "conversations.sqlite3").exists()
        if policy == "agent":
            config = {"configurable": {"thread_id": str(record.execution_id)}}
            store = root._get_conversation_checkpoint_store()
            assert store.checkpointer.get_tuple(config) is None
    finally:
        root.close()


def test_followup_uses_actual_previous_answer_in_production_history_window(
    tmp_path: Path,
) -> None:
    """把gold参考答案作为历史、清空历史或串入别的策略会导致此断言失败。

    Args:
        tmp_path: 隔离数据目录。
    """
    from evaluation.lanes.models import LaneEpisode, LaneTurn
    from evaluation.lanes.runtime import LaneEpisodeRunner

    histories = []

    class RecordingProvider(OfflineUnderstandingProvider):
        """只观察真实理解模型边界，保留真实历史选择和编排。"""

        def generate(
            self, messages: list[dict[str, object]],
            response_model: type[ResponseModel],
        ) -> ResponseModel:
            """记录发送给理解层的实际历史。

            Args:
                messages: 生产构造的完整模型输入。
                response_model: 生产要求的结构化schema。

            Returns:
                保守知识意图。
            """
            payload = json.loads(str(messages[-1]["content"]))
            histories.append(payload["history"])
            return super().generate(messages, response_model)

    app, _, _ = _app(
        tmp_path, understanding_factory=lambda *args: RecordingProvider(),
    )
    root = app.state.container
    try:
        episode = LaneEpisode(id="followup", tags=("multi_turn",), turns=(
            LaneTurn(question="needle", reference="不允许充当历史的gold答案",
                     expected_lane="fast", expected_sources=("guide.md",)),
            LaneTurn(question="它如何持久化？", reference="SQLite保存状态",
                     expected_lane="fast", expected_sources=("guide.md",)),
        ))
        runner = LaneEpisodeRunner(
            root.query_orchestrator, root.history_window,
            root.trace_repository, root.cleanup_temporary_execution,
        )
        first = runner.run(episode, "fast_only", RetrievalStrategy.HYBRID)
        second = runner.run(episode, "auto", RetrievalStrategy.HYBRID)
        assert len(first) == len(second) == 2
        assert histories[0] == histories[2] == []
        assert "资料[1]" in json.dumps(histories[1], ensure_ascii=False)
        assert "gold答案" not in json.dumps(histories, ensure_ascii=False)
        assert first[0].execution_id == first[1].execution_id
        assert first[0].execution_id != second[0].execution_id
    finally:
        root.close()


def test_nonknowledge_case_cannot_silently_enter_forced_knowledge_baseline(
    tmp_path: Path,
) -> None:
    """闲聊强制检索不得混进单跳知识基线的分母。

    Args:
        tmp_path: 隔离数据目录。
    """
    from evaluation.lanes.models import LaneEpisode, LaneTurn
    from evaluation.lanes.runtime import LaneEpisodeRunner

    app, _, _ = _app(tmp_path)
    root = app.state.container
    try:
        episode = LaneEpisode(id="chat", tags=("chitchat",), turns=(
            LaneTurn(question="你好", reference="友好问候", expected_lane="direct",
                     requires_retrieval=False),
        ))
        runner = LaneEpisodeRunner(
            root.query_orchestrator, root.history_window,
            root.trace_repository, root.cleanup_temporary_execution,
        )
        with pytest.raises(ValueError, match="知识"):
            runner.run(episode, "fast_only", RetrievalStrategy.HYBRID)
        record = runner.run(episode, "auto", RetrievalStrategy.HYBRID)[0]
        assert record.result.metadata.lane == "direct"
        assert record.result.retrieval_calls == 0
        assert record.retrieval_compliant is True
    finally:
        root.close()


def test_resume_reuses_generated_answer_without_regeneration_and_cleans_failed_run(
    tmp_path: Path,
) -> None:
    """评分或落盘中断后不能把成功答案重新生成，缓存历史也必须真实。

    Args:
        tmp_path: 隔离数据目录。
    """
    from evaluation.lanes.models import LaneEpisode, LaneObservation, LaneTurn
    from evaluation.lanes.runtime import LaneEpisodeRunner

    app, _, quality = _app(tmp_path)
    root = app.state.container
    saved = []

    def interrupted(record: LaneObservation) -> None:
        saved.append(LaneObservation.model_validate_json(record.model_dump_json()))
        raise RuntimeError("模拟已持久化后中断")

    try:
        episode = LaneEpisode(id="resume", tags=("multi_turn",), turns=(
            LaneTurn(question="needle", reference="SQLite", expected_lane="fast"),
            LaneTurn(question="needle怎么用", reference="SQLite", expected_lane="fast"),
        ))
        runner = LaneEpisodeRunner(
            root.query_orchestrator, root.history_window,
            root.trace_repository, root.cleanup_temporary_execution,
        )
        with pytest.raises(RuntimeError, match="模拟"):
            runner.run(episode, "agent", RetrievalStrategy.HYBRID, on_turn=interrupted)
        assert len(saved) == 1
        store = root._get_conversation_checkpoint_store()
        config = {"configurable": {"thread_id": str(saved[0].execution_id)}}
        assert store.checkpointer.get_tuple(config) is None
        assert saved[0].result.execution is None
        assert "execution" not in json.loads(saved[0].model_dump_json())["result"]
        assert quality.evidence_calls == 1
        resumed = runner.run(episode, "agent", RetrievalStrategy.HYBRID, cached=tuple(saved))
        assert len(resumed) == 2
        assert quality.evidence_calls == 2
        assert resumed[0].trace.trace_id == saved[0].trace.trace_id
        assert store.checkpointer.get_tuple(config) is None
        again = runner.run(episode, "agent", RetrievalStrategy.HYBRID, cached=resumed)
        assert len(again) == 2
        assert quality.evidence_calls == 2
        with pytest.raises(ValueError, match="缓存"):
            runner.run(episode, "auto", RetrievalStrategy.HYBRID, cached=resumed)
    finally:
        root.close()

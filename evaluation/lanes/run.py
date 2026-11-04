# -*- coding: utf-8 -*-
"""串行执行小样本，生成和独立评分分别持久化并可恢复。"""

from collections.abc import Callable
from typing import Literal

from evaluation.checkpoints import classify_remote_error
from evaluation.lanes.checkpoints import LaneRunManifest
from evaluation.lanes.checkpoints import LaneRunStore
from evaluation.lanes.dataset import LaneDataset
from evaluation.lanes.judge import LaneJudge
from evaluation.lanes.reporting import LaneReport
from evaluation.lanes.reporting import LaneRunIssue
from evaluation.lanes.reporting import build_lane_report
from evaluation.lanes.runtime import LaneEpisodeRunner


EvaluationPhase = Literal["all", "generate", "score"]


def execute_lane_runs(
    manifest: LaneRunManifest, dataset: LaneDataset,
    runner: LaneEpisodeRunner, judge: LaneJudge | None, store: LaneRunStore,
    *, smoke: bool, phase: EvaluationPhase = "all",
    on_progress: Callable[[str], None] | None = None,
) -> LaneReport:
    """恢复成功答案与分数，只请求尚未完成的步骤。

    Args:
        manifest: 与存储完全匹配的运行身份。
        dataset: 完整固定标签。
        runner: 生产编排器的片段执行器。
        judge: 独立评分器，generate阶段可省略。
        store: 已持有运行独占锁下的短事务成功存储。
        smoke: 是否仅为连通性烟雾。
        phase: all、generate或score；score绝不调用生成器。
        on_progress: 不含问题正文、凭据与原始异常的阶段输出。

    Returns:
        含计划覆盖率、独立评分分母和安全失败分类的报告。
    """
    if phase not in {"all", "generate", "score"}:
        raise ValueError("评估阶段无效")
    if phase != "generate" and judge is None:
        raise ValueError("评分阶段必须配置独立裁判")
    catalog = {episode.id: episode for episode in dataset.episodes}
    observations = []
    scores = []
    issues = []
    for episode_id, policy in manifest.selections:
        episode = catalog[episode_id]
        cached = store.load(episode_id, policy)
        if phase != "score":
            if on_progress is not None:
                on_progress(f"生成 {episode_id}/{policy}，复用 {len(cached)}/{len(episode.turns)} 轮")
            try:
                runner.run(
                    episode, policy, manifest.strategy,
                    cached=cached, on_turn=store.save,
                )
            except Exception as error:
                persisted = store.load(episode_id, policy)
                issues.append(LaneRunIssue(
                    episode_id=episode_id, policy=policy,
                    turn_index=len(persisted), stage="generation",
                    error_category=classify_remote_error(error).value,
                ))
            cached = store.load(episode_id, policy)
        observations.extend(cached)
        for observation in cached:
            score = store.load_score(episode_id, policy, observation.turn_index)
            if score is None and phase != "generate":
                if on_progress is not None:
                    on_progress(f"独立评分 {episode_id}/{policy}/{observation.turn_index + 1}")
                try:
                    score = judge.score(episode.turns[observation.turn_index], observation)
                    store.save_score(score)
                except Exception as error:
                    issues.append(LaneRunIssue(
                        episode_id=episode_id, policy=policy,
                        turn_index=observation.turn_index, stage="scoring",
                        error_category=classify_remote_error(error).value,
                    ))
                    score = None
            if score is not None:
                scores.append(score)
    return build_lane_report(
        manifest, dataset.episodes, tuple(observations), tuple(scores),
        smoke=smoke, issues=tuple(issues),
    )

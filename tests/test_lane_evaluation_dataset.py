# -*- coding: utf-8 -*-
"""固定分通道标签与小规模选择，不允许悄悄扩成150次生成。"""

from pathlib import Path

import pytest

from evaluation.datasets import DatasetValidationError
from evaluation.datasets import load_corpus


ROOT = Path(__file__).resolve().parents[1] / "evaluation"


def test_fixed_dataset_has_grounded_sources_and_thirteen_turn_smoke() -> None:
    """来源拼错、片段丢失或默认全量执行都会改变人工推导的13轮。
    """
    from evaluation.lanes.dataset import load_lane_dataset
    from evaluation.lanes.dataset import select_runs

    corpus = load_corpus(ROOT / "corpus", ROOT / "corpus_manifest.json")
    dataset = load_lane_dataset(ROOT / "lane_dataset.json", corpus)
    runs = select_runs(dataset, smoke=True)
    assert len(dataset.episodes) == 10
    assert len(runs) == 10
    assert sum(len(episode.turns) for episode, _ in runs) == 13
    chat = [(episode.id, policy) for episode, policy in runs if episode.id == "greeting"]
    assert chat == [("greeting", "auto")]
    tags = set()
    for episode in dataset.episodes:
        tags.update(episode.tags)
    assert {"single_hop", "multi_hop", "comparison", "aggregation", "multi_turn",
            "clarification", "retry", "refusal", "chitchat", "meta"} <= tags



def test_policy_filter_keeps_only_requested_policies() -> None:
    """只跑 auto 时每个片段只保留一次自动路由，不丢片段也不留对照组。
    """
    from evaluation.lanes.dataset import load_lane_dataset
    from evaluation.lanes.dataset import select_runs

    corpus = load_corpus(ROOT / "corpus", ROOT / "corpus_manifest.json")
    dataset = load_lane_dataset(ROOT / "lane_dataset.json", corpus)
    everything = select_runs(dataset, smoke=False)
    auto = select_runs(dataset, smoke=False, policies=("auto",))
    assert {policy for _, policy in auto} == {"auto"}
    assert [episode.id for episode, _ in auto] == [episode.id for episode in dataset.episodes]
    assert auto == tuple(run for run in everything if run[1] == "auto")
    assert select_runs(dataset, smoke=False, policies=("auto", "fast_only", "agent")) == everything
    agent = select_runs(dataset, smoke=False, policies=("agent",))
    assert all(all(turn.requires_retrieval for turn in episode.turns) for episode, _ in agent)

@pytest.mark.parametrize("mutation", ["duplicate", "unknown_source", "bad_smoke"])
def test_invalid_labels_are_rejected_before_model_or_storage(
    tmp_path: Path, mutation: str,
) -> None:
    """重复ID、错来源或未知烟雾选择不能带着错误标签运行。

    Args:
        tmp_path: 测试文件目录。
        mutation: 故意破坏的独立标签约束。
    """
    import json

    from evaluation.lanes.dataset import load_lane_dataset

    corpus = load_corpus(ROOT / "corpus", ROOT / "corpus_manifest.json")
    payload = json.loads((ROOT / "lane_dataset.json").read_text(encoding="utf-8"))
    if mutation == "duplicate":
        payload["episodes"][1]["id"] = payload["episodes"][0]["id"]
    elif mutation == "unknown_source":
        payload["episodes"][0]["turns"][0]["expected_sources"] = ["missing.md"]
    else:
        payload["smoke_ids"] = ["missing"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        load_lane_dataset(path, corpus)

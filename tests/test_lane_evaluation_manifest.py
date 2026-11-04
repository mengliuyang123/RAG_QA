# -*- coding: utf-8 -*-
"""运行指纹必须取实际配置与脏源码，不能只用HEAD或把密钥纳入。"""

from pathlib import Path

from app.config import Settings
from evaluation.datasets import load_corpus
from evaluation.lanes.dataset import load_lane_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_actual_manifest_changes_for_models_dirty_code_and_judge_not_keys(
    tmp_path: Path,
) -> None:
    """配置变化仍复用旧答案、忽略脏源码或将密钥轮换当模型变化都应失败。

    Args:
        tmp_path: 安全的源码和数据试验目录。
    """
    from evaluation.lanes.manifest import build_lane_manifest

    source = tmp_path / "source"
    (source / "app").mkdir(parents=True)
    code = source / "app" / "example.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    corpus = load_corpus(ROOT / "evaluation/corpus", ROOT / "evaluation/corpus_manifest.json")
    dataset = load_lane_dataset(ROOT / "evaluation/lane_dataset.json", corpus)
    settings = Settings(
        _env_file=None, llm_api_key="generator-secret", llm_model="generator",
        llm_base_url="https://generator.example/v1",
        embedding_api_key="embed-secret",
        reranker_base_url=None,
        evaluation_judge_api_key="judge-secret",
        evaluation_judge_model="judge",
        evaluation_judge_base_url="https://judge.example/v1",
    )

    def manifest(config: Settings):
        return build_lane_manifest(config, dataset, corpus, source, smoke=True)

    first = manifest(settings)
    assert first.strategy.value == "hybrid"
    assert len(first.selections) == 10
    changed_key = manifest(settings.model_copy(update={
        "llm_api_key": "rotated", "evaluation_judge_api_key": "rotated-judge",
        "embedding_api_key": "rotated-embed",
    }))
    assert first.fingerprint() == changed_key.fingerprint()
    changed_model = manifest(settings.model_copy(update={"llm_model": "generator-v2"}))
    assert first.configuration_version != changed_model.configuration_version
    changed_judge = manifest(settings.model_copy(update={"evaluation_judge_model": "judge-v2"}))
    assert first.judge_configuration_sha256 != changed_judge.judge_configuration_sha256
    assert first.configuration_version == changed_judge.configuration_version
    code.write_text("VALUE = 2\n", encoding="utf-8")
    changed_code = manifest(settings)
    assert first.code_sha256 != changed_code.code_sha256
    assert first.index_fingerprint == changed_code.index_fingerprint

# -*- coding: utf-8 -*-
"""从实际配置、固定标签和脏源码生成非敏感运行清单。"""

import hashlib
import json
from pathlib import Path

from app.bootstrap.trace_configuration import trace_configuration_version
from app.config import Settings
from evaluation.lanes.checkpoints import LaneRunManifest
from evaluation.lanes.dataset import LaneDataset
from evaluation.lanes.dataset import select_runs
from evaluation.lanes.models import LanePolicy
from evaluation.models import CorpusDataset
from evaluation.runtime import IndexConfig
from evaluation.runtime import compute_index_fingerprint


def build_lane_manifest(
    settings: Settings, dataset: LaneDataset, corpus: CorpusDataset,
    source_root: Path, *, smoke: bool,
    policies: tuple[LanePolicy, ...] | None = None,
) -> LaneRunManifest:
    """在创建远程资源前固定清单，不通过整个Settings哈希暴露凭据关系。

    Args:
        settings: 本次真实生效的环境设置。
        dataset: 已校验的固定标签。
        corpus: 已校验文件内容及来源哈希的语料。
        source_root: 当前仓库，用相对路径和内容覆盖未提交代码及依赖清单。
        smoke: 选择13轮烟雾或完整能力标签。
        policies: 只评估这些策略，None 为全部；选择写入清单，参与指纹。

    Returns:
        可校验成功轮次复用边界的冻结清单。
    """
    settings.validate_evaluation_settings(require_judge=True)
    sources = []
    for directory in ("app", "evaluation"):
        sources.extend((source_root / directory).rglob("*.py"))
    if not sources:
        raise ValueError("运行指纹找不到相关源码")
    for name in ("pyproject.toml", "uv.lock"):
        path = source_root / name
        if path.is_file():
            sources.append(path)
    source_hashes = {}
    for path in sorted(sources):
        relative = path.relative_to(source_root).as_posix()
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    selections = []
    for episode, policy in select_runs(dataset, smoke=smoke, policies=policies):
        selections.append((episode.id, policy))
    judge = {
        "origin": Settings._service_origin(settings.evaluation_judge_base_url),
        "model": settings.evaluation_judge_model,
        "temperature": settings.evaluation_judge_temperature,
        "reasoning_effort": settings.evaluation_judge_reasoning_effort,
        "output_mode": settings.evaluation_judge_output_mode,
        "timeout_seconds": settings.evaluation_judge_timeout_seconds,
        "max_output_tokens": settings.evaluation_judge_max_output_tokens,
        "output_token_parameter": settings.evaluation_judge_output_token_parameter,
        "input_context_tokens": settings.conversation_context_tokens,
        "max_retries": 0,
    }
    return LaneRunManifest(
        dataset_sha256=_digest(dataset.model_dump(mode="json")),
        index_fingerprint=compute_index_fingerprint(IndexConfig.from_settings(settings), corpus),
        configuration_version=trace_configuration_version(
            settings, settings.evaluation_local_temperature,
        ),
        code_sha256=_digest(source_hashes),
        judge_configuration_sha256=_digest(judge),
        selections=tuple(selections), strategy=settings.knowledge_retrieval_strategy,
    )


def _digest(value: object) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

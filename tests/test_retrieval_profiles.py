# -*- coding: utf-8 -*-
"""启动时确定检索配置档，不能在在线重排失败后偷偷切换策略。"""

import pytest

from app.config import Settings
from app.domain.retrieval import RetrievalStrategy


def test_auto_profile_without_reranker_uses_explicit_hybrid_policy() -> None:
    """没有重排服务时仍可启动并使用混合通道。"""
    settings = Settings(_env_file=None, reranker_base_url=None)
    assert settings.knowledge_retrieval_profile == "auto"
    assert settings.knowledge_retrieval_strategy is RetrievalStrategy.HYBRID


def test_auto_profile_with_qwen_configuration_selects_rerank() -> None:
    """已配置Qwen重排时，新知识执行器应默认使用精排而不是忽略它。"""
    settings = Settings(
        _env_file=None, reranker_base_url="https://rerank.example/v1",
        reranker_api_key="offline-test-key",
    )
    assert settings.knowledge_retrieval_strategy is RetrievalStrategy.HYBRID_RERANK


def test_explicit_hybrid_profile_does_not_depend_on_reranker_configuration() -> None:
    """明确禁用精排时不能因为保留了旧URL就强制调用它。"""
    settings = Settings(
        _env_file=None, knowledge_retrieval_profile="hybrid",
        reranker_base_url="https://rerank.example/v1", reranker_api_key=None,
    )
    assert settings.knowledge_retrieval_strategy is RetrievalStrategy.HYBRID


@pytest.mark.parametrize("base_url,key", [(None, None), ("https://rerank.example", None)])
def test_explicit_rerank_profile_rejects_incomplete_configuration(
    base_url: str | None, key: str | None,
) -> None:
    """要求精排时在配置阶段报告缺失项，不把每轮请求都变成502。

    Args:
        base_url: 缺失或存在的重排地址。
        key: 本例缺失的重排密钥。
    """
    with pytest.raises(ValueError, match="RERANKER"):
        Settings(
            _env_file=None, knowledge_retrieval_profile="hybrid_rerank",
            reranker_base_url=base_url, reranker_api_key=key,
        )

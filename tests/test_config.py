# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_settings_reject_overlap_not_smaller_than_chunk_size() -> None:
    """重叠长度达到分块长度时应拒绝配置。"""
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100, _env_file=None)


def test_settings_allows_missing_api_key_for_offline_work() -> None:
    """离线开发和测试不应强制要求 OpenAI API Key。"""
    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None


def test_embedding_credentials_fall_back_to_openai_api_key() -> None:
    """未单独配置 Embedding 凭据时应复用 OpenAI API Key。"""
    settings = Settings(
        openai_api_key="openai-key",
        embedding_api_key=None,
        _env_file=None,
    )

    assert settings.resolved_embedding_api_key == "openai-key"


def test_llm_credentials_fall_back_to_openai_api_key() -> None:
    """官方 OpenAI LLM 地址可复用旧的 OpenAI API Key。"""
    settings = Settings(
        openai_api_key="openai-key",
        llm_api_key=None,
        _env_file=None,
    )

    assert settings.resolved_llm_api_key == "openai-key"


def test_embedding_specific_key_has_priority() -> None:
    """Embedding 专用凭据应优先于 LLM 使用的 OpenAI Key。"""
    settings = Settings(
        openai_api_key="openai-key",
        embedding_api_key="ollama",
        _env_file=None,
    )

    assert settings.resolved_embedding_api_key == "ollama"


def test_llm_specific_key_has_priority() -> None:
    """LLM 专用凭据应优先于旧的 OpenAI API Key。"""
    settings = Settings(
        openai_api_key="openai-key",
        llm_api_key="ollama",
        _env_file=None,
    )

    assert settings.resolved_llm_api_key == "ollama"


def test_blank_embedding_key_falls_back_to_openai_api_key() -> None:
    """空白专用凭据不应覆盖有效的 OpenAI Key。"""
    settings = Settings(
        openai_api_key="openai-key",
        embedding_api_key="   ",
        _env_file=None,
    )

    assert settings.resolved_embedding_api_key == "openai-key"


def test_default_embedding_base_url_targets_openai() -> None:
    """默认兼容地址应保持现有 OpenAI 行为。"""
    settings = Settings(_env_file=None)

    assert settings.embedding_base_url == "https://api.openai.com/v1"


def test_default_llm_base_url_targets_openai() -> None:
    """默认 LLM 兼容地址应保持现有 OpenAI 行为。"""
    settings = Settings(_env_file=None)

    assert settings.llm_base_url == "https://api.openai.com/v1"


def test_llm_reasoning_effort_is_optional_and_configurable() -> None:
    """显式推理强度应可配置，未设置时不得改变兼容接口请求。"""
    default_settings = Settings(_env_file=None)
    configured_settings = Settings(
        llm_reasoning_effort="none",
        _env_file=None,
    )

    assert default_settings.llm_reasoning_effort is None
    assert configured_settings.llm_reasoning_effort == "none"


@pytest.mark.parametrize(
    ("field", "resolved_field"),
    [
        ("embedding_base_url", "resolved_embedding_api_key"),
        ("llm_base_url", "resolved_llm_api_key"),
    ],
)
def test_third_party_model_service_never_receives_openai_fallback_key(
    field: str,
    resolved_field: str,
) -> None:
    """第三方模型地址不得自动获得 OpenAI 凭据。"""
    settings = Settings(
        openai_api_key="openai-secret",
        **{field: "http://localhost:11434/v1"},
        _env_file=None,
    )

    assert getattr(settings, resolved_field) is None


def test_official_openai_url_allows_trailing_slash_for_key_fallback() -> None:
    """官方地址末尾斜杠不应影响安全回退判断。"""
    settings = Settings(
        openai_api_key="openai-key",
        llm_base_url="https://api.openai.com/v1/",
        embedding_base_url="https://api.openai.com/v1/",
        _env_file=None,
    )

    assert settings.resolved_llm_api_key == "openai-key"
    assert settings.resolved_embedding_api_key == "openai-key"


def test_reranker_has_independent_optional_service_configuration() -> None:
    """Reranker 地址和凭据必须独立，且不得回退 OpenAI Key。"""
    settings = Settings(openai_api_key="openai-secret", _env_file=None)

    assert settings.reranker_base_url is None
    assert settings.reranker_api_key is None
    assert settings.resolved_reranker_api_key is None
    assert settings.reranker_model == "qwen3-rerank"
    assert settings.reranker_endpoint == "reranks"


def test_reranker_returns_only_its_own_api_key() -> None:
    """远程 Reranker 只能使用显式配置的专用凭据。"""
    settings = Settings(
        openai_api_key="openai-secret",
        reranker_base_url="http://localhost:8000/v1",
        reranker_api_key="reranker-token",
        _env_file=None,
    )

    assert settings.resolved_reranker_api_key == "reranker-token"


def test_default_data_paths_are_absolute_project_paths() -> None:
    """默认数据目录不应随进程启动目录变化。"""
    settings = Settings(_env_file=None)

    assert settings.docs_path.is_absolute()
    assert settings.chroma_path.is_absolute()
    assert settings.docs_path == Path(__file__).parents[1] / "data" / "docs"


def test_default_conversation_configuration_is_local_and_bounded() -> None:
    """会话历史、Agent 上限和两个 SQLite 路径应有安全默认值。"""
    settings = Settings(_env_file=None)

    assert settings.conversation_db_path == (
        Path(__file__).parents[1]
        / "data"
        / "conversations"
        / "conversations.sqlite3"
    )
    assert settings.agent_checkpoint_db_path.name == "checkpoints.sqlite3"
    assert settings.conversation_history_turns == 6
    assert settings.agent_max_retrieval_calls == 3
    assert settings.agent_temperature == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conversation_history_turns", 0),
        ("agent_max_retrieval_calls", 0),
        ("agent_max_retrieval_calls", 4),
        ("agent_temperature", -0.1),
    ],
)
def test_conversation_configuration_rejects_unsafe_bounds(
    field: str,
    value: int | float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_get_settings_returns_cached_instance() -> None:
    """应用内重复获取配置时应复用同一个只读配置实例。"""
    get_settings.cache_clear()

    assert get_settings() is get_settings()


def test_default_evaluation_configuration_is_isolated() -> None:
    """默认评估目录必须与生产目录分离并使用固定随机参数。"""
    settings = Settings(_env_file=None)

    assert settings.evaluation_chroma_path.is_absolute()
    assert settings.evaluation_chroma_path != settings.chroma_path
    assert settings.evaluation_collection_prefix == "rag_eval"
    assert settings.evaluation_random_seed == 20260820
    assert settings.evaluation_local_temperature == 0.0
    assert settings.evaluation_rgb_temperature == 0.2
    assert settings.evaluation_judge_temperature == 0.0
    assert settings.evaluation_judge_reasoning_effort is None


def test_evaluation_judge_reasoning_effort_is_configurable() -> None:
    """独立裁判推理强度应能与回答生成模型分别配置。"""
    settings = Settings(
        evaluation_judge_reasoning_effort="none",
        _env_file=None,
    )

    assert settings.evaluation_judge_reasoning_effort == "none"


def test_evaluation_validation_rejects_production_chroma_path(
    tmp_path: Path,
) -> None:
    """评估路径解析后与生产路径相同时必须在远程调用前失败。"""
    database_path = tmp_path / "shared"
    settings = Settings(
        chroma_path=database_path,
        evaluation_chroma_path=database_path / ".",
        evaluation_judge_api_key="judge-key",
        evaluation_judge_base_url="https://judge.example/v1",
        evaluation_judge_model="judge-model",
        _env_file=None,
    )

    with pytest.raises(ValueError, match="评估 Chroma 路径必须与生产路径隔离"):
        settings.validate_evaluation_settings()


def test_evaluation_validation_rejects_incomplete_judge_configuration() -> None:
    """裁判三元组缺少任一字段时不得开始评估。"""
    settings = Settings(
        evaluation_judge_api_key="judge-key",
        evaluation_judge_base_url="https://judge.example/v1",
        evaluation_judge_model=None,
        _env_file=None,
    )

    with pytest.raises(ValueError, match="独立裁判配置不完整"):
        settings.validate_evaluation_settings()


def test_evaluation_validation_rejects_generator_as_judge() -> None:
    """相同服务 origin 与模型名不能伪装成独立裁判。"""
    settings = Settings(
        llm_api_key="generator-key",
        llm_base_url="https://models.example/v1",
        llm_model="qwen-plus",
        evaluation_judge_api_key="judge-key",
        evaluation_judge_base_url=(
            "https://models.example/compatible-api/v1"
        ),
        evaluation_judge_model="qwen-plus",
        _env_file=None,
    )

    with pytest.raises(ValueError, match="裁判模型必须独立于生成模型"):
        settings.validate_evaluation_settings()


def test_evaluation_validation_accepts_independent_judge() -> None:
    """不同服务 origin 的显式裁判配置应通过预检。"""
    settings = Settings(
        llm_api_key="generator-key",
        llm_base_url="https://generator.example/v1",
        llm_model="qwen-plus",
        evaluation_judge_api_key="judge-key",
        evaluation_judge_base_url="https://judge.example/v1",
        evaluation_judge_model="judge-model",
        _env_file=None,
    )

    settings.validate_evaluation_settings()

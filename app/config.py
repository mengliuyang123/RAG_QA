# -*- coding: utf-8 -*-
"""应用配置及其校验规则。"""

from functools import lru_cache
from pathlib import Path
from typing import Self
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain.retrieval import RetrievalStrategy
from app.domain.tracing import ModelPrice

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENAI_BASE_URL = "https://api.openai.com/v1"


class Settings(BaseSettings):
    """集中管理环境变量和应用默认配置。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    openai_api_key: str | None = None
    auth_db_path: Path = PROJECT_ROOT / "data" / "auth" / "auth.sqlite3"
    auth_secure_cookie: bool = True
    auth_session_seconds: int = Field(default=43200, ge=60)
    auth_idle_seconds: int = Field(default=1800, ge=1)
    auth_login_attempts: int = Field(default=10, ge=1, le=1000)
    auth_login_window_seconds: int = Field(default=60, ge=1)
    http_allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]")
    cors_allowed_origins: tuple[str, ...] = (
        "http://localhost:5173", "http://127.0.0.1:5173",
    )

    llm_api_key: str | None = None
    llm_base_url: str = OPENAI_BASE_URL
    llm_model: str = "gpt-4o-mini"
    llm_reasoning_effort: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_output_tokens: int = Field(default=4096, ge=1)
    llm_output_token_parameter: Literal[
        "max_tokens", "max_completion_tokens",
    ] = "max_tokens"
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)
    model_max_retries: int = Field(default=0, ge=0, le=2)
    quality_output_mode: Literal["json_schema", "json_object", "prompt"] = (
        "json_schema"
    )
    understanding_api_key: str | None = None
    understanding_base_url: str | None = None
    understanding_model: str | None = None
    understanding_reasoning_effort: str | None = None
    understanding_temperature: float | None = Field(default=None, ge=0)
    # 以下理解模型参数未配置时沿用对应的 LLM_* / 通用模型配置，见 resolved_* 属性。
    understanding_timeout_seconds: float | None = Field(default=None, gt=0)
    understanding_max_output_tokens: int | None = Field(default=None, ge=1)
    understanding_output_token_parameter: Literal[
        "max_tokens", "max_completion_tokens",
    ] | None = None
    understanding_context_tokens: int | None = Field(default=None, ge=1)
    understanding_output_mode: Literal[
        "json_schema", "json_object", "prompt",
    ] | None = None
    understanding_confidence_threshold: float = Field(
        default=0.8, gt=0, le=1, allow_inf_nan=False,
    )
    # 证据判断与答案审核的独立模型；未配置的项沿用 LLM_*。
    judge_api_key: str | None = None
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_reasoning_effort: str | None = None
    judge_timeout_seconds: float | None = Field(default=None, gt=0)

    embedding_api_key: str | None = None
    embedding_base_url: str = OPENAI_BASE_URL
    embedding_model: str = "text-embedding-3-small"

    reranker_api_key: str | None = None
    reranker_base_url: str | None = None
    reranker_model: str = "qwen3-rerank"
    reranker_endpoint: str = "reranks"

    evaluation_judge_api_key: str | None = None
    evaluation_judge_base_url: str | None = None
    evaluation_judge_model: str | None = None
    evaluation_judge_reasoning_effort: str | None = None
    evaluation_judge_output_mode: Literal[
        "json_schema", "json_object", "prompt",
    ] = "json_schema"
    evaluation_judge_timeout_seconds: float = Field(
        default=60.0, gt=0, allow_inf_nan=False,
    )
    evaluation_judge_max_output_tokens: int = Field(default=1024, ge=1)
    evaluation_judge_output_token_parameter: Literal[
        "max_tokens", "max_completion_tokens",
    ] = "max_tokens"

    evaluation_chroma_path: Path = (
        PROJECT_ROOT / "data" / "evaluation" / "chroma"
    )
    evaluation_collection_prefix: str = "rag_eval"
    evaluation_results_path: Path = PROJECT_ROOT / "evaluation" / "results"
    refusal_cluster_similarity_threshold: float = Field(
        default=0.85, ge=0, le=1, allow_inf_nan=False,
    )
    evaluation_random_seed: int = 20260820
    evaluation_local_temperature: float = Field(default=0.0, ge=0)
    evaluation_rgb_temperature: float = Field(default=0.2, ge=0)
    evaluation_judge_temperature: float = Field(default=0.0, ge=0)

    conversation_db_path: Path = (
        PROJECT_ROOT / "data" / "conversations" / "conversations.sqlite3"
    )
    knowledge_base_db_path: Path = (
        PROJECT_ROOT / "data" / "knowledge" / "catalog.sqlite3"
    )
    trace_db_path: Path = PROJECT_ROOT / "data" / "traces" / "traces.sqlite3"
    trace_capture_queries: bool = False
    trace_retention_days: int = Field(default=30, ge=1, le=3650)
    trace_cleanup_interval_seconds: float = Field(
        default=3600.0, gt=0, allow_inf_nan=False,
    )
    trace_model_prices: tuple[ModelPrice, ...] = ()
    agent_checkpoint_db_path: Path = (
        PROJECT_ROOT / "data" / "conversations" / "checkpoints.sqlite3"
    )
    conversation_history_turns: int = Field(default=6, ge=1)
    conversation_context_tokens: int = Field(default=32768, ge=1)
    conversation_reserved_tokens: int = Field(default=16384, ge=0)
    conversation_assistant_token_limit: int = Field(default=512, ge=1)
    conversation_history_batch_size: int = Field(default=32, ge=1, le=200)
    agent_max_retrieval_calls: int = Field(default=3, ge=1, le=3)
    agent_temperature: float = Field(default=0.0, ge=0)
    agent_turn_timeout_seconds: float = Field(default=180.0, gt=0)
    agent_retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    knowledge_retrieval_profile: Literal[
        "auto", "vector", "hybrid", "hybrid_rerank",
    ] = "auto"

    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    embedding_batch_size: int = Field(default=100, gt=0)

    top_k: int = Field(default=5, ge=1, le=20)
    top_k_rerank: int = Field(default=20, ge=1)

    docs_path: Path = PROJECT_ROOT / "data" / "docs"
    chroma_path: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "knowledge_base"
    max_upload_size: int = Field(default=20 * 1024 * 1024, gt=0)
    ingestion_queue_capacity: int = Field(default=16, ge=1, le=1000)
    ingestion_worker_enabled: bool = True
    ingestion_poll_seconds: float = Field(
        default=0.5, ge=0.05, le=60, allow_inf_nan=False,
    )
    question_rate_limit_per_minute: int = Field(default=10, ge=1, le=10000)
    question_max_active: int = Field(default=3, ge=1, le=16)
    question_max_waiting: int = Field(default=20, ge=0, le=32)
    question_queue_timeout_seconds: float = Field(
        default=60.0, gt=0, le=3600, allow_inf_nan=False,
    )

    log_level: str = "INFO"

    @field_validator("http_allowed_hosts")
    @classmethod
    def validate_http_hosts(cls, hosts: tuple[str, ...]) -> tuple[str, ...]:
        """要求显式Host白名单，避免恶意Host参与同源判断。

        Args:
            hosts: 配置的主机名，不含协议或端口。

        Returns:
            已验证的白名单。
        """
        if not hosts:
            raise ValueError("HTTP Host白名单不能为空")
        for host in hosts:
            if not host or host != host.strip() or any(
                character in host for character in "*/@?#\\"
            ) or "://" in host:
                raise ValueError("HTTP Host必须是显式主机名")
        return hosts

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_origins(cls, origins: tuple[str, ...]) -> tuple[str, ...]:
        """Cookie认证只允许完整且不带凭据的显式来源。

        Args:
            origins: 协议、主机和可选端口组成的来源。

        Returns:
            已验证来源，空元组表示只允许同源。
        """
        for origin in origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname or "*" in origin
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment
                or origin != f"{parsed.scheme}://{parsed.netloc}"
            ):
                raise ValueError("CORS必须使用不带路径或凭据的显式HTTP来源")
            # 提前验证非法端口，避免来源配置一直无法匹配浏览器请求。
            parsed.port
        return origins

    @field_validator(
        "understanding_api_key", "understanding_base_url",
        "understanding_model", "understanding_reasoning_effort",
        "understanding_timeout_seconds", "understanding_max_output_tokens",
        "understanding_output_token_parameter", "understanding_context_tokens",
        "understanding_output_mode",
        "judge_api_key", "judge_base_url", "judge_model",
        "judge_reasoning_effort", "judge_timeout_seconds",
        mode="before",
    )
    @classmethod
    def normalize_understanding_options(cls, value: object) -> object:
        """把可选环境变量空值视为未指定，而非错误的独立端点。

        Args:
            value: 环境变量或显式参数。

        Returns:
            去除首尾空白的配置或 None。
        """
        if isinstance(value, str):
            return value.strip() or None
        return value

    @property
    def knowledge_retrieval_strategy(self) -> RetrievalStrategy:
        """在启动配置中选择统一知识通道策略，不做运行时错误后降级。

        Returns:
            明确的三种检索策略之一。
        """
        if self.knowledge_retrieval_profile == "auto":
            if self.reranker_base_url and self.reranker_base_url.strip():
                return RetrievalStrategy.HYBRID_RERANK
            return RetrievalStrategy.HYBRID
        return RetrievalStrategy(self.knowledge_retrieval_profile)

    @property
    def resolved_llm_api_key(self) -> str | None:
        """返回 LLM 专用凭据，官方 OpenAI 地址允许复用旧凭据。"""
        return self._resolve_openai_compatible_api_key(
            self.llm_api_key,
            self.llm_base_url,
        )

    @property
    def resolved_understanding_api_key(self) -> str | None:
        """仅在未另选端点时继承生成凭据，避免跨服务泄漏。

        Returns:
            独立密钥、同服务继承密钥或未配置状态。
        """
        if self.understanding_api_key and self.understanding_api_key.strip():
            return self.understanding_api_key
        if self.understanding_base_url is not None:
            return None
        return self.resolved_llm_api_key

    @property
    def resolved_understanding_base_url(self) -> str:
        """返回显式理解地址或生成服务地址。

        Returns:
            理解模型实际访问的端点。
        """
        return self.understanding_base_url or self.llm_base_url

    @property
    def resolved_understanding_model(self) -> str:
        """允许独立选择轻量模型，未配置时复用生成模型。

        Returns:
            理解模型标识，不在代码中写死云供应商或本地模型。
        """
        return self.understanding_model or self.llm_model

    @property
    def resolved_understanding_reasoning_effort(self) -> str | None:
        """未配置时沿用生成的推理强度，避免本地思考模型在理解步骤超时。

        Returns:
            理解推理强度或未指定。
        """
        return self.understanding_reasoning_effort or self.llm_reasoning_effort

    @property
    def resolved_understanding_timeout_seconds(self) -> float:
        """返回理解单次调用超时，未配置时沿用生成超时。

        Returns:
            正数秒数。
        """
        if self.understanding_timeout_seconds is not None:
            return self.understanding_timeout_seconds
        return self.llm_timeout_seconds

    @property
    def resolved_understanding_max_output_tokens(self) -> int:
        """返回理解输出上限，未配置时沿用生成输出上限。

        Returns:
            正整数 token 数。
        """
        if self.understanding_max_output_tokens is not None:
            return self.understanding_max_output_tokens
        return self.llm_max_output_tokens

    @property
    def resolved_understanding_output_token_parameter(self) -> str:
        """返回理解请求使用的输出上限参数名，未配置时沿用生成配置。

        Returns:
            max_tokens 或 max_completion_tokens。
        """
        return (
            self.understanding_output_token_parameter
            or self.llm_output_token_parameter
        )

    @property
    def resolved_understanding_context_tokens(self) -> int:
        """返回理解模型上下文预算，未配置时沿用会话上下文预算。

        Returns:
            正整数 token 数。
        """
        if self.understanding_context_tokens is not None:
            return self.understanding_context_tokens
        return self.conversation_context_tokens

    @property
    def resolved_understanding_output_mode(self) -> str:
        """返回理解的结构化输出协议，未配置时沿用在线质量判断的协议。

        Returns:
            json_schema、json_object 或 prompt。
        """
        return self.understanding_output_mode or self.quality_output_mode

    @property
    def judge_configured(self) -> bool:
        """是否显式配置了任一裁判项；全部未配置时裁判与生成共用模型。

        Returns:
            存在任一 JUDGE_* 配置时为真。
        """
        return any(
            value is not None
            for value in (
                self.judge_api_key, self.judge_base_url, self.judge_model,
                self.judge_reasoning_effort, self.judge_timeout_seconds,
            )
        )

    @property
    def resolved_judge_api_key(self) -> str | None:
        """仅在未另选端点时继承生成凭据，避免把密钥发给另一个服务。

        Returns:
            独立密钥、同服务继承密钥或未配置状态。
        """
        if self.judge_api_key and self.judge_api_key.strip():
            return self.judge_api_key
        if self.judge_base_url is not None:
            return None
        return self.resolved_llm_api_key

    @property
    def resolved_judge_base_url(self) -> str:
        """返回显式裁判地址或生成服务地址。

        Returns:
            证据判断与答案审核实际访问的端点。
        """
        return self.judge_base_url or self.llm_base_url

    @property
    def resolved_judge_model(self) -> str:
        """允许为判断类步骤选择更强的模型，未配置时复用生成模型。

        Returns:
            裁判模型标识。
        """
        return self.judge_model or self.llm_model

    @property
    def resolved_judge_reasoning_effort(self) -> str | None:
        """未配置时沿用生成的推理强度，保持未使用裁判配置时的原行为。

        Returns:
            裁判推理强度或未指定。
        """
        return self.judge_reasoning_effort or self.llm_reasoning_effort

    @property
    def resolved_judge_timeout_seconds(self) -> float:
        """返回裁判单次调用超时，未配置时沿用生成超时。

        Returns:
            正数秒数。
        """
        if self.judge_timeout_seconds is not None:
            return self.judge_timeout_seconds
        return self.llm_timeout_seconds

    @property
    def resolved_embedding_api_key(self) -> str | None:
        """返回 Embedding 专用凭据，官方地址允许复用旧凭据。"""
        return self._resolve_openai_compatible_api_key(
            self.embedding_api_key,
            self.embedding_base_url,
        )

    @property
    def resolved_reranker_api_key(self) -> str | None:
        """只返回显式配置的 Reranker 凭据。"""
        if self.reranker_api_key and self.reranker_api_key.strip():
            return self.reranker_api_key
        return None

    def _resolve_openai_compatible_api_key(
        self,
        dedicated_api_key: str | None,
        base_url: str,
    ) -> str | None:
        """解析兼容服务凭据，并阻止旧密钥泄露给第三方地址。"""
        if dedicated_api_key and dedicated_api_key.strip():
            return dedicated_api_key
        if base_url.rstrip("/") == OPENAI_BASE_URL:
            return self.openai_api_key
        return None

    def validate_evaluation_settings(
        self,
        require_judge: bool = True,
    ) -> None:
        """校验评估路径隔离和独立裁判配置。

        Args:
            require_judge: 是否要求裁判地址、模型和凭据全部存在。

        Returns:
            None。

        Raises:
            ValueError: 评估路径未隔离或裁判配置不完整、不独立。
        """
        production_path = self.chroma_path.resolve()
        evaluation_path = self.evaluation_chroma_path.resolve()
        if production_path == evaluation_path:
            raise ValueError("评估 Chroma 路径必须与生产路径隔离")

        judge_values = (
            self.evaluation_judge_api_key,
            self.evaluation_judge_base_url,
            self.evaluation_judge_model,
        )
        configured_fields = []
        for value in judge_values:
            is_configured = isinstance(value, str) and bool(value.strip())
            configured_fields.append(is_configured)

        if require_judge and not all(configured_fields):
            raise ValueError("独立裁判配置不完整")
        if any(configured_fields) and not all(configured_fields):
            raise ValueError("独立裁判配置不完整")
        if not any(configured_fields):
            return

        judge_base_url = self.evaluation_judge_base_url
        judge_model = self.evaluation_judge_model
        if judge_base_url is None or judge_model is None:
            raise ValueError("独立裁判配置不完整")
        generator_identity = (
            self._service_origin(self.llm_base_url),
            self.llm_model.strip(),
        )
        judge_identity = (
            self._service_origin(judge_base_url),
            judge_model.strip(),
        )
        if generator_identity == judge_identity:
            raise ValueError("裁判模型必须独立于生成模型")

    @staticmethod
    def _service_origin(base_url: str) -> str:
        """返回不包含路径、查询和凭据的规范服务 origin。"""
        parsed = urlsplit(base_url.strip())
        if not parsed.scheme or parsed.hostname is None:
            raise ValueError("模型服务地址必须包含 scheme 和 host")
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname.lower()
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("模型服务地址端口无效") from exc
        is_default_port = (
            scheme == "https" and port == 443
        ) or (
            scheme == "http" and port == 80
        )
        if port is None or is_default_port:
            return f"{scheme}://{hostname}"
        return f"{scheme}://{hostname}:{port}"

    @model_validator(mode="after")
    def validate_question_admission_settings(self) -> Self:
        """为同步问答入口在默认线程池中保留其他路由的执行余量。

        Returns:
            保留至少八个默认Starlette线程槽的配置。
        """
        if self.question_max_active + self.question_max_waiting > 32:
            raise ValueError("问答执行和等待容量合计不得超过32")
        return self

    @model_validator(mode="after")
    def validate_retrieval_and_chunk_settings(self) -> Self:
        """校验配置项之间的大小关系。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        if self.top_k_rerank < self.top_k:
            raise ValueError("top_k_rerank 必须大于或等于 top_k")
        if self.conversation_reserved_tokens >= self.conversation_context_tokens:
            raise ValueError("上下文预留量必须小于总预算")
        if self.llm_max_output_tokens > self.conversation_reserved_tokens:
            raise ValueError("输出上限不得超过上下文预留量")
        if (
            self.resolved_understanding_max_output_tokens
            >= self.resolved_understanding_context_tokens
        ):
            raise ValueError("理解模型输出上限必须小于其上下文预算")
        needs_rerank = (
            self.agent_retrieval_strategy is RetrievalStrategy.HYBRID_RERANK
            or self.knowledge_retrieval_strategy is RetrievalStrategy.HYBRID_RERANK
        )
        if needs_rerank:
            if not self.reranker_base_url or not self.reranker_base_url.strip():
                raise ValueError(
                    "Agent hybrid_rerank 需要配置 RERANKER_BASE_URL"
                )
            if not self.resolved_reranker_api_key:
                raise ValueError("hybrid_rerank需要配置RERANKER_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    """返回进程内缓存的应用配置。"""
    return Settings()

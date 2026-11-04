# -*- coding: utf-8 -*-
"""定义组合根使用的显式基础设施 adapter factories。"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from langchain_openai import ChatOpenAI

from app.application.ports.chunking import Chunker
from app.application.ports.document_citations import DocumentCitationReader
from app.application.ports.knowledge_deletion import KnowledgeDeletionRepository
from app.application.ports.embedding import EmbeddingProvider
from app.application.ports.generation import LLMProvider
from app.application.ports.ingestion_jobs import IngestionJobRepository
from app.application.ports.ingestion_sources import IngestionSources
from app.application.ports.ingestion import IngestionExecutionGuard
from app.application.ports.keyword_index import KeywordIndex
from app.application.ports.knowledge_bases import KnowledgeBaseRepository
from app.application.ports.knowledge_content import KnowledgeContentReader
from app.application.ports.document_lifecycle import DocumentLifecycle
from app.application.ports.suggestions import CuratedQuestionRepository
from app.application.ports.parsing import ParserResolver
from app.application.ports.repair_log import RepairLogWriter
from app.application.ports.reranking import Reranker
from app.application.ports.token_counting import TokenCounter
from app.application.ports.structured_output import StructuredOutputProvider
from app.application.ports.tracing import TraceRepository
from app.application.ports.trace_queries import TraceReader
from app.application.ports.feedback_admin import FeedbackAdminRepository
from app.application.ports.shares import ShareRepository
from app.application.ports.evaluation_reports import EvaluationReportReader
from app.application.ports.insights import RefusalReader
from app.application.ports.insights import InsightMetricReader
from app.application.ports.auth import AuthRepository
from app.application.ports.auth import PasswordHasher
from app.application.query.context_budget import PromptBudgetGuard
from app.application.ports.vector_store import VectorStoreAdmin
from app.application.ports.vector_store import VectorStoreReader
from app.application.ports.vector_store import VectorStoreWriter
from app.application.conversations.ports import ConversationAgent
from app.application.conversations.ports import ConversationCheckpointStore
from app.application.conversations.ports import ConversationRepository
from app.application.conversations.quality import ConversationQualityEvaluator
from app.application.errors import LLMMissingAPIKeyError
from app.application.retrieval.service import RetrievalService
from app.config import Settings
from app.domain.retrieval import RetrievalStrategy
from app.infrastructure.chunking.recursive import RecursiveDocumentChunker
from app.infrastructure.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from app.infrastructure.generation.openai_compatible import (
    OpenAICompatibleLLMProvider,
)
from app.infrastructure.generation.structured_output import (
    ModelStructuredOutputProvider,
)
from app.infrastructure.generation.token_counting import (
    Utf8TokenUpperBoundCounter,
)
from app.infrastructure.conversations.quality import ModelQualityEvaluator
from app.infrastructure.indexes.bm25 import BM25Index
from app.infrastructure.parsing.factory import ParserFactory
from app.infrastructure.persistence.repair_log import RepairLog
from app.infrastructure.persistence.traces import SQLiteTraceRepository
from app.infrastructure.persistence.trace_queries import SQLiteTraceReader
from app.infrastructure.persistence.feedback_admin import SQLiteFeedbackAdminRepository
from app.infrastructure.persistence.shares import SQLiteShareRepository
from app.infrastructure.persistence.evaluation_reports import FileEvaluationReportReader
from app.infrastructure.persistence.insights import SQLiteRefusalReader
from app.infrastructure.persistence.insights import SQLiteInsightMetricReader
from app.infrastructure.persistence.document_citations import SQLiteDocumentCitationReader
from app.infrastructure.persistence.knowledge_deletion import SQLiteKnowledgeDeletionRepository
from app.infrastructure.auth.passwords import Argon2PasswordHasher
from app.infrastructure.auth.sqlite_repository import SQLiteAuthRepository
from app.infrastructure.persistence.knowledge_bases import (
    SQLiteKnowledgeBaseRepository,
)
from app.infrastructure.persistence.knowledge_registry import SQLiteKnowledgeReadRegistry
from app.infrastructure.persistence.ingestion_jobs import (
    SQLiteIngestionJobRepository,
    has_pending_ingestion_jobs,
)
from app.infrastructure.persistence.ingestion_sources import LocalIngestionSources
from app.infrastructure.persistence.ingestion_execution import (
    LocalIngestionExecutionGuard,
)
from app.infrastructure.reranking.dashscope import RemoteReranker
from app.infrastructure.vectorstores.chroma import ChromaVectorStore
from app.infrastructure.conversations.checkpoint import SQLiteCheckpointStore
from app.infrastructure.conversations.langgraph_agent import (
    LangGraphConversationAgent,
)
from app.infrastructure.conversations.sqlite_repository import (
    SQLiteConversationRepository,
)


class VectorStoreAdapter(
    VectorStoreReader,
    VectorStoreWriter,
    VectorStoreAdmin,
    KnowledgeContentReader,
    Protocol,
):
    """聚合组合根所需的向量存储读、写和管理能力。"""


class ConversationCheckpointAdapter(
    ConversationCheckpointStore,
    Protocol,
):
    """聚合组合根需要的 checkpoint、图编译和关闭能力。"""

    @property
    def checkpointer(self) -> object:
        """返回供 LangGraph 编译使用的 checkpoint saver。"""

    def close(self) -> None:
        """释放 checkpoint 数据库连接。"""


class KnowledgeReadRegistry(DocumentLifecycle, CuratedQuestionRepository, Protocol):
    """仅在组合根聚合共享物理仓储的两个独立应用端口。"""


VectorStoreFactory = Callable[
    [
        Path,
        str,
        Mapping[str, str | int] | None,
        bool,
    ],
    VectorStoreAdapter,
]
EmbeddingProviderFactory = Callable[
    [str | None, str, str, int],
    EmbeddingProvider,
]
ChunkerFactory = Callable[[int, int], Chunker]
ParserResolverFactory = Callable[[], ParserResolver]
KeywordIndexFactory = Callable[[VectorStoreAdapter], KeywordIndex]
RepairLogFactory = Callable[[Path], RepairLogWriter]
RerankerFactory = Callable[
    [str | None, str | None, str, str],
    Reranker | None,
]
LLMProviderFactory = Callable[
    [str | None, str, str, float | None, str | None],
    LLMProvider,
]
ConversationRepositoryFactory = Callable[[Path], ConversationRepository]
KnowledgeBaseRepositoryFactory = Callable[[Path], KnowledgeBaseRepository]
TraceRepositoryFactory = Callable[[Path], TraceRepository]
AuthRepositoryFactory = Callable[[Path], AuthRepository]
QualityEvaluatorFactory = Callable[
    [str | None, str, str, float | None, str | None],
    ConversationQualityEvaluator,
]
UnderstandingProviderFactory = Callable[
    [str | None, str, str, float | None, str | None],
    StructuredOutputProvider,
]
ConversationCheckpointStoreFactory = Callable[
    [Path], ConversationCheckpointAdapter
]
ConversationAgentFactory = Callable[
    [
        str | None,
        str,
        str,
        float | None,
        str | None,
        RetrievalService,
        object,
        int,
        int,
    ],
    ConversationAgent,
]


def _create_vector_store(
    path: Path,
    collection_name: str,
    collection_metadata: Mapping[str, str | int] | None,
    rebuild_collection: bool,
) -> VectorStoreAdapter:
    vector_store = ChromaVectorStore(
        path=path,
        collection_name=collection_name,
        collection_metadata=collection_metadata,
    )
    if rebuild_collection:
        vector_store.delete_collection()
        vector_store = ChromaVectorStore(
            path=path,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
        )
    return vector_store


def _create_embedding_provider(
    api_key: str | None,
    base_url: str,
    model: str,
    batch_size: int,
    *,
    timeout_seconds: float = 30.0,
    max_retries: int = 0,
) -> EmbeddingProvider:
    return OpenAICompatibleEmbeddingProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def _create_chunker(chunk_size: int, chunk_overlap: int) -> Chunker:
    return RecursiveDocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _create_parser_resolver() -> ParserResolver:
    return ParserFactory()


def _create_keyword_index(
    vector_store: VectorStoreAdapter,
) -> KeywordIndex:
    keyword_index = BM25Index()
    keyword_index.rebuild(vector_store.list_chunks())
    return keyword_index


def _create_repair_log(path: Path) -> RepairLogWriter:
    return RepairLog(path)


def _create_reranker(
    base_url: str | None,
    api_key: str | None,
    model: str,
    endpoint: str,
) -> Reranker | None:
    if base_url is None or not base_url.strip():
        return None
    return RemoteReranker(
        base_url=base_url,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
    )


def _create_llm_provider(
    api_key: str | None,
    base_url: str,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    *,
    timeout_seconds: float = 60.0,
    max_retries: int = 0,
    max_output_tokens: int = 4096,
    output_token_parameter: str = "max_tokens",
) -> LLMProvider:
    return OpenAICompatibleLLMProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_output_tokens=max_output_tokens,
        output_token_parameter=output_token_parameter,
    )


def _create_conversation_repository(
    path: Path,
) -> ConversationRepository:
    return SQLiteConversationRepository(path)


def _create_quality_evaluator(
    api_key: str | None, base_url: str, model: str,
    temperature: float | None, reasoning_effort: str | None, *,
    timeout_seconds: float = 60.0, max_retries: int = 0,
    max_output_tokens: int = 4096, output_token_parameter: str = "max_tokens",
    quality_output_mode: str = "json_schema",
    prompt_guard: PromptBudgetGuard | None = None,
) -> ModelQualityEvaluator:
    if not api_key or not api_key.strip():
        raise LLMMissingAPIKeyError("未配置LLM_API_KEY或JUDGE_API_KEY，无法创建在线裁判")
    host = urlsplit(base_url).hostname
    client = httpx.Client(trust_env=False) if host in {"localhost", "127.0.0.1", "::1"} else None
    try:
        options = {
            "api_key": api_key, "base_url": base_url, "model": model,
            "timeout": timeout_seconds, "max_retries": max_retries,
        }
        if client is not None:
            options["http_client"] = client
        if temperature is not None:
            options["temperature"] = temperature
        if reasoning_effort is not None:
            options["reasoning_effort"] = reasoning_effort
        if output_token_parameter == "max_tokens":
            options["extra_body"] = {"max_tokens": max_output_tokens}
        elif output_token_parameter == "max_completion_tokens":
            options["max_completion_tokens"] = max_output_tokens
        else:
            raise ValueError("输出上限参数名无效")
        chat_model = ChatOpenAI(**options)
        return ModelQualityEvaluator(
            chat_model,
            structured_provider=ModelStructuredOutputProvider(chat_model, mode=quality_output_mode),
            prompt_guard=prompt_guard,
            close_callback=client.close if client is not None else chat_model.root_client.close,
        )
    except BaseException:
        if client is not None:
            client.close()
        raise


def _create_knowledge_base_repository(path: Path) -> KnowledgeBaseRepository:
    return SQLiteKnowledgeBaseRepository(path)


def _create_knowledge_read_registry(path: Path) -> KnowledgeReadRegistry:
    return SQLiteKnowledgeReadRegistry(path)


def _create_ingestion_job_repository(path: Path) -> IngestionJobRepository:
    return SQLiteIngestionJobRepository(path)


def _create_trace_repository(path: Path) -> TraceRepository:
    return SQLiteTraceRepository(path)


def _create_trace_reader(trace_path: Path, conversation_path: Path) -> TraceReader:
    """为组合根创建不写业务数据的只读投影。

    Args:
        trace_path: 当前运行追踪路径。
        conversation_path: 当前运行业务路径。

    Returns:
        双库只读查询适配器。
    """
    return SQLiteTraceReader(trace_path, conversation_path)


def _create_auth_repository(path: Path) -> AuthRepository:
    return SQLiteAuthRepository(path)


def _create_feedback_admin_repository(path: Path) -> FeedbackAdminRepository:
    return SQLiteFeedbackAdminRepository(path)


def _create_share_repository(path: Path) -> ShareRepository:
    """装配当前运行业务库的分享存储。

    Args:
        path: 正式会话库路径。

    Returns:
        事务化的分享存储。
    """
    return SQLiteShareRepository(path)


def _create_refusal_reader(path: Path) -> RefusalReader:
    return SQLiteRefusalReader(path)


def _create_insight_metric_reader(path: Path) -> InsightMetricReader:
    return SQLiteInsightMetricReader(path)


def _create_document_citation_reader(path: Path) -> DocumentCitationReader:
    return SQLiteDocumentCitationReader(path)


def _create_knowledge_deletion_repository(path: Path) -> KnowledgeDeletionRepository:
    return SQLiteKnowledgeDeletionRepository(path)


def _create_understanding_provider(
    api_key: str | None,
    base_url: str,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    *,
    timeout_seconds: float = 10.0,
    max_output_tokens: int = 1024,
    output_token_parameter: str = "max_tokens",
    output_mode: str = "json_schema",
) -> ModelStructuredOutputProvider:
    if not api_key or not api_key.strip():
        raise LLMMissingAPIKeyError("未配置理解模型凭据")
    host = urlsplit(base_url).hostname
    is_local = host in {"localhost", "127.0.0.1", "::1"}
    client = httpx.Client(trust_env=False) if is_local else None
    try:
        options: dict[str, object] = {
            "api_key": api_key, "base_url": base_url, "model": model,
            "timeout": timeout_seconds, "max_retries": 0,
        }
        if client is not None:
            options["http_client"] = client
        if temperature is not None:
            options["temperature"] = temperature
        if reasoning_effort is not None:
            options["reasoning_effort"] = reasoning_effort
        if output_token_parameter == "max_tokens":
            options["extra_body"] = {"max_tokens": max_output_tokens}
        elif output_token_parameter == "max_completion_tokens":
            options["max_completion_tokens"] = max_output_tokens
        else:
            raise ValueError("输出上限参数名无效")
        chat_model = ChatOpenAI(**options)
        return ModelStructuredOutputProvider(
            chat_model, mode=output_mode,
            close_callback=(
                client.close if client is not None
                else chat_model.root_client.close
            ),
        )
    except BaseException:
        if client is not None:
            client.close()
        raise


def create_evaluation_judge_provider(
    settings: Settings,
) -> ModelStructuredOutputProvider:
    """创建独立评分模型，不继承生成器凭据或失败后猜测输出协议。

    Args:
        settings: 已显式配置独立地址、模型与密钥的评估设置。

    Returns:
        由调用方负责关闭的结构化评分端口，SDK隐藏重试固定为零。
    """
    settings.validate_evaluation_settings(require_judge=True)
    return _create_understanding_provider(
        settings.evaluation_judge_api_key,
        settings.evaluation_judge_base_url,
        settings.evaluation_judge_model,
        settings.evaluation_judge_temperature,
        settings.evaluation_judge_reasoning_effort,
        timeout_seconds=settings.evaluation_judge_timeout_seconds,
        max_output_tokens=settings.evaluation_judge_max_output_tokens,
        output_token_parameter=settings.evaluation_judge_output_token_parameter,
        output_mode=settings.evaluation_judge_output_mode,
    )


def _create_conversation_checkpoint_store(
    path: Path,
) -> ConversationCheckpointAdapter:
    return SQLiteCheckpointStore(path)


@dataclass(frozen=True)
class JudgeModelOptions:
    """证据判断与答案审核使用的独立模型配置。

    Attributes:
        api_key: 裁判服务凭据。
        base_url: 裁判服务地址。
        model: 裁判模型标识。
        reasoning_effort: 可选推理强度。
        timeout_seconds: 单次调用超时秒数。
    """

    api_key: str | None
    base_url: str
    model: str
    reasoning_effort: str | None
    timeout_seconds: float


def _create_conversation_agent(
    api_key: str | None,
    base_url: str,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    retrieval_service: RetrievalService,
    checkpointer: object,
    top_k: int,
    max_retrieval_calls: int,
    *,
    timeout_seconds: float = 60.0,
    max_retries: int = 0,
    turn_timeout_seconds: float = 180.0,
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    quality_output_mode: str = "json_schema",
    max_output_tokens: int = 4096,
    prompt_guard: PromptBudgetGuard | None = None,
    output_token_parameter: str = "max_tokens",
    judge: JudgeModelOptions | None = None,
) -> LangGraphConversationAgent:
    """创建支持标准 tool calling 的 ChatOpenAI 与 LangGraph 图。

    judge 为 None 时，证据判断和答案审核沿用 Agent 自身的聊天模型；
    提供时改用独立的裁判模型，并在 Agent 关闭时一并释放。
    """
    if api_key is None or not api_key.strip():
        raise LLMMissingAPIKeyError(
            "未配置 LLM_API_KEY，无法创建会话 Agent"
        )
    host = urlsplit(base_url).hostname
    is_local = host in {"localhost", "127.0.0.1", "::1"}
    http_client = httpx.Client(trust_env=False) if is_local else None
    model_options: dict[str, object] = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout": timeout_seconds,
        "max_retries": max_retries,
    }
    if output_token_parameter == "max_tokens":
        # 当前 LangChain 会重命名 max_tokens，用 extra_body 保留兼容端点契约。
        model_options["extra_body"] = {"max_tokens": max_output_tokens}
    elif output_token_parameter == "max_completion_tokens":
        model_options["max_completion_tokens"] = max_output_tokens
    else:
        if http_client is not None:
            http_client.close()
        raise ValueError("输出上限参数名无效")
    if temperature is not None:
        model_options["temperature"] = temperature
    if reasoning_effort is not None:
        model_options["reasoning_effort"] = reasoning_effort
    if http_client is not None:
        model_options["http_client"] = http_client
    judge_evaluator: ModelQualityEvaluator | None = None
    try:
        chat_model = ChatOpenAI(**model_options)
        if judge is not None:
            judge_evaluator = _create_quality_evaluator(
                judge.api_key, judge.base_url, judge.model, temperature,
                judge.reasoning_effort, timeout_seconds=judge.timeout_seconds,
                max_retries=max_retries, max_output_tokens=max_output_tokens,
                output_token_parameter=output_token_parameter,
                quality_output_mode=quality_output_mode, prompt_guard=prompt_guard,
            )
        agent_close = (
            http_client.close if http_client is not None
            else chat_model.root_client.close
        )

        def close_clients() -> None:
            try:
                agent_close()
            finally:
                if judge_evaluator is not None:
                    judge_evaluator.close()

        return LangGraphConversationAgent(
            model=chat_model,
            retrieval_service=retrieval_service,
            checkpointer=checkpointer,
            top_k=top_k,
            max_retrieval_calls=max_retrieval_calls,
            turn_timeout_seconds=turn_timeout_seconds,
            retrieval_strategy=retrieval_strategy,
            prompt_guard=prompt_guard,
            quality_evaluator=judge_evaluator or ModelQualityEvaluator(
                chat_model,
                structured_provider=ModelStructuredOutputProvider(
                    chat_model, mode=quality_output_mode,
                ),
                prompt_guard=prompt_guard,
            ),
            close_callback=close_clients,
        )
    except BaseException:
        if judge_evaluator is not None:
            judge_evaluator.close()
        if http_client is not None:
            http_client.close()
        raise


@dataclass(frozen=True)
class AdapterFactories:
    """保存组合根可替换且签名明确的 adapter factory 集合。

    Attributes:
        vector_store: 创建并按需重建向量 collection 的 factory。
        embedding_provider: 创建文本向量 provider 的 factory。
        chunker: 创建文档分块 adapter 的 factory。
        parser_resolver: 创建解析器选择 adapter 的 factory。
        keyword_index: 从共享向量存储创建关键词索引的 factory。
        repair_log: 创建文档删除修复日志的 factory。
        reranker: 创建可选远程重排 adapter 的 factory。
        llm_provider: 创建语言模型生成 provider 的 factory。
        token_counter: 创建可替换的模型输入预算计数器。
        knowledge_base_repository: 创建独立知识库目录的工厂。
        ingestion_job_repository: 创建同运行时的持久化任务队列工厂。
    """

    vector_store: VectorStoreFactory = _create_vector_store
    embedding_provider: EmbeddingProviderFactory = (
        _create_embedding_provider
    )
    chunker: ChunkerFactory = _create_chunker
    parser_resolver: ParserResolverFactory = _create_parser_resolver
    keyword_index: KeywordIndexFactory = _create_keyword_index
    repair_log: RepairLogFactory = _create_repair_log
    reranker: RerankerFactory = _create_reranker
    llm_provider: LLMProviderFactory = _create_llm_provider
    token_counter: Callable[[], TokenCounter] = Utf8TokenUpperBoundCounter
    knowledge_base_repository: KnowledgeBaseRepositoryFactory = (
        _create_knowledge_base_repository
    )
    knowledge_read_registry: Callable[[Path], KnowledgeReadRegistry] = (
        _create_knowledge_read_registry
    )
    ingestion_job_repository: Callable[[Path], IngestionJobRepository] = (
        _create_ingestion_job_repository
    )
    ingestion_sources: Callable[[Path], IngestionSources] = LocalIngestionSources
    ingestion_pending_probe: Callable[[Path], bool] = has_pending_ingestion_jobs
    ingestion_execution_guard: Callable[[Path], IngestionExecutionGuard] = (
        LocalIngestionExecutionGuard
    )
    quality_evaluator: QualityEvaluatorFactory = _create_quality_evaluator
    trace_repository: TraceRepositoryFactory = _create_trace_repository
    trace_reader: Callable[[Path, Path], TraceReader] = _create_trace_reader
    feedback_admin_repository: Callable[[Path], FeedbackAdminRepository] = _create_feedback_admin_repository
    share_repository: Callable[[Path], ShareRepository] = _create_share_repository
    evaluation_report_reader: Callable[[Path], EvaluationReportReader] = FileEvaluationReportReader
    refusal_reader: Callable[[Path], RefusalReader] = _create_refusal_reader
    insight_metric_reader: Callable[[Path], InsightMetricReader] = _create_insight_metric_reader
    document_citation_reader: Callable[[Path], DocumentCitationReader] = _create_document_citation_reader
    knowledge_deletion_repository: Callable[[Path], KnowledgeDeletionRepository] = _create_knowledge_deletion_repository
    auth_repository: AuthRepositoryFactory = _create_auth_repository
    password_hasher: Callable[[], PasswordHasher] = Argon2PasswordHasher
    understanding_provider: UnderstandingProviderFactory = (
        _create_understanding_provider
    )
    conversation_repository: ConversationRepositoryFactory = (
        _create_conversation_repository
    )
    conversation_checkpoint_store: ConversationCheckpointStoreFactory = (
        _create_conversation_checkpoint_store
    )
    conversation_agent: ConversationAgentFactory = (
        _create_conversation_agent
    )


def configured_adapter_factories(settings: Settings) -> AdapterFactories:
    """将请求策略绑定到默认工厂，保持可替换工厂的既有签名。

    Args:
        settings: 已校验的服务端配置。

    Returns:
        使用有限请求等待和明确检索策略的 adapter 工厂集合。
    """
    return AdapterFactories(
        understanding_provider=partial(
            _create_understanding_provider,
            timeout_seconds=settings.resolved_understanding_timeout_seconds,
            max_output_tokens=settings.resolved_understanding_max_output_tokens,
            output_token_parameter=settings.resolved_understanding_output_token_parameter,
            output_mode=settings.resolved_understanding_output_mode,
        ),
        quality_evaluator=partial(
            _create_quality_evaluator,
            timeout_seconds=settings.resolved_judge_timeout_seconds,
            max_retries=settings.model_max_retries,
            max_output_tokens=settings.llm_max_output_tokens,
            output_token_parameter=settings.llm_output_token_parameter,
            quality_output_mode=settings.quality_output_mode,
            prompt_guard=PromptBudgetGuard(
                Utf8TokenUpperBoundCounter(),
                settings.conversation_context_tokens - settings.llm_max_output_tokens,
            ),
        ),
        embedding_provider=partial(
            _create_embedding_provider,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.model_max_retries,
        ),
        llm_provider=partial(
            _create_llm_provider,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.model_max_retries,
            max_output_tokens=settings.llm_max_output_tokens,
            output_token_parameter=settings.llm_output_token_parameter,
        ),
        conversation_agent=partial(
            _create_conversation_agent,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.model_max_retries,
            turn_timeout_seconds=settings.agent_turn_timeout_seconds,
            retrieval_strategy=settings.agent_retrieval_strategy,
            quality_output_mode=settings.quality_output_mode,
            max_output_tokens=settings.llm_max_output_tokens,
            output_token_parameter=settings.llm_output_token_parameter,
            prompt_guard=PromptBudgetGuard(
                Utf8TokenUpperBoundCounter(),
                settings.conversation_context_tokens
                - settings.llm_max_output_tokens,
            ),
            # 未配置任何 JUDGE_* 时不传，保持 Agent 用自身模型判断的原行为。
            judge=JudgeModelOptions(
                api_key=settings.resolved_judge_api_key,
                base_url=settings.resolved_judge_base_url,
                model=settings.resolved_judge_model,
                reasoning_effort=settings.resolved_judge_reasoning_effort,
                timeout_seconds=settings.resolved_judge_timeout_seconds,
            ) if settings.judge_configured else None,
        ),
    )

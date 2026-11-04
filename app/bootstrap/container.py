# -*- coding: utf-8 -*-
"""按需装配、共享并关闭应用运行时组件。"""

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from threading import Condition
from threading import get_ident
from threading import RLock
from threading import Event
from typing import Generic
from typing import Protocol
from typing import TypeVar
from typing import cast
from typing import runtime_checkable
from uuid import UUID

from app.application.documents.service import DocumentService
from app.application.documents.listing import AdminDocumentListing
from app.application.documents.citations import CitationStatisticsService
from app.application.documents.chunks import AdminDocumentChunksService
from app.application.conversations.ports import ConversationAgent
from app.application.conversations.ports import ConversationRepository
from app.application.conversations.quality import ConversationQualityEvaluator
from app.application.knowledge.scopes import ScopeResolver
from app.application.knowledge.catalog import KnowledgeCatalogService
from app.application.knowledge.deletion import KnowledgeBaseDeletionService
from app.application.knowledge.browse import KnowledgeBrowseService
from app.application.conversations.service import ConversationService
from app.application.conversations.history_window import HistoryWindowBuilder
from app.application.ingestion.pipeline import IngestionPipeline
from app.application.ingestion.service import UploadService
from app.application.ingestion.jobs import IngestionJobService
from app.application.ingestion.job_pipeline import JobIngestionPipeline
from app.application.ingestion.job_runner import IngestionJobRunner
from app.application.ingestion.worker import IngestionWorker
from app.application.admission import QuestionAdmissionController
from app.application.ports.chunking import Chunker
from app.application.ports.embedding import EmbeddingProvider
from app.application.ports.generation import LLMProvider
from app.application.ports.ingestion_jobs import IngestionJobRepository
from app.application.ports.ingestion_sources import IngestionSources
from app.application.ports.keyword_index import KeywordIndex
from app.application.ports.knowledge_bases import KnowledgeBaseRepository
from app.application.ports.parsing import ParserResolver
from app.application.ports.repair_log import RepairLogWriter
from app.application.ports.reranking import Reranker
from app.application.ports.structured_output import StructuredOutputProvider
from app.application.ports.tracing import TraceRepository
from app.application.tracing import TraceService
from app.application.auth.service import AuthService
from app.application.feedback.service import FeedbackService
from app.application.shares import ShareService
from app.application.evaluation_reports import EvaluationReportService
from app.application.insights.clusters import RefusalClusterService
from app.application.errors import ShareStorageError
from app.application.tracing.queries import TraceQueryService
from app.application.tracing.usage import UsageQueryService
from app.application.ports.trace_queries import TraceReader
from app.bootstrap.trace_configuration import trace_configuration_version
from app.bootstrap.capabilities import DeferredEmbeddingProvider
from app.application.feedback.admin import FeedbackAdminService
from app.application.insights.refusals import RefusalInsightService
from app.application.insights.overview import InsightOverviewService
from app.application.tracing.auditing import AdminAuditService
from app.application.errors import ConversationStorageError
from app.application.errors import TraceStorageError
from app.application.errors import InsightStorageError
from app.application.errors import KnowledgeContentError
from app.application.query.generator import AnswerGenerator
from app.application.query.context_budget import PromptBudgetGuard
from app.application.query.prompting import PromptBuilder
from app.application.query.service import QueryService
from app.application.query.routed_service import RoutedQueryService
from app.application.query.orchestrator import QueryOrchestrator
from app.application.query.routing import DeterministicRouter
from app.application.query.understanding import ModelUnderstandingService
from app.application.query.fast_lane import FastLane
from app.application.query.knowledge_runner import KnowledgeRunner
from app.application.retrieval.service import RetrievalService
from app.application.retrieval.service import VectorRetriever
from app.bootstrap.factories import AdapterFactories
from app.bootstrap.factories import configured_adapter_factories
from app.bootstrap.factories import ConversationCheckpointAdapter
from app.bootstrap.factories import KnowledgeReadRegistry
from app.bootstrap.factories import VectorStoreAdapter
from app.config import Settings


@runtime_checkable
class _Closable(Protocol):
    def close(self) -> None:
        """释放 adapter 持有的资源。"""


class _ContainerState(Enum):
    """表示容器是否仍接受组件发布。"""

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


_UNSET = object()
_CONTAINER_CLOSED_MESSAGE = "应用容器正在关闭或已经关闭"
_T = TypeVar("_T")


@dataclass
class _LazySlot(Generic[_T]):
    """保存一个惰性组件及当前构造线程。"""

    value: object = _UNSET
    creator_thread_id: int | None = None


@dataclass(frozen=True)
class RuntimeOptions:
    """定义单次运行可覆盖的隔离存储和生成选项。

    Attributes:
        chroma_path: 可选 Chroma 持久化目录覆盖。
        collection_name: 可选 collection 名称覆盖。
        collection_metadata: 创建 collection 时写入的可选元数据。
        rebuild_collection: 是否重建当前精确 collection。
        llm_temperature: 可选语言模型生成温度覆盖。
        knowledge_base_db_path: 可选的独立知识库目录路径。
        conversation_db_path: 可选正式会话路径；评估默认随Chroma隔离。
        agent_checkpoint_db_path: 可选图状态路径；评估默认随Chroma隔离。
        trace_db_path: 可选追踪路径；评估默认随Chroma隔离。
        auth_db_path: 可选认证路径；隔离运行不得读取生产账号。
        evaluation_results_path: 可选发布报告目录，隔离运行默认不读生产报告。
    """

    chroma_path: Path | None = None
    collection_name: str | None = None
    collection_metadata: Mapping[str, str | int] | None = None
    rebuild_collection: bool = False
    llm_temperature: float | None = None
    knowledge_base_db_path: Path | None = None
    conversation_db_path: Path | None = None
    agent_checkpoint_db_path: Path | None = None
    trace_db_path: Path | None = None
    auth_db_path: Path | None = None
    evaluation_results_path: Path | None = None


class ApplicationContainer:
    """线程安全地惰性创建并共享应用组件。

    Attributes:
        vector_store: 共享向量存储 adapter。
        embedding_provider: 上传和查询共享的向量 provider。
        chunker: 共享文档分块 adapter。
        keyword_index: 与向量存储同步的关键词索引。
        ingestion_pipeline: 持有共享文档写锁的入库流水线。
        upload_service: 文档上传应用服务。
        document_service: 文档列表与删除应用服务。
        retrieval_service: 统一检索应用服务。
        query_service: 查询、生成和引用编排应用服务。
        knowledge_base_repository: 惰性创建的知识库目录，不依赖模型凭据。
    """

    def __init__(
        self,
        settings: Settings,
        runtime_options: RuntimeOptions,
        factories: AdapterFactories,
    ) -> None:
        """保存配置并初始化彼此独立的惰性锁和文档写锁。

        Args:
            settings: 环境和应用默认配置。
            runtime_options: 本次运行允许覆盖的隔离参数。
            factories: 签名明确且可替换的 adapter factories。

        Returns:
            None。
        """
        self._settings = settings
        self._runtime_options = runtime_options
        self._factories = factories
        self._lazy_lock = RLock()
        self._lazy_condition = Condition(self._lazy_lock)
        self._document_mutation_lock = RLock()
        self._ingestion_stop = Event()

        self._state = _ContainerState.OPEN
        self._active_build_count = 0
        self._close_release_in_progress = False
        self._closable_adapters: list[_Closable] = []
        self._owned_adapter_ids: set[int] = set()
        # 保留强引用，避免已清理候选的 id 被后续并发候选复用。
        self._owned_adapter_references: list[_Closable] = []

        self._vector_store_slot: _LazySlot[VectorStoreAdapter] = (
            _LazySlot()
        )
        self._embedding_provider_slot: _LazySlot[EmbeddingProvider] = (
            _LazySlot()
        )
        self._chunker_slot: _LazySlot[Chunker] = _LazySlot()
        self._parser_resolver_slot: _LazySlot[ParserResolver] = (
            _LazySlot()
        )
        self._keyword_index_slot: _LazySlot[KeywordIndex] = _LazySlot()
        self._repair_log_slot: _LazySlot[RepairLogWriter] = _LazySlot()
        self._reranker_slot: _LazySlot[Reranker | None] = _LazySlot()
        self._llm_provider_slot: _LazySlot[LLMProvider] = _LazySlot()
        self._understanding_provider_slot: _LazySlot[
            StructuredOutputProvider
        ] = _LazySlot()
        self._conversation_repository_slot: _LazySlot[
            ConversationRepository
        ] = _LazySlot()
        self._history_window_slot: _LazySlot[HistoryWindowBuilder] = _LazySlot()
        self._conversation_checkpoint_store_slot: _LazySlot[
            ConversationCheckpointAdapter
        ] = _LazySlot()
        self._conversation_agent_slot: _LazySlot[ConversationAgent] = (
            _LazySlot()
        )
        self._knowledge_base_repository_slot: _LazySlot[
            KnowledgeBaseRepository
        ] = _LazySlot()
        self._knowledge_catalog_service_slot: _LazySlot[KnowledgeCatalogService] = _LazySlot()
        self._knowledge_deletion_service_slot: _LazySlot[KnowledgeBaseDeletionService] = _LazySlot()
        self._knowledge_browse_service_slot: _LazySlot[KnowledgeBrowseService] = _LazySlot()
        self._knowledge_read_registry_slot: _LazySlot[KnowledgeReadRegistry] = _LazySlot()
        self._ingestion_job_repository_slot: _LazySlot[
            IngestionJobRepository
        ] = _LazySlot()
        self._ingestion_sources_slot: _LazySlot[IngestionSources] = _LazySlot()
        self._ingestion_job_service_slot: _LazySlot[IngestionJobService] = _LazySlot()
        self._job_ingestion_pipeline_slot: _LazySlot[JobIngestionPipeline] = _LazySlot()
        self._ingestion_job_runner_slot: _LazySlot[IngestionJobRunner] = _LazySlot()
        self._ingestion_worker_slot: _LazySlot[IngestionWorker] = _LazySlot()
        self._question_admission_slot: _LazySlot[
            QuestionAdmissionController
        ] = _LazySlot()
        self._trace_repository_slot: _LazySlot[TraceRepository] = _LazySlot()
        self._trace_service_slot: _LazySlot[TraceService] = _LazySlot()
        self._trace_query_service_slot: _LazySlot[TraceQueryService] = _LazySlot()
        self._trace_reader_slot: _LazySlot[TraceReader] = _LazySlot()
        self._usage_query_service_slot: _LazySlot[UsageQueryService] = _LazySlot()
        self._feedback_admin_service_slot: _LazySlot[FeedbackAdminService] = _LazySlot()
        self._refusal_insight_service_slot: _LazySlot[
            RefusalInsightService
        ] = _LazySlot()
        self._insight_overview_service_slot: _LazySlot[
            InsightOverviewService
        ] = _LazySlot()
        self._admin_audit_slot: _LazySlot[AdminAuditService] = _LazySlot()
        self._citation_statistics_slot: _LazySlot[CitationStatisticsService] = _LazySlot()
        self._admin_document_chunks_slot: _LazySlot[AdminDocumentChunksService] = _LazySlot()
        self._auth_service_slot: _LazySlot[AuthService] = _LazySlot()
        self._feedback_service_slot: _LazySlot[FeedbackService] = _LazySlot()
        self._share_service_slot: _LazySlot[ShareService] = _LazySlot()
        self._evaluation_report_slot: _LazySlot[EvaluationReportService] = _LazySlot()
        self._refusal_cluster_slot: _LazySlot[RefusalClusterService] = _LazySlot()

        self._ingestion_pipeline_slot: _LazySlot[IngestionPipeline] = (
            _LazySlot()
        )
        self._upload_service_slot: _LazySlot[UploadService] = _LazySlot()
        self._document_service_slot: _LazySlot[DocumentService] = (
            _LazySlot()
        )
        self._retrieval_service_slot: _LazySlot[RetrievalService] = (
            _LazySlot()
        )
        self._query_service_slot: _LazySlot[RoutedQueryService] = _LazySlot()
        self._baseline_query_service_slot: _LazySlot[QueryService] = _LazySlot()
        self._query_orchestrator_slot: _LazySlot[QueryOrchestrator] = _LazySlot()
        self._knowledge_runner_slot: _LazySlot[KnowledgeRunner] = _LazySlot()
        self._quality_evaluator_slot: _LazySlot[ConversationQualityEvaluator] = _LazySlot()
        self._conversation_service_slot: _LazySlot[ConversationService] = (
            _LazySlot()
        )

    @property
    def knowledge_base_repository(self) -> KnowledgeBaseRepository:
        """返回本运行时独立的共享知识库目录。

        Returns:
            不初始化向量或聊天模型的目录端口。
        """
        return self._get_or_create(
            self._knowledge_base_repository_slot,
            self._create_knowledge_base_repository,
            owned_adapter=False,
        )

    @property
    def knowledge_catalog_service(self) -> KnowledgeCatalogService:
        """只从组合根装配目录用例，不触发Embedding或聊天模型。

        Returns:
            与入库／删除共享锁的元数据目录服务。
        """
        return self._get_or_create(
            self._knowledge_catalog_service_slot,
            lambda: KnowledgeCatalogService(
                self.knowledge_base_repository, self.vector_store,
                self._document_mutation_lock,
                publication=self.ingestion_job_repository,
            ), owned_adapter=False,
        )

    @property
    def knowledge_deletion_service(self) -> KnowledgeBaseDeletionService:
        """返回有持久恢复计划的知识库维护用例，不装配模型。

        Returns:
            与所有文档维护共用写锁的删除服务。
        """
        return self._get_or_create(
            self._knowledge_deletion_service_slot,
            self._create_knowledge_deletion_service, owned_adapter=False,
        )

    def _create_knowledge_deletion_service(self) -> KnowledgeBaseDeletionService:
        catalog = self.knowledge_base_repository
        jobs = self.ingestion_job_repository
        path = self._knowledge_database_path()
        return KnowledgeBaseDeletionService(
            catalog, self.vector_store, jobs,
            self._factories.knowledge_deletion_repository(path),
            self.document_service, self._job_sources_path(),
            self._document_mutation_lock, self.admin_audit,
        )

    @property
    def knowledge_browse_service(self) -> KnowledgeBrowseService:
        """返回不加载模型的只读文档分页与原文窗口服务。

        Returns:
            与文档入库／删除共享锁的浏览用例。
        """
        return self._get_or_create(
            self._knowledge_browse_service_slot,
            lambda: KnowledgeBrowseService(
                self.knowledge_base_repository, self.vector_store,
                self.knowledge_read_registry, self.knowledge_read_registry,
                self._document_mutation_lock,
                publication=self.ingestion_job_repository,
            ), owned_adapter=False,
        )

    def _create_knowledge_base_repository(self) -> KnowledgeBaseRepository:
        return self._factories.knowledge_base_repository(self._knowledge_database_path())

    @property
    def knowledge_read_registry(self) -> KnowledgeReadRegistry:
        """共享目录路径上的预设与删除记录，不触发向量模型。

        Returns:
            由两个独立应用端口使用的持久化登记。
        """
        return self._get_or_create(
            self._knowledge_read_registry_slot,
            self._create_knowledge_read_registry, owned_adapter=False,
        )

    def _create_knowledge_read_registry(self) -> KnowledgeReadRegistry:
        # 预设的外键依赖既有目录表；同一路径派生规则也必须用于评估。
        self.knowledge_base_repository
        return self._factories.knowledge_read_registry(self._knowledge_database_path())

    @property
    def ingestion_job_repository(self) -> IngestionJobRepository:
        """惰性共享任务登记，不因读取任务而构造模型或打开生产路径。

        Returns:
            与本运行时知识库目录同路径的持久化任务端口。
        """
        return self._get_or_create(
            self._ingestion_job_repository_slot,
            self._create_ingestion_job_repository,
            owned_adapter=False,
        )

    def _create_ingestion_job_repository(self) -> IngestionJobRepository:
        # 任务外键必须引用本runtime已初始化的目录，而非生产目录。
        self.knowledge_base_repository
        return self._factories.ingestion_job_repository(
            self._knowledge_database_path(),
        )

    @property
    def ingestion_sources(self) -> IngestionSources:
        """惰性取得本运行时源文件仓储，不构造模型或索引。

        Returns:
            受控源文件及已验证私有快照端口。
        """
        return self._get_or_create(
            self._ingestion_sources_slot,
            lambda: self._factories.ingestion_sources(self._job_sources_path()),
            owned_adapter=False,
        )

    @property
    def ingestion_job_service(self) -> IngestionJobService:
        """仅装配入队和进度读取所需组件，不执行解析或向量化。

        Returns:
            共享持久化任务应用服务。
        """
        return self._get_or_create(
            self._ingestion_job_service_slot,
            lambda: IngestionJobService(
                self.knowledge_base_repository, self.ingestion_job_repository,
                self.ingestion_sources, self._get_parser_resolver(),
                max_upload_size=self._settings.max_upload_size,
                capacity=self._settings.ingestion_queue_capacity,
            ), owned_adapter=False,
        )

    @property
    def ingestion_job_runner(self) -> IngestionJobRunner:
        """装配惰性执行器，真正认领任务之后才构造向量客户端。

        Returns:
            共享写锁及物理索引排他边界的单任务执行器。
        """
        return self._get_or_create(
            self._ingestion_job_runner_slot,
            self._create_ingestion_job_runner, owned_adapter=False,
        )

    def _create_ingestion_job_runner(self) -> IngestionJobRunner:
        chroma_path = self._runtime_options.chroma_path or self._settings.chroma_path
        collection = self._runtime_options.collection_name or self._settings.collection_name
        runtime_id = sha256(collection.encode()).hexdigest()[:16]
        guard = self._factories.ingestion_execution_guard(
            chroma_path.resolve() / ".ingestion-locks" / f"{runtime_id}.lock",
        )
        return IngestionJobRunner(
            self.ingestion_job_repository, self.ingestion_sources,
            lambda: self.job_ingestion_pipeline,
            lambda: self.ingestion_pipeline, guard,
            mutation_lock=self._document_mutation_lock,
        )

    @property
    def job_ingestion_pipeline(self) -> JobIngestionPipeline:
        """复用现有解析、分块及双索引，增加真实阶段进度。

        Returns:
            只有后台执行时才打开模型的分阶段流水线。
        """
        return self._get_or_create(
            self._job_ingestion_pipeline_slot,
            lambda: JobIngestionPipeline(
                self.chunker, self.embedding_provider, self.vector_store,
                self.keyword_index, self._get_parser_resolver(),
                self.ingestion_pipeline,
                embedding_batch_size=self._settings.embedding_batch_size,
                mutation_lock=self._document_mutation_lock,
            ), owned_adapter=False,
        )

    def _job_sources_path(self) -> Path:
        if self._runtime_options.chroma_path is None:
            return self._settings.docs_path
        collection = self._runtime_options.collection_name or self._settings.collection_name
        runtime_id = sha256(collection.encode()).hexdigest()[:16]
        return self._runtime_options.chroma_path / "sources" / runtime_id

    @property
    def ingestion_worker(self) -> IngestionWorker:
        """取得尚未自动启动的生命周期组件，不隐式装配模型。

        Returns:
            单消费线程的后台执行组件。
        """
        return self._get_or_create(
            self._ingestion_worker_slot,
            lambda: IngestionWorker(
                lambda: self._factories.ingestion_pending_probe(
                    self._knowledge_database_path(),
                ),
                lambda: self.ingestion_job_runner, self._ingestion_stop,
                poll_seconds=self._settings.ingestion_poll_seconds,
            ), owned_adapter=False,
        )

    def start_ingestion_worker(self) -> None:
        """仅由HTTP生命周期启动，禁用时不创建消费组件。

        Returns:
            None。
        """
        if self._settings.ingestion_worker_enabled:
            self.ingestion_worker.start()

    @property
    def question_admission(self) -> QuestionAdmissionController:
        """两个入口及当前用户额度读取共享同一准入控制器。

        Returns:
            不保存问题正文、不打开模型的进程内窗口和队列。
        """
        return self._get_or_create(
            self._question_admission_slot,
            lambda: QuestionAdmissionController(
                per_user_limit=self._settings.question_rate_limit_per_minute,
                max_active=self._settings.question_max_active,
                max_waiting=self._settings.question_max_waiting,
                queue_timeout_seconds=self._settings.question_queue_timeout_seconds,
            ), owned_adapter=False,
        )

    def _knowledge_database_path(self) -> Path:
        selected_path = self._runtime_options.knowledge_base_db_path
        if selected_path is None and self._runtime_options.chroma_path is not None:
            collection_name = (
                self._runtime_options.collection_name
                or self._settings.collection_name
            )
            # 评估的collection名只作为隔离标识，不可直接参与文件系统路径。
            catalog_id = sha256(collection_name.encode("utf-8")).hexdigest()[:16]
            selected_path = (
                self._runtime_options.chroma_path
                / "catalogs" / f"{catalog_id}.sqlite3"
            )
        if selected_path is None:
            selected_path = self._settings.knowledge_base_db_path
        return selected_path

    @property
    def trace_repository(self) -> TraceRepository:
        """返回当前运行隔离的追踪存储。

        Returns:
            不依赖模型凭据的存储端口。
        """
        return self._get_or_create(
            self._trace_repository_slot,
            lambda: self._factories.trace_repository(self._runtime_database_path(
                "traces", self._runtime_options.trace_db_path,
                self._settings.trace_db_path,
            )),
            owned_adapter=False,
        )

    @property
    def feedback_service(self) -> FeedbackService:
        """返回与正式会话数据库共享事务边界的反馈服务。

        Returns:
            惰性共享反馈用例。
        """
        return self._get_or_create(
            self._feedback_service_slot,
            lambda: FeedbackService(self._get_conversation_repository()),
            owned_adapter=False,
        )

    @property
    def trace_query_service(self) -> TraceQueryService:
        """获取与写入侧使用同一隔离路径的后台查询用例。

        Returns:
            不依赖模型凭据的只读追踪查询服务。
        """
        return self._get_or_create(
            self._trace_query_service_slot,
            lambda: TraceQueryService(self._get_trace_reader()),
            owned_adapter=False,
        )

    @property
    def usage_query_service(self) -> UsageQueryService:
        """返回使用同一只读快照来源的用量统计服务。

        Returns:
            惰性共享且不依赖模型凭据的统计用例。
        """
        return self._get_or_create(
            self._usage_query_service_slot,
            lambda: UsageQueryService(
                self._get_trace_reader(),
                retention_days=self._settings.trace_retention_days,
                catalog=self.knowledge_base_repository,
            ),
            owned_adapter=False,
        )

    def _get_trace_reader(self) -> TraceReader:
        return self._get_or_create(
            self._trace_reader_slot, self._create_trace_reader,
            owned_adapter=False,
        )

    @property
    def admin_audit(self) -> AdminAuditService:
        """返回同一运行、同一保留策略的后台访问审计。

        Returns:
            不保存正文的审计用例。
        """
        return self._get_or_create(
            self._admin_audit_slot,
            lambda: AdminAuditService(self.trace_repository, self._settings.trace_retention_days),
            owned_adapter=False,
        )

    @property
    def feedback_admin_service(self) -> FeedbackAdminService:
        """只为后台构建反馈投影，不改变普通会话归属规则。

        Returns:
            带访问审计的分类与查询用例。
        """
        return self._get_or_create(
            self._feedback_admin_service_slot, self._create_feedback_admin_service,
            owned_adapter=False,
        )

    @property
    def refusal_insight_service(self) -> RefusalInsightService:
        """获取隔离且带访问审计的拒答洞察，不需要模型凭据。

        Returns:
            惰性共享的后台拒答用例。
        """
        return self._get_or_create(
            self._refusal_insight_service_slot,
            self._create_refusal_insight_service, owned_adapter=False,
        )

    def _create_refusal_insight_service(self) -> RefusalInsightService:
        try:
            self._get_conversation_repository()
        except (ConversationStorageError, OSError) as error:
            raise InsightStorageError("拒答洞察存储不可用") from error
        path = self._runtime_database_path(
            "conversations", self._runtime_options.conversation_db_path,
            self._settings.conversation_db_path,
        )
        return RefusalInsightService(
            self._factories.refusal_reader(path), self.admin_audit,
        )

    @property
    def refusal_cluster_service(self) -> RefusalClusterService:
        """获取审计与模型均惰性、存储隔离的语义聚类用例。

        Returns:
            当前运行固定模型配置的聚类服务。
        """
        return self._get_or_create(
            self._refusal_cluster_slot, self._create_refusal_cluster_service,
            owned_adapter=False,
        )

    def _create_refusal_cluster_service(self) -> RefusalClusterService:
        try:
            self._get_conversation_repository()
        except (ConversationStorageError, OSError) as error:
            raise InsightStorageError("拒答聚类存储不可用") from error
        path = self._runtime_database_path(
            "conversations", self._runtime_options.conversation_db_path,
            self._settings.conversation_db_path,
        )
        return RefusalClusterService(
            self._factories.refusal_reader(path), self.admin_audit,
            DeferredEmbeddingProvider(lambda: self.embedding_provider),
            self._settings.embedding_model,
            self._settings.refusal_cluster_similarity_threshold,
        )

    @property
    def insight_overview_service(self) -> InsightOverviewService:
        """返回不依赖模型或追踪正文的正式问答概览。

        Returns:
            带后台访问审计的惰性共享统计用例。
        """
        return self._get_or_create(
            self._insight_overview_service_slot,
            self._create_insight_overview_service, owned_adapter=False,
        )

    def _create_insight_overview_service(self) -> InsightOverviewService:
        try:
            self._get_conversation_repository()
        except (ConversationStorageError, OSError) as error:
            raise InsightStorageError("概览统计存储不可用") from error
        path = self._runtime_database_path(
            "conversations", self._runtime_options.conversation_db_path,
            self._settings.conversation_db_path,
        )
        return InsightOverviewService(
            self._factories.insight_metric_reader(path),
            self.knowledge_base_repository, self.admin_audit,
        )

    def _create_feedback_admin_service(self) -> FeedbackAdminService:
        try:
            self._get_conversation_repository()
        except (ConversationStorageError, OSError) as error:
            raise InsightStorageError("后台反馈存储不可用") from error
        path = self._runtime_database_path(
            "conversations", self._runtime_options.conversation_db_path,
            self._settings.conversation_db_path,
        )
        return FeedbackAdminService(self._factories.feedback_admin_repository(path), self.admin_audit)

    @property
    def share_service(self) -> ShareService:
        """获取只依赖正式业务存储的分享用例。

        Returns:
            本运行隔离的共享用例。
        """
        return self._get_or_create(
            self._share_service_slot, self._create_share_service,
            owned_adapter=False,
        )

    def _create_share_service(self) -> ShareService:
        try:
            self._get_conversation_repository()
        except (ConversationStorageError, OSError) as error:
            raise ShareStorageError("分享存储暂时不可用") from error
        path = self._runtime_database_path(
            "conversations", self._runtime_options.conversation_db_path,
            self._settings.conversation_db_path,
        )
        return ShareService(self._factories.share_repository(path))

    @property
    def evaluation_report_service(self) -> EvaluationReportService:
        """返回不启动模型、数据库或评估任务的报告服务。

        Returns:
            当前运行的只读报告服务。
        """
        return self._get_or_create(
            self._evaluation_report_slot, self._create_evaluation_report_service,
            owned_adapter=False,
        )

    def _create_evaluation_report_service(self) -> EvaluationReportService:
        path = self._runtime_options.evaluation_results_path
        if path is None:
            path = self._settings.evaluation_results_path
            if self._runtime_options.chroma_path is not None:
                collection = (
                    self._runtime_options.collection_name
                    or self._settings.collection_name
                )
                runtime_id = sha256(collection.encode()).hexdigest()[:16]
                path = (
                    self._runtime_options.chroma_path / "runtimes"
                    / runtime_id / "evaluation-results"
                )
        return EvaluationReportService(
            self._factories.evaluation_report_reader(path),
        )

    def _create_trace_reader(self) -> TraceReader:
        # 先初始化空表；读取器自身只用mode=ro，不承担建库或迁移职责。
        self.trace_repository
        try:
            self._get_conversation_repository()
        except (ConversationStorageError, OSError) as error:
            raise TraceStorageError("追踪关联存储不可用") from error
        return self._factories.trace_reader(
            self._runtime_database_path(
                "traces", self._runtime_options.trace_db_path,
                self._settings.trace_db_path,
            ),
            self._runtime_database_path(
                "conversations", self._runtime_options.conversation_db_path,
                self._settings.conversation_db_path,
            ),
        )

    @property
    def settings(self) -> Settings:
        """返回只读配置，供HTTP安全策略使用，不向客户端序列化。

        Returns:
            当前运行的冻结配置。
        """
        return self._settings

    def prune_expired_traces(self) -> int:
        """维护当前运行追踪库，只清理已经到期的记录。

        Returns:
            实际删除的过期追踪数量。
        """
        return self.trace_repository.prune_expired()

    @property
    def auth_service(self) -> AuthService:
        """返回惰性共享的认证服务，不自动创建用户。

        Returns:
            具有独立运行路径的认证用例。
        """
        return self._get_or_create(
            self._auth_service_slot,
            lambda: AuthService(
                self._factories.auth_repository(self._runtime_database_path(
                    "auth", self._runtime_options.auth_db_path,
                    self._settings.auth_db_path,
                )),
                self._factories.password_hasher(),
                session_seconds=self._settings.auth_session_seconds,
                idle_seconds=self._settings.auth_idle_seconds,
                login_attempts=self._settings.auth_login_attempts,
                login_window_seconds=self._settings.auth_login_window_seconds,
            ),
            owned_adapter=False,
        )

    @property
    def trace_service(self) -> TraceService:
        """返回按配置保存内容和费用版本的共享追踪服务。

        Returns:
            请求间不保存常驻会话字典的应用服务。
        """
        return self._get_or_create(
            self._trace_service_slot,
            lambda: TraceService(
                self.trace_repository,
                capture_queries=self._settings.trace_capture_queries,
                retention_days=self._settings.trace_retention_days,
                prices=self._settings.trace_model_prices,
                configuration_version=trace_configuration_version(
                    self._settings, self._runtime_options.llm_temperature,
                ),
            ),
            owned_adapter=False,
        )

    def _runtime_database_path(
        self, name: str, override: Path | None, configured: Path,
    ) -> Path:
        if override is not None:
            return override
        if self._runtime_options.chroma_path is None:
            return configured
        collection = self._runtime_options.collection_name or self._settings.collection_name
        runtime_id = sha256(collection.encode()).hexdigest()[:16]
        return self._runtime_options.chroma_path / "runtimes" / runtime_id / f"{name}.sqlite3"

    @property
    def vector_store(self) -> VectorStoreAdapter:
        """返回本次运行共享的惰性向量存储。

        Returns:
            使用运行时存储覆盖或 Settings 默认值创建的
            adapter。
        """
        return self._get_or_create(
            self._vector_store_slot,
            self._create_vector_store,
            owned_adapter=True,
        )

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """返回上传和查询共享的惰性向量 provider。

        Returns:
            使用 Settings 模型配置创建的向量 provider。
        """
        return self._get_or_create(
            self._embedding_provider_slot,
            self._create_embedding_provider,
            owned_adapter=True,
        )

    @property
    def chunker(self) -> Chunker:
        """返回使用 Settings 分块参数创建的惰性 adapter。

        Returns:
            共享文档分块 adapter。
        """
        return self._get_or_create(
            self._chunker_slot,
            self._create_chunker,
            owned_adapter=True,
        )

    @property
    def keyword_index(self) -> KeywordIndex:
        """返回绑定共享向量存储的惰性关键词索引。

        Returns:
            已从当前向量文本块初始化的关键词索引。
        """
        return self._get_or_create(
            self._keyword_index_slot,
            self._create_keyword_index,
            owned_adapter=True,
        )

    @property
    def ingestion_pipeline(self) -> IngestionPipeline:
        """返回注入 ParserResolver 和共享写锁的入库流水线。

        Returns:
            上传和文档删除共同使用的入库流水线。
        """
        return self._get_or_create(
            self._ingestion_pipeline_slot,
            self._create_ingestion_pipeline,
            owned_adapter=False,
        )

    @property
    def upload_service(self) -> UploadService:
        """返回复用共享入库流水线的上传服务。

        Returns:
            使用 Settings 上传约束创建的应用服务。
        """
        return self._get_or_create(
            self._upload_service_slot,
            self._create_upload_service,
            owned_adapter=False,
        )

    @property
    def document_service(self) -> DocumentService:
        """返回复用共享存储和入库流水线的文档服务。

        Returns:
            文档列表与删除应用服务。
        """
        return self._get_or_create(
            self._document_service_slot,
            self._create_document_service,
            owned_adapter=False,
        )

    @property
    def retrieval_service(self) -> RetrievalService:
        """返回共享向量、关键词和可选重排的检索服务。

        Returns:
            支持全部已配置策略的统一检索应用服务。
        """
        return self._get_or_create(
            self._retrieval_service_slot,
            self._create_retrieval_service,
            owned_adapter=False,
        )

    @property
    def query_service(self) -> RoutedQueryService:
        """返回共享检索和生成 adapter 的查询服务。

        Returns:
            同时支持普通、评估和流式查询的应用服务。
        """
        return self._get_or_create(
            self._query_service_slot,
            self._create_query_service,
            owned_adapter=False,
        )

    @property
    def conversation_service(self) -> ConversationService:
        """返回共享正式历史、checkpoint 与 Agent 的会话服务。"""
        return self._get_or_create(
            self._conversation_service_slot,
            self._create_conversation_service,
            owned_adapter=False,
        )

    def close(self) -> None:
        """按创建逆序在惰性锁外关闭已实例化 adapter。

        Returns:
            None。

        Raises:
            BaseException: 所有 adapter 尝试关闭后重新抛出的首个
                关闭异常。
        """
        # 先发停止信号并等执行退出，再关闭其仍在使用的模型／SQLite适配器。
        # 共享信号也防止close与后台组件构造并发时启动一个游离线程。
        self._ingestion_stop.set()
        with self._lazy_condition:
            worker = self._ingestion_worker_slot.value
            admission = self._question_admission_slot.value
        if isinstance(admission, QuestionAdmissionController):
            admission.close()
        if isinstance(worker, IngestionWorker):
            worker.close()
        with self._lazy_condition:
            if self._state is not _ContainerState.OPEN:
                return
            self._state = _ContainerState.CLOSING
            self._close_release_in_progress = True
            adapters = list(reversed(self._closable_adapters))
            self._closable_adapters.clear()
            self._lazy_condition.notify_all()

        first_error: BaseException | None = None
        try:
            for adapter in adapters:
                try:
                    adapter.close()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        finally:
            with self._lazy_condition:
                self._close_release_in_progress = False
                self._finish_closing_if_ready_locked()
                self._lazy_condition.notify_all()

        if first_error is not None:
            raise first_error

    def _create_vector_store(self) -> VectorStoreAdapter:
        chroma_path = self._runtime_options.chroma_path
        if chroma_path is None:
            chroma_path = self._settings.chroma_path
        collection_name = self._runtime_options.collection_name
        if collection_name is None:
            collection_name = self._settings.collection_name
        return self._factories.vector_store(
            chroma_path,
            collection_name,
            self._runtime_options.collection_metadata,
            self._runtime_options.rebuild_collection,
        )

    def _create_embedding_provider(self) -> EmbeddingProvider:
        return self._factories.embedding_provider(
            self._settings.resolved_embedding_api_key,
            self._settings.embedding_base_url,
            self._settings.embedding_model,
            self._settings.embedding_batch_size,
        )

    def _create_chunker(self) -> Chunker:
        return self._factories.chunker(
            self._settings.chunk_size,
            self._settings.chunk_overlap,
        )

    def _create_keyword_index(self) -> KeywordIndex:
        return self._factories.keyword_index(self.vector_store)

    def _create_ingestion_pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            chunker=self.chunker,
            embedder=DeferredEmbeddingProvider(lambda: self.embedding_provider),
            vector_store=self.vector_store,
            embedding_batch_size=self._settings.embedding_batch_size,
            parser_resolver=self._get_parser_resolver(),
            keyword_index=self.keyword_index,
            mutation_lock=self._document_mutation_lock,
        )

    def _create_upload_service(self) -> UploadService:
        return UploadService(
            docs_path=self._job_sources_path(),
            max_upload_size=self._settings.max_upload_size,
            ingestion_pipeline=self.ingestion_pipeline,
        )

    @property
    def citation_statistics(self) -> CitationStatisticsService:
        """返回当前业务库的只读正式引用统计，不初始化模型。

        Returns:
            惰性共享的统计用例。
        """
        return self._get_or_create(
            self._citation_statistics_slot, self._create_citation_statistics,
            owned_adapter=False,
        )

    def _create_citation_statistics(self) -> CitationStatisticsService:
        try:
            self._get_conversation_repository()
        except (ConversationStorageError, OSError) as error:
            raise KnowledgeContentError("正式引用统计暂时不可用") from error
        path = self._runtime_database_path(
            "conversations", self._runtime_options.conversation_db_path,
            self._settings.conversation_db_path,
        )
        return CitationStatisticsService(self._factories.document_citation_reader(path))

    @property
    def admin_document_chunks(self) -> AdminDocumentChunksService:
        """返回仅供管理员的真实分页正文诊断。

        Returns:
            共享变更锁和审计的诊断用例。
        """
        return self._get_or_create(
            self._admin_document_chunks_slot,
            lambda: AdminDocumentChunksService(
                self.vector_store, self.ingestion_job_repository,
                self.citation_statistics, self.admin_audit,
                self._document_mutation_lock,
            ), owned_adapter=False,
        )

    def _create_document_service(self) -> DocumentService:
        return DocumentService(
            docs_path=self._job_sources_path(),
            vector_store=self.vector_store,
            ingestion_pipeline=lambda: self.ingestion_pipeline,
            repair_log=self._get_repair_log(),
            mutation_lock=self._document_mutation_lock,
            deletion_registry=self.knowledge_read_registry,
            jobs=self.ingestion_job_repository,
            listing=AdminDocumentListing(
                self._job_sources_path(), self.vector_store,
                self.ingestion_job_repository, self._document_mutation_lock,
                statistics=lambda: self.citation_statistics,
            ),
        )

    def _create_retrieval_service(self) -> RetrievalService:
        vector_retriever = VectorRetriever(
            vector_store=self.vector_store,
            embedder=self.embedding_provider,
        )
        return RetrievalService(
            vector_retriever=vector_retriever,
            keyword_retriever=self.keyword_index,
            reranker=self._get_reranker(),
            top_k_rerank=self._settings.top_k_rerank,
            publication=self.ingestion_job_repository,
        )

    @property
    def baseline_query_service(self) -> QueryService:
        """返回只供离线评估的原始三策略基线，不作为HTTP入口。

        Returns:
            没有被HTTP依赖引用的基线服务。
        """
        return self._get_or_create(
            self._baseline_query_service_slot, self._create_baseline_query_service,
            owned_adapter=False,
        )

    @property
    def query_orchestrator(self) -> QueryOrchestrator:
        """返回两种入口共用的唯一分层编排器实例。

        Returns:
            不提前创建知识模型客户端的应用编排器。
        """
        return self._get_or_create(
            self._query_orchestrator_slot, self._create_query_orchestrator,
            owned_adapter=False,
        )

    def _create_query_orchestrator(self) -> QueryOrchestrator:
        return QueryOrchestrator(
            ModelUnderstandingService(
                self._get_understanding_provider,
                confidence_threshold=(
                    self._settings.understanding_confidence_threshold
                ),
                prompt_guard=PromptBudgetGuard(
                    self._factories.token_counter(),
                    self._settings.resolved_understanding_context_tokens
                    - self._settings.resolved_understanding_max_output_tokens,
                ),
            ),
            DeterministicRouter(
                self._settings.understanding_confidence_threshold,
            ),
            self.knowledge_base_repository, ScopeResolver(self.knowledge_base_repository),
            self._get_knowledge_runner,
            strategy=self._settings.knowledge_retrieval_strategy,
            top_k=self._settings.top_k,
            turn_timeout_seconds=self._settings.agent_turn_timeout_seconds,
            traces=self.trace_service,
        )

    def _get_understanding_provider(self) -> StructuredOutputProvider:
        return self._get_or_create(
            self._understanding_provider_slot,
            lambda: self._factories.understanding_provider(
                self._settings.resolved_understanding_api_key,
                self._settings.resolved_understanding_base_url,
                self._settings.resolved_understanding_model,
                self._settings.understanding_temperature,
                self._settings.resolved_understanding_reasoning_effort,
            ),
            owned_adapter=True,
        )

    def _get_knowledge_runner(self) -> KnowledgeRunner:
        return self._get_or_create(
            self._knowledge_runner_slot, self._create_knowledge_runner,
            owned_adapter=False,
        )

    def _create_knowledge_runner(self) -> KnowledgeRunner:
        quality = self._get_or_create(
            self._quality_evaluator_slot,
            # 证据判断与答案审核使用 JUDGE_*，未配置的项沿用 LLM_*。
            lambda: self._factories.quality_evaluator(
                self._settings.resolved_judge_api_key,
                self._settings.resolved_judge_base_url,
                self._settings.resolved_judge_model,
                self._settings.agent_temperature,
                self._settings.resolved_judge_reasoning_effort,
            ), owned_adapter=True,
        )
        generator = AnswerGenerator(
            PromptBuilder(), self._get_llm_provider(),
            prompt_guard=PromptBudgetGuard(
                self._factories.token_counter(),
                self._settings.conversation_context_tokens - self._settings.llm_max_output_tokens,
            ),
        )
        return KnowledgeRunner(
            FastLane(
                self.retrieval_service, quality, generator,
                strategy=self._settings.knowledge_retrieval_strategy,
            ), self._get_conversation_agent(),
        )

    def _create_query_service(self) -> RoutedQueryService:
        return RoutedQueryService(
            self.query_orchestrator, self._cleanup_ephemeral_thread,
            admission=self.question_admission,
            traces=self.trace_service,
        )

    def _cleanup_ephemeral_thread(self, request_id: UUID) -> None:
        with self._lazy_lock:
            store = self._conversation_checkpoint_store_slot.value
        if store is not _UNSET:
            cast(ConversationCheckpointAdapter, store).delete_thread(request_id)

    def _create_baseline_query_service(self) -> QueryService:
        answer_generator = AnswerGenerator(
            prompt_builder=PromptBuilder(),
            llm_provider=self._get_llm_provider(),
            prompt_guard=PromptBudgetGuard(
                self._factories.token_counter(),
                self._settings.conversation_context_tokens
                - self._settings.llm_max_output_tokens,
            ),
        )
        return QueryService(
            retriever=self.retrieval_service,
            answer_generator=answer_generator,
        )

    @property
    def history_window(self) -> HistoryWindowBuilder:
        """让线上会话和隔离多轮评估共用同一历史预算装配。

        Returns:
            不改写正式原文的历史选择器。
        """
        return self._get_or_create(
            self._history_window_slot,
            lambda: HistoryWindowBuilder(
                counter=self._factories.token_counter(),
                token_budget=(self._settings.conversation_context_tokens - self._settings.conversation_reserved_tokens),
                assistant_token_limit=self._settings.conversation_assistant_token_limit,
            ), owned_adapter=False,
        )

    def cleanup_temporary_execution(self, request_id: UUID) -> None:
        """仅供可信无状态用例清理自己创建的临时图线程。

        Args:
            request_id: 调用方新分配的临时执行ID，不能来自HTTP用户输入。

        Returns:
            None。
        """
        self._cleanup_ephemeral_thread(request_id)

    def _create_conversation_service(self) -> ConversationService:
        return ConversationService(
            repository=self._get_conversation_repository(),
            agent=self.query_orchestrator,
            scope_validator=self.query_orchestrator.validate_scope,
            traces=self.trace_service,
            checkpoint_store=self._get_conversation_checkpoint_store(),
            history_turns=self._settings.conversation_history_batch_size,
            history_window=self.history_window,
            admission=self.question_admission,
        )

    def _get_conversation_repository(self) -> ConversationRepository:
        return self._get_or_create(
            self._conversation_repository_slot,
            lambda: self._factories.conversation_repository(
                self._runtime_database_path(
                    "conversations", self._runtime_options.conversation_db_path,
                    self._settings.conversation_db_path,
                ),
            ),
            owned_adapter=False,
        )

    def _get_conversation_checkpoint_store(
        self,
    ) -> ConversationCheckpointAdapter:
        return self._get_or_create(
            self._conversation_checkpoint_store_slot,
            lambda: self._factories.conversation_checkpoint_store(
                self._runtime_database_path(
                    "checkpoints", self._runtime_options.agent_checkpoint_db_path,
                    self._settings.agent_checkpoint_db_path,
                ),
            ),
            owned_adapter=True,
        )

    def _get_conversation_agent(self) -> ConversationAgent:
        return self._get_or_create(
            self._conversation_agent_slot,
            self._create_conversation_agent,
            owned_adapter=True,
        )

    def _create_conversation_agent(self) -> ConversationAgent:
        checkpoint_store = self._get_conversation_checkpoint_store()
        return self._factories.conversation_agent(
            self._settings.resolved_llm_api_key,
            self._settings.llm_base_url,
            self._settings.llm_model,
            self._settings.agent_temperature,
            self._settings.llm_reasoning_effort,
            self.retrieval_service,
            checkpoint_store.checkpointer,
            self._settings.top_k,
            self._settings.agent_max_retrieval_calls,
        )

    def _get_parser_resolver(self) -> ParserResolver:
        return self._get_or_create(
            self._parser_resolver_slot,
            self._factories.parser_resolver,
            owned_adapter=True,
        )

    def _get_repair_log(self) -> RepairLogWriter:
        return self._get_or_create(
            self._repair_log_slot,
            self._create_repair_log,
            owned_adapter=True,
        )

    def _create_repair_log(self) -> RepairLogWriter:
        repair_root = self._settings.docs_path.parent
        if self._runtime_options.chroma_path is not None:
            # 不同隔离collection的修复记录不能混入生产或彼此的源文件目录。
            repair_root = self._job_sources_path()
        repair_path = repair_root / "repair" / "document_deletions.jsonl"
        return self._factories.repair_log(repair_path)

    def _get_reranker(self) -> Reranker | None:
        return self._get_or_create(
            self._reranker_slot,
            self._create_reranker,
            owned_adapter=True,
        )

    def _create_reranker(self) -> Reranker | None:
        return self._factories.reranker(
            self._settings.reranker_base_url,
            self._settings.resolved_reranker_api_key,
            self._settings.reranker_model,
            self._settings.reranker_endpoint,
        )

    def _get_llm_provider(self) -> LLMProvider:
        return self._get_or_create(
            self._llm_provider_slot,
            self._create_llm_provider,
            owned_adapter=True,
        )

    def _create_llm_provider(self) -> LLMProvider:
        return self._factories.llm_provider(
            self._settings.resolved_llm_api_key,
            self._settings.llm_base_url,
            self._settings.llm_model,
            self._runtime_options.llm_temperature,
            self._settings.llm_reasoning_effort,
        )

    def _get_or_create(
        self,
        slot: _LazySlot[_T],
        factory: Callable[[], _T],
        *,
        owned_adapter: bool,
    ) -> _T:
        current_thread_id = get_ident()
        with self._lazy_condition:
            while True:
                self._ensure_open_locked()
                if slot.value is not _UNSET:
                    return cast(_T, slot.value)
                if slot.creator_thread_id is None:
                    slot.creator_thread_id = current_thread_id
                    self._start_build_locked()
                    break
                if slot.creator_thread_id == current_thread_id:
                    raise RuntimeError("检测到应用组件循环构造")
                self._lazy_condition.wait()

        try:
            candidate = factory()
        except BaseException:
            self._finish_failed_build(slot)
            raise

        should_dispose = False
        with self._lazy_condition:
            slot.creator_thread_id = None
            if self._state is _ContainerState.OPEN:
                if owned_adapter:
                    self._register_adapter_locked(candidate)
                slot.value = candidate
                self._finish_build_locked()
                self._lazy_condition.notify_all()
                return candidate
            should_dispose = owned_adapter
            self._lazy_condition.notify_all()

        try:
            if should_dispose:
                self._dispose_unpublished_candidate(candidate)
        finally:
            with self._lazy_condition:
                self._finish_build_locked()
                self._finish_closing_if_ready_locked()
                self._lazy_condition.notify_all()

        raise RuntimeError(_CONTAINER_CLOSED_MESSAGE)

    def _finish_failed_build(
        self,
        slot: _LazySlot[_T],
    ) -> None:
        with self._lazy_condition:
            slot.creator_thread_id = None
            self._finish_build_locked()
            self._finish_closing_if_ready_locked()
            self._lazy_condition.notify_all()

    def _start_build_locked(self) -> None:
        self._active_build_count += 1

    def _finish_build_locked(self) -> None:
        self._active_build_count -= 1

    def _register_adapter_locked(self, adapter: object | None) -> None:
        if not isinstance(adapter, _Closable):
            return
        adapter_id = id(adapter)
        if adapter_id in self._owned_adapter_ids:
            return
        self._owned_adapter_ids.add(adapter_id)
        self._owned_adapter_references.append(adapter)
        self._closable_adapters.append(adapter)

    def _dispose_unpublished_candidate(
        self,
        candidate: object | None,
    ) -> None:
        with self._lazy_condition:
            if not isinstance(candidate, _Closable):
                return
            candidate_id = id(candidate)
            if candidate_id in self._owned_adapter_ids:
                return
            self._owned_adapter_ids.add(candidate_id)
            self._owned_adapter_references.append(candidate)

        candidate.close()

    def _finish_closing_if_ready_locked(self) -> None:
        can_finish = (
            self._state is _ContainerState.CLOSING
            and not self._close_release_in_progress
            and self._active_build_count == 0
        )
        if can_finish:
            self._state = _ContainerState.CLOSED

    def _ensure_open_locked(self) -> None:
        if self._state is not _ContainerState.OPEN:
            raise RuntimeError(_CONTAINER_CLOSED_MESSAGE)


def build_application_container(
    settings: Settings,
    runtime_options: RuntimeOptions | None = None,
    factories: AdapterFactories | None = None,
) -> ApplicationContainer:
    """使用应用配置和可选运行时覆盖创建惰性容器。

    Args:
        settings: 已加载且不由容器修改的应用配置。
        runtime_options: 单次运行的隔离存储和温度覆盖。
        factories: 测试或替换基础设施实现使用的 factory 集合。

    Returns:
        尚未实例化任何运行时组件的应用容器。
    """
    options = runtime_options or RuntimeOptions()
    adapter_factories = factories or configured_adapter_factories(settings)
    return ApplicationContainer(settings, options, adapter_factories)

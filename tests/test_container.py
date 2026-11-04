# -*- coding: utf-8 -*-
"""共享应用容器的惰性装配、配置覆盖和生命周期测试。"""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Thread

import pytest

from app.application.documents.service import DocumentService
from app.application.ingestion.pipeline import IngestionPipeline
from app.application.ingestion.service import UploadService
from app.application.query.service import QueryService
from app.application.query.routed_service import RoutedQueryService
from app.application.retrieval.service import RetrievalService
from app.application.conversations.service import ConversationService
from app.bootstrap import (
    AdapterFactories,
    RuntimeOptions,
    build_application_container,
)
from app.config import Settings
from app.infrastructure.chunking.recursive import RecursiveDocumentChunker
from app.infrastructure.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from app.infrastructure.indexes.bm25 import BM25Index
from app.infrastructure.reranking.dashscope import RemoteReranker
from app.infrastructure.vectorstores.chroma import ChromaVectorStore


class _ClosableAdapter:
    """记录容器触发的 adapter 关闭顺序。"""

    def __init__(self, name: str, closed: list[str]) -> None:
        """保存 adapter 名称和共享关闭记录。

        Args:
            name: 关闭记录使用的稳定名称。
            closed: 所有测试 adapter 共享的关闭顺序列表。
        """
        self._name = name
        self._closed = closed

    def close(self) -> None:
        """记录一次关闭。

        Returns:
            None。
        """
        self._closed.append(self._name)


class _RecordingFactories:
    """提供不访问磁盘或远程服务的显式 adapter factories。"""

    def __init__(
        self,
        closable_names: set[str] | None = None,
        shared_adapter_names: set[str] | None = None,
    ) -> None:
        """初始化创建、参数和关闭记录。

        Args:
            closable_names: 需要返回独立可关闭对象的 factory
                名称。
            shared_adapter_names: 需要复用同一可关闭对象的 factory
                名称。
        """
        self.created: list[str] = []
        self.closed: list[str] = []
        self.arguments: dict[str, dict[str, object]] = {}
        self.adapters: dict[str, object] = {}
        self._closable_names = closable_names or set()
        self._shared_adapter_names = shared_adapter_names or set()
        self._shared_adapter = _ClosableAdapter("shared", self.closed)

    def _adapter(self, name: str) -> object:
        self.created.append(name)
        if name in self._shared_adapter_names:
            adapter: object = self._shared_adapter
        elif name in self._closable_names:
            adapter = _ClosableAdapter(name, self.closed)
        else:
            adapter = object()
        self.adapters[name] = adapter
        return adapter

    def _create_vector_store(
        self,
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        self.arguments["vector_store"] = {
            "path": path,
            "collection_name": collection_name,
            "collection_metadata": collection_metadata,
            "rebuild_collection": rebuild_collection,
        }
        return self._adapter("vector_store")

    def _create_embedding_provider(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        batch_size: int,
    ) -> object:
        self.arguments["embedding_provider"] = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "batch_size": batch_size,
        }
        return self._adapter("embedding_provider")

    def _create_chunker(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ) -> object:
        self.arguments["chunker"] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        return self._adapter("chunker")

    def _create_parser_resolver(self) -> object:
        return self._adapter("parser_resolver")

    def _create_keyword_index(self, vector_store: object) -> object:
        self.arguments["keyword_index"] = {"vector_store": vector_store}
        return self._adapter("keyword_index")

    def _create_repair_log(self, path: Path) -> object:
        self.arguments["repair_log"] = {"path": path}
        return self._adapter("repair_log")

    def _create_reranker(
        self,
        base_url: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
    ) -> object:
        self.arguments["reranker"] = {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "endpoint": endpoint,
        }
        return self._adapter("reranker")

    def _create_llm_provider(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        temperature: float | None,
        reasoning_effort: str | None,
    ) -> object:
        self.arguments["llm_provider"] = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
        }
        return self._adapter("llm_provider")

    def _create_conversation_repository(self, path: Path) -> object:
        self.arguments["conversation_repository"] = {"path": path}
        return self._adapter("conversation_repository")

    def _create_conversation_checkpoint_store(self, path: Path) -> object:
        self.arguments["conversation_checkpoint_store"] = {"path": path}
        adapter = self._adapter("conversation_checkpoint_store")
        setattr(adapter, "checkpointer", object())
        return adapter

    def _create_conversation_agent(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        temperature: float | None,
        reasoning_effort: str | None,
        retrieval_service: object,
        checkpointer: object,
        top_k: int,
        max_retrieval_calls: int,
    ) -> object:
        self.arguments["conversation_agent"] = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "retrieval_service": retrieval_service,
            "checkpointer": checkpointer,
            "top_k": top_k,
            "max_retrieval_calls": max_retrieval_calls,
        }
        return self._adapter("conversation_agent")

    def _adapter_factories(self) -> AdapterFactories:
        return AdapterFactories(
            vector_store=self._create_vector_store,
            embedding_provider=self._create_embedding_provider,
            chunker=self._create_chunker,
            parser_resolver=self._create_parser_resolver,
            keyword_index=self._create_keyword_index,
            repair_log=self._create_repair_log,
            reranker=self._create_reranker,
            llm_provider=self._create_llm_provider,
            conversation_repository=self._create_conversation_repository,
            conversation_checkpoint_store=(
                self._create_conversation_checkpoint_store
            ),
            conversation_agent=self._create_conversation_agent,
        )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_api_key="test-llm",
        llm_base_url="https://llm.example/v1",
        llm_model="test-generator",
        llm_reasoning_effort="medium",
        embedding_api_key="test-embedding",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="test-embedder",
        reranker_api_key="test-reranker",
        reranker_base_url="https://reranker.example/v1",
        reranker_model="test-reranker-model",
        reranker_endpoint="reranks",
        chunk_size=240,
        chunk_overlap=24,
        embedding_batch_size=7,
        top_k=4,
        top_k_rerank=12,
        docs_path=tmp_path / "production-docs",
        chroma_path=tmp_path / "production-chroma",
        collection_name="production_collection",
        max_upload_size=4096,
        conversation_db_path=tmp_path / "conversations.db",
        knowledge_base_db_path=tmp_path / "catalog.sqlite3",
        trace_db_path=tmp_path / "traces.sqlite3",
        agent_checkpoint_db_path=tmp_path / "checkpoints.db",
        conversation_history_turns=6,
        agent_max_retrieval_calls=3,
        agent_temperature=0.0,
    )


def test_conversation_service_is_lazy_shared_and_uses_existing_retrieval(
    tmp_path: Path,
) -> None:
    """会话装配应复用检索服务并隔离两个 SQLite 数据库。"""
    recording = _RecordingFactories(
        closable_names={
            "conversation_checkpoint_store",
            "conversation_agent",
        }
    )
    settings = _settings(tmp_path)
    container = build_application_container(
        settings,
        factories=recording._adapter_factories(),
    )

    service = container.conversation_service

    assert isinstance(service, ConversationService)
    assert container.conversation_service is service
    assert recording.arguments["conversation_repository"] == {
        "path": settings.conversation_db_path
    }
    assert recording.arguments["conversation_checkpoint_store"] == {
        "path": settings.agent_checkpoint_db_path
    }
    assert service._agent is container.query_orchestrator
    assert "conversation_agent" not in recording.created
    container._get_conversation_agent()
    assert recording.arguments["conversation_agent"] == {
        "api_key": settings.resolved_llm_api_key,
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "temperature": settings.agent_temperature,
        "reasoning_effort": settings.llm_reasoning_effort,
        "retrieval_service": container.retrieval_service,
        "checkpointer": recording.adapters[
            "conversation_checkpoint_store"
        ].checkpointer,
        "top_k": settings.top_k,
        "max_retrieval_calls": settings.agent_max_retrieval_calls,
    }
    assert service._history_turns == settings.conversation_history_batch_size


def test_lazy_properties_reuse_one_instance(tmp_path: Path) -> None:
    """重复访问公开组件属性不得再次运行其构造链。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    container = build_application_container(
        _settings(tmp_path),
        factories=recording._adapter_factories(),
    )

    assert container.vector_store is container.vector_store
    assert container.embedding_provider is container.embedding_provider
    assert container.chunker is container.chunker
    assert container.keyword_index is container.keyword_index
    assert container.ingestion_pipeline is container.ingestion_pipeline
    assert container.upload_service is container.upload_service
    assert container.document_service is container.document_service
    assert container.retrieval_service is container.retrieval_service
    assert container.query_service is container.query_service
    for name in recording.created:
        assert recording.created.count(name) == 1


def test_document_use_cases_share_one_mutation_lock(tmp_path: Path) -> None:
    """上传与删除必须复用同一流水线及其进程内写锁。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    container = build_application_container(
        _settings(tmp_path),
        factories=recording._adapter_factories(),
    )

    pipeline = container.ingestion_pipeline
    upload_service = container.upload_service
    document_service = container.document_service

    assert upload_service._ingestion_pipeline is pipeline
    assert document_service._ingestion_pipeline is pipeline
    assert pipeline._mutation_lock is container._document_mutation_lock
    assert container._lazy_lock is not container._document_mutation_lock
    assert pipeline._parser_resolver is recording.adapters["parser_resolver"]


def test_runtime_options_override_only_requested_fields(
    tmp_path: Path,
) -> None:
    """评估覆盖应只替换隔离存储和生成温度字段。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    recording = _RecordingFactories()
    metadata = {"fingerprint": "abc123", "schema_version": 2}
    options = RuntimeOptions(
        chroma_path=tmp_path / "evaluation-chroma",
        collection_name="evaluation_collection",
        collection_metadata=metadata,
        rebuild_collection=True,
        llm_temperature=0.15,
    )
    container = build_application_container(
        settings,
        runtime_options=options,
        factories=recording._adapter_factories(),
    )

    vector_store = container.vector_store
    container.chunker
    container.embedding_provider
    container.document_service
    container.baseline_query_service

    assert recording.arguments["vector_store"] == {
        "path": tmp_path / "evaluation-chroma",
        "collection_name": "evaluation_collection",
        "collection_metadata": metadata,
        "rebuild_collection": True,
    }
    assert recording.arguments["chunker"] == {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    assert recording.arguments["embedding_provider"] == {
        "api_key": settings.resolved_embedding_api_key,
        "base_url": settings.embedding_base_url,
        "model": settings.embedding_model,
        "batch_size": settings.embedding_batch_size,
    }
    assert recording.arguments["keyword_index"] == {
        "vector_store": vector_store,
    }
    assert recording.arguments["repair_log"] == {
        "path": (
            container.document_service._docs_path
            / "repair"
            / "document_deletions.jsonl"
        ),
    }
    assert recording.arguments["repair_log"]["path"].is_relative_to(options.chroma_path)
    assert recording.arguments["reranker"] == {
        "base_url": settings.reranker_base_url,
        "api_key": settings.resolved_reranker_api_key,
        "model": settings.reranker_model,
        "endpoint": settings.reranker_endpoint,
    }
    assert recording.arguments["llm_provider"] == {
        "api_key": settings.resolved_llm_api_key,
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "temperature": 0.15,
        "reasoning_effort": settings.llm_reasoning_effort,
    }
    assert settings.chroma_path == tmp_path / "production-chroma"
    assert settings.collection_name == "production_collection"


def test_runtime_options_are_frozen_and_fall_back_field_by_field(
    tmp_path: Path,
) -> None:
    """部分运行时覆盖应保持不可变并逐字段回退到 Settings。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    settings = _settings(tmp_path)
    recording = _RecordingFactories()
    options = RuntimeOptions(collection_name="evaluation_collection")
    container = build_application_container(
        settings,
        runtime_options=options,
        factories=recording._adapter_factories(),
    )

    with pytest.raises(FrozenInstanceError):
        setattr(options, "collection_name", "mutated_collection")

    container.vector_store
    container.baseline_query_service

    assert recording.arguments["vector_store"] == {
        "path": settings.chroma_path,
        "collection_name": "evaluation_collection",
        "collection_metadata": None,
        "rebuild_collection": False,
    }
    assert recording.arguments["llm_provider"]["temperature"] is None


def test_close_is_idempotent_without_constructing_unused_resources(
    tmp_path: Path,
) -> None:
    """关闭只处理已创建资源，重复关闭不产生额外动作。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    container = build_application_container(
        _settings(tmp_path),
        factories=recording._adapter_factories(),
    )

    container.vector_store
    container.close()
    container.close()

    assert recording.created == ["vector_store"]
    assert recording.closed == []


def test_close_reverses_creation_order_and_closes_shared_adapter_once(
    tmp_path: Path,
) -> None:
    """可关闭 adapter 按首次创建逆序且按身份只关闭一次。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories(
        closable_names={"vector_store", "reranker"},
        shared_adapter_names={"embedding_provider", "llm_provider"},
    )
    container = build_application_container(
        _settings(tmp_path),
        factories=recording._adapter_factories(),
    )

    container.embedding_provider
    container.vector_store
    container.baseline_query_service
    container.close()
    container.close()

    assert recording.closed == ["reranker", "vector_store", "shared"]


def test_factory_candidate_is_rejected_after_reentrant_close(
    tmp_path: Path,
) -> None:
    """factory 内重入 close 后不得发布或泄漏候选资源。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    base_factories = recording._adapter_factories()
    candidate = _ClosableAdapter("candidate", recording.closed)
    factory_calls = 0

    def create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        nonlocal factory_calls
        factory_calls += 1
        container.close()
        return candidate

    factories = replace(
        base_factories,
        vector_store=create_vector_store,
    )
    container = build_application_container(
        _settings(tmp_path),
        factories=factories,
    )

    with pytest.raises(RuntimeError, match="关闭"):
        container.vector_store
    with pytest.raises(RuntimeError, match="关闭"):
        container.vector_store
    container.close()

    assert factory_calls == 1
    assert recording.closed == ["candidate"]


@pytest.mark.parametrize(
    "fatal_error",
    [
        KeyboardInterrupt("interrupt candidate cleanup"),
        SystemExit("exit candidate cleanup"),
    ],
    ids=["keyboard_interrupt", "system_exit"],
)
def test_candidate_cleanup_finalizes_before_base_exception_propagates(
    tmp_path: Path,
    fatal_error: BaseException,
) -> None:
    """候选清理异常传播前必须结束活动构造并完成关闭。

    Args:
        tmp_path: pytest 提供的临时目录。
        fatal_error: 候选资源关闭时抛出的进程级异常。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    base_factories = recording._adapter_factories()
    close_calls = 0

    class FailingCandidate:
        """关闭时抛出进程级异常的未发布候选。"""

        def close(self) -> None:
            """记录清理并抛出预设异常。

            Returns:
                None。

            Raises:
                BaseException: 参数化测试注入的异常。
            """
            nonlocal close_calls
            close_calls += 1
            raise fatal_error

    candidate = FailingCandidate()

    def create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        container.close()
        return candidate

    container = build_application_container(
        _settings(tmp_path),
        factories=replace(
            base_factories,
            vector_store=create_vector_store,
        ),
    )

    with pytest.raises(type(fatal_error)) as exc_info:
        container.vector_store

    assert exc_info.value is fatal_error
    assert close_calls == 1
    assert container._active_build_count == 0
    assert container._state.value == "closed"
    container.close()


def test_close_does_not_hold_lazy_lock_during_adapter_release(
    tmp_path: Path,
) -> None:
    """adapter 等待访问容器的工作线程时 close 不得形成锁死。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    base_factories = recording._adapter_factories()
    worker_started = Event()
    worker_finished = Event()
    worker_errors: list[Exception] = []

    class WorkerAccessAdapter:
        """关闭时等待一个会访问容器的工作线程。"""

        def close(self) -> None:
            """启动并等待工作线程完成容器访问。

            Returns:
                None。
            """
            def access_container() -> None:
                worker_started.set()
                try:
                    container.embedding_provider
                except Exception as exc:
                    worker_errors.append(exc)
                finally:
                    worker_finished.set()

            worker = Thread(target=access_container)
            worker.start()
            assert worker_started.wait(timeout=2)
            if not worker_finished.wait(timeout=1):
                raise RuntimeError("工作线程被惰性锁阻塞")
            worker.join(timeout=2)

    adapter = WorkerAccessAdapter()

    def create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        return adapter

    container = build_application_container(
        _settings(tmp_path),
        factories=replace(
            base_factories,
            vector_store=create_vector_store,
        ),
    )
    container.vector_store

    container.close()

    assert worker_finished.is_set()
    assert len(worker_errors) == 1
    assert isinstance(worker_errors[0], RuntimeError)


def test_adapter_close_can_join_worker_that_calls_container_close(
    tmp_path: Path,
) -> None:
    """adapter 应能等待派生线程回调同一容器 close。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    base_factories = recording._adapter_factories()
    worker_started = Event()
    worker_finished = Event()
    worker_errors: list[BaseException] = []

    class WorkerCloseAdapter:
        """关闭时等待派生线程回调容器 close。"""

        def close(self) -> None:
            """启动并等待容器关闭回调完成。

            Returns:
                None。
            """
            def close_container() -> None:
                worker_started.set()
                try:
                    container.close()
                except BaseException as exc:
                    worker_errors.append(exc)
                finally:
                    worker_finished.set()

            worker = Thread(target=close_container)
            worker.start()
            assert worker_started.wait(timeout=2)
            if not worker_finished.wait(timeout=1):
                raise RuntimeError("派生线程 close 被阻塞")
            worker.join(timeout=2)

    adapter = WorkerCloseAdapter()

    def create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        return adapter

    container = build_application_container(
        _settings(tmp_path),
        factories=replace(
            base_factories,
            vector_store=create_vector_store,
        ),
    )
    container.vector_store

    container.close()

    assert worker_finished.is_set()
    assert worker_errors == []


def test_adapter_close_can_reenter_container_close_on_same_thread(
    tmp_path: Path,
) -> None:
    """adapter 同线程重入 close 应立即返回且不重复释放。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    base_factories = recording._adapter_factories()

    class ReentrantAdapter:
        """记录同线程重入容器 close 前后的执行顺序。"""

        def close(self) -> None:
            """在 adapter 释放期间重入容器 close。

            Returns:
                None。
            """
            recording.closed.append("before_reentrant_close")
            container.close()
            recording.closed.append("after_reentrant_close")

    adapter = ReentrantAdapter()

    def create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        return adapter

    container = build_application_container(
        _settings(tmp_path),
        factories=replace(
            base_factories,
            vector_store=create_vector_store,
        ),
    )
    container.vector_store

    container.close()
    container.close()

    assert recording.closed == [
        "before_reentrant_close",
        "after_reentrant_close",
    ]


def test_construction_race_with_close_never_publishes_candidate(
    tmp_path: Path,
) -> None:
    """close 抢占正在运行的 factory 后候选只能被清理并拒绝。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    base_factories = recording._adapter_factories()
    factory_started = Event()
    allow_factory = Event()
    close_started = Event()
    allow_existing_close = Event()
    candidate_closed = Event()
    access_results: list[object] = []
    access_errors: list[Exception] = []
    close_errors: list[Exception] = []

    class ExistingAdapter:
        """让测试观察容器已进入 closing 状态。"""

        def close(self) -> None:
            """标记关闭已开始并等待测试继续。

            Returns:
                None。
            """
            close_started.set()
            assert allow_existing_close.wait(timeout=2)

    class CandidateAdapter:
        """记录竞态期间返回的候选是否被安全释放。"""

        def close(self) -> None:
            """记录候选资源清理。

            Returns:
                None。
            """
            candidate_closed.set()

    candidate = CandidateAdapter()

    def create_embedding_provider(
        api_key: str | None,
        base_url: str,
        model: str,
        batch_size: int,
    ) -> object:
        del api_key
        del base_url
        del model
        del batch_size
        return ExistingAdapter()

    def create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        factory_started.set()
        assert allow_factory.wait(timeout=2)
        return candidate

    factories = replace(
        base_factories,
        embedding_provider=create_embedding_provider,
        vector_store=create_vector_store,
    )
    container = build_application_container(
        _settings(tmp_path),
        factories=factories,
    )
    container.embedding_provider

    def access_vector_store() -> None:
        try:
            access_results.append(container.vector_store)
        except Exception as exc:
            access_errors.append(exc)

    def close_container() -> None:
        try:
            container.close()
        except Exception as exc:
            close_errors.append(exc)

    access_thread = Thread(target=access_vector_store)
    close_thread = Thread(target=close_container)
    access_thread.start()
    assert factory_started.wait(timeout=2)
    close_thread.start()
    observed_closing = close_started.wait(timeout=1)
    try:
        allow_factory.set()
        if observed_closing:
            assert candidate_closed.wait(timeout=2)
    finally:
        allow_existing_close.set()
        access_thread.join(timeout=2)
        close_thread.join(timeout=2)

    assert observed_closing
    assert not access_thread.is_alive()
    assert not close_thread.is_alive()
    assert access_results == []
    assert len(access_errors) == 1
    assert isinstance(access_errors[0], RuntimeError)
    assert close_errors == []
    assert candidate_closed.is_set()


def test_close_continues_after_errors_and_rethrows_first_error_once(
    tmp_path: Path,
) -> None:
    """首个 close 应继续逆序释放并只向拥有者抛出首错。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    base_factories = recording._adapter_factories()
    newer_error = RuntimeError("newer close failed")
    older_error = RuntimeError("older close failed")

    class FailingAdapter:
        """关闭时记录顺序并抛出预设异常。"""

        def __init__(
            self,
            name: str,
            error: Exception,
        ) -> None:
            """保存关闭记录与异常。

            Args:
                name: 关闭顺序记录名称。
                error: 关闭结束时抛出的异常。

            Returns:
                None。
            """
            self._name = name
            self._error = error

        def close(self) -> None:
            """记录关闭并抛出预设异常。

            Returns:
                None。

            Raises:
                Exception: 构造时注入的关闭异常。
            """
            recording.closed.append(self._name)
            raise self._error

    older_adapter = FailingAdapter("embedding", older_error)
    newer_adapter = FailingAdapter("vector_store", newer_error)

    def create_embedding_provider(
        api_key: str | None,
        base_url: str,
        model: str,
        batch_size: int,
    ) -> object:
        del api_key
        del base_url
        del model
        del batch_size
        return older_adapter

    def create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        return newer_adapter

    container = build_application_container(
        _settings(tmp_path),
        factories=replace(
            base_factories,
            embedding_provider=create_embedding_provider,
            vector_store=create_vector_store,
        ),
    )
    container.embedding_provider
    container.vector_store

    with pytest.raises(RuntimeError) as exc_info:
        container.close()

    assert exc_info.value is newer_error
    assert recording.closed == ["vector_store", "embedding"]
    container.close()


@pytest.mark.parametrize(
    "fatal_error",
    [
        KeyboardInterrupt("interrupt close"),
        SystemExit("exit close"),
    ],
    ids=["keyboard_interrupt", "system_exit"],
)
def test_close_finalizes_after_base_exception(
    tmp_path: Path,
    fatal_error: BaseException,
) -> None:
    """主关闭栈遇 BaseException 仍应释放剩余资源并进入 CLOSED。

    Args:
        tmp_path: pytest 提供的临时目录。
        fatal_error: adapter 关闭时抛出的进程级异常。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    base_factories = recording._adapter_factories()

    class BaseExceptionAdapter:
        """记录关闭并按需抛出进程级异常。"""

        def __init__(
            self,
            name: str,
            error: BaseException | None = None,
        ) -> None:
            """保存关闭名称与可选异常。

            Args:
                name: 关闭顺序记录名称。
                error: 关闭时抛出的可选异常。

            Returns:
                None。
            """
            self._name = name
            self._error = error

        def close(self) -> None:
            """记录关闭，并在配置时抛出异常。

            Returns:
                None。

            Raises:
                BaseException: 构造时注入的异常。
            """
            recording.closed.append(self._name)
            if self._error is not None:
                raise self._error

    older_adapter = BaseExceptionAdapter("embedding")
    newer_adapter = BaseExceptionAdapter(
        "vector_store",
        fatal_error,
    )

    def create_embedding_provider(
        api_key: str | None,
        base_url: str,
        model: str,
        batch_size: int,
    ) -> object:
        del api_key
        del base_url
        del model
        del batch_size
        return older_adapter

    def create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        return newer_adapter

    container = build_application_container(
        _settings(tmp_path),
        factories=replace(
            base_factories,
            embedding_provider=create_embedding_provider,
            vector_store=create_vector_store,
        ),
    )
    container.embedding_provider
    container.vector_store

    with pytest.raises(type(fatal_error)) as exc_info:
        container.close()

    assert exc_info.value is fatal_error
    assert recording.closed == ["vector_store", "embedding"]
    assert container._state.value == "closed"
    container.close()


def test_access_after_close_fails_without_constructing_resource(
    tmp_path: Path,
) -> None:
    """关闭后的公开属性访问应稳定失败且不触发 factory。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    container = build_application_container(
        _settings(tmp_path),
        factories=recording._adapter_factories(),
    )

    container.close()

    with pytest.raises(RuntimeError, match="关闭"):
        container.vector_store
    assert recording.created == []


def test_lazy_creation_is_thread_safe(tmp_path: Path) -> None:
    """并发首次访问同一属性时 factory 仍只能执行一次。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    base_factories = recording._adapter_factories()
    barrier = Barrier(8)
    factory_started = Event()
    allow_factory = Event()
    call_count = 0
    adapter = object()

    def create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        nonlocal call_count
        call_count += 1
        factory_started.set()
        assert allow_factory.wait(timeout=2)
        return adapter

    factories = replace(
        base_factories,
        vector_store=create_vector_store,
    )
    container = build_application_container(
        _settings(tmp_path),
        factories=factories,
    )

    def read_vector_store(_: int) -> object:
        barrier.wait()
        return container.vector_store

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for index in range(8):
            futures.append(executor.submit(read_vector_store, index))
        assert factory_started.wait(timeout=2)
        allow_factory.set()
        results = []
        for future in futures:
            results.append(future.result(timeout=2))

    assert call_count == 1
    for result in results:
        assert result is adapter


def test_concurrent_close_returns_while_owner_releases_resource(
    tmp_path: Path,
) -> None:
    """观察到 CLOSING 的并发 close 应立即成功返回。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        None。
    """
    recording = _RecordingFactories()
    base_factories = recording._adapter_factories()
    close_started = Event()
    allow_close = Event()
    second_started = Event()

    class BlockingClosableAdapter:
        """在测试允许前阻塞 adapter 关闭。"""

        def close(self) -> None:
            """等待测试释放关闭流程。

            Returns:
                None。
            """
            close_started.set()
            assert allow_close.wait(timeout=5)

    adapter = BlockingClosableAdapter()

    def create_vector_store(
        path: Path,
        collection_name: str,
        collection_metadata: Mapping[str, str | int] | None,
        rebuild_collection: bool,
    ) -> object:
        del path
        del collection_name
        del collection_metadata
        del rebuild_collection
        return adapter

    factories = replace(
        base_factories,
        vector_store=create_vector_store,
    )
    container = build_application_container(
        _settings(tmp_path),
        factories=factories,
    )
    container.vector_store

    def close_again() -> None:
        second_started.set()
        container.close()

    first_thread = Thread(target=container.close)
    second_thread = Thread(target=close_again)
    first_thread.start()
    assert close_started.wait(timeout=2)
    second_thread.start()
    assert second_started.wait(timeout=2)
    try:
        second_thread.join(timeout=1)
        returned_while_closing = not second_thread.is_alive()
    finally:
        allow_close.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert returned_while_closing
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()


@pytest.mark.parametrize("reranker_base_url", [None, "   "])
def test_default_factories_match_current_constructor_chain(
    tmp_path: Path,
    reranker_base_url: str | None,
) -> None:
    """默认装配应构造真实组件，未配置地址时不创建 reranker。

    Args:
        tmp_path: pytest 提供的临时目录。
        reranker_base_url: 未配置或仅含空白的重排服务地址。

    Returns:
        None。
    """
    settings = _settings(tmp_path).model_copy(
        update={"reranker_base_url": reranker_base_url},
    )
    container = build_application_container(settings)

    assert isinstance(container.vector_store, ChromaVectorStore)
    assert isinstance(
        container.embedding_provider,
        OpenAICompatibleEmbeddingProvider,
    )
    assert isinstance(container.chunker, RecursiveDocumentChunker)
    assert isinstance(container.keyword_index, BM25Index)
    assert isinstance(container.ingestion_pipeline, IngestionPipeline)
    assert isinstance(container.upload_service, UploadService)
    assert isinstance(container.document_service, DocumentService)
    assert isinstance(container.retrieval_service, RetrievalService)
    assert container.retrieval_service._reranker is None
    assert isinstance(container.query_service, RoutedQueryService)
    assert isinstance(container.baseline_query_service, QueryService)

    container.close()


def test_default_provider_factories_release_internal_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """容器关闭应释放默认远程 adapter 创建的内部客户端。

    Args:
        tmp_path: pytest 提供的临时目录。
        monkeypatch: pytest 提供的属性替换工具。

    Returns:
        None。
    """
    class RecordingClient:
        """记录 OpenAI 兼容客户端的关闭次数。"""

        def __init__(self) -> None:
            """初始化未关闭的测试客户端。

            Returns:
                None。
            """
            self.close_calls = 0

        def close(self) -> None:
            """记录一次客户端关闭。

            Returns:
                None。
            """
            self.close_calls += 1

    embedding_client = RecordingClient()
    llm_client = RecordingClient()
    reranker_client = RecordingClient()

    def _create_embedding_client(
        *,
        api_key: str,
        base_url: str,
        **options: object,
    ) -> RecordingClient:
        del api_key
        del base_url
        return embedding_client

    def _create_llm_client(
        *,
        api_key: str,
        base_url: str,
        **options: object,
    ) -> RecordingClient:
        del api_key
        del base_url
        return llm_client

    def _create_reranker_client(
        *,
        timeout: float,
    ) -> RecordingClient:
        assert timeout == 10.0
        return reranker_client

    monkeypatch.setattr(
        "app.infrastructure.embeddings.openai_compatible.OpenAI",
        _create_embedding_client,
    )
    monkeypatch.setattr(
        "app.infrastructure.generation.openai_compatible.OpenAI",
        _create_llm_client,
    )
    monkeypatch.setattr(
        "app.infrastructure.reranking.dashscope.httpx.Client",
        _create_reranker_client,
    )
    default_factories = AdapterFactories()
    recording = _RecordingFactories()
    factories = replace(
        recording._adapter_factories(),
        embedding_provider=default_factories.embedding_provider,
        llm_provider=default_factories.llm_provider,
        reranker=default_factories.reranker,
    )
    container = build_application_container(
        _settings(tmp_path),
        factories=factories,
    )

    container.embedding_provider
    container.baseline_query_service
    container.close()
    container.close()

    assert embedding_client.close_calls == 1
    assert llm_client.close_calls == 1
    assert reranker_client.close_calls == 1

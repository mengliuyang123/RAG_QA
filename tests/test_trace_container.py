# -*- coding: utf-8 -*-
"""组合根必须隔离追踪、会话和checkpoint，不因评估覆盖不全触碰生产库。"""

from pathlib import Path
from uuid import UUID
from uuid import uuid4

from app.bootstrap.container import RuntimeOptions
from app.bootstrap.container import build_application_container
from app.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        conversation_db_path=tmp_path / "production-conversations.sqlite3",
        agent_checkpoint_db_path=tmp_path / "production-checkpoints.sqlite3",
        knowledge_base_db_path=tmp_path / "production-catalog.sqlite3",
        trace_db_path=tmp_path / "production-traces.sqlite3",
        auth_db_path=tmp_path / "production-auth.sqlite3",
    )


def test_trace_is_lazy_shared_and_independent_of_model_credentials(tmp_path: Path) -> None:
    """追踪存储不需要调用模型，也不能在容器构造时就创建文件。

    Args:
        tmp_path: 全部路径的隔离目录。
    """
    settings = _settings(tmp_path)
    container = build_application_container(settings)
    try:
        assert not settings.trace_db_path.exists()
        assert container.trace_service is container.trace_service
        assert settings.trace_db_path.exists()
    finally:
        container.close()


def test_chroma_runtime_override_isolates_all_request_databases(tmp_path: Path) -> None:
    """只传旧Chroma覆盖选项的新评估通道也必须隔离全部业务持久化。

    Args:
        tmp_path: 隔离的生产和评估目录。
    """
    settings = _settings(tmp_path)
    container = build_application_container(
        settings,
        RuntimeOptions(chroma_path=tmp_path / "evaluation", collection_name="one"),
    )
    try:
        conversation = container.conversation_service.create()
        result = container.conversation_service.query(conversation.id, "你好")
        assert result.metadata.trace_id is not None
        projected = container.trace_query_service.get(UUID(result.metadata.trace_id))
        assert projected.turn_id == result.turn_id
        assert projected.commit_status == "committed"
        usage = container.usage_query_service.get()
        assert usage.totals.questions == 1
        feedback = container.feedback_admin_service.list(uuid4())
        assert feedback.feedback == ()
        assert container.auth_service is container.auth_service
        for path in (
            settings.trace_db_path, settings.conversation_db_path,
            settings.agent_checkpoint_db_path, settings.knowledge_base_db_path,
            settings.auth_db_path,
        ):
            assert not path.exists()
    finally:
        container.close()

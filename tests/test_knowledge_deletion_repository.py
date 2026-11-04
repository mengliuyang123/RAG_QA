# -*- coding: utf-8 -*-
"""删除冻结与真实队列必须在同一SQLite事务边界生效。"""

import sqlite3
from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from app.application.errors import KnowledgeBaseIngestionBusyError
from app.application.errors import KnowledgeBaseNotFoundError
from app.application.errors import KnowledgeBaseStorageError
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.knowledge_deletion import KnowledgeDeletionPlan
from app.domain.suggestions import CuratedQuestionsUpdate
from app.infrastructure.persistence.ingestion_jobs import SQLiteIngestionJobRepository
from app.infrastructure.persistence.knowledge_bases import SQLiteKnowledgeBaseRepository
from app.infrastructure.persistence.knowledge_deletion import SQLiteKnowledgeDeletionRepository
from app.infrastructure.persistence.knowledge_registry import SQLiteKnowledgeReadRegistry
from tests.test_ingestion_job_repository import _submission


def test_delete_freeze_rejects_stale_enqueue_and_keeps_terminal_history(tmp_path: Path) -> None:
    """只在应用检查KB、目录物理删除或删除历史任务都会破坏本契约。

    Args:
        tmp_path: 真实临时目录库。

    Returns:
        None。
    """
    path = tmp_path / "catalog.sqlite3"
    catalog = SQLiteKnowledgeBaseRepository(path)
    catalog.create(KnowledgeBase(id="ops", name="运维"))
    jobs = SQLiteIngestionJobRepository(path)
    old = jobs.enqueue(_submission(), capacity=10)
    claim = jobs.claim_next(lease_seconds=60)
    jobs.fail(claim, code="parse_failed")
    registry = SQLiteKnowledgeReadRegistry(path)
    registry.replace_curated("ops", CuratedQuestionsUpdate(questions=("公开原问题",)))
    plans = SQLiteKnowledgeDeletionRepository(path)
    plan = KnowledgeDeletionPlan(kb_id="ops", force=False, started_at=datetime.now(UTC))
    assert plans.begin(plan) == plan
    assert plans.get("ops") == plan
    with pytest.raises(KnowledgeBaseNotFoundError):
        # 模拟已通过旧catalog.get的上传者直接提交队列，仍必须在事务内被拒绝。
        jobs.enqueue(_submission(), capacity=10)
    with pytest.raises(KnowledgeBaseNotFoundError):
        registry.replace_curated("ops", CuratedQuestionsUpdate(questions=("不能发布",)))
    assert registry.list_curated(None, 20) == ()
    assert [row.id for row in catalog.list()] == ["default"]
    plans.complete(plan)
    assert plans.get("ops").state == "deleted"
    assert jobs.get(old.job_id).status == "failed"
    # 重开目录不能重建已删除ID；历史外键依然完整，不通过级联删任务回避约束。
    reopened = SQLiteKnowledgeBaseRepository(path)
    assert [row.id for row in reopened.list()] == ["default"]
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_begin_cannot_cross_a_pending_job_commit(tmp_path: Path) -> None:
    """prepare后、begin前入队的真实任务也必须阻止冻结与删除。

    Args:
        tmp_path: 真实共享任务目录库。

    Returns:
        None。
    """
    path = tmp_path / "catalog.sqlite3"
    catalog = SQLiteKnowledgeBaseRepository(path)
    catalog.create(KnowledgeBase(id="ops", name="运维"))
    jobs = SQLiteIngestionJobRepository(path)
    plans = SQLiteKnowledgeDeletionRepository(path)
    plan = KnowledgeDeletionPlan(kb_id="ops", force=True, started_at=datetime.now(UTC))
    jobs.enqueue(_submission(), capacity=10)
    with pytest.raises(KnowledgeBaseIngestionBusyError):
        plans.begin(plan)
    assert plans.get("ops") is None
    assert catalog.get("ops").id == "ops"


@pytest.mark.parametrize("corrupt", ["json", "identity", "state"])
def test_corrupt_plan_cannot_become_a_new_or_completed_operation(tmp_path: Path, corrupt: str) -> None:
    """JSON、身份或目录／计划状态冲突必须失败，不允许伪造空库成功。

    Args:
        tmp_path: 隔离真实目录库。
        corrupt: 故障种类。

    Returns:
        None。
    """
    path = tmp_path / "catalog.sqlite3"
    catalog = SQLiteKnowledgeBaseRepository(path)
    catalog.create(KnowledgeBase(id="ops", name="运维"))
    SQLiteIngestionJobRepository(path)
    plans = SQLiteKnowledgeDeletionRepository(path)
    plan = KnowledgeDeletionPlan(kb_id="ops", force=False, started_at=datetime.now(UTC))
    plans.begin(plan)
    with sqlite3.connect(path) as connection:
        if corrupt == "state":
            connection.execute("UPDATE knowledge_bases SET lifecycle='active' WHERE id='ops'")
        else:
            value = "not json"
            if corrupt == "identity":
                value = plan.model_copy(update={"kb_id": "another"}).model_dump_json()
            connection.execute("UPDATE knowledge_deletion_plans SET plan_json=?", (value,))
    with pytest.raises(KnowledgeBaseStorageError):
        plans.get("ops")

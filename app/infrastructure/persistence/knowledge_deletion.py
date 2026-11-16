# -*- coding: utf-8 -*-
"""在目录数据库原子冻结新入队并保存跨存储删除恢复身份。"""

import psycopg
from app.infrastructure.persistence.database import Database, Record
from collections.abc import Iterator
from contextlib import closing
from contextlib import contextmanager
from pathlib import Path

from app.application.errors import KnowledgeBaseIngestionBusyError
from app.application.errors import KnowledgeBaseNotFoundError
from app.application.errors import KnowledgeBaseProtectedError
from app.application.errors import KnowledgeBaseStorageError
from app.domain.knowledge_bases import DEFAULT_KB_ID
from app.domain.knowledge_deletion import KnowledgeDeletionPlan


class PostgresKnowledgeDeletionRepository:
    """计划先落库，历史任务的外键与正式引用无需删除或重写。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    def get(self, kb_id: str) -> KnowledgeDeletionPlan | None:
        """核对目录状态与计划，不允许损坏计划被误认为首次请求。

        Args:
            kb_id: 稳定知识库ID。

        Returns:
            已提交计划或真实不存在的None。
        """
        with self._connection() as connection:
            return self._get(connection, kb_id)

    def begin(self, plan: KnowledgeDeletionPlan) -> KnowledgeDeletionPlan:
        """BEGIN IMMEDIATE与enqueue竞争，确保不存在可继续运行的任务。

        Args:
            plan: 已通过用例身份与源路径检查的首次计划。

        Returns:
            只在事务提交后可执行的实际计划。
        """
        if plan.kb_id == DEFAULT_KB_ID:
            raise KnowledgeBaseProtectedError("默认知识库不可删除")
        if plan.state != "deleting":
            raise KnowledgeBaseStorageError("首次删除计划状态不正确")
        with self._connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || 'business_write', 0))")
            previous = self._get(connection, plan.kb_id)
            if previous is not None:
                return previous
            row = connection.execute(
                "SELECT lifecycle FROM knowledge_bases WHERE id = %s",
                (plan.kb_id,),
            ).fetchone()
            if row is None:
                raise KnowledgeBaseNotFoundError("知识库不存在")
            if row[0] != "active":
                raise KnowledgeBaseStorageError("知识库状态缺少对应删除计划")
            busy = connection.execute(
                "SELECT 1 FROM ingestion_jobs WHERE kb_id = %s "
                "AND (status NOT IN ('succeeded','failed') OR status IS NULL) LIMIT 1",
                (plan.kb_id,),
            ).fetchone()
            if busy is not None:
                raise KnowledgeBaseIngestionBusyError("知识库仍有入库任务")
            connection.execute(
                "INSERT INTO knowledge_deletion_plans VALUES (%s, %s)",
                (plan.kb_id, plan.model_dump_json()),
            )
            connection.execute(
                "UPDATE knowledge_bases SET lifecycle='deleting' WHERE id=%s",
                (plan.kb_id,),
            )
        return plan

    def complete(self, plan: KnowledgeDeletionPlan) -> None:
        """原计划身份未改变时才提交完成事实，不删除历史任务。

        Args:
            plan: 用例刚确认所有资源均已清理的计划。

        Returns:
            None。
        """
        with self._connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(current_schema() || 'business_write', 0))")
            actual = self._get(connection, plan.kb_id)
            if actual != plan or plan.state != "deleting":
                raise KnowledgeBaseStorageError("删除计划状态已经变化")
            completed = plan.model_copy(update={"state": "deleted"})
            connection.execute(
                "UPDATE knowledge_deletion_plans SET plan_json=%s WHERE kb_id=%s",
                (completed.model_dump_json(), plan.kb_id),
            )
            connection.execute(
                "UPDATE knowledge_bases SET lifecycle='deleted' WHERE id=%s",
                (plan.kb_id,),
            )

    def _get(self, connection: psycopg.Connection, kb_id: str) -> KnowledgeDeletionPlan | None:
        row = connection.execute(
            "SELECT p.plan_json, k.lifecycle FROM knowledge_deletion_plans p "
            "JOIN knowledge_bases k ON k.id=p.kb_id WHERE p.kb_id=%s", (kb_id,),
        ).fetchone()
        if row is None:
            return None
        plan = KnowledgeDeletionPlan.model_validate_json(row[0])
        if plan.kb_id != kb_id or plan.state != row[1]:
            raise KnowledgeBaseStorageError("知识库状态与删除计划不一致")
        return plan

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        try:
            with self._database.connection() as connection:
                yield connection
        except (psycopg.Error, ValueError, OSError) as exc:
            raise KnowledgeBaseStorageError("PostgreSQL 仓储操作失败") from exc

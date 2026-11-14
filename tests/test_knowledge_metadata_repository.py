from tests.postgres_support import pg_database, pg_connect
# -*- coding: utf-8 -*-
"""知识库负责人元数据迁移及部分更新不能覆盖并发管理员的其他字段。"""

import psycopg
from app.infrastructure.persistence.database import record_row, Record
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from app.infrastructure.persistence.knowledge_bases import PostgresKnowledgeBaseRepository


def test_postgres_migration_preserves_catalog_metadata(tmp_path: Path) -> None:
    from app.infrastructure.persistence.migrations import migrate_database
    from app.domain.knowledge_bases import KnowledgeBase, KnowledgeBasePatch
    database=pg_database(tmp_path/'catalog.db')
    repository=PostgresKnowledgeBaseRepository(database)
    repository.create(KnowledgeBase(id='ops',name='历史语料',description='保留说明',owner_name='Owner'))
    before=repository.get('ops')
    migrate_database(database._dsn,database.schema)
    assert PostgresKnowledgeBaseRepository(database).get('ops')==before


def test_concurrent_patch_keeps_both_independent_field_updates(tmp_path: Path) -> None:
    """读整条再覆盖写回，会让两个管理员的独立修改之一丢失。

    Args:
        tmp_path: SQLite隔离目录。
    """
    from app.domain.knowledge_bases import KnowledgeBasePatch

    path = tmp_path / "catalog.db"
    PostgresKnowledgeBaseRepository(pg_database(path))
    barrier = Barrier(2)

    def update(field: str, value: str) -> None:
        repository = PostgresKnowledgeBaseRepository(pg_database(path))
        barrier.wait()
        repository.patch("default", KnowledgeBasePatch(**{field: value}))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(update, "name", "新标题"),
                   pool.submit(update, "description", "新的选库说明")]
        for future in futures:
            future.result()
    result = PostgresKnowledgeBaseRepository(pg_database(path)).get("default")
    assert result.name == "新标题"
    assert result.description == "新的选库说明"

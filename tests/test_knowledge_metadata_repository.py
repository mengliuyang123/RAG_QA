# -*- coding: utf-8 -*-
"""知识库负责人元数据迁移及部分更新不能覆盖并发管理员的其他字段。"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from app.infrastructure.persistence.knowledge_bases import SQLiteKnowledgeBaseRepository


def test_legacy_catalog_migrates_owner_without_rewriting_metadata(tmp_path: Path) -> None:
    """重建旧库、重置名称或时间、缺失owner迁移都会破坏此断言。

    Args:
        tmp_path: 旧数据库的隔离目录。
    """
    from app.domain.knowledge_bases import KnowledgeBasePatch

    path = tmp_path / "catalog.sqlite3"
    original = ("default", "历史语料", "保留说明", "book",
                "2024-01-01T00:00:00+00:00", "2024-02-01T00:00:00+00:00")
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE knowledge_bases (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL, icon TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        connection.execute("INSERT INTO knowledge_bases VALUES (?, ?, ?, ?, ?, ?)", original)
    repository = SQLiteKnowledgeBaseRepository(path)
    assert repository.get("default").owner_name is None
    with sqlite3.connect(path) as connection:
        stored = connection.execute("SELECT id,name,description,icon,created_at,updated_at FROM knowledge_bases").fetchone()
    assert stored == original
    updated = repository.patch("default", KnowledgeBasePatch(owner_name="运维组"))
    assert updated.owner_name == "运维组"
    assert updated.name == "历史语料"
    assert updated.created_at.isoformat() == original[4]
    reopened = SQLiteKnowledgeBaseRepository(path)
    assert reopened.get("default").owner_name == "运维组"


def test_concurrent_patch_keeps_both_independent_field_updates(tmp_path: Path) -> None:
    """读整条再覆盖写回，会让两个管理员的独立修改之一丢失。

    Args:
        tmp_path: SQLite隔离目录。
    """
    from app.domain.knowledge_bases import KnowledgeBasePatch

    path = tmp_path / "catalog.sqlite3"
    SQLiteKnowledgeBaseRepository(path)
    barrier = Barrier(2)

    def update(field: str, value: str) -> None:
        repository = SQLiteKnowledgeBaseRepository(path)
        barrier.wait()
        repository.patch("default", KnowledgeBasePatch(**{field: value}))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(update, "name", "新标题"),
                   pool.submit(update, "description", "新的选库说明")]
        for future in futures:
            future.result()
    result = SQLiteKnowledgeBaseRepository(path).get("default")
    assert result.name == "新标题"
    assert result.description == "新的选库说明"

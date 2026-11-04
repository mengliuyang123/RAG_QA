# -*- coding: utf-8 -*-
"""使用独立SQLite保存知识库目录，兼容没有分库的旧语料。"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from contextlib import contextmanager
from datetime import datetime
from datetime import timezone
from pathlib import Path

from app.application.errors import KnowledgeBaseConflictError
from app.application.errors import KnowledgeBaseNotFoundError
from app.application.errors import KnowledgeBaseStorageError
from app.domain.knowledge_bases import DEFAULT_KB_ID
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.knowledge_bases import KnowledgeBasePatch


class SQLiteKnowledgeBaseRepository:
    """持久化空知识库及其描述，默认库只创建一次且不覆盖管理员修改。"""

    def __init__(self, path: Path) -> None:
        """初始化本地目录和幂等默认知识库。

        Args:
            path: 由组合根提供的独立数据库路径。

        Returns:
            None。
        """
        self._path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            default = KnowledgeBase(
                id=DEFAULT_KB_ID, name="默认知识库",
                description="未分库的历史文档和默认上传文档。",
            )
            with self._connection() as connection:
                # 惰性组合根可能同时创建候选，迁移必须在同一写事务内检查。
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_bases (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        icon TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                columns = connection.execute("PRAGMA table_info(knowledge_bases)").fetchall()
                names = {column["name"] for column in columns}
                if "owner_name" not in names:
                    connection.execute("ALTER TABLE knowledge_bases ADD COLUMN owner_name TEXT")
                if "lifecycle" not in names:
                    connection.execute(
                        "ALTER TABLE knowledge_bases ADD COLUMN lifecycle "
                        "TEXT NOT NULL DEFAULT 'active' "
                        "CHECK(lifecycle IN ('active','deleting','deleted'))"
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO knowledge_bases "
                    "(id, name, description, icon, created_at, updated_at, owner_name) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    self._values(default),
                )
        except OSError as exc:
            raise KnowledgeBaseStorageError("知识库目录初始化失败") from exc

    def list(self) -> list[KnowledgeBase]:
        """返回按稳定ID排序的全部目录记录。

        Returns:
            不依赖当前文档数量的目录列表。
        """
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM knowledge_bases WHERE lifecycle = 'active' ORDER BY id",
            ).fetchall()
        return [self._decode(row) for row in rows]

    def get(self, kb_id: str) -> KnowledgeBase:
        """读取一项目录元数据。

        Args:
            kb_id: 稳定知识库ID。

        Returns:
            已持久化的元数据。
        """
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_bases WHERE id = ? AND lifecycle = 'active'", (kb_id,),
            ).fetchone()
        if row is None:
            raise KnowledgeBaseNotFoundError("知识库不存在")
        return self._decode(row)

    def create(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        """创建空知识库而不覆盖同ID记录。

        Args:
            knowledge_base: 待保存的元数据。

        Returns:
            新建的元数据。
        """
        validated = KnowledgeBase.model_validate(knowledge_base.model_dump())
        with self._connection() as connection:
            try:
                connection.execute(
                    "INSERT INTO knowledge_bases "
                    "(id, name, description, icon, created_at, updated_at, owner_name) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    self._values(validated),
                )
            except sqlite3.IntegrityError as exc:
                raise KnowledgeBaseConflictError("知识库ID已存在") from exc
        return validated

    def update(self, knowledge_base: KnowledgeBase) -> KnowledgeBase:
        """修改可编辑元数据，不改写原创建时间。

        Args:
            knowledge_base: 修改后的元数据副本。

        Returns:
            数据库中的最新记录。
        """
        validated = KnowledgeBase.model_validate(knowledge_base.model_dump())
        with self._connection() as connection:
            result = connection.execute(
                """UPDATE knowledge_bases
                   SET name = ?, description = ?, icon = ?, owner_name = ?, updated_at = ?
                   WHERE id = ? AND lifecycle = 'active'""",
                (
                    validated.name, validated.description, validated.icon,
                    validated.owner_name,
                    datetime.now(timezone.utc).isoformat(), validated.id,
                ),
            )
            if result.rowcount != 1:
                raise KnowledgeBaseNotFoundError("知识库不存在")
        return self.get(validated.id)

    def patch(self, kb_id: str, changes: KnowledgeBasePatch) -> KnowledgeBase:
        """仅在事务中更新传入字段，不覆盖并发修改的其他列。

        Args:
            kb_id: 既有知识库ID。
            changes: 严格校验的可编辑字段集合。

        Returns:
            同一写事务内读取的最新元数据。
        """
        validated = KnowledgeBasePatch.model_validate(changes.model_dump(exclude_unset=True))
        assignments = []
        values = []
        for name in ("name", "description", "icon", "owner_name"):
            if name in validated.model_fields_set:
                assignments.append(f"{name} = ?")
                values.append(getattr(validated, name))
        assignments.append("updated_at = ?")
        values.extend((datetime.now(timezone.utc).isoformat(), kb_id))
        query = "UPDATE knowledge_bases SET " + ", ".join(assignments) + " WHERE id = ? AND lifecycle = 'active'"
        with self._connection() as connection:
            result = connection.execute(query, values)
            if result.rowcount != 1:
                raise KnowledgeBaseNotFoundError("知识库不存在")
            row = connection.execute(
                "SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,),
            ).fetchone()
            return self._decode(row)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            with closing(sqlite3.connect(self._path, timeout=5.0)) as connection:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.row_factory = sqlite3.Row
                with connection:
                    yield connection
        except sqlite3.Error as exc:
            raise KnowledgeBaseStorageError("知识库目录存储失败") from exc

    @staticmethod
    def _values(knowledge_base: KnowledgeBase) -> tuple[object, ...]:
        return (
            knowledge_base.id, knowledge_base.name, knowledge_base.description,
            knowledge_base.icon, knowledge_base.created_at.isoformat(),
            knowledge_base.updated_at.isoformat(),
            knowledge_base.owner_name,
        )

    @staticmethod
    def _decode(row: sqlite3.Row) -> KnowledgeBase:
        try:
            return KnowledgeBase(
                id=row["id"], name=row["name"], description=row["description"],
                icon=row["icon"], created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                owner_name=row["owner_name"],
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise KnowledgeBaseStorageError("知识库目录记录无效") from exc

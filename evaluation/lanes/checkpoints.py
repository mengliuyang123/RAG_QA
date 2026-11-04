# -*- coding: utf-8 -*-
"""独立命名空间的分通道成功记录，不覆盖旧150条评估记录。"""

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.domain.retrieval import RetrievalStrategy
from evaluation.lanes.models import LaneObservation
from evaluation.lanes.models import LanePolicy
from evaluation.lanes.models import LaneScoreRecord


Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class LaneRunManifest(BaseModel):
    """精确绑定复用边界，不凭HEAD或模型名猜测未提交代码是否相同。

    Attributes:
        dataset_sha256: 规范标签集指纹。
        index_fingerprint: 已校验语料与分块／Embedding配置指纹。
        configuration_version: 实际生产执行配置指纹。
        code_sha256: 相关Python源码指纹，包含未提交修改。
        judge_configuration_sha256: 独立裁判的非敏感配置指纹。
        selections: 本轮允许的案例与评估策略组合。
        strategy: 与通道独立的检索策略。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_sha256: Digest
    index_fingerprint: Digest
    configuration_version: str = Field(min_length=1)
    code_sha256: Digest
    judge_configuration_sha256: Digest
    selections: tuple[tuple[str, LanePolicy], ...] = Field(min_length=1)
    strategy: RetrievalStrategy

    def fingerprint(self) -> str:
        """用规范JSON计算完整运行标识，不包含API Key或本地路径。

        Returns:
            清单SHA256。
        """
        return hashlib.sha256(_canonical(self).encode("utf-8")).hexdigest()


class LaneRunStore:
    """短事务保存成功轮次；进程级网络执行独占锁由运行入口持有。"""

    def __init__(
        self, path: Path, manifest: LaneRunManifest, *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """打开新运行或校验已有运行，拒绝自动覆盖不匹配的清单。

        Args:
            path: 隔离评估目录中的专用SQLite文件。
            manifest: 已从显式非敏感字段生成的复用身份。
            clock: 有时区的当前时间；用于同生产追踪一致的到期控制。

        Returns:
            None。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._manifest = manifest
        self._clock = clock or _utc_now
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS lane_manifest "
                    "(id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL)",
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS lane_observations "
                    "(episode_id TEXT NOT NULL, policy TEXT NOT NULL, "
                    "turn_index INTEGER NOT NULL, payload TEXT NOT NULL, "
                    "expires_at REAL NOT NULL, "
                    "PRIMARY KEY (episode_id, policy, turn_index))",
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS lane_scores "
                    "(episode_id TEXT NOT NULL, policy TEXT NOT NULL, "
                    "turn_index INTEGER NOT NULL, payload TEXT NOT NULL, "
                    "expires_at REAL NOT NULL, "
                    "PRIMARY KEY (episode_id, policy, turn_index), "
                    "FOREIGN KEY (episode_id, policy, turn_index) REFERENCES "
                    "lane_observations (episode_id, policy, turn_index) "
                    "ON DELETE CASCADE)",
                )
                payload = _canonical(manifest)
                connection.execute(
                    "INSERT OR IGNORE INTO lane_manifest VALUES (1, ?)",
                    (payload,),
                )
                current = connection.execute(
                    "SELECT payload FROM lane_manifest WHERE id = 1",
                ).fetchone()
                if current is None or current[0] != payload:
                    raise ValueError("运行指纹不匹配，不能复用或覆盖原记录")
        self.prune_expired()

    def save(self, record: LaneObservation) -> None:
        """立即持久化已验证结果，完全相同重试幂等，冲突绝不覆盖。

        Args:
            record: 本轮真实编排结果及脱敏追踪。

        Returns:
            None。
        """
        if (
            record.strategy != self._manifest.strategy
            or (record.episode_id, record.policy) not in self._manifest.selections
        ):
            raise ValueError("案例或策略不属于当前运行")
        if record.trace.expires_at.timestamp() <= self._now_timestamp():
            raise ValueError("评估追踪已过期，不能重新保存或复用")
        key = (record.episode_id, record.policy, record.turn_index)
        payload = _canonical(record)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO lane_observations VALUES (?, ?, ?, ?, ?)",
                    (*key, payload, record.trace.expires_at.timestamp()),
                )
                current = connection.execute(
                    "SELECT payload FROM lane_observations "
                    "WHERE episode_id = ? AND policy = ? AND turn_index = ?",
                    key,
                ).fetchone()
                if current is None or current[0] != payload:
                    raise ValueError("同一成功轮次记录冲突，不允许覆盖或重新生成")

    def load(
        self, episode_id: str, policy: LanePolicy,
    ) -> tuple[LaneObservation, ...]:
        """按轮次读取成功前缀，连续性由生产片段执行器再次核对。

        Args:
            episode_id: 固定案例标识。
            policy: 评估策略。

        Returns:
            可恢复真实历史的冻结观察；无记录时为空。
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT payload FROM lane_observations "
                "WHERE episode_id = ? AND policy = ? AND expires_at > ? "
                "ORDER BY turn_index",
                (episode_id, policy, self._now_timestamp()),
            ).fetchall()
        records = []
        for row in rows:
            records.append(LaneObservation.model_validate_json(row[0]))
        return tuple(records)

    def save_score(self, record: LaneScoreRecord) -> None:
        """评分必须绑定已保存的实际答案，不能给同一轮替换另一份结论。

        Args:
            record: 独立裁判的完整成功记录。

        Returns:
            None；缺少对应答案、评分冲突或已过期时拒绝保存。
        """
        key = (record.episode_id, record.policy, record.turn_index)
        payload = _canonical(record)
        now = self._now_timestamp()
        with closing(self._connect()) as connection:
            with connection:
                source = connection.execute(
                    "SELECT payload, expires_at FROM lane_observations "
                    "WHERE episode_id = ? AND policy = ? AND turn_index = ? "
                    "AND expires_at > ?", (*key, now),
                ).fetchone()
                if source is None:
                    raise ValueError("评分缺少对应的有效答案")
                observation = LaneObservation.model_validate_json(source[0])
                if (
                    observation.trace.trace_id != record.source_trace_id
                    or observation.strategy != record.strategy
                ):
                    raise ValueError("评分与对应答案的执行或策略不一致")
                expires_at = min(source[1], record.trace.expires_at.timestamp())
                if expires_at <= now:
                    raise ValueError("独立评分追踪已过期")
                connection.execute(
                    "INSERT OR IGNORE INTO lane_scores VALUES (?, ?, ?, ?, ?)",
                    (*key, payload, expires_at),
                )
                current = connection.execute(
                    "SELECT payload FROM lane_scores WHERE episode_id = ? "
                    "AND policy = ? AND turn_index = ?", key,
                ).fetchone()
                if current is None or current[0] != payload:
                    raise ValueError("同一成功评分冲突，不允许覆盖")

    def load_score(
        self, episode_id: str, policy: LanePolicy, turn_index: int,
    ) -> LaneScoreRecord | None:
        """恢复对应答案的评分，未成功评分不制造默认分数。

        Args:
            episode_id: 固定案例标识。
            policy: 独立评估策略。
            turn_index: 零起始轮次。

        Returns:
            未过期的成功评分，无记录时为None。
        """
        now = self._now_timestamp()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT s.payload FROM lane_scores s JOIN lane_observations o "
                "USING (episode_id, policy, turn_index) "
                "WHERE s.episode_id = ? AND s.policy = ? AND s.turn_index = ? "
                "AND s.expires_at > ? AND o.expires_at > ?",
                (episode_id, policy, turn_index, now, now),
            ).fetchone()
        if row is None:
            return None
        return LaneScoreRecord.model_validate_json(row[0])

    def prune_expired(self) -> int:
        """只删除此运行中到期的追踪副本，不让评估缓存旁路保留策略。

        Returns:
            实际删除的过期轮次数。
        """
        with closing(self._connect()) as connection:
            with connection:
                now = self._now_timestamp()
                connection.execute(
                    "DELETE FROM lane_scores WHERE expires_at <= ?", (now,),
                )
                cursor = connection.execute(
                    "DELETE FROM lane_observations WHERE expires_at <= ?",
                    (now,),
                )
                return cursor.rowcount

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _now_timestamp(self) -> float:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("评估时钟必须含时区")
        return now.timestamp()


def _canonical(value: BaseModel) -> str:
    return json.dumps(
        value.model_dump(mode="json"), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)

# -*- coding: utf-8 -*-
"""独立命名空间的分通道成功记录，不覆盖旧150条评估记录。"""

import hashlib
import json
from app.bootstrap.database import Database
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
    """按运行身份保存 PostgreSQL 评估记录，重复成功记录不可覆盖。"""
    def __init__(self, database: Database, manifest: LaneRunManifest, *,
                 clock: Callable[[], datetime] | None = None, run_id: str = 'default') -> None:
        self._database = database
        self._clock = clock or _utc_now
        self._run_id = run_id
        self._manifest = manifest
        payload = _canonical(manifest)
        with database.connection() as connection:
            connection.execute('INSERT INTO lane_manifest VALUES (%s,%s) ON CONFLICT DO NOTHING', (run_id, payload))
            current = connection.execute('SELECT payload FROM lane_manifest WHERE run_id=%s', (run_id,)).fetchone()
            if current[0] != payload:
                raise ValueError('运行指纹不匹配，不能复用或覆盖原记录')
        self.prune_expired()

    def save(self, record: LaneObservation) -> None:
        if record.strategy != self._manifest.strategy or (record.episode_id, record.policy) not in self._manifest.selections:
            raise ValueError('案例或策略不属于当前运行')
        if record.trace.expires_at.timestamp() <= self._now_timestamp():
            raise ValueError('评估追踪已过期，不能重新保存或复用')
        key = (self._run_id, record.episode_id, record.policy, record.turn_index)
        payload = _canonical(record)
        with self._database.connection() as connection:
            connection.execute('INSERT INTO lane_observations VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                               (*key, payload, record.trace.expires_at.timestamp()))
            current = connection.execute('SELECT payload FROM lane_observations WHERE '
                'run_id=%s AND episode_id=%s AND policy=%s AND turn_index=%s', key).fetchone()
            if current[0] != payload:
                raise ValueError('同一成功轮次记录冲突，不允许覆盖或重新生成')

    def load(self, episode_id: str, policy: LanePolicy) -> tuple[LaneObservation, ...]:
        with self._database.connection(read_only=True) as connection:
            rows = connection.execute('SELECT payload FROM lane_observations WHERE run_id=%s '
                'AND episode_id=%s AND policy=%s AND expires_at>%s ORDER BY turn_index',
                (self._run_id, episode_id, policy, self._now_timestamp())).fetchall()
            return tuple(LaneObservation.model_validate_json(row[0]) for row in rows)

    def save_score(self, record: LaneScoreRecord) -> None:
        if record.strategy != self._manifest.strategy or (record.episode_id, record.policy) not in self._manifest.selections:
            raise ValueError('案例或策略不属于当前运行')
        key = (self._run_id, record.episode_id, record.policy, record.turn_index)
        payload = _canonical(record)
        now = self._now_timestamp()
        with self._database.connection() as connection:
            source = connection.execute('SELECT payload,expires_at FROM lane_observations WHERE '
                'run_id=%s AND episode_id=%s AND policy=%s AND turn_index=%s AND expires_at>%s FOR UPDATE',
                (*key, now)).fetchone()
            if source is None:
                raise ValueError('评分缺少对应的有效答案')
            observation = LaneObservation.model_validate_json(source[0])
            if observation.trace.trace_id != record.source_trace_id or observation.strategy != record.strategy:
                raise ValueError('评分与对应答案的执行或策略不一致')
            expires = min(source[1], record.trace.expires_at.timestamp())
            if expires <= now:
                raise ValueError('独立评分追踪已过期')
            connection.execute('INSERT INTO lane_scores VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                               (*key, payload, expires))
            current = connection.execute('SELECT payload FROM lane_scores WHERE '
                'run_id=%s AND episode_id=%s AND policy=%s AND turn_index=%s', key).fetchone()
            if current[0] != payload:
                raise ValueError('同一成功评分冲突，不允许覆盖')

    def load_score(self, episode_id: str, policy: LanePolicy, turn_index: int) -> LaneScoreRecord | None:
        now = self._now_timestamp()
        with self._database.connection(read_only=True) as connection:
            row = connection.execute('SELECT s.payload FROM lane_scores s JOIN lane_observations o '
                'USING (run_id,episode_id,policy,turn_index) WHERE s.run_id=%s AND s.episode_id=%s '
                'AND s.policy=%s AND s.turn_index=%s AND s.expires_at>%s AND o.expires_at>%s',
                (self._run_id, episode_id, policy, turn_index, now, now)).fetchone()
            return LaneScoreRecord.model_validate_json(row[0]) if row else None

    def prune_expired(self) -> int:
        now = self._now_timestamp()
        with self._database.connection() as connection:
            connection.execute('DELETE FROM lane_scores WHERE run_id=%s AND expires_at<=%s', (self._run_id, now))
            return connection.execute('DELETE FROM lane_observations WHERE run_id=%s AND expires_at<=%s', (self._run_id, now)).rowcount

    def _now_timestamp(self) -> float:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError('评估时钟必须含时区')
        return now.timestamp()


def _canonical(value: BaseModel) -> str:
    return json.dumps(
        value.model_dump(mode="json"), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)

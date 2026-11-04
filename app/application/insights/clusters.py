# -*- coding: utf-8 -*-
"""有审计、有界模型调用和快照复核的拒答语义聚类用例。"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import time
from collections import OrderedDict
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from threading import Lock
from uuid import UUID

from app.application.errors import EmbeddingResponseError
from app.application.errors import InsightStorageError
from app.application.errors import RefusalClustersBusyError
from app.application.errors import RefusalClustersChangedError
from app.application.errors import RefusalClustersTimeoutError
from app.application.insights.clustering import cluster_refusals
from app.application.insights.clustering import normalize_vector
from app.application.ports.embedding import EmbeddingProvider
from app.application.ports.insights import RefusalReader
from app.application.tracing.auditing import AdminAuditService
from app.domain.insights import RefusalItem
from app.domain.refusal_clusters import RefusalClusterPage
from app.domain.refusal_clusters import RefusalClusterPosition
from app.domain.refusal_clusters import RefusalClusterSample


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _sample_digest(sample: RefusalClusterSample) -> str:
    return _digest(sample.model_dump(mode="json"))


class RefusalClusterService:
    """单运行串行聚类与有界内存向量缓存，不持久化问题副本。"""

    def __init__(
        self, reader: RefusalReader, audit: AdminAuditService,
        embedder: EmbeddingProvider, model: str, threshold: float = 0.85,
    ) -> None:
        """绑定同一运行存储与固定模型配置。

        Args:
            reader: 只读正式拒答端口。
            audit: 模型调用之前的强制审计。
            embedder: 惰性真实Embedding端口。
            model: 部署模型名。
            threshold: 完整链接相似度阈值。

        Returns:
            None。
        """
        self._reader = reader
        self._audit = audit
        self._embedder = embedder
        self._model = model
        self._threshold = threshold
        self._lock = Lock()
        self._cache: OrderedDict[str, tuple[float, tuple[float, ...]]] = OrderedDict()

    def list(
        self, actor_id: UUID, *, days: int = 30, limit: int = 20,
        kb_id: str | None = None, cursor: str | None = None,
    ) -> RefusalClusterPage:
        """管理员分页读取真实语义分组，旧快照改变时要求刷新。

        Args:
            actor_id: 已认证管理员。
            days: 1–365个UTC日历天。
            limit: 返回1–100个组。
            kb_id: 实际检索范围过滤。
            cursor: 绑定条件和快照的短时游标。

        Returns:
            带窗口、截断及模型阈值说明的聚类页。
        """
        if not 1 <= days <= 365 or not 1 <= limit <= 100:
            raise ValueError("聚类窗口或分页参数无效")
        if not self._lock.acquire(blocking=False):
            raise RefusalClustersBusyError("拒答聚类正在运行")
        try:
            return self._list(actor_id, days, limit, kb_id, cursor)
        finally:
            # 失败请求同样可能已经缓存了前几批，不能只在成功路径淘汰。
            while len(self._cache) > 1000:
                self._cache.popitem(last=False)
            self._lock.release()

    def _list(
        self, actor_id: UUID, days: int, limit: int,
        kb_id: str | None, cursor: str | None,
    ) -> RefusalClusterPage:
        deadline = time.monotonic() + 60
        now = datetime.now(UTC)
        filters = _digest({
            "days": days, "kb_id": kb_id,
            "model": self._model, "threshold": self._threshold,
        })
        position = self._position(cursor, filters, now)
        as_of = position.as_of if position is not None else now
        start = as_of.astimezone(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0,
        ) - timedelta(days=days - 1)
        sample = self._reader.cluster_sample(start, as_of, kb_id)
        sample_hash = _sample_digest(sample)
        identifiers = tuple(item.turn_id for item in sample.items)
        if not identifiers:
            self._audit.record(actor_id, "read", "refusals", ())
        for offset in range(0, len(identifiers), 100):
            self._audit.record(
                actor_id, "read", "refusals", identifiers[offset:offset + 100],
            )
        vectors = self._vectors(sample.items, deadline)
        self._check_deadline(deadline)
        clusters = cluster_refusals(sample.items, vectors, self._threshold)
        self._check_deadline(deadline)
        current = self._reader.cluster_sample(start, as_of, kb_id)
        if _sample_digest(current) != sample_hash:
            raise RefusalClustersChangedError("拒答快照已改变")
        snapshot = _digest({
            "sample": sample_hash,
            "groups": [item.model_dump(mode="json") for item in clusters],
        })
        offset = 0
        if position is not None:
            if position.snapshot != snapshot:
                raise RefusalClustersChangedError("拒答聚类已改变")
            offset = position.offset
            if offset >= len(clusters):
                raise ValueError("聚类游标位置无效")
        page = clusters[offset:offset + limit]
        next_cursor = None
        if offset + limit < len(clusters):
            next_position = RefusalClusterPosition(
                as_of=as_of, offset=offset + limit,
                filters=filters, snapshot=snapshot,
            )
            next_cursor = base64.urlsafe_b64encode(
                next_position.model_dump_json().encode(),
            ).decode()
        return RefusalClusterPage(
            clusters=page, next_cursor=next_cursor,
            total_clusters=len(clusters), total_refusals=sample.total,
            clustered_refusals=len(sample.items),
            truncated=sample.total > len(sample.items), period_days=days,
            window_started_at=start, window_ended_at=as_of,
            similarity_threshold=self._threshold, embedding_model=self._model,
        )

    @staticmethod
    def _position(
        cursor: str | None, filters: str, now: datetime,
    ) -> RefusalClusterPosition | None:
        if cursor is None:
            return None
        try:
            position = RefusalClusterPosition.model_validate_json(
                base64.b64decode(cursor, altchars=b"-_", validate=True),
            )
            if position.filters != filters:
                raise ValueError("聚类筛选条件不一致")
            age = (now - position.as_of).total_seconds()
            if age < 0 or age > 300:
                raise ValueError("聚类游标已过期或时间无效")
            return position
        except (ValueError, binascii.Error) as error:
            raise ValueError("聚类游标无效") from error

    def _vectors(
        self, items: tuple[RefusalItem, ...], deadline: float,
    ) -> list[tuple[float, ...]]:
        now = time.monotonic()
        for key, (expires_at, vector) in list(self._cache.items()):
            if expires_at <= now:
                del self._cache[key]
        missing = {}
        keys = []
        for item in items:
            if len(item.question) > 16000:
                raise InsightStorageError("旧拒答原问题超过聚类批次上限")
            key = hashlib.sha256(item.question.encode()).hexdigest()
            keys.append(key)
            if key not in self._cache:
                missing[key] = item.question
        batch_keys = []
        batch_texts = []
        characters = 0
        for key, question in missing.items():
            if batch_texts and (len(batch_texts) >= 20 or characters + len(question) > 16000):
                self._embed_batch(batch_keys, batch_texts, deadline)
                batch_keys, batch_texts, characters = [], [], 0
            batch_keys.append(key)
            batch_texts.append(question)
            characters += len(question)
        if batch_texts:
            self._embed_batch(batch_keys, batch_texts, deadline)
        vectors = [self._cache[key][1] for key in keys]
        if vectors and any(len(vector) != len(vectors[0]) for vector in vectors):
            self._cache.clear()
            raise EmbeddingResponseError("聚类缓存与模型维度不一致")
        return vectors

    def _embed_batch(
        self, keys: list[str], texts: list[str], deadline: float,
    ) -> None:
        self._check_deadline(deadline)
        vectors = self._embedder.embed_documents(texts)
        if len(vectors) != len(texts):
            raise EmbeddingResponseError("聚类向量数量不一致")
        normalized = [normalize_vector(vector) for vector in vectors]
        self._check_deadline(deadline)
        expires_at = time.monotonic() + 300
        for key, vector in zip(keys, normalized):
            self._cache[key] = (expires_at, vector)

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise RefusalClustersTimeoutError("拒答聚类超过时间预算")

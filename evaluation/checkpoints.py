# -*- coding: utf-8 -*-
"""长任务检查点、稳定错误分类、重试与单进程锁。"""

import errno
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from app.application.errors import (
    EmbeddingAuthenticationError,
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
    LLMAuthenticationError,
    LLMGenerationError,
    LLMMissingAPIKeyError,
    LLMRateLimitError,
    LLMTimeoutError,
    MissingAPIKeyError,
    RerankerUnavailableError,
)


T = TypeVar("T")


class CheckpointChannel(StrEnum):
    """一次评估运行的三个 append-only 记录通道。"""

    LOCAL = "local"
    RGB = "rgb"
    RAGAS = "ragas"


class StableErrorCategory(StrEnum):
    """允许进入检查点和报告的稳定错误类别。"""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    INVALID_DATA = "invalid_data"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


class InvalidRemoteResponseError(RuntimeError):
    """远程模型响应不符合固定结构时抛出。"""


@dataclass(frozen=True)
class RetryPolicy:
    """远程调用的有界指数退避策略。

    Attributes:
        max_attempts: 包含首次调用在内的最大尝试次数。
        initial_delay_seconds: 第一次重试前的等待秒数。
        multiplier: 每次重试的等待倍率。
        sleeper: 实际执行等待的可注入函数。
    """

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    multiplier: float = 2.0
    sleeper: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        """校验策略不会产生无界或负数等待。

        Raises:
            ValueError: 尝试次数、初始延迟或倍率无效。
        """
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须大于 0")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds 不能小于 0")
        if self.multiplier < 1:
            raise ValueError("multiplier 不能小于 1")


class RunCheckpoint:
    """一次评估运行的清单与 append-only JSONL 文件。

    Attributes:
        run_id: 稳定运行标识。
        run_path: `work` 下本次运行的目录。
        manifest_hash: 规范运行清单的 SHA-256。
    """

    def __init__(
        self,
        run_id: str,
        run_path: Path,
        manifest_hash: str,
        successful_keys: dict[CheckpointChannel, set[str]],
    ) -> None:
        """保存已经验证的运行目录状态。

        Args:
            run_id: 稳定运行标识。
            run_path: 本次运行目录。
            manifest_hash: 规范运行清单 SHA-256。
            successful_keys: 每个通道已有的成功记录 key。
        """
        self.run_id = run_id
        self.run_path = run_path
        self.manifest_hash = manifest_hash
        self._successful_keys = successful_keys

    @classmethod
    def create(
        cls,
        work_root: Path,
        run_id: str,
        manifest: BaseModel | dict[str, Any],
    ) -> "RunCheckpoint":
        """创建全新的运行目录并持久化规范清单。

        Args:
            work_root: 所有恢复目录的根路径。
            run_id: 新运行的唯一标识。
            manifest: 非敏感运行配置与输入哈希。

        Returns:
            尚无成功记录的检查点管理器。

        Raises:
            FileExistsError: 相同 run ID 已存在。
        """
        work_root.mkdir(parents=True, exist_ok=True)
        run_path = work_root / run_id
        run_path.mkdir(exist_ok=False)
        manifest_payload = _manifest_payload(manifest)
        manifest_json = _canonical_json(manifest_payload)
        _write_durable_text(run_path / "manifest.json", manifest_json + "\n")
        return cls(
            run_id,
            run_path,
            _sha256_text(manifest_json),
            _empty_successful_keys(),
        )

    @classmethod
    def resume(
        cls,
        work_root: Path,
        run_id: str,
        manifest: BaseModel | dict[str, Any],
    ) -> "RunCheckpoint":
        """在清单完全匹配时恢复已有运行。

        Args:
            work_root: 所有恢复目录的根路径。
            run_id: 要恢复的运行标识。
            manifest: 当前非敏感运行配置与输入哈希。

        Returns:
            已加载每个通道成功 key 的检查点管理器。

        Raises:
            ValueError: 保存清单损坏或与当前清单不一致。
        """
        run_path = work_root / run_id
        saved_path = run_path / "manifest.json"
        try:
            saved_payload = json.loads(saved_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("恢复运行清单不存在或损坏") from exc
        current_payload = _manifest_payload(manifest)
        saved_json = _canonical_json(saved_payload)
        current_json = _canonical_json(current_payload)
        if saved_json != current_json:
            raise ValueError("恢复运行清单不匹配")

        successful_keys = _empty_successful_keys()
        for channel in CheckpointChannel:
            successful_keys[channel] = _load_successful_keys(
                run_path / f"{channel.value}.jsonl"
            )
        return cls(
            run_id,
            run_path,
            _sha256_text(saved_json),
            successful_keys,
        )

    def append(
        self,
        channel: CheckpointChannel,
        record: BaseModel,
    ) -> None:
        """追加一条记录并在返回前 flush、fsync。

        Args:
            channel: 目标 JSONL 通道。
            record: 至少包含 `key` 和 `status` 的严格记录。

        Returns:
            None。

        Raises:
            ValueError: 记录缺少 key/status 或成功 key 已存在。
        """
        payload = record.model_dump(mode="json")
        key = payload.get("key")
        status = payload.get("status")
        if not isinstance(key, str) or not key:
            raise ValueError("检查点记录缺少非空 key")
        if not isinstance(status, str) or not status:
            raise ValueError("检查点记录缺少非空 status")
        if status == "success" and key in self._successful_keys[channel]:
            raise ValueError("检查点成功 key 重复")

        line = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        path = self.run_path / f"{channel.value}.jsonl"
        with path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(line + "\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        if status == "success":
            self._successful_keys[channel].add(key)

    def successful_keys(self, channel: CheckpointChannel) -> frozenset[str]:
        """返回指定通道已经完成的成功 key。

        Args:
            channel: 要查询的检查点通道。

        Returns:
            不可变成功 key 集合。
        """
        return frozenset(self._successful_keys[channel])

    def records(
        self,
        channel: CheckpointChannel,
    ) -> dict[str, dict[str, Any]]:
        """返回指定通道按成功 key 索引的完整记录。

        Args:
            channel: 要读取的检查点通道。

        Returns:
            每个成功 key 对应的 JSON 兼容记录副本。

        Raises:
            ValueError: 检查点文件损坏或包含重复成功 key。
        """
        path = self.run_path / f"{channel.value}.jsonl"
        return _load_successful_records(path)


@dataclass(frozen=True)
class EvaluationRunLock:
    """由当前进程独占的评估路径锁。

    Attributes:
        path: 锁文件路径。
        run_id: 当前运行标识。
        manifest_hash: 当前运行清单哈希。
    """

    path: Path
    run_id: str
    manifest_hash: str

    @classmethod
    def acquire(
        cls,
        path: Path,
        resume: bool,
        run_id: str = "evaluation",
        manifest_hash: str = "",
    ) -> "EvaluationRunLock":
        """独占创建锁，或显式替换已确认失效的锁。

        Args:
            path: 单一评估路径共享的锁文件。
            resume: 是否允许替换已确认 PID 不存在的旧锁。
            run_id: 当前运行标识。
            manifest_hash: 当前运行清单哈希。

        Returns:
            可作为上下文管理器使用的已持有锁。

        Raises:
            RuntimeError: 锁由活跃进程持有，或旧锁未显式恢复。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "started_at": _utc_now_text(),
            "run_id": run_id,
            "manifest_hash": manifest_hash,
        }
        try:
            _create_lock_file(path, payload)
        except FileExistsError:
            existing = _read_lock_file(path)
            existing_pid = existing.get("pid")
            if isinstance(existing_pid, int) and _process_exists(existing_pid):
                raise RuntimeError("已有评估进程持有锁")
            if not resume:
                raise RuntimeError("失效评估锁必须显式使用 resume 恢复")
            path.unlink()
            try:
                _create_lock_file(path, payload)
            except FileExistsError as exc:
                raise RuntimeError("评估锁恢复时被其他进程占用") from exc
        return cls(path=path, run_id=run_id, manifest_hash=manifest_hash)

    def __enter__(self) -> "EvaluationRunLock":
        """返回当前已持有锁。

        Returns:
            当前锁实例。
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """仅在锁仍属于当前运行时删除锁文件。

        Args:
            exc_type: 上下文异常类型。
            exc_value: 上下文异常实例。
            traceback: 上下文异常 traceback。

        Returns:
            None。
        """
        del exc_type
        del exc_value
        del traceback
        try:
            payload = _read_lock_file(self.path)
        except RuntimeError:
            return
        is_owner = (
            payload.get("pid") == os.getpid()
            and payload.get("run_id") == self.run_id
            and payload.get("manifest_hash") == self.manifest_hash
        )
        if is_owner:
            self.path.unlink(missing_ok=True)


def classify_remote_error(error: Exception) -> StableErrorCategory:
    """把远程与领域异常映射为不泄露详情的稳定类别。

    Args:
        error: 捕获的调用异常。

    Returns:
        允许写入检查点和报告的稳定类别枚举。
    """
    if isinstance(
        error,
        (EmbeddingTimeoutError, LLMTimeoutError, TimeoutError),
    ):
        return StableErrorCategory.TIMEOUT
    if isinstance(error, (EmbeddingRateLimitError, LLMRateLimitError)):
        return StableErrorCategory.RATE_LIMIT
    if isinstance(
        error,
        (EmbeddingAuthenticationError, LLMAuthenticationError),
    ):
        return StableErrorCategory.AUTHENTICATION
    if isinstance(error, (LLMMissingAPIKeyError, MissingAPIKeyError)):
        return StableErrorCategory.CONFIGURATION
    if isinstance(
        error,
        (EmbeddingResponseError, InvalidRemoteResponseError),
    ):
        return StableErrorCategory.INVALID_RESPONSE
    if isinstance(error, (LLMGenerationError, EmbeddingError)):
        return StableErrorCategory.INVALID_RESPONSE
    if isinstance(error, RerankerUnavailableError):
        return _classify_http_cause(error.__cause__)
    if isinstance(error, (ValueError, TypeError, json.JSONDecodeError)):
        return StableErrorCategory.INVALID_DATA
    return _classify_http_cause(error)


def run_with_retry(
    operation: Callable[[], T],
    policy: RetryPolicy,
) -> T:
    """只对超时和限流执行有界指数退避。

    Args:
        operation: 无参数远程调用。
        policy: 最大尝试次数和等待策略。

    Returns:
        首次成功调用的返回值。

    Raises:
        Exception: 永久错误或最后一次瞬时错误，保留原异常类型。
    """
    delay = policy.initial_delay_seconds
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            category = classify_remote_error(exc)
            retryable = category in {
                StableErrorCategory.TIMEOUT,
                StableErrorCategory.RATE_LIMIT,
            }
            if not retryable or attempt == policy.max_attempts:
                raise
            policy.sleeper(delay)
            delay *= policy.multiplier
    raise RuntimeError("重试循环未返回结果")


def _classify_http_cause(error: BaseException | None) -> StableErrorCategory:
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return StableErrorCategory.TIMEOUT
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        if status_code == 429:
            return StableErrorCategory.RATE_LIMIT
        if status_code in {401, 403}:
            return StableErrorCategory.AUTHENTICATION
        return StableErrorCategory.INVALID_RESPONSE
    return StableErrorCategory.UNKNOWN


def _manifest_payload(
    manifest: BaseModel | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(manifest, BaseModel):
        return manifest.model_dump(mode="json")
    return dict(manifest)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_durable_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as file_handle:
        file_handle.write(content)
        file_handle.flush()
        os.fsync(file_handle.fileno())


def _empty_successful_keys() -> dict[CheckpointChannel, set[str]]:
    return {channel: set() for channel in CheckpointChannel}


def _load_successful_keys(path: Path) -> set[str]:
    return set(_load_successful_records(path))


def _load_successful_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    successful_records: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("检查点记录必须是对象")
            if payload.get("status") != "success":
                continue
            key = payload.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError("成功记录缺少 key")
            if key in successful_records:
                raise ValueError("检查点成功 key 重复")
            successful_records[key] = payload
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(f"检查点文件损坏：{path.name}") from exc
    return successful_records


def _create_lock_file(path: Path, payload: dict[str, object]) -> None:
    content = _canonical_json(payload) + "\n"
    _write_durable_text(path, content)


def _read_lock_file(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("评估锁文件不存在或损坏") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("评估锁文件格式无效")
    return payload


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        raise
    return True


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

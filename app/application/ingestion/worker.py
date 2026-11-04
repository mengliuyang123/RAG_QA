# -*- coding: utf-8 -*-
"""一个后台消费线程读取持久队列，不使用无界Future或仅内存任务。"""

import logging
from collections.abc import Callable
from math import isfinite
from threading import Event
from threading import RLock
from threading import Thread
from threading import current_thread

from app.application.errors import IngestionWorkerUnavailableError
from app.application.ingestion.job_runner import IngestionJobRunner


logger = logging.getLogger(__name__)


class IngestionWorker:
    """空闲时只做只读队列探测，存在任务才惰性打开处理依赖。"""

    def __init__(
        self, has_pending: Callable[[], bool],
        runner_loader: Callable[[], IngestionJobRunner], stop: Event, *,
        poll_seconds: float = 0.5,
    ) -> None:
        """保存后台生命周期及有界轮询配置，不立即启动线程。

        Args:
            has_pending: 不创建生产表或模型的只读队列探测。
            runner_loader: 有任务时才装配的执行器。
            stop: 容器关闭后永久置位的停止信号。
            poll_seconds: 有限正数轮询间隔。

        Returns:
            None。
        """
        if not isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("后台入库轮询间隔无效")
        self._has_pending = has_pending
        self._runner_loader = runner_loader
        self._stop = stop
        self._poll_seconds = poll_seconds
        self._wake = Event()
        self._lock = RLock()
        self._thread: Thread | None = None

    def start(self) -> None:
        """幂等启动单个消费线程，已关闭容器不得重新启动。

        Returns:
            None。
        """
        with self._lock:
            if self._stop.is_set():
                raise IngestionWorkerUnavailableError("入库执行器已停止")
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = Thread(target=self._run, name="rag-ingestion-worker")
            self._thread.start()

    def ensure_running(self) -> None:
        """前台202接收必须处于真实后台生命周期内。

        Returns:
            None。
        """
        with self._lock:
            if self._stop.is_set() or self._thread is None:
                raise IngestionWorkerUnavailableError("入库执行器尚未就绪")
            if not self._thread.is_alive():
                raise IngestionWorkerUnavailableError("入库执行器尚未就绪")

    def wake(self) -> None:
        """提交后提前唤醒轮询，队列事实仍在SQLite而不是此Event。

        Returns:
            None。
        """
        self._wake.set()

    def close(self) -> None:
        """停止新批次并等待当前线程与其租约心跳退出，再关闭模型连接。

        Returns:
            None。
        """
        self._stop.set()
        self._wake.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._has_pending():
                    if self._runner_loader().run_next(self._stop):
                        continue
            except Exception:
                # 只输出安全状态，不打印正文／凭据或上游异常；留待下个有界周期。
                logger.warning("后台入库暂时不可用，将重试队列探测")
            self._wake.wait(self._poll_seconds)
            self._wake.clear()

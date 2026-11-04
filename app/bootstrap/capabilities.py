# -*- coding: utf-8 -*-
"""组合根的惰性能力桥接：本地维护不应要求远程模型配置。"""

from collections.abc import Callable

from app.application.ports.embedding import EmbeddingProvider


class DeferredEmbeddingProvider:
    """真正向量化时才获取同一容器provider，生命周期仍由组合根负责。"""

    def __init__(self, provider: Callable[[], EmbeddingProvider]) -> None:
        """保存惰性获取函数，不创建SDK客户端。

        Args:
            provider: 组合根提供的线程安全共享实例获取函数。

        Returns:
            None。
        """
        self._provider = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """实际向量化时获取模型，不伪造缺失模型的结果。

        Args:
            texts: 实际需要向量化的文本。

        Returns:
            真实provider按输入顺序生成的向量。
        """
        return self._provider().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """保留完整端口语义，实际查询才初始化共享provider。

        Args:
            text: 实际查询文本。

        Returns:
            真实查询向量。
        """
        return self._provider().embed_query(text)

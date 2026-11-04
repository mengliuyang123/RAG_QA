# -*- coding: utf-8 -*-
"""通过兼容 DashScope ``/rerank`` 协议的服务重排候选。"""

from math import isfinite
from typing import Any

import httpx

from app.application.errors import RerankerUnavailableError
from app.application.ports.reranking import Reranker
from app.application.tracing import provider_call
from app.domain.retrieval import RetrievalResult
from app.infrastructure.generation.usage import extract_token_usage


class RemoteReranker(Reranker):
    """调用远程 Cross-Encoder 服务重排候选。

    Attributes:
        _base_url: 远程服务的基础地址。
        _api_key: 可选的 Bearer 凭据。
        _model: 远程重排模型名称。
        _endpoint: 去除首尾斜杠的相对端点。
        _owns_client: 是否由本适配器创建并关闭 HTTP 客户端。
        _client: 执行远程请求的同步 HTTP 客户端。
    """

    def __init__(
        self,
        base_url: str | None,
        api_key: str | None,
        model: str,
        endpoint: str = "rerank",
        client: httpx.Client | None = None,
    ) -> None:
        """保存远程服务配置并按需创建 HTTP 客户端。

        Args:
            base_url: 远程重排服务基础地址。
            api_key: 用于 Bearer 认证的可选密钥。
            model: 远程服务使用的重排模型。
            endpoint: 基础地址后的重排端点。
            client: 测试或共享场景可注入的 HTTP 客户端。

        Returns:
            None。
        """
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint.strip().strip("/")
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=10.0)

    def rerank(
        self,
        question: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """调用远程服务并返回按相关性重新评分的候选。

        Args:
            question: 用于比较候选相关性的用户问题。
            candidates: 保持融合顺序的待重排候选。
            top_k: 最多返回的重排候选数量。

        Returns:
            使用远程相关性分数并按降序排列的候选。

        Raises:
            ValueError: ``top_k`` 不是正数。
            RerankerUnavailableError: 配置、请求或远程响应无效。
        """
        if not candidates:
            return []
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        base_url = self._normalized_base_url()
        if not self._endpoint:
            raise RerankerUnavailableError("Reranker endpoint 不能为空")
        expected_count = min(top_k, len(candidates))
        api_key = self._api_key.strip() if self._api_key else ""
        try:
            with provider_call("reranking", base_url, self._model) as call:
                response = self._client.post(
                    f"{base_url}/{self._endpoint}",
                    headers=(
                        {"Authorization": f"Bearer {api_key}"}
                        if api_key else None
                    ),
                    json={
                        "model": self._model,
                        "query": question,
                        "documents": [candidate.text for candidate in candidates],
                        "top_n": expected_count,
                    },
                )
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise RerankerUnavailableError("重排服务返回无效响应") from exc
                if isinstance(payload, dict):
                    call.usage = extract_token_usage(payload.get("usage"), no_generation=True)
                    call.response_model = payload.get("model")
        except httpx.HTTPError as exc:
            raise RerankerUnavailableError("重排服务请求失败") from exc

        try:
            ranked = self._parse_results(
                payload,
                candidate_count=len(candidates),
                expected_count=expected_count,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankerUnavailableError("重排服务返回无效响应") from exc

        return [
            candidates[index].model_copy(update={"score": score})
            for index, score in ranked[:top_k]
        ]

    def close(self) -> None:
        """关闭本适配器自行创建的 HTTP 连接池。

        Returns:
            None。
        """
        if self._owns_client:
            self._client.close()

    def _normalized_base_url(self) -> str:
        """返回无末尾斜杠的服务地址，并拒绝缺失配置。"""
        if self._base_url is None or not self._base_url.strip():
            raise RerankerUnavailableError(
                "未配置 RERANKER_BASE_URL，重排服务不可用"
            )
        return self._base_url.strip().rstrip("/")

    @staticmethod
    def _parse_results(
        payload: Any,
        candidate_count: int,
        expected_count: int,
    ) -> list[tuple[int, float]]:
        """验证响应索引与分数，并返回确定性相关度排序。"""
        if not isinstance(payload, dict):
            raise ValueError("响应必须是对象")
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("响应缺少 results 列表")

        parsed: list[tuple[int, float]] = []
        seen_indices: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("results 项必须是对象")
            index = item.get("index")
            score = item.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("响应索引必须是整数")
            if not 0 <= index < candidate_count or index in seen_indices:
                raise ValueError("响应索引重复或越界")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("响应缺少有效相关度分数")
            try:
                normalized_score = float(score)
            except (OverflowError, ValueError) as exc:
                raise ValueError("响应缺少有效相关度分数") from exc
            if not isfinite(normalized_score):
                raise ValueError("响应缺少有效相关度分数")
            seen_indices.add(index)
            parsed.append((index, normalized_score))

        if len(parsed) < expected_count:
            raise ValueError("响应结果数量少于请求数量")

        return sorted(parsed, key=lambda item: (-item[1], item[0]))

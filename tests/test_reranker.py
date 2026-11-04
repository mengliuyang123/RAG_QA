import httpx
import pytest

from app.application.errors import RerankerUnavailableError
from app.infrastructure.reranking.dashscope import RemoteReranker
from app.domain.retrieval import RetrievalResult


def make_candidate(
    chunk_id: str,
    text: str,
    *,
    score: float,
) -> RetrievalResult:
    """创建具有不同来源信息的待重排候选。"""
    doc_id, index_text = chunk_id.split(":")
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        filename=f"{doc_id}.md",
        page=int(index_text) + 1,
        chunk_index=int(index_text),
        score=score,
    )


def test_remote_reranker_maps_indices_and_sorts_by_relevance() -> None:
    """服务端结果乱序时，应按 index 映射候选并按新分数降序返回。"""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.2},
                    {"index": 1, "relevance_score": 0.95},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reranker = RemoteReranker(
        base_url="http://reranker.test/v1/",
        api_key="secret-token",
        model="BAAI/bge-reranker-v2-m3",
        client=client,
    )
    first = make_candidate("doc-a:0", "普通文本", score=0.8)
    second = make_candidate("doc-b:1", "与问题高度相关", score=0.1)

    results = reranker.rerank("什么最相关？", [first, second], top_k=2)

    assert [result.chunk_id for result in results] == ["doc-b:1", "doc-a:0"]
    assert [result.score for result in results] == pytest.approx([0.95, 0.2])
    assert results[0].model_dump(exclude={"score"}) == second.model_dump(
        exclude={"score"}
    )
    assert captured == {
        "url": "http://reranker.test/v1/rerank",
        "authorization": "Bearer secret-token",
        "payload": (
            '{"model":"BAAI/bge-reranker-v2-m3",'
            '"query":"什么最相关？",'
            '"documents":["普通文本","与问题高度相关"],'
            '"top_n":2}'
        ),
    }


def test_remote_reranker_supports_dashscope_plural_endpoint() -> None:
    """DashScope 模式应请求 OpenAI 兼容的复数 `/reranks` 端点。"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://reranker.test/v1/reranks"
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.9}]},
        )

    reranker = RemoteReranker(
        base_url="https://reranker.test/v1/",
        api_key="dashscope-key",
        model="qwen3-rerank",
        endpoint="/reranks",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    results = reranker.rerank(
        "什么是重排序模型？",
        [make_candidate("doc-a:0", "重排序模型用于重新打分。", score=0.1)],
        top_k=1,
    )

    assert results[0].score == pytest.approx(0.9)


def test_remote_reranker_returns_empty_candidates_without_request() -> None:
    """空候选必须直接返回，不能要求服务配置或访问网络。"""

    def reject_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("空候选不应发送请求")

    reranker = RemoteReranker(
        base_url=None,
        api_key=None,
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(reject_request)),
    )

    assert reranker.rerank("问题", [], top_k=5) == []


def test_remote_reranker_limits_response_to_top_k() -> None:
    """上游多返回结果时，组件仍只能暴露调用方请求的数量。"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://reranker.test/rerank"
        assert b'"top_n":1' in request.content
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ]
            },
        )

    candidates = [
        make_candidate("doc-a:0", "A", score=0.3),
        make_candidate("doc-b:0", "B", score=0.2),
        make_candidate("doc-c:0", "C", score=0.1),
    ]
    reranker = RemoteReranker(
        base_url="http://reranker.test",
        api_key=None,
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    results = reranker.rerank("问题", candidates, top_k=1)

    assert [result.chunk_id for result in results] == ["doc-b:0"]


def test_remote_reranker_returns_all_candidates_when_top_k_is_larger() -> None:
    """top_k 超过候选数时，应把实际候选数作为 top_n 并返回全部命中。"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert b'"top_n":2' in request.content
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 0, "relevance_score": 0.4},
                ]
            },
        )

    candidates = [
        make_candidate("doc-a:0", "A", score=0.2),
        make_candidate("doc-b:0", "B", score=0.1),
    ]
    reranker = RemoteReranker(
        base_url="http://reranker.test/v1",
        api_key=None,
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    results = reranker.rerank("问题", candidates, top_k=20)

    assert [result.chunk_id for result in results] == ["doc-b:0", "doc-a:0"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_remote_reranker_rejects_non_positive_top_k_without_request(
    top_k: int,
) -> None:
    """非正 top_k 不得形成无效 top_n 或触发负切片语义。"""

    def reject_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("无效 top_k 不应发送请求")

    reranker = RemoteReranker(
        base_url="http://reranker.test/v1",
        api_key=None,
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(reject_request)),
    )

    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank(
            "问题",
            [make_candidate("doc-a:0", "候选", score=0.1)],
            top_k=top_k,
        )


def test_remote_reranker_requires_configured_base_url() -> None:
    """非空候选缺少地址时，应抛出稳定的服务不可用异常。"""
    reranker = RemoteReranker(
        base_url="   ",
        api_key=None,
        model="test-model",
    )

    with pytest.raises(RerankerUnavailableError, match="RERANKER_BASE_URL"):
        reranker.rerank(
            "问题",
            [make_candidate("doc-a:0", "候选", score=0.1)],
            top_k=1,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
            ]
        },
        {"results": [{"index": 1, "relevance_score": 0.9}]},
        {"results": [{"index": 0}]},
        {},
    ],
    ids=[
        "duplicate-index",
        "out-of-range-index",
        "missing-score",
        "missing-results",
    ],
)
def test_remote_reranker_rejects_invalid_result_mapping(
    payload: dict[str, object],
) -> None:
    """候选映射不可信时，应拒绝整份响应而不是返回错配文本。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    reranker = RemoteReranker(
        base_url="http://reranker.test/v1",
        api_key=None,
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RerankerUnavailableError, match="无效响应"):
        reranker.rerank(
            "问题",
            [make_candidate("doc-a:0", "候选", score=0.1)],
            top_k=1,
        )


@pytest.mark.parametrize("result_count", [0, 1])
def test_remote_reranker_rejects_fewer_results_than_requested(
    result_count: int,
) -> None:
    """上游返回数量不足时，不得静默缩减调用方请求的结果集。"""
    response_results = [
        {"index": 0, "relevance_score": 0.9},
    ][:result_count]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": response_results})

    reranker = RemoteReranker(
        base_url="http://reranker.test/v1",
        api_key=None,
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    candidates = [
        make_candidate("doc-a:0", "候选 A", score=0.2),
        make_candidate("doc-b:0", "候选 B", score=0.1),
    ]

    with pytest.raises(RerankerUnavailableError, match="无效响应"):
        reranker.rerank("问题", candidates, top_k=2)


@pytest.mark.parametrize(
    "encoded_score",
    [str(10**400), "NaN", "Infinity", "true", '"0.5"'],
    ids=["huge-integer", "nan", "infinity", "boolean", "string"],
)
def test_remote_reranker_rejects_invalid_numeric_scores(
    encoded_score: str,
) -> None:
    """无法安全转换为有限浮点数的分数必须转换为领域异常。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                '{"results":[{"index":0,"relevance_score":'
                f"{encoded_score}"
                "}]}"
            ).encode(),
            headers={"content-type": "application/json"},
        )

    reranker = RemoteReranker(
        base_url="http://reranker.test/v1",
        api_key=None,
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RerankerUnavailableError, match="无效响应"):
        reranker.rerank(
            "问题",
            [make_candidate("doc-a:0", "候选", score=0.1)],
            top_k=1,
        )


def test_remote_reranker_converts_invalid_json_to_domain_error() -> None:
    """上游返回非 JSON 内容时，不得向调用方泄漏 JSON 解析异常。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    reranker = RemoteReranker(
        base_url="http://reranker.test/v1",
        api_key=None,
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RerankerUnavailableError, match="无效响应"):
        reranker.rerank(
            "问题",
            [make_candidate("doc-a:0", "候选", score=0.1)],
            top_k=1,
        )


def test_remote_reranker_converts_timeout_to_domain_error() -> None:
    """远程请求超时应统一转换为 Reranker 领域异常。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    reranker = RemoteReranker(
        base_url="http://reranker.test/v1",
        api_key=None,
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RerankerUnavailableError, match="请求失败"):
        reranker.rerank(
            "问题",
            [make_candidate("doc-a:0", "候选", score=0.1)],
            top_k=1,
        )


def test_remote_reranker_converts_http_error_to_domain_error() -> None:
    """认证失败等非成功状态应统一转换为 Reranker 领域异常。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid token"})

    reranker = RemoteReranker(
        base_url="http://reranker.test/v1",
        api_key="bad-token",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RerankerUnavailableError, match="请求失败"):
        reranker.rerank(
            "问题",
            [make_candidate("doc-a:0", "候选", score=0.1)],
            top_k=1,
        )


def test_remote_reranker_closes_only_internally_owned_client(
    monkeypatch,
) -> None:
    """close 应释放组件创建的连接池，但不能关闭调用方注入的客户端。"""
    internal_client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200))
    )
    external_client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200))
    )
    monkeypatch.setattr(
        "app.infrastructure.reranking.dashscope.httpx.Client",
        lambda **_kwargs: internal_client,
    )

    internally_owned = RemoteReranker(
        base_url="http://reranker.test/v1",
        api_key=None,
        model="test-model",
    )
    externally_owned = RemoteReranker(
        base_url="http://reranker.test/v1",
        api_key=None,
        model="test-model",
        client=external_client,
    )

    internally_owned.close()
    externally_owned.close()

    assert internal_client.is_closed
    assert not external_client.is_closed
    external_client.close()

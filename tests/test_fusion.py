import pytest

from app.application.retrieval.fusion import reciprocal_rank_fusion
from app.domain.retrieval import RetrievalResult


def make_result(chunk_id: str, *, score: float = 0.5) -> RetrievalResult:
    """创建保留完整来源字段的融合测试结果。"""
    doc_id, index_text = chunk_id.split(":")
    index = int(index_text)
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"文本块 {chunk_id}",
        filename=f"{doc_id}.md",
        page=None,
        chunk_index=index,
        score=score,
    )


def test_rrf_accumulates_exact_scores_across_rankings() -> None:
    """同一文本块位于第 1、2 名时应累加两个倒数排名分数。"""
    first = make_result("doc-a:0", score=0.99)
    second = make_result("doc-b:0", score=100.0)

    results = reciprocal_rank_fusion([
        [first, second],
        [second, first],
    ])

    assert [result.chunk_id for result in results] == ["doc-a:0", "doc-b:0"]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert results[1].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_deduplicates_chunk_ids_and_ignores_raw_scores() -> None:
    """融合应按 chunk_id 去重，并完全忽略不可比较的原始分数。"""
    vector_result = make_result("doc-a:0", score=-500.0)
    keyword_result = make_result("doc-a:0", score=9999.0)

    results = reciprocal_rank_fusion([[vector_result], [keyword_result]])

    assert len(results) == 1
    assert results[0].chunk_id == "doc-a:0"
    assert results[0].score == pytest.approx(2 / 61)


def test_rrf_counts_duplicate_id_only_once_per_ranking() -> None:
    """同一路意外返回重复 ID 时只能贡献其最佳排名一次。"""
    duplicate = make_result("doc-a:0")

    results = reciprocal_rank_fusion([[duplicate, duplicate]])

    assert results[0].score == pytest.approx(1 / 61)


def test_rrf_allows_empty_rankings() -> None:
    """任一路合法为空时，另一结果集应可独立参与融合。"""
    only_result = make_result("doc-a:0")

    assert reciprocal_rank_fusion([[], [only_result]])[0].chunk_id == "doc-a:0"
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_breaks_complete_ties_by_chunk_id() -> None:
    """总分和最佳单路排名相同时，应按 chunk_id 确定性排序。"""
    result_a = make_result("doc-a:0")
    result_b = make_result("doc-b:0")

    results = reciprocal_rank_fusion([
        [result_b, result_a],
        [result_a, result_b],
    ])

    assert [result.chunk_id for result in results] == ["doc-a:0", "doc-b:0"]

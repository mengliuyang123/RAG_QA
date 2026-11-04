# -*- coding: utf-8 -*-
"""RGB 固定缓存、passage 组装和官方评分测试。"""

import hashlib
import json
from pathlib import Path

import pytest


def _jsonl_bytes(records: list[dict[str, object]]) -> bytes:
    lines = []
    for record in records:
        lines.append(json.dumps(record, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_rgb_cache_parses_jsonl_and_preserves_asnwer1(tmp_path: Path) -> None:
    """扩展名为 JSON 的文件应逐行解析，并保留上游拼写字段。"""
    from evaluation.models import RGBFile, RGBManifest
    from evaluation.rgb import RGBDatasetCache

    refine_bytes = _jsonl_bytes(
        [{"id": 1, "query": "q", "answer": ["a"],
          "positive": ["p"], "negative": ["n"]}]
    )
    int_bytes = _jsonl_bytes(
        [{"id": 2, "query": "iq", "answer": ["a", "b"],
          "asnwer1": ["a"], "answer2": ["b"],
          "positive": [["p1"], ["p2"]], "negative": ["n"]}]
    )
    instruction_bytes = (
        'zh:\n  system: "系统"\n  instruction: "文档：{DOCS} 问题：{QUERY}"\n'
    ).encode("utf-8")
    payloads = {
        "data/zh_refine.json": refine_bytes,
        "data/zh_int.json": int_bytes,
        "config/instruction.yaml": instruction_bytes,
    }
    files = {}
    for name, content in payloads.items():
        key = "instruction"
        count = None
        if "refine" in name:
            key = "zh_refine"
            count = 1
        elif "zh_int" in name:
            key = "zh_int"
            count = 1
        files[key] = RGBFile(
            path=name,
            sha256=hashlib.sha256(content).hexdigest(),
            records_count=count,
        )
    manifest = RGBManifest.model_construct(
        repository_url="https://github.com/chen700564/RGB",
        commit="a" * 40,
        license="CC BY-NC-SA 4.0",
        noncommercial=True,
        passage_num=5,
        passage_seed=2333,
        noise_rates=[0.0, 0.4, 0.8, 1.0],
        files=files,
        zh_refine_ids=[1] * 30,
        zh_int_ids=[2] * 30,
    )
    calls: list[str] = []

    def downloader(url: str) -> bytes:
        calls.append(url)
        for path, content in payloads.items():
            if url.endswith(path):
                return content
        raise AssertionError(url)

    dataset = RGBDatasetCache(tmp_path, downloader=downloader).prepare(manifest)

    assert dataset.integration[2].asnwer1 == ["a"]
    assert dataset.system_prompt == "系统"
    assert len(calls) == 3
    RGBDatasetCache(tmp_path, downloader=downloader).prepare(manifest)
    assert len(calls) == 3


def test_rgb_integration_allows_upstream_empty_answer2_helper() -> None:
    """真实上游后段记录允许辅助 answer2 为空，主 answer 仍须非空。"""
    from evaluation.rgb import RGBIntegrationRecord

    record = RGBIntegrationRecord(
        id=84,
        query="q",
        answer=["primary"],
        asnwer1=["primary"],
        answer2=[],
        positive=[["passage"]],
        negative=[],
    )

    assert record.answer == ["primary"]
    assert record.answer2 == []


def test_rgb_cache_rejects_wrong_hash(tmp_path: Path) -> None:
    """下载字节与锁定哈希不同时不得进入缓存。"""
    from evaluation.models import RGBFile, RGBManifest
    from evaluation.rgb import RGBDatasetCache, RGBDatasetError

    file_model = RGBFile(
        path="data/zh_refine.json",
        sha256="a" * 64,
        records_count=1,
    )
    manifest = RGBManifest.model_construct(
        repository_url="https://github.com/chen700564/RGB",
        commit="a" * 40,
        files={"zh_refine": file_model},
    )

    with pytest.raises(RGBDatasetError, match="SHA-256"):
        RGBDatasetCache(
            tmp_path,
            downloader=lambda _url: b"wrong",
        ).prepare(manifest)


def test_refine_passages_use_exact_negative_counts_and_seed() -> None:
    """五篇 passage 在四档噪声下应分别使用 0、2、4、5 条负例。"""
    from evaluation.rgb import RGBRefineRecord, build_refine_case

    record = RGBRefineRecord(
        id=1,
        query="q",
        answer=["a"],
        positive=[f"p{i}" for i in range(5)],
        negative=[f"n{i}" for i in range(5)],
    )

    for noise_rate, negative_count in ((0.0, 0), (0.4, 2), (0.8, 4), (1.0, 5)):
        first = build_refine_case(record, noise_rate, passage_num=5, seed=2333)
        second = build_refine_case(record, noise_rate, passage_num=5, seed=2333)
        assert first.passages == second.passages
        actual_negatives = 0
        for passage in first.passages:
            if passage.startswith("n"):
                actual_negatives += 1
        assert actual_negatives == negative_count


def test_integration_passages_take_one_per_group_before_filling() -> None:
    """信息整合 case 应先覆盖每个 positive 分组再补足五篇。"""
    from evaluation.rgb import RGBIntegrationRecord, build_integration_case

    record = RGBIntegrationRecord(
        id=2,
        query="q",
        answer=["a", "b"],
        asnwer1=["a"],
        answer2=["b"],
        positive=[["a1", "a2", "a3"], ["b1", "b2", "b3"]],
        negative=["n1"],
    )

    case = build_integration_case(record, passage_num=5, seed=2333)

    assert len(case.passages) == 5
    assert any(passage.startswith("a") for passage in case.passages)
    assert any(passage.startswith("b") for passage in case.passages)
    assert all(not passage.startswith("n") for passage in case.passages)


@pytest.mark.parametrize(
    ("answer", "response", "expected"),
    [
        (["北京"], "答案是 北 京。", True),
        ([["北京", "北京市"]], "地点是北京市。", True),
        (["1月3日", "1月12日"], "1月3日举行。", False),
        (["1月3日", "1月12日"], "1月3日宣誓，1月12日开会。", True),
    ],
)
def test_rgb_official_score_requires_every_answer_component(
    answer: list[object],
    response: str,
    expected: bool,
) -> None:
    """普通与整合问题都应要求每个答案组成部分命中。"""
    from evaluation.rgb import RGBCase, score_rgb_official

    case = RGBCase(
        key="rgb-1",
        source="zh_refine",
        record_id=1,
        query="q",
        answer=answer,
        passages=["p"] * 5,
        passage_hashes=["a" * 64] * 5,
        noise_rate=0.0,
        capability="noise_robustness",
    )

    assert score_rgb_official(case, response) is expected


def test_rgb_rejection_requires_information_insufficient_marker() -> None:
    """纯噪声 case 只有包含“信息不足”时才算拒答成功。"""
    from evaluation.rgb import RGBCase, score_rgb_official

    case = RGBCase(
        key="rgb-1",
        source="zh_refine",
        record_id=1,
        query="q",
        answer=["a"],
        passages=["n"] * 5,
        passage_hashes=["a" * 64] * 5,
        noise_rate=1.0,
        capability="rejection",
    )

    assert score_rgb_official(case, "文档信息不足，无法回答。") is True
    assert score_rgb_official(case, "我不知道。") is False

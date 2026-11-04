# -*- coding: utf-8 -*-
"""CRUD-RAG 适配：固定哈希缓存、按事件分层且互不重叠的确定性抽样。"""

import hashlib
import json
from pathlib import Path

import pytest

from app.config import Settings
from evaluation import crud_rag
from evaluation.crud_rag import CrudRagError
from evaluation.crud_rag import build_crud_sample
from evaluation.crud_rag import fetch_crud_split


def _write_split(path: Path, events: int = 12) -> Path:
    """写一个结构与上游一致的小样本：每个事件在三档中都出现。"""
    data: dict[str, list[dict[str, str]]] = {
        "questanswer_1doc": [], "questanswer_2docs": [], "questanswer_3docs": [],
    }
    for number in range(events):
        event_id = f"{number:024x}"
        for key, count in (("questanswer_1doc", 1), ("questanswer_2docs", 2), ("questanswer_3docs", 3)):
            item = {
                "ID": event_id, "event": f"事件{number}",
                "questions": f"{key} 问题{number}", "answers": f"{key} 答案{number}",
            }
            for index in range(1, count + 1):
                item[f"news{index}"] = f"事件{number} {key} 新闻{index} 正文"
            data[key].append(item)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_sample_is_stratified_disjoint_and_deterministic(tmp_path: Path) -> None:
    """三档各取同样题数，事件不重叠，来源都在语料中，相同参数结果相同。

    Args:
        tmp_path: 临时目录。
    """
    split = _write_split(tmp_path / "split.json")
    corpus, dataset = build_crud_sample(split, per_tier=2, distractors=4, seed=7)

    tiers = [episode.tags[1] for episode in dataset.episodes]
    assert tiers.count("crud-1doc") == tiers.count("crud-2docs") == tiers.count("crud-3docs") == 2
    events = [episode.id.split("-")[-1] for episode in dataset.episodes]
    assert len(set(events)) == 6
    assert dataset.smoke_ids == tuple(episode.id for episode in dataset.episodes)

    lanes = {episode.tags[1]: episode.turns[0].expected_lane for episode in dataset.episodes}
    assert lanes == {"crud-1doc": "fast", "crud-2docs": "agent", "crud-3docs": "agent"}
    for episode in dataset.episodes:
        turn = episode.turns[0]
        expected_count = int(episode.tags[1][len("crud-")])
        assert len(turn.expected_sources) == expected_count
        assert set(turn.expected_sources) <= set(corpus.documents)

    sources = {source for episode in dataset.episodes for source in episode.turns[0].expected_sources}
    assert len(corpus.entries) == len(sources) + 4
    for entry in corpus.entries:
        text = corpus.documents[entry.filename]
        assert entry.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert entry.license == "Apache-2.0"

    assert build_crud_sample(split, per_tier=2, distractors=4, seed=7) == (corpus, dataset)
    assert build_crud_sample(split, per_tier=2, distractors=4, seed=8)[1] != dataset


def test_distractors_never_come_from_sampled_events(tmp_path: Path) -> None:
    """干扰新闻只能取自未抽中的事件，避免答案藏在别的题目来源里。

    Args:
        tmp_path: 临时目录。
    """
    split = _write_split(tmp_path / "split.json")
    corpus, dataset = build_crud_sample(split, per_tier=2, distractors=10, seed=3)
    sampled = {f"事件{int(episode.id.split('-')[-1], 16)} " for episode in dataset.episodes}
    sources = {source for episode in dataset.episodes for source in episode.turns[0].expected_sources}
    for filename, text in corpus.documents.items():
        if filename not in sources:
            assert not any(text.startswith(prefix) for prefix in sampled)


def test_sample_rejects_impossible_requests(tmp_path: Path) -> None:
    """事件或干扰新闻不够时直接失败，不悄悄缩小样本。

    Args:
        tmp_path: 临时目录。
    """
    split = _write_split(tmp_path / "split.json", events=5)
    with pytest.raises(CrudRagError, match="可用事件不足"):
        build_crud_sample(split, per_tier=2, distractors=0, seed=1)
    with pytest.raises(CrudRagError, match="干扰新闻不足"):
        build_crud_sample(split, per_tier=1, distractors=1000, seed=1)
    with pytest.raises(CrudRagError):
        build_crud_sample(split, per_tier=0, distractors=0, seed=1)


def test_fetch_uses_valid_cache_and_rejects_wrong_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """哈希匹配时不下载；下载内容哈希不符时拒绝写入缓存。

    Args:
        tmp_path: 缓存目录。
        monkeypatch: 替换固定哈希以使用小文件。
    """
    content = b'{"questanswer_1doc": []}'
    monkeypatch.setattr(crud_rag, "CRUD_SPLIT_SHA256", hashlib.sha256(content).hexdigest())
    urls: list[str] = []

    def download(url: str) -> bytes:
        urls.append(url)
        return content

    path = fetch_crud_split(tmp_path / "cache", downloader=download)
    assert path.read_bytes() == content
    assert urls == [
        "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/"
        f"{crud_rag.CRUD_COMMIT}/data/crud_split/split_merged.json",
    ]
    assert fetch_crud_split(tmp_path / "cache", downloader=download) == path
    assert len(urls) == 1

    with pytest.raises(CrudRagError, match="SHA-256"):
        fetch_crud_split(tmp_path / "other", downloader=lambda url: b"tampered")
    assert not (tmp_path / "other" / "split_merged.json").exists()


def test_cli_dry_run_plans_three_policies_per_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CRUD 预检：每题三种策略，不创建模型、索引或运行目录。

    Args:
        tmp_path: 临时目录。
        monkeypatch: 替换下载与有副作用的资源入口。
        capsys: 捕获预检输出。
    """
    from evaluation.lanes import cli

    split = _write_split(tmp_path / "split.json")
    monkeypatch.setattr(cli, "fetch_crud_split", lambda cache_root: split)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run不得创建远程或存储资源")

    for name in ("build_evaluation_runtime", "build_application_container",
                 "create_evaluation_judge_provider"):
        monkeypatch.setattr(cli, name, forbidden)
    settings = Settings(
        _env_file=None, llm_api_key="generator-secret", llm_model="generator",
        llm_base_url="https://generator.example/v1", embedding_api_key="embedding-secret",
        reranker_base_url=None, evaluation_judge_api_key="judge-secret",
        evaluation_judge_model="judge", evaluation_judge_base_url="https://judge.example/v1",
    )
    work = tmp_path / "work"
    code = cli.main([
        "--dry-run", "--benchmark", "crud-rag", "--crud-per-tier", "2",
        "--crud-distractors", "3", "--work-root", str(work),
    ], settings=settings)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark"] == "crud-rag"
    assert payload["purpose"] == "capability_evaluation"
    assert payload["planned_episodes"] == 6 * 3
    assert payload["planned_turns"] == 18
    assert payload["corpus_documents"] == 2 * (1 + 2 + 3) + 3
    assert not work.exists()


def test_cli_dry_run_can_limit_to_auto_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--policies auto 每题只计划一次，且指纹与三种策略不同，不会复用对方记录。

    Args:
        tmp_path: 临时目录。
        monkeypatch: 替换下载入口。
        capsys: 捕获预检输出。
    """
    from evaluation.lanes import cli

    split = _write_split(tmp_path / "split.json")
    monkeypatch.setattr(cli, "fetch_crud_split", lambda cache_root: split)
    settings = Settings(
        _env_file=None, llm_api_key="generator-secret", llm_model="generator",
        llm_base_url="https://generator.example/v1", embedding_api_key="embedding-secret",
        reranker_base_url=None, evaluation_judge_api_key="judge-secret",
        evaluation_judge_model="judge", evaluation_judge_base_url="https://judge.example/v1",
    )
    base = ["--dry-run", "--benchmark", "crud-rag", "--crud-per-tier", "2",
            "--crud-distractors", "3", "--work-root", str(tmp_path / "work")]
    assert cli.main(base, settings=settings) == 0
    everything = json.loads(capsys.readouterr().out)
    assert cli.main([*base, "--policies", "auto"], settings=settings) == 0
    auto = json.loads(capsys.readouterr().out)
    assert auto["planned_episodes"] == auto["planned_turns"] == 6
    assert auto["fingerprint"] != everything["fingerprint"]

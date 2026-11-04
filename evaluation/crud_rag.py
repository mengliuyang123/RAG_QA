# -*- coding: utf-8 -*-
"""把固定 commit 的 CRUD-RAG 问答子集转换为分通道评估的语料与标签。

CRUD-RAG（IAAR-Shanghai，Apache-2.0）的问答题自带来源新闻，因此只缓存
`split_merged.json`，不下载约 190MB 的完整新闻库。抽样按事件分层：同一
事件在单篇、两篇、三篇三档中都会出现，三档抽取的事件互不重叠，干扰文档
也只取未被抽中事件的新闻，避免一道题的答案藏在另一道题的来源里。
"""

import hashlib
import json
import os
import random
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

from evaluation.lanes.dataset import LaneDataset
from evaluation.lanes.models import LaneEpisode
from evaluation.lanes.models import LaneTurn
from evaluation.models import CorpusDataset
from evaluation.models import CorpusEntry


CRUD_REPOSITORY = "https://github.com/IAAR-Shanghai/CRUD_RAG"
CRUD_COMMIT = "1aace383994e1f68efa12cf2a8e2dadfb4102ceb"
CRUD_SPLIT_PATH = "data/crud_split/split_merged.json"
CRUD_SPLIT_SHA256 = "b199fb281ac526c1c8da7364cad7ea22b7f6fe1d4167e511edbdda8ad3bd943a"
CRUD_LICENSE = "Apache-2.0"

# (原始分组, 标签, 来源篇数, 预期通道)。多篇题需要整合多处证据，预期升级到 Agent；
# 这是本适配器的假设，路由准确率指标应结合各档正确率一起解读。
CRUD_TIERS = (
    ("questanswer_1doc", "1doc", 1, "fast"),
    ("questanswer_2docs", "2docs", 2, "agent"),
    ("questanswer_3docs", "3docs", 3, "agent"),
)


class CrudRagError(ValueError):
    """CRUD-RAG 缓存、哈希或抽样不满足约定。"""


def fetch_crud_split(
    cache_root: Path,
    downloader: Callable[[str], bytes] | None = None,
) -> Path:
    """返回已校验的固定问答文件，缺失或哈希不符时重新下载。

    Args:
        cache_root: CRUD-RAG 缓存目录。
        downloader: 接收 raw URL 返回完整字节的函数，测试可替换。

    Returns:
        SHA-256 与固定值一致的本地文件路径。

    Raises:
        CrudRagError: 下载内容的 SHA-256 与固定值不一致。
    """
    destination = cache_root / "split_merged.json"
    if destination.is_file() and _sha256(destination.read_bytes()) == CRUD_SPLIT_SHA256:
        return destination
    repository = CRUD_REPOSITORY.removeprefix("https://github.com/")
    url = f"https://raw.githubusercontent.com/{repository}/{CRUD_COMMIT}/{CRUD_SPLIT_PATH}"
    content = (downloader or _download_bytes)(url)
    if _sha256(content) != CRUD_SPLIT_SHA256:
        raise CrudRagError("CRUD-RAG 文件 SHA-256 不匹配")
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_root / f".split_merged.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_crud_sample(
    split_path: Path, *, per_tier: int, distractors: int, seed: int,
) -> tuple[CorpusDataset, LaneDataset]:
    """确定性地分层抽样，并生成可直接交给分通道评估的语料与标签。

    Args:
        split_path: 已校验的 `split_merged.json`。
        per_tier: 单篇、两篇、三篇每档抽取的题数。
        distractors: 额外放入索引、与题目无关的新闻篇数。
        seed: 固定随机种子，相同参数总得到相同样本。

    Returns:
        语料（来源新闻加干扰新闻）与全部抽中题目组成的标签集。

    Raises:
        CrudRagError: 参数无效或可用事件、干扰新闻不足。
    """
    if per_tier < 1 or distractors < 0:
        raise CrudRagError("每档题数必须为正，干扰篇数不能为负")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    tiers: dict[str, dict[str, dict[str, str]]] = {}
    for key, _, count, _ in CRUD_TIERS:
        usable = {}
        for item in data[key]:
            fields = [item.get("questions"), item.get("answers")]
            fields += [item.get(f"news{index}") for index in range(1, count + 1)]
            if all(isinstance(value, str) and value.strip() for value in fields):
                usable[item["ID"]] = item
        tiers[key] = usable

    # 只用三档都可用的事件，打乱后按档切分，保证三档事件互不重叠。
    shared = sorted(set.intersection(*(set(items) for items in tiers.values())))
    if len(shared) < per_tier * len(CRUD_TIERS):
        raise CrudRagError("可用事件不足以按档抽样")
    rng = random.Random(seed)
    rng.shuffle(shared)

    entries: dict[str, CorpusEntry] = {}
    documents: dict[str, str] = {}
    episodes: list[LaneEpisode] = []
    sampled_ids: set[str] = set()
    for offset, (key, label, count, lane) in enumerate(CRUD_TIERS):
        for event_id in shared[offset * per_tier:(offset + 1) * per_tier]:
            sampled_ids.add(event_id)
            item = tiers[key][event_id]
            sources = []
            for index in range(1, count + 1):
                path = f"{CRUD_SPLIT_PATH}#{key}/{event_id}/news{index}"
                sources.append(_add_document(
                    entries, documents, item[f"news{index}"],
                    title=f"CRUD-RAG {label} {event_id} news{index}", source_path=path,
                ))
            episodes.append(LaneEpisode(
                id=f"crud-{label}-{event_id.lower()}",
                tags=("crud-rag", f"crud-{label}"),
                turns=(LaneTurn(
                    question=item["questions"].strip(),
                    reference=item["answers"].strip(),
                    expected_lane=lane,
                    expected_sources=tuple(dict.fromkeys(sources)),
                ),),
            ))

    source_texts = set(documents.values())
    pool: dict[str, str] = {}
    for key, _, count, _ in CRUD_TIERS:
        for event_id, item in sorted(tiers[key].items()):
            if event_id in sampled_ids:
                continue
            for index in range(1, count + 1):
                text = item[f"news{index}"]
                if text not in source_texts:
                    pool.setdefault(text, f"{CRUD_SPLIT_PATH}#{key}/{event_id}/news{index}")
    if len(pool) < distractors:
        raise CrudRagError("可用干扰新闻不足")
    for text in rng.sample(sorted(pool), distractors):
        _add_document(
            entries, documents, text,
            title="CRUD-RAG distractor", source_path=pool[text],
        )

    corpus = CorpusDataset(
        entries=sorted(entries.values(), key=lambda entry: entry.filename),
        documents=documents,
    )
    dataset = LaneDataset(
        version=f"crud-rag-qa-{CRUD_COMMIT[:7]}-n{per_tier}-d{distractors}-s{seed}",
        smoke_ids=tuple(episode.id for episode in episodes),
        episodes=tuple(episodes),
    )
    return corpus, dataset


def _add_document(
    entries: dict[str, CorpusEntry], documents: dict[str, str], text: str,
    *, title: str, source_path: str,
) -> str:
    """按正文去重登记一篇新闻，返回稳定文件名。"""
    digest = _sha256(text.encode("utf-8"))
    filename = f"crud_{digest[:16]}.txt"
    if filename not in entries:
        entries[filename] = CorpusEntry(
            filename=filename, title=title, source_url=CRUD_REPOSITORY,
            source_revision=CRUD_COMMIT, source_path=source_path,
            source_sha256=CRUD_SPLIT_SHA256, license=CRUD_LICENSE, sha256=digest,
        )
        documents[filename] = text
    return filename


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()

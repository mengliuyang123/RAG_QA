# -*- coding: utf-8 -*-
"""已记录索引版本不得在召回、裁判、图或HTTP／正式历史中丢失。"""

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.application.query.citations import extract_citations
from app.application.query.fast_lane import FastLane
from app.application.query.generator import AnswerGenerator
from app.application.query.prompting import PromptBuilder
from app.application.retrieval.service import RetrievalService
from app.application.retrieval.service import VectorRetriever
from app.domain.agent_quality import EvidenceItem
from app.domain.agent_quality import EvidenceAssessment
from app.domain.documents import DocumentChunk
from app.domain.execution import EvidenceBundle
from app.domain.execution import ExecutionBudget
from app.domain.execution import RetrievalAttempt
from app.domain.retrieval import RetrievalResult
from app.application.errors import VectorStoreError
from app.infrastructure.conversations.langgraph_agent import LangGraphConversationAgent
from app.infrastructure.conversations.quality import ModelQualityEvaluator
from app.infrastructure.indexes.bm25 import BM25Index
from app.infrastructure.reranking.dashscope import RemoteReranker
from app.infrastructure.vectorstores.chroma import ChromaVectorStore
from tests.test_agent_quality import _JSONModel
from tests.test_fast_lane import _Provider
from tests.test_fast_lane import _Quality
from tests.test_fast_lane import _Retriever
from tests.test_retriever import FakeQueryEmbedder
from tests.test_scoped_agent import _Model
from tests.test_scoped_agent import _Quality as ApprovedQuality
from tests.test_unified_query_api import _app
from tests.test_unified_query_api import _events
from tests.test_unified_query_api import _Model as EntryModel


VERSION = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_agent_repairs_versioned_seed_without_losing_scope_or_budget(
    repair_succeeds: bool,
) -> None:
    """版本时间不能让真实图修复分支崩溃，也不能泄漏未批准来源。

    Args:
        repair_succeeds: 唯一修复返回有效或仍无效的引用。

    Returns:
        None。
    """
    budget = ExecutionBudget(
        deadline=100.0, evidence_assessments=1,
    ).record_retrieval(
        RetrievalAttempt(query="needle", kb_ids=("ops",)), now=1.0,
    )
    seed = EvidenceBundle(
        budget=budget,
        evidence=(
            EvidenceItem(
                number=2, chunk_id="noise", doc_id="noise",
                filename="noise.md", page=None, text="UNAPPROVED_SECRET",
                kb_id="ops", doc_updated_at=VERSION,
            ),
            EvidenceItem(
                number=5, chunk_id="ops:0", doc_id="ops",
                filename="guide.md", page=None, text="使用SQLite保存状态",
                kb_id="ops", doc_updated_at=VERSION,
            ),
        ),
        approved_numbers=(5,),
        assessment=EvidenceAssessment(
            sufficient=True, relevant_numbers=[5],
            missing_information=[], next_query=None,
        ),
    )

    class RepairModel(_Model):
        """只替代外部生成，检查真正发给修复模型的协议内容。"""

        def invoke(self, messages: list[dict[str, object]]) -> AIMessage:
            """确保版本被序列化，未获批证据不参与修复。

            Args:
                messages: 图实际构造的生成或修复消息。

            Returns:
                预设的生成或修复响应。
            """
            if "现有证据：" in str(messages[-1]["content"]):
                text = str(messages[-1]["content"])
                assert "UNAPPROVED_SECRET" not in text
                evidence = json.loads(text.split("现有证据：\n", 1)[1])
                assert [item["number"] for item in evidence] == [5]
                assert datetime.fromisoformat(evidence[0]["doc_updated_at"]) == VERSION
            return super().invoke(messages)

    retriever = _Retriever()
    repaired = "使用SQLite保存状态[5]" if repair_succeeds else "仍引用错误[99]"
    model = RepairModel([
        AIMessage(content="首份草稿引用错误[99]"),
        AIMessage(content=repaired),
    ])
    result = LangGraphConversationAgent(
        model, retriever, InMemorySaver(), top_k=5,
        quality_evaluator=ApprovedQuality(), clock=lambda: 2.0,
    ).run(uuid4(), "如何保存状态？", [], seed=seed, kb_ids=("ops",))
    assert result.retrieval_calls == 1
    assert retriever.calls == []
    assert result.execution.budget.repair_attempted is True
    assert result.execution.budget.deadline == 100.0
    if repair_succeeds:
        assert result.answer == repaired
        assert result.citations[0].position == 5
        assert result.citations[0].doc_updated_at == VERSION
    else:
        assert result.citations == []
        assert "[99]" not in result.answer


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_versioned_repair_reaches_both_http_entries_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    conversation: bool, stream: bool,
) -> None:
    """真实组合根、图和存储中，错误草稿不能因修复异常变502或泄漏。

    Args:
        tmp_path: 隔离全部生产存储适配器。
        monkeypatch: 只在外部生成响应边界注入草稿和修复结果。
        conversation: 是否使用正式多轮入口。
        stream: 是否使用SSE外壳。

    Returns:
        None。
    """
    outputs = iter([
        AIMessage(content="DRAFT_NOT_PUBLIC[99]"),
        AIMessage(content="SQLite保存状态[1]"),
    ])

    def invoke(
        self: EntryModel, messages: list[dict[str, object]],
    ) -> AIMessage:
        return next(outputs)

    monkeypatch.setattr(EntryModel, "invoke", invoke)
    app, chunk, _ = _app(tmp_path)
    root = app.state.container
    try:
        root.vector_store.delete_document(chunk.doc_id)
        root.vector_store.add_chunks([
            chunk.model_copy(update={"doc_updated_at": VERSION}),
        ], [[1.0, 0.0]])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            path = "/api/query"
            detail_path = None
            if conversation:
                created = (await client.post("/api/conversations")).json()
                detail_path = "/api/conversations/" + created["id"]
                path = detail_path + "/query"
            if stream:
                path += "/stream"
            response = await client.post(path, json={"question": "比较needle"})
            assert response.status_code == 200
            assert "DRAFT_NOT_PUBLIC" not in response.text
            if stream:
                events = _events(response.text)
                terminal = [item for item in events if item[0] in {"complete", "error"}]
                assert [name for name, _ in terminal] == ["complete"]
                result = terminal[0][1]
                tokens = [data["token"] for name, data in events if name == "token"]
                assert "".join(tokens) == "SQLite保存状态[1]"
            else:
                result = response.json()
            assert result["answer"] == "SQLite保存状态[1]"
            assert result["lane"] == "agent"
            assert result["retrieval_calls"] == 1
            assert result["citations"][0]["position"] == 1
            assert datetime.fromisoformat(result["citations"][0]["doc_updated_at"]) == VERSION
            if detail_path is not None:
                turns = (await client.get(detail_path)).json()["turns"]
                assert len(turns) == 1
                assert turns[0]["assistant_message"] == result["answer"]
                assert turns[0]["citations"] == result["citations"]
    finally:
        root.close()


@pytest.mark.parametrize("strategy", ["vector", "hybrid", "hybrid_rerank"])
def test_all_retrieval_strategies_and_legacy_citation_keep_index_version(tmp_path: Path, strategy: str) -> None:
    """任何召回分支丢字段、RRF重建元数据或重排覆盖来源都应失败。

    Args:
        tmp_path: 真实临时Chroma。
        strategy: 三个生产检索策略。

    Returns:
        None。
    """
    identifier = str(uuid4())
    chunk = DocumentChunk(id=f"opaque/{identifier}/a", doc_id=identifier, filename="guide.md", text="needle 使用SQLite", page=None, chunk_index=0, doc_updated_at=VERSION)
    store = ChromaVectorStore(tmp_path / "chroma", "versions")
    store.add_chunks([chunk], [[1.0, 0.0]])
    index = BM25Index()
    index.rebuild([chunk])

    def _rerank(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    with httpx.Client(transport=httpx.MockTransport(_rerank)) as client:
        service = RetrievalService(VectorRetriever(store, FakeQueryEmbedder()), index, RemoteReranker(base_url="http://offline.invalid/v1", api_key="offline", model="offline", client=client))
        results = service.retrieve("needle", strategy=strategy, top_k=1)
        assert len(results) == 1
        assert results[0].model_dump().get("doc_updated_at") == VERSION
        assert extract_citations("资料[1]", results)[0].doc_updated_at == VERSION


@pytest.mark.anyio
@pytest.mark.parametrize("lane", ["fast", "agent"])
@pytest.mark.parametrize("stream", [False, True])
async def test_two_entry_points_stream_and_formal_history_keep_citation_version(
    tmp_path: Path, lane: str, stream: bool,
) -> None:
    """真实组合根及HTTP外壳不能把已知版本变null，正式历史保留回答时版本。

    Args:
        tmp_path: 所有真实数据库与索引的隔离目录。
        lane: 管理查询强制开始的通道。
        stream: SSE和普通JSON均检查complete结果。

    Returns:
        None。
    """
    app, chunk, _ = _app(tmp_path)
    root = app.state.container
    root.vector_store.delete_document(chunk.doc_id)
    root.vector_store.add_chunks([chunk.model_copy(update={"doc_updated_at": VERSION})], [[1.0, 0.0]])
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            suffix = "/stream" if stream else ""
            response = await client.post("/api/query" + suffix, json={"question": "needle", "strategy": "hybrid", "lane": lane})
            assert response.status_code == 200
            if stream:
                payload = next(data for name, data in _events(response.text) if name == "complete")
            else:
                payload = response.json()
            assert payload["lane"] == lane
            assert payload["citations"][0]["doc_updated_at"] is not None
            assert datetime.fromisoformat(payload["citations"][0]["doc_updated_at"]) == VERSION
            conversation = (await client.post("/api/conversations")).json()["id"]
            turn = await client.post(f"/api/conversations/{conversation}/query" + suffix, json={"question": "needle"})
            assert turn.status_code == 200
            if stream:
                turn_payload = next(data for name, data in _events(turn.text) if name == "complete")
            else:
                turn_payload = turn.json()
            assert turn_payload["citations"][0]["doc_updated_at"] is not None
            assert datetime.fromisoformat(turn_payload["citations"][0]["doc_updated_at"]) == VERSION
            # 修改当前索引不能重写已经提交的正式引用版本。
            root.vector_store.delete_document(chunk.doc_id)
            root.vector_store.add_chunks([chunk.model_copy(update={"doc_updated_at": datetime(2026, 1, 2, tzinfo=UTC)})], [[1.0, 0.0]])
            detail = (await client.get(f"/api/conversations/{conversation}")).json()
            assert datetime.fromisoformat(detail["turns"][0]["citations"][0]["doc_updated_at"]) == VERSION
    finally:
        root.close()


def test_fast_seed_and_agent_added_evidence_keep_versions_without_duplicate_retrieval() -> None:
    """版本不能使种子JSON失败、被补查证据覆盖，或逼迫重复首次检索。

    Returns:
        None。
    """
    class VersionedRetriever(_Retriever):
        """仅给离线召回边界返回完整真实形状的版本元数据。"""

        def retrieve(self, question: str, strategy: str, top_k: int) -> list[RetrievalResult]:
            """保留父类脚本与调用记录，加入持久版本而非模型生成日期。

            Args:
                question: 实际查询。
                strategy: 实际策略。
                top_k: 上限。

            Returns:
                带已知来源版本的候选。
            """
            values = super().retrieve(question, strategy, top_k)
            return [RetrievalResult.model_validate({**item.model_dump(), "doc_updated_at": VERSION}) for item in values]

    retriever = VersionedRetriever()
    quality = _Quality(sufficient=False)
    output = FastLane(retriever, quality, AnswerGenerator(PromptBuilder(), _Provider()), clock=lambda: 1.0).run("比较 A/B", [], query="A 存储", bundle=EvidenceBundle(budget=ExecutionBudget(deadline=100.0)))
    result = LangGraphConversationAgent(_Model([AIMessage(content="A 使用 SQLite[2]，B 使用 PostgreSQL[3]")]), retriever, InMemorySaver(), top_k=5, quality_evaluator=quality, clock=lambda: 2.0).run(uuid4(), "比较 A/B", [], seed=output.bundle)
    assert [citation.doc_updated_at for citation in result.citations] == [VERSION, VERSION]
    assert [call[0] for call in retriever.calls] == ["A 存储", "B 存储方式"]


def test_quality_model_payload_serializes_version_as_data() -> None:
    """加入datetime后裁判JSON不能抛错，也不能把来源版本误当模型输出字段。

    Returns:
        None。
    """
    evidence = EvidenceItem(number=1, chunk_id="a", doc_id="a", filename="a.md", page=None, text="资料", doc_updated_at=VERSION)
    model = _JSONModel(json.dumps({"sufficient": True, "relevant_numbers": [1], "missing_information": [], "next_query": None}))
    result = ModelQualityEvaluator(model).assess_evidence("问题", [], [evidence])
    assert result.sufficient is True
    payload = json.loads(model.messages[-1]["content"])
    assert datetime.fromisoformat(payload["evidence"][0]["doc_updated_at"]) == VERSION


@pytest.mark.anyio
async def test_unknown_legacy_version_is_not_filled_from_source_mtime_or_request(tmp_path: Path) -> None:
    """缺版本时即使源文件有mtime，也不能给旧索引或正式引用补造当前日期。

    Args:
        tmp_path: 真实索引、文件与全部临时数据库。

    Returns:
        None。
    """
    app, chunk, _ = _app(tmp_path)
    root = app.state.container
    source = root.settings.docs_path / f"{chunk.doc_id}_{chunk.filename}"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("later source metadata", encoding="utf-8")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            single = await client.post("/api/query", json={"question": "needle", "strategy": "hybrid"})
            assert single.status_code == 200
            assert single.json()["citations"][0]["doc_updated_at"] is None
            conversation = (await client.post("/api/conversations")).json()["id"]
            turn = await client.post(f"/api/conversations/{conversation}/query", json={"question": "needle"})
            assert turn.status_code == 200
            assert turn.json()["citations"][0]["doc_updated_at"] is None
            detail = (await client.get(f"/api/conversations/{conversation}")).json()
            assert detail["turns"][0]["citations"][0]["doc_updated_at"] is None
            context = await client.get(f"/api/documents/{chunk.doc_id}/chunks/{chunk.id}")
            assert context.status_code == 200
            assert context.json()["doc_updated_at"] is None
    finally:
        root.close()


@pytest.mark.parametrize("stored", ["2026-10-25T12:00:00", "bad-version"])
def test_corrupt_index_version_is_not_downgraded_to_unknown(tmp_path: Path, stored: str) -> None:
    """损坏时间不是旧字段缺失，真实读取必须失败而非以null悄悄继续回答。

    Args:
        tmp_path: 临时真实索引及隔离业务库。
        stored: 无时区或非法ISO时间。

    Returns:
        None。
    """
    app, chunk, _ = _app(tmp_path)
    root = app.state.container
    try:
        root.vector_store._collection.update(ids=[chunk.id], metadatas=[{"doc_updated_at": stored}])
        retriever = VectorRetriever(root.vector_store, FakeQueryEmbedder())
        with pytest.raises(VectorStoreError):
            retriever.retrieve("needle", top_k=1)
    finally:
        root.close()

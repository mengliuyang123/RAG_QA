# -*- coding: utf-8 -*-
"""审查问题的并发、生产装配与数据边界回归测试。"""

import asyncio
import subprocess
from pathlib import Path
from threading import Event
from threading import Timer

import httpx
import pytest
from pydantic import ValidationError

from app.bootstrap.factories import AdapterFactories
from app.application.errors import EmbeddingError
from app.application.errors import VectorStoreError
from app.domain.documents import DocumentChunk
from app.domain.documents import UploadResult
from app.interfaces.http.dependencies import get_upload_service
from app.interfaces.http.dependencies import get_query_service
from app.interfaces.http.schemas import ConversationQueryRequest
from app.interfaces.http.schemas import QueryRequest
from tests.http_auth import create_test_app as create_app


@pytest.fixture
def anyio_backend() -> str:
    """固定异步后端。

    Returns:
        asyncio 后端名称。
    """
    return "asyncio"


@pytest.mark.anyio
async def test_health_remains_responsive_during_upload() -> None:
    """同步入库尚未完成时，其他 HTTP 请求必须能继续执行。"""
    entered = Event()
    released = Event()

    class BlockingUpload:
        """等待测试信号后返回的上传服务。"""

        max_upload_size = 100

        def save_and_parse(
            self, filename: str, content: bytes,
        ) -> UploadResult:
            """模拟等待外部 Embedding 的同步入库。

            Args:
                filename: 上传文件名。
                content: 上传内容。

            Returns:
                完整上传结果。
            """
            entered.set()
            assert released.wait(3)
            return UploadResult(
                doc_id="doc", filename=filename, chunks_count=1,
            )

    app = create_app()
    app.dependency_overrides[get_upload_service] = BlockingUpload
    timer = Timer(1.0, released.set)
    timer.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            upload = asyncio.create_task(client.post(
                "/api/upload", files={"file": ("a.txt", b"hello")},
            ))
            assert await asyncio.to_thread(entered.wait, 2)
            health = await client.get("/api/health")
            was_released = released.is_set()
            released.set()
            response = await upload
        assert health.status_code == 200
        assert not was_released
        assert response.status_code == 201
    finally:
        released.set()
        timer.cancel()
        timer.join()
        app.state.container.close()


def test_production_bm25_queries_do_not_reload_chroma() -> None:
    """生产 factory 的查询必须复用预建索引而非加载整个向量库。"""
    class Store:
        """禁止查询期间重复加载语料的存储。"""

        calls = 0

        def list_chunks(self) -> list[DocumentChunk]:
            """只允许初始化时读取一次。

            Returns:
                有完整来源信息的语料快照。
            """
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("查询不应重新扫描 Chroma")
            return [DocumentChunk(
                id="doc:0", doc_id="doc", text="SQLite storage",
                filename="a.md", page=None, chunk_index=0,
            )]

    store = Store()
    index = AdapterFactories().keyword_index(store)
    assert index.search("SQLite", 1)[0].chunk_id == "doc:0"
    assert index.search("storage", 1)[0].chunk_id == "doc:0"


def test_git_ignores_uploaded_and_conversation_data() -> None:
    """批量暂存不得意外加入上传正文和 SQLite/WAL 数据。"""
    root = Path(__file__).resolve().parents[1]
    paths = [
        "data/docs/private.pdf", "data/conversations/history.sqlite3",
        "data/conversations/history.sqlite3-wal",
        "data/conversations/checkpoints.sqlite3-shm",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=root, input="\n".join(paths) + "\n", text=True,
        capture_output=True, check=False,
    )
    assert result.stdout.splitlines() == paths


@pytest.mark.anyio
@pytest.mark.parametrize("error_type", [EmbeddingError, VectorStoreError])
async def test_query_backend_failures_have_stable_502(
    error_type: type[Exception],
) -> None:
    """后端不可用应作为依赖故障返回，不暴露异常细节。

    Args:
        error_type: 需要验证 HTTP 映射的后端异常类型。
    """
    class FailingService:
        """始终报告指定依赖错误的查询服务。"""

        def query(self, *args: object, **kwargs: object) -> None:
            """模拟依赖故障。

            Args:
                args: 位置参数。
                kwargs: 查询参数。

            Returns:
                不返回，始终抛出测试异常。
            """
            raise error_type("private endpoint detail")

    app = create_app()
    app.dependency_overrides[get_query_service] = FailingService
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/query", json={"question": "问题"},
            )
        assert response.status_code == 502
        assert "private" not in response.text
    finally:
        app.state.container.close()


def test_question_lengths_are_bounded_at_http_boundary() -> None:
    """单轮与会话请求均应拒绝超长输入。"""
    for request_type in (QueryRequest, ConversationQueryRequest):
        with pytest.raises(ValidationError):
            request_type(question="q" * 8_001)

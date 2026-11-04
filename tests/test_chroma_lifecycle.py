# -*- coding: utf-8 -*-
"""真实Chroma客户端必须随拥有它的组合根释放，不能累积原生运行时。"""

from pathlib import Path

from chromadb.api.shared_system_client import SharedSystemClient

from app.infrastructure.vectorstores.chroma import ChromaVectorStore
from tests.test_unified_query_api import _app
from tests.test_vectorstore import make_chunks


def test_container_close_releases_owned_chroma_runtime(tmp_path: Path) -> None:
    """遗漏适配器close时，组合根退出后共享缓存仍持有原生数据库。

    Args:
        tmp_path: 隔离全部存储。

    Returns:
        None。
    """
    app, _, _ = _app(tmp_path)
    store = app.state.container.vector_store
    identifier = str(tmp_path / "chroma")
    assert identifier in SharedSystemClient._identifier_to_system
    try:
        app.state.container.close()
        assert identifier not in SharedSystemClient._identifier_to_system
    finally:
        store._client.close()


def test_store_close_keeps_other_client_alive_and_reopens_data(
    tmp_path: Path,
) -> None:
    """关闭一个拥有者不能停掉同路径其他客户端或删除持久数据。

    Args:
        tmp_path: 隔离真实向量库。

    Returns:
        None。
    """
    first = ChromaVectorStore(tmp_path, "lifecycle")
    second = ChromaVectorStore(tmp_path, "lifecycle")
    try:
        first.add_chunks(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
        first.close()
        first.close()
        assert second.list_chunks() == make_chunks()
        assert str(tmp_path) in SharedSystemClient._identifier_to_system
        second.close()
        assert str(tmp_path) not in SharedSystemClient._identifier_to_system
        reopened = ChromaVectorStore(tmp_path, "lifecycle")
        try:
            assert reopened.list_chunks() == make_chunks()
        finally:
            reopened._client.close()
    finally:
        first._client.close()
        second._client.close()

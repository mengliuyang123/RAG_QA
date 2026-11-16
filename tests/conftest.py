# -*- coding: utf-8 -*-
"""离线测试强制禁止DNS和套接字连接，防止漏注入模型替身时真实计费。"""

import socket

import pytest

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def isolated_postgres_storage(monkeypatch, tmp_path):
    """延迟创建随机测试 schema；任何仓储连接均限制到独立测试库。"""
    import threading
    from uuid import uuid4
    import psycopg
    from psycopg import sql
    from app.infrastructure.persistence.database import Database
    from app.infrastructure.persistence.migrations import migrate_database
    from tests import postgres_support as support
    dsn = support.test_database_url()
    schema = 'test_' + uuid4().hex[:20]
    original_init = Database.__init__
    databases = []
    schemas = set()
    lock = threading.RLock()

    def initialize(instance, requested_dsn, requested_schema='rag_qa'):
        selected = schema if requested_schema == 'rag_qa' else requested_schema
        if not selected.startswith('test_'):
            selected = schema + '_' + __import__('hashlib').sha256(selected.encode()).hexdigest()[:8]
        with lock:
            if selected not in schemas:
                migrate_database(dsn, selected)
                schemas.add(selected)
            original_init(instance, dsn, selected)
            databases.append(instance)

    monkeypatch.setattr(Database, '__init__', initialize)
    support._active_databases = {}
    support._active_root = tmp_path.resolve()
    support._active_schema = schema
    try:
        yield
    finally:
        for database in databases:
            database.close()
        support._active_databases = None
        with psycopg.connect(dsn, autocommit=True) as connection:
            for selected in schemas:
                connection.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(selected)))


@pytest.fixture(autouse=True)
def forbid_network_in_offline_tests(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """仅integration测试可以使用真实网络。

    Args:
        request: 当前测试节点。
        monkeypatch: 本测试结束后自动恢复原函数。

    Returns:
        None。
    """
    if request.node.get_closest_marker("integration") is not None:
        return

    def deny(*args: object, **kwargs: object) -> object:
        raise AssertionError("离线测试不允许真实网络连接")

    def connect(sock: socket.socket, address: object, *args: object, **kwargs: object) -> object:
        # Windows 的 asyncio 事件循环用回环套接字自唤醒，不属于真实外部网络。
        if isinstance(address, tuple) and address and address[0] in LOOPBACK_HOSTS:
            return original_connect(sock, address, *args, **kwargs)
        raise AssertionError("离线测试不允许真实网络连接")

    original_connect = socket.socket.connect
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)

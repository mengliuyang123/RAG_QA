# -*- coding: utf-8 -*-
"""离线测试强制禁止DNS和套接字连接，防止漏注入模型替身时真实计费。"""

import socket

import pytest


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

    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)

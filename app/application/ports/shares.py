# -*- coding: utf-8 -*-
"""分享授权与正式轮次同事务存储的端口。"""

from datetime import datetime
from typing import Protocol

from app.domain.shares import SharedTurn
from app.domain.shares import StoredShare


class ShareRepository(Protocol):
    """原子检查归属，并只读一轮被授权的正式内容。"""

    def create(self, share: StoredShare) -> None:
        """检查活动会话的本人归属后持久化摘要。

        Args:
            share: 已认证身份和新链接授权。

        Returns:
            None。
        """
        ...

    def read(self, token_hash: str, now: datetime) -> SharedTurn:
        """在同一快照检查有效性并读取单轮引用。

        Args:
            token_hash: 链接摘要。
            now: 当前UTC时间。

        Returns:
            仍有效的单轮内容。
        """
        ...

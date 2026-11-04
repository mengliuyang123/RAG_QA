# -*- coding: utf-8 -*-
"""签发不可猜测的团队链接，令牌明文不进入持久存储。"""

import hashlib
import re
import secrets
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from uuid import UUID

from app.application.errors import ShareNotFoundError
from app.application.ports.shares import ShareRepository
from app.domain.auth import User
from app.domain.shares import ShareLink
from app.domain.shares import ShareRequest
from app.domain.shares import SharedTurn
from app.domain.shares import StoredShare


class ShareService:
    """所有签发受正式会话归属约束，没有管理员越权分支。"""

    def __init__(self, repository: ShareRepository) -> None:
        """注入当前运行业务存储。

        Args:
            repository: 分享授权端口。

        Returns:
            None。
        """
        self._repository = repository

    def create(
        self, turn_id: UUID, user: User, request: ShareRequest,
    ) -> ShareLink:
        """用随机256位令牌签发有限期链接。

        Args:
            turn_id: 要主动分享的正式轮次。
            user: 服务端认证身份。
            request: 有限有效期。

        Returns:
            事务提交后的链接。
        """
        now = datetime.now(UTC)
        token = "shr_" + secrets.token_urlsafe(32)
        expires_at = now + timedelta(days=request.expires_in_days)
        self._repository.create(StoredShare(
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            turn_id=turn_id, owner_id=user.id, shared_by=user.name,
            created_at=now, expires_at=expires_at,
        ))
        return ShareLink(
            token=token, url="/s/" + token, expires_at=expires_at,
        )

    def read(self, token: str) -> SharedTurn:
        """只将令牌摘要传入存储；登录由HTTP边界强制执行。

        Args:
            token: 请求链接中的令牌。

        Returns:
            当前仍有效的单轮内容。
        """
        if re.fullmatch(r"shr_[A-Za-z0-9_-]{43}", token) is None:
            raise ShareNotFoundError("分享不存在")
        return self._repository.read(
            hashlib.sha256(token.encode()).hexdigest(), datetime.now(UTC),
        )

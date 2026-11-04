# -*- coding: utf-8 -*-
"""只描述SQLite键集位置；游标不能替代归属校验。"""

import base64
import hashlib
from datetime import datetime
from datetime import timedelta

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from app.domain.errors import InvalidConversationCursorError


class PagePosition(BaseModel):
    """保存同一排序键的位置及范围摘要。

    Attributes:
        scope: 用户、资源和页类型的摘要。
        timestamp: 原始UTC数据库时间字符串，不经格式转换改变比较值。
        rowid: 同一时间戳下的稳定位置。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    scope: str = Field(pattern=r"^[0-9a-f]{64}$")
    timestamp: str = Field(min_length=1, max_length=64)
    rowid: int = Field(ge=1, le=2 ** 63 - 1)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        """拒绝无时区和非UTC游标，保留数据库字面值。

        Args:
            value: 游标中的时间。

        Returns:
            验证后的原值。
        """
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() != timedelta(0):
            raise ValueError("分页时间必须为UTC")
        return value


def decode_cursor(cursor: str | None, scope: str) -> PagePosition | None:
    """验证不透明位置并绑定用户和资源。

    Args:
        cursor: 客户端原样回传的游标。
        scope: 服务端确定的页类型及归属组合。

    Returns:
        已验证位置；首页为None。
    """
    if cursor is None:
        return None
    try:
        if not cursor or len(cursor) > 2048:
            raise ValueError("游标长度非法")
        raw = base64.b64decode(cursor, altchars=b"-_", validate=True)
        position = PagePosition.model_validate_json(raw)
        digest = hashlib.sha256(scope.encode()).hexdigest()
        if position.scope != digest:
            raise ValueError("游标范围不符")
        return position
    except (ValueError, UnicodeError) as error:
        raise InvalidConversationCursorError("会话分页参数无效") from error


def encode_cursor(timestamp: str, rowid: int, scope: str) -> str:
    """编码一页末尾的真实数据库位置。

    Args:
        timestamp: 实际记录的UTC时间。
        rowid: 实际记录行号。
        scope: 后端确定的用户和资源范围。

    Returns:
        不含正文的不透明游标。
    """
    position = PagePosition(
        scope=hashlib.sha256(scope.encode()).hexdigest(),
        timestamp=timestamp, rowid=rowid,
    )
    return base64.urlsafe_b64encode(position.model_dump_json().encode()).decode()

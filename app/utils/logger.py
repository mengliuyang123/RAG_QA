# -*- coding: utf-8 -*-
"""应用日志配置与分享链接脱敏。"""

import logging
import re


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_SHARE_PATH = re.compile(r"(/(?:api/shares|s)/)[^/\s?\"'#]+")


def _redact_share_path(value: object) -> object:
    original = str(value)
    redacted = _SHARE_PATH.sub(r"\1[redacted]", original)
    if redacted != original:
        return redacted
    return value


class _SharePathFilter(logging.Filter):
    """在访问日志进入任何处理器前遮盖链接令牌。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """保留Uvicorn的五元参数，避免破坏其专用格式器。

        Args:
            record: Uvicorn或HTTPX的请求日志。

        Returns:
            始终保留脱敏后的日志。
        """
        record.msg = _redact_share_path(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_redact_share_path(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {
                key: _redact_share_path(value)
                for key, value in record.args.items()
            }
        return True


def configure_logging(level: str = "INFO") -> None:
    """使用统一格式并隐藏分享路径中的敏感令牌。

    Args:
        level: 日志等级。

    Returns:
        None。
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
    # basicConfig 在根记录器已有处理器时不会更新级别，因此需要显式设置。
    logging.getLogger().setLevel(numeric_level)
    for name in ("uvicorn.access", "httpx"):
        logger = logging.getLogger(name)
        if not any(isinstance(item, _SharePathFilter) for item in logger.filters):
            logger.addFilter(_SharePathFilter())

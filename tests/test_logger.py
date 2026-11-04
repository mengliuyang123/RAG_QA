import logging

from app.utils.logger import configure_logging


def test_configure_logging_updates_root_level_when_handlers_exist() -> None:
    """已有日志处理器时，应用配置的日志级别仍应生效。"""
    root_logger = logging.getLogger()
    original_level = root_logger.level
    handler = logging.NullHandler()
    root_logger.addHandler(handler)

    try:
        configure_logging("DEBUG")

        assert root_logger.level == logging.DEBUG
    finally:
        root_logger.setLevel(original_level)
        root_logger.removeHandler(handler)

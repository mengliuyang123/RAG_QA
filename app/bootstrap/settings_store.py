# -*- coding: utf-8 -*-
"""组合根提供的运行配置存储；HTTP 层经由这里访问 .env，不直接依赖基础设施。"""

from pathlib import Path

from app.infrastructure.settings.env_file import EnvFile


def open_env_file(path: Path) -> EnvFile:
    """返回指定路径的 .env 编辑器。

    Args:
        path: .env 文件路径。

    Returns:
        只改动指定键、原子写入的 .env 编辑器。
    """
    return EnvFile(path)

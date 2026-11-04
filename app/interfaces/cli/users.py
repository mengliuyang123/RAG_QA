# -*- coding: utf-8 -*-
"""隐藏输入口令创建初始账号，不提供默认管理员或匿名注册。"""

import argparse
import sys
import warnings
from getpass import GetPassWarning
from getpass import getpass

from app.application.errors import AuthStorageError
from app.application.errors import UserAlreadyExistsError
from app.bootstrap.container import build_application_container
from app.config import Settings
from app.config import get_settings


def main(argv: list[str] | None = None, *, settings: Settings | None = None) -> int:
    """从受信终端创建账号，口令不允许出现在进程参数中。

    Args:
        argv: 命令参数；None读取当前进程参数。
        settings: 可选显式配置，默认沿用项目环境配置。

    Returns:
        成功返回0，输入不一致或创建失败返回2。
    """
    parser = argparse.ArgumentParser(description="本地账号管理")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="隐藏输入口令创建账号")
    create.add_argument("--username", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--role", choices=("member", "admin"), default="member")
    arguments = parser.parse_args(argv)
    try:
        # 无法关闭回显时拒绝读取，避免getpass的回退行为暴露口令。
        with warnings.catch_warnings():
            warnings.simplefilter("error", GetPassWarning)
            password = getpass("口令（15–128字符）：")
            confirmation = getpass("再次输入口令：")
    except (EOFError, KeyboardInterrupt, GetPassWarning):
        print("无法安全读取口令，请在交互终端重试。", file=sys.stderr)
        return 2
    if password != confirmation:
        print("两次口令不一致，未创建账号。", file=sys.stderr)
        return 2
    container = build_application_container(settings or get_settings())
    try:
        user = container.auth_service.create_user(
            arguments.username, password, name=arguments.name, role=arguments.role,
        )
    except UserAlreadyExistsError:
        print("账号已存在，未修改已有账号。", file=sys.stderr)
        return 2
    except ValueError:
        print("账号、展示名或口令格式不符合要求。", file=sys.stderr)
        return 2
    except AuthStorageError:
        print("认证存储不可用，账号创建失败。", file=sys.stderr)
        return 2
    finally:
        container.close()
    print(user.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

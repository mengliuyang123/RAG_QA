"""显式初始化应用数据库，不输出连接凭据。"""
from app.config import get_settings
from app.bootstrap.database import initialize_databases


def main() -> None:
    settings = get_settings()
    initialize_databases(settings)
    print('PostgreSQL schema 初始化完成')


if __name__ == '__main__':
    main()

"""数据库初始化及评估入口的基础设施装配边界。"""
from app.config import Settings
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.migrations import migrate_database


def initialize_databases(settings: Settings) -> None:
    migrate_database(settings.database_url, settings.database_schema)
    migrate_database(settings.database_url, settings.evaluation_database_schema)

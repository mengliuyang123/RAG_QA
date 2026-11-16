"""真实 PostgreSQL 连接生命周期与迁移契约。"""
import pytest
from tests.postgres_support import isolated_database, test_database_url as database_url

pytestmark = pytest.mark.database


def test_migration_is_idempotent_and_failed_transaction_rolls_back():
    from app.infrastructure.persistence.migrations import migrate_database
    with isolated_database() as database:
        migrate_database(database_url(), database.schema)
        with database.connection() as connection:
            connection.execute('CREATE TABLE rollback_probe (id INTEGER PRIMARY KEY)')
        with pytest.raises(RuntimeError):
            with database.connection() as connection:
                connection.execute('INSERT INTO rollback_probe VALUES (1)')
                raise RuntimeError('abort')
        with database.connection(read_only=True) as connection:
            assert connection.execute('SELECT COUNT(*) AS total FROM rollback_probe').fetchone()['total'] == 0
        database.close()
        database.close()


def test_test_database_cannot_point_to_application_database(monkeypatch):
    from dotenv import dotenv_values
    monkeypatch.setenv('TEST_DATABASE_URL', dotenv_values('.env')['DATABASE_URL'])
    with pytest.raises(ValueError, match='分离'):
        database_url()


def test_timestamp_upgrade_rebuilds_feedback_expression_index():
    from app.infrastructure.persistence.migrations import migrate_database
    with isolated_database() as database:
        with database.connection() as connection:
            connection.execute('DROP INDEX idx_turns_feedback_time')
            for column in ('created_at','feedback_updated_at','feedback_triaged_at'):
                connection.execute(f'ALTER TABLE conversation_turns ALTER COLUMN {column} TYPE TEXT USING {column}::text')
            connection.execute('CREATE INDEX idx_turns_feedback_time ON conversation_turns(COALESCE(feedback_updated_at,created_at),id) WHERE feedback_json IS NOT NULL')
            connection.execute('DELETE FROM schema_migrations WHERE version=3')
        migrate_database(database._dsn,database.schema)
        with database.connection(read_only=True) as connection:
            columns=connection.execute("SELECT data_type FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='conversation_turns' AND column_name IN ('created_at','feedback_updated_at')").fetchall()
            assert len(columns)==2
            assert all(row[0]=='timestamp with time zone' for row in columns)

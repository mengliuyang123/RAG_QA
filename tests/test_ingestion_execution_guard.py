"""独立进程验证 PostgreSQL 执行咨询锁及进程退出释放。"""
import os
import subprocess
import sys
from pathlib import Path
from app.infrastructure.persistence.ingestion_execution import PostgresIngestionExecutionGuard
from tests.postgres_support import pg_database, test_database_url as database_url

CHILD = """
import os,sys
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.ingestion_execution import PostgresIngestionExecutionGuard
from app.application.errors import IngestionWorkerUnavailableError
try:
    database = Database(os.environ['TEST_DATABASE_URL'], sys.argv[1])
    with PostgresIngestionExecutionGuard(database).hold():
        if len(sys.argv)>2: os._exit(0)
        print('acquired',flush=True)
    database.close()
except IngestionWorkerUnavailableError:
    print('busy',flush=True)
    sys.exit(3)
"""


def _child(database, abrupt=False):
    env = {**os.environ, 'TEST_DATABASE_URL':database_url()}
    return subprocess.run([sys.executable,'-c',CHILD,database.schema]+(['abrupt'] if abrupt else []),
        env=env,capture_output=True,text=True,timeout=15,check=False)


def test_another_process_cannot_enter_held_execution_guard(tmp_path: Path):
    database=pg_database(tmp_path/'worker.db')
    with PostgresIngestionExecutionGuard(database).hold():
        blocked=_child(database)
        assert blocked.returncode==3, blocked.stderr
        assert blocked.stdout.strip()=='busy'
    acquired=_child(database)
    assert acquired.returncode==0, acquired.stderr
    assert acquired.stdout.strip()=='acquired'


def test_process_exit_releases_guard_even_without_context_cleanup(tmp_path: Path):
    database=pg_database(tmp_path/'worker.db')
    result=_child(database,abrupt=True)
    assert result.returncode==0,result.stderr
    with PostgresIngestionExecutionGuard(database).hold():
        pass

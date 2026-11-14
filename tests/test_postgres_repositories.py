"""PostgreSQL 仓储真实事务集成。"""
from uuid import uuid4
import pytest
from tests.postgres_support import isolated_database
from app.domain.knowledge_bases import KnowledgeBase

pytestmark = pytest.mark.database


def test_knowledge_catalog_persists_and_rejects_duplicates():
    from app.infrastructure.persistence.knowledge_bases import PostgresKnowledgeBaseRepository
    from app.application.errors import KnowledgeBaseConflictError
    with isolated_database() as database:
        repository = PostgresKnowledgeBaseRepository(database)
        repository.create(KnowledgeBase(id='ops', name='运维'))
        assert PostgresKnowledgeBaseRepository(database).get('ops').name == '运维'
        with pytest.raises(KnowledgeBaseConflictError):
            repository.create(KnowledgeBase(id='ops', name='重复'))


def test_auth_roundtrip_and_login():
    from app.infrastructure.auth.postgres_repository import PostgresAuthRepository
    from app.application.auth.service import AuthService
    from app.infrastructure.auth.passwords import Argon2PasswordHasher
    from app.config import Settings
    with isolated_database() as database:
        service = AuthService(PostgresAuthRepository(database), Argon2PasswordHasher())
        user = service.create_user('postgres_smoke', 'Postgres-Test-Password!', name='test', role='admin')
        result = service.login('postgres_smoke', 'Postgres-Test-Password!', client_id='test')
        assert result.user.id == user.id
        assert service.authenticate(result.session_token).user.role == 'admin'


def test_parallel_enqueue_enforces_capacity_and_single_claim():
    from concurrent.futures import ThreadPoolExecutor
    from app.infrastructure.persistence.ingestion_jobs import PostgresIngestionJobRepository
    from app.domain.ingestion_jobs import IngestionJobSubmission
    from app.application.errors import IngestionQueueFullError
    with isolated_database() as database:
        repository = PostgresIngestionJobRepository(database)
        def submit(_):
            item = IngestionJobSubmission(job_id='job_'+uuid4().hex, doc_id=uuid4(), kb_id='default',
                filename='test.md', size_bytes=5, uploaded_by=uuid4(), content_sha256='a'*64)
            try:
                repository.enqueue(item, capacity=1)
                return True
            except IngestionQueueFullError:
                return False
        with ThreadPoolExecutor(2) as workers:
            assert sum(workers.map(submit, range(2))) == 1
            claims = list(workers.map(lambda _: repository.claim_next(lease_seconds=60), range(2)))
        assert sum(claim is not None for claim in claims) == 1


def test_conversation_roundtrip_and_checkpoint_deletion():
    from app.infrastructure.conversations.postgres_repository import PostgresConversationRepository
    from app.infrastructure.conversations.checkpoint import PostgresCheckpointStore
    from app.domain.conversations import AgentResult, AnswerMode
    with isolated_database() as database:
        repository = PostgresConversationRepository(database)
        conversation = repository.create(owner_id=uuid4())
        turn = repository.save_turn(conversation.id, 'hello',
            AgentResult(answer='你好', answer_mode=AnswerMode.DIRECT, retrieval_calls=0, citations=[]), 1.0)
        assert PostgresConversationRepository(database).get(conversation.id).turns[0].id == turn.id
        store = PostgresCheckpointStore(database)
        store.delete_thread(conversation.id)
        store.close()
        store.close()


def test_default_container_uses_postgres_for_all_read_models():
    from app.bootstrap.container import build_application_container, RuntimeOptions
    from app.config import Settings
    with isolated_database() as database:
        container = build_application_container(Settings(_env_file=None, ingestion_worker_enabled=False),
            runtime_options=RuntimeOptions(database=database))
        try:
            assert container.knowledge_catalog_service.list().knowledge_bases[0].id == 'default'
            for name in ('auth_service', 'conversation_service', 'trace_query_service', 'usage_query_service',
                         'feedback_admin_service', 'share_service', 'refusal_insight_service',
                         'insight_overview_service', 'citation_statistics'):
                # Only construct services which do not call real models.
                if name != 'conversation_service':
                    assert getattr(container, name) is not None
        finally:
            container.close()


def test_delete_mark_waits_for_inflight_save(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    from threading import Event
    from app.infrastructure.conversations.postgres_repository import PostgresConversationRepository
    from app.domain.conversations import AgentResult, AnswerMode
    with isolated_database() as database:
        writer=PostgresConversationRepository(database)
        deleter=PostgresConversationRepository(database)
        conversation=writer.create()
        entered,release=Event(),Event()
        original=writer._conversation_row
        def pause(connection, identifier):
            result=original(connection,identifier)
            entered.set()
            assert release.wait(5)
            return result
        monkeypatch.setattr(writer,'_conversation_row',pause)
        with ThreadPoolExecutor(2) as workers:
            saving=workers.submit(writer.save_turn,conversation.id,'question',
                AgentResult(answer='answer',answer_mode=AnswerMode.DIRECT,retrieval_calls=0,citations=[]),1.0)
            assert entered.wait(3)
            deleting=workers.submit(deleter.mark_deleting,conversation.id)
            try:
                with pytest.raises(FutureTimeout):
                    deleting.result(timeout=.3)
            finally:
                release.set()
            saving.result(timeout=3)
            deleting.result(timeout=3)


def test_share_authorization_and_body_use_one_snapshot():
    from datetime import datetime, timedelta, UTC
    from app.infrastructure.conversations.postgres_repository import PostgresConversationRepository
    from app.infrastructure.persistence.shares import PostgresShareRepository
    from app.domain.conversations import AgentResult, AnswerMode
    from app.domain.shares import StoredShare
    from tests.postgres_support import observe_queries
    with isolated_database() as database:
        repository=PostgresConversationRepository(database)
        owner=uuid4()
        conversation=repository.create(owner_id=owner)
        turn=repository.save_turn(conversation.id,'question',
            AgentResult(answer='answer',answer_mode=AnswerMode.DIRECT,retrieval_calls=0,citations=[]),1.0)
        shares=PostgresShareRepository(database)
        now=datetime.now(UTC)
        shares.create(StoredShare(token_hash='a'*64,turn_id=turn.id,owner_id=owner,
            shared_by='test',created_at=now,expires_at=now+timedelta(days=1)))
        def before(query):
            if query.startswith('SELECT user_message'):
                repository.delete(conversation.id)
        with observe_queries(database,before):
            assert shares.read('a'*64,now).answer=='answer'


def test_index_initialization_is_serialized_across_containers():
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    from threading import Event
    from app.bootstrap.container import build_application_container, RuntimeOptions
    from app.config import Settings
    with isolated_database() as database:
        roots=[build_application_container(Settings(_env_file=None), RuntimeOptions(database=database,collection_name='same')) for _ in range(2)]
        guard=roots[0].hold_index_initialization
        entered,release=Event(),Event()
        def first():
            with guard():
                entered.set()
                assert release.wait(5)
        def second():
            with roots[1].hold_index_initialization():
                return True
        try:
            with ThreadPoolExecutor(2) as workers:
                one=workers.submit(first)
                assert entered.wait(3)
                two=workers.submit(second)
                try:
                    with pytest.raises(FutureTimeout): two.result(timeout=.3)
                finally:
                    release.set()
                one.result(timeout=3)
                assert two.result(timeout=3)
        finally:
            for root in roots: root.close()

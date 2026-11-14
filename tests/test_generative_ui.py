from uuid import uuid4

import pytest

from app.domain.conversations import AgentResult, AnswerMode, TurnCitation
from app.domain.retrieval import RetrievalResult
from tests.test_conversation_service import _service


def hit(kb='ops', identity='c1', text='数据库说明'):
    return RetrievalResult(chunk_id=identity, doc_id='doc', kb_id=kb,
                           filename='manual.md', page=3, chunk_index=0, text=text, score=.8)


class UIAgent:
    def __init__(self, fail=False):
        self.fail = fail

    def run(self, conversation_id, question, history, on_status=None, **options):
        from app.application.conversations.generative_ui import track_retrieval
        from app.application.conversations.progress import notify_status
        for attempt, phase in enumerate(('retrieving', 'expanding', 'escalating'), 1):
            notify_status(on_status, phase, attempt, ('ops',))
            with track_retrieval(question, ('ops',)) as rows:
                if self.fail:
                    raise RuntimeError('secret internal path C:/private/password')
                rows.extend([hit(text='C:\\private\\password ' + '中文资料' * 100), hit('private', 'hidden')])
        return AgentResult(answer='回答 [1]', answer_mode=AnswerMode.RETRIEVAL, retrieval_calls=3,
                           citations=[TurnCitation(position=1, doc_id='doc', chunk_id='c1', kb_id='ops',
                                                   filename='manual.md', page=3, chunk_text='资料')])


def test_tool_order_scope_safety_and_complete_history_roundtrip(tmp_path):
    service = _service(tmp_path, UIAgent())
    owner=uuid4()
    conversation = service.create(owner_id=owner)
    events = list(service.stream_query(conversation.id, '数据库'))
    tools = [e.data for e in events if e.event == 'tool']
    assert len(tools) == 6
    assert [t.args.reason for t in tools[:3]] == ['initial', 'expand_scope', 'escalate']
    assert len({t.tool_call_id for t in tools}) == 3
    first_token = next(i for i,e in enumerate(events) if e.event == 'token')
    assert all(i < first_token for i,e in enumerate(events) if e.event == 'tool')
    for running, final in zip(tools[:3], tools[3:]):
        assert running.status == 'running' and final.status == 'complete'
        assert running.tool_call_id == final.tool_call_id and running.args == final.args
        assert len(final.result.hits) == 1
        item = final.result.hits[0]
        assert item.used and item.kb_id == 'ops' and len(item.snippet) <= 200
        assert 'private' not in item.snippet
    complete = events[-1].data
    assert complete.tool_calls == tools[3:]
    assert complete.data_blocks == []
    history = service.get(conversation.id).turns[0]
    assert history.tool_calls == complete.tool_calls and history.data_blocks == []
    assert service.get_page(conversation.id, owner_id=owner, limit=1).turns[0].tool_calls == complete.tool_calls


def test_retrieval_failure_has_one_safe_terminal_then_stream_error(tmp_path):
    service = _service(tmp_path, UIAgent(fail=True))
    events = list(service.stream_query(service.create().id, '数据库'))
    tools = [e.data for e in events if e.event == 'tool']
    assert [t.status for t in tools] == ['running', 'error']
    assert tools[0].tool_call_id == tools[1].tool_call_id
    assert tools[1].error.code == 'retrieval_failed'
    assert 'private' not in str(tools[1].model_dump())
    assert events[-1].event == 'error'
    assert not any(e.event == 'token' for e in events)


def test_cancel_after_running_does_not_emit_terminal_or_save(tmp_path):
    service = _service(tmp_path, UIAgent())
    conversation = service.create()
    stream = service.stream_query(conversation.id, '数据库')
    while next(stream).event != 'tool':
        pass
    stream.close()
    assert list(stream) == []
    assert service.get(conversation.id).turns == []


def test_legacy_defaults_and_migration_idempotence(tmp_path):
    from tests.postgres_support import pg_database
    from tests.test_conversation_service import _FakeAgent
    from app.infrastructure.persistence.migrations import migrate_database
    database = pg_database(tmp_path / 'conversations.db')
    service = _service(tmp_path, _FakeAgent())
    conversation = service.create()
    service.query(conversation.id, '你好')
    with database.connection() as connection:
        connection.execute('ALTER TABLE conversation_turns DROP COLUMN tool_calls, DROP COLUMN data_blocks')
        connection.execute('DELETE FROM schema_migrations WHERE version=5')
    migrate_database(database._dsn, database.schema)
    migrate_database(database._dsn, database.schema)
    turn = service.get(conversation.id).turns[0]
    assert turn.tool_calls == [] and turn.data_blocks == []


def test_real_fast_and_graph_nodes_are_instrumented():
    from app.application.conversations.generative_ui import ToolCollector, capture_tools
    from tests.test_fast_lane import _run
    from tests.test_langgraph_conversation_agent import _agent, _ScriptedModel, _FakeRetriever, _chunk
    from langchain_core.messages import AIMessage
    for execute in (
        lambda: _run()[0].result,
        lambda: _agent(_ScriptedModel([AIMessage(content='直接猜测'), AIMessage(content='回答[1]')]),
                       _FakeRetriever({'问题': [_chunk(1)]})).run(uuid4(), '问题', []),
    ):
        emitted = []
        collector = ToolCollector(emitted.append)
        with capture_tools(collector):
            result = execute()
        finals = collector.finish(result)
        assert len(finals) == 1
        assert [e.data.status for e in emitted] == ['running', 'complete']
        assert emitted[0].data.args.query


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
async def test_http_complete_and_history_have_same_final_tools(tmp_path):
    import json
    import httpx
    from fastapi import FastAPI
    from tests.http_auth import allow_test_identity, TEST_USER_ID
    from app.interfaces.http.router import router
    from app.interfaces.http.dependencies import get_conversation_service
    service = _service(tmp_path, UIAgent())
    conversation = service.create(owner_id=TEST_USER_ID)
    app = allow_test_identity(FastAPI())
    app.include_router(router)
    app.dependency_overrides[get_conversation_service] = lambda: service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post(f'/api/conversations/{conversation.id}/query/stream', json={'question': '数据库'})
        assert response.status_code == 200
        frames = [frame for frame in response.text.split('\n\n') if frame.startswith('event:')]
        complete = json.loads(frames[-1].split('data: ', 1)[1])
        assert frames[-1].startswith('event: complete')
        assert len(complete['tool_calls']) == 3 and complete['data_blocks'] == []
        cited = {(c['doc_id'], c['chunk_id']) for c in complete['citations']}
        assert all((h['doc_id'], h['chunk_id']) in cited
                   for call in complete['tool_calls'] for h in call.get('result', {}).get('hits', [])
                   if h['used'])
        terminal = [json.loads(frame.split('data: ', 1)[1]) for frame in frames
                    if frame.startswith('event: tool') and '"status":"running"' not in frame]
        assert complete['tool_calls'] == terminal
        for suffix in ('', '?turns_limit=1'):
            history = await client.get(f'/api/conversations/{conversation.id}{suffix}')
            assert history.status_code == 200
            assert history.json()['turns'][0]['tool_calls'] == terminal


def test_hits_limit_paths_and_score_omission():
    from app.application.conversations.generative_ui import ToolCollector, capture_tools, track_retrieval
    collector = ToolCollector()
    with capture_tools(collector):
        with track_retrieval('查询', ('ops',)) as rows:
            rows.extend(hit(identity=str(i), text='/private/path C:\\secret\\file '+ '字'*500)
                        for i in range(15))
    final = collector.finish(None)[0]
    assert final.result.retrieved == 15 and len(final.result.hits) == 10
    assert all('private' not in h.snippet and 'secret' not in h.snippet and len(h.snippet)<=200
               and not h.used and 'score' not in h.model_dump() for h in final.result.hits)


def test_table_model_limits_and_cell_types():
    from pydantic import ValidationError
    from app.domain.generative_ui import TableData
    with pytest.raises(ValidationError):
        TableData(caption='表', columns=['a'], rows=[[True]], source_positions=[1])
    with pytest.raises(ValidationError):
        TableData(caption='表', columns=['a'], rows=[['x']]*51, source_positions=[1])


@pytest.mark.parametrize('path', [r'C:\Users\John Doe\AppData\Local\secret.txt',
                                  '/var/private folder/config.yaml',
                                  '"C:\\Program Files\\Service\\config.json"'])
def test_paths_with_spaces_cannot_leak_remainders(path):
    from app.application.conversations.generative_ui import safe_text
    cleaned = safe_text('文档说明 ' + path)
    assert cleaned == '文档说明 [路径已隐藏]' or cleaned == '文档说明 "[路径已隐藏]'


def test_invisible_controls_do_not_bypass_path_redaction():
    from app.application.conversations.generative_ui import safe_text
    assert safe_text('资料\x7f\u200b') == '资料'
    assert safe_text('路径 C:\u200b\\private\\config') == '路径 [路径已隐藏]'


@pytest.mark.parametrize('scope', [None, ()])
def test_empty_or_unknown_tool_scope_never_means_all_libraries(scope):
    from app.application.conversations.generative_ui import ToolCollector, capture_tools, track_retrieval
    collector = ToolCollector()
    with capture_tools(collector):
        with track_retrieval('查询', scope) as rows:
            rows.extend([hit('ops'), hit('private', 'private-chunk')])
    final = collector.finish(None)[0]
    assert final.args.kb_ids == []
    assert final.result.hits == []
    assert final.result.retrieved == 0 and final.result.relevant == 0

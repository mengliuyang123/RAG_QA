"""请求级公开检索记录；终态延后到引用确定，绝不传播模型 scratchpad。"""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import re

from app.application.execution_control import check_cancellation, StreamCancelled
from app.application.conversations.events import ToolConversationEvent
from app.domain.conversations import AgentResult
from app.domain.generative_ui import RetrievalToolArgs, RetrievalToolResult, RetrievalHit, ToolCall, ToolError
from app.domain.retrieval import RetrievalResult


_CURRENT: ContextVar['ToolCollector | None'] = ContextVar('conversation_ui', default=None)
# 路径可能含空格/引号；保守隐藏到行尾，不能留下用户名或后续目录。
_PATH = re.compile(r'(?:[a-zA-Z]:[\\/]|\\\\|(?<!\w)/[\w.-])[^\n]*')


def safe_text(text: str) -> str:
    return _PATH.sub('[路径已隐藏]', ''.join(c for c in text if c.isprintable() or c in '\n\t'))


@dataclass
class _Call:
    running: ToolCall
    rows: list[RetrievalResult] = field(default_factory=list)
    failed: bool = False


class ToolCollector:
    def __init__(self, publish: Callable[[ToolConversationEvent], None] | None = None):
        self.publish = publish
        self.calls: list[_Call] = []
        self.reason = 'initial'

    def status(self, phase: str) -> None:
        if phase == 'expanding':
            self.reason = 'expand_scope'
        elif phase == 'escalating':
            self.reason = 'escalate'

    def emit(self, call: ToolCall) -> None:
        check_cancellation()
        if self.publish is not None:
            self.publish(ToolConversationEvent(data=call))

    def finish(self, result: AgentResult | None) -> list[ToolCall]:
        used = {(c.kb_id, c.doc_id, c.chunk_id) for c in result.citations} if result else set()
        approved = set()
        if result and result.execution:
            approved = {(e.kb_id, e.doc_id, e.chunk_id) for e in result.execution.evidence
                        if e.number in result.execution.approved_numbers}
        else:
            approved = used
        finals = []
        for call in self.calls:
            base = dict(tool_call_id=call.running.tool_call_id, args=call.running.args)
            if call.failed:
                final = ToolCall(**base, status='error', error=ToolError())
            else:
                # kb_ids 来自经 ScopeResolver 校验的实际调用范围；未知范围拒绝公开。
                rows = [r for r in call.rows if r.kb_id in call.running.args.kb_ids]
                rows.sort(key=lambda r: r.score, reverse=True)
                hits = [RetrievalHit(doc_id=r.doc_id, chunk_id=r.chunk_id, kb_id=r.kb_id,
                                     filename=safe_text(re.split(r'[\\/]', r.filename)[-1]), page=r.page,
                                     snippet=safe_text(r.text)[:200], used=(r.kb_id,r.doc_id,r.chunk_id) in used)
                        for r in rows[:10]]
                final = ToolCall(**base, status='complete', result=RetrievalToolResult(
                    retrieved=len(rows), relevant=sum((r.kb_id,r.doc_id,r.chunk_id) in approved for r in rows), hits=hits))
            self.emit(final)
            finals.append(final)
        return finals


@contextmanager
def capture_tools(collector: ToolCollector) -> Iterator[None]:
    token = _CURRENT.set(collector)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def observe_status(phase: str) -> None:
    collector = _CURRENT.get()
    if collector:
        collector.status(phase)


@contextmanager
def track_retrieval(query: str, kb_ids: tuple[str, ...] | None) -> Iterator[list[RetrievalResult]]:
    collector = _CURRENT.get()
    rows: list[RetrievalResult] = []
    if collector is None:
        yield rows
        return
    check_cancellation()
    call = _Call(ToolCall(tool_call_id=f'tc_{len(collector.calls)+1:02}', status='running',
                         args=RetrievalToolArgs(query=query, kb_ids=list(kb_ids or ()),
                                                attempt=len(collector.calls)+1, reason=collector.reason)), rows)
    collector.calls.append(call)
    collector.emit(call.running)
    try:
        yield rows
        check_cancellation()
    except StreamCancelled:
        raise
    except BaseException:
        call.failed = True
        raise

# -*- coding: utf-8 -*-
"""统一执行器把Fast错误带入会话后，仍须发送安全的唯一终止事件。"""

from pathlib import Path

import pytest

from app.application.errors import EmbeddingError
from app.application.errors import EmbeddingTimeoutError
from app.application.errors import KnowledgeBaseStorageError
from app.application.errors import LLMGenerationError
from app.application.errors import LLMTimeoutError
from app.application.errors import RerankerUnavailableError
from app.application.errors import VectorStoreError
from tests.test_conversation_service import _FakeAgent
from tests.test_conversation_service import _service


@pytest.mark.parametrize("error,code", [
    (EmbeddingTimeoutError("private"), "timeout"),
    (LLMTimeoutError("private"), "timeout"),
    (EmbeddingError("private"), "retrieval_error"),
    (VectorStoreError("private"), "retrieval_error"),
    (RerankerUnavailableError("private"), "retrieval_error"),
    (LLMGenerationError("private"), "generation_error"),
    (KnowledgeBaseStorageError("private"), "storage_error"),
])
def test_fast_executor_errors_terminate_stream_without_saving(
    tmp_path: Path, error: Exception, code: str,
) -> None:
    """不能仅捕获旧Graph包装过的异常而让Fast异常直接断开流。

    Args:
        tmp_path: 隔离SQLite目录。
        error: Fast或目录端口可能抛出的具体错误。
        code: 对外稳定错误分类。
    """
    service = _service(tmp_path, _FakeAgent(error=error))
    conversation = service.create()
    events = list(service.stream_query(conversation.id, "问题"))
    assert [event.event for event in events] == ["status", "error"]
    assert events[-1].data.code == code
    assert "private" not in events[-1].data.message
    assert service.get(conversation.id).turns_count == 0
    assert not service._conversation_locks

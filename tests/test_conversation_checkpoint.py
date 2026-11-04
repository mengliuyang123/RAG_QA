# -*- coding: utf-8 -*-
"""LangGraph SQLite checkpoint 生命周期测试。"""

from pathlib import Path
from uuid import uuid4

from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from app.infrastructure.conversations.checkpoint import SQLiteCheckpointStore


def test_checkpoint_store_persists_and_deletes_one_thread(
    tmp_path: Path,
) -> None:
    store = SQLiteCheckpointStore(tmp_path / "nested" / "checkpoints.db")
    builder = StateGraph(dict)
    builder.add_node("answer", lambda state: {**state, "answer": "ok"})
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=store.checkpointer)
    first_thread = uuid4()
    second_thread = uuid4()

    for thread_id in (first_thread, second_thread):
        graph.invoke(
            {"question": "test"},
            {"configurable": {"thread_id": str(thread_id)}},
        )

    store.delete_thread(first_thread)

    first_config = {"configurable": {"thread_id": str(first_thread)}}
    second_config = {"configurable": {"thread_id": str(second_thread)}}
    assert list(store.checkpointer.list(first_config)) == []
    assert list(store.checkpointer.list(second_config)) != []
    store.close()
    store.close()

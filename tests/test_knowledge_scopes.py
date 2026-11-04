# -*- coding: utf-8 -*-
"""范围必须来自持久化目录，模型建议不能创造知识库或覆盖显式选择。"""

from pathlib import Path

import pytest

from app.application.knowledge.scopes import ScopeResolver
from app.domain.knowledge_bases import KnowledgeBase
from app.domain.errors import InvalidKnowledgeScopeError
from app.domain.routing import KnowledgeScope
from app.infrastructure.persistence.knowledge_bases import (
    SQLiteKnowledgeBaseRepository,
)


def _catalog(tmp_path: Path) -> SQLiteKnowledgeBaseRepository:
    catalog = SQLiteKnowledgeBaseRepository(tmp_path / "catalog.sqlite3")
    catalog.create(KnowledgeBase(id="ops", name="运维手册", description="部署配置"))
    catalog.create(KnowledgeBase(id="dev", name="研发文档", description="架构开发"))
    return catalog


def test_explicit_scope_ignores_model_suggestion_and_never_expands(tmp_path: Path) -> None:
    """模型即使建议研发库，也不能覆盖用户只选运维库的边界。

    Args:
        tmp_path: 隔离目录。
    """
    plan = ScopeResolver(_catalog(tmp_path)).resolve(
        KnowledgeScope(mode="explicit", kb_ids=("ops",)),
        preferred_kb_ids=("dev",),
    )
    assert plan.initial_kb_ids == ("ops",)
    assert plan.expansion_kb_ids is None


def test_auto_scope_can_expand_to_all_registered_kbs(tmp_path: Path) -> None:
    """自动选库只从目录取ID，扩大范围使用同一目录快照。

    Args:
        tmp_path: 隔离目录。
    """
    plan = ScopeResolver(_catalog(tmp_path)).resolve(
        KnowledgeScope(), preferred_kb_ids=("ops",),
    )
    assert plan.initial_kb_ids == ("ops",)
    assert plan.expansion_kb_ids == ("default", "dev", "ops")


@pytest.mark.parametrize("suggestion", [(), ("invented",), ("ops", "invented")])
def test_uncertain_or_invalid_model_scope_falls_back_to_all(
    tmp_path: Path, suggestion: tuple[str, ...],
) -> None:
    """非法模型建议保守查全库，而不是悄悄接受部分虚构范围。

    Args:
        tmp_path: 隔离目录。
        suggestion: 空或含未知库的模型建议。
    """
    plan = ScopeResolver(_catalog(tmp_path)).resolve(
        KnowledgeScope(), preferred_kb_ids=suggestion,
    )
    assert plan.initial_kb_ids == ("default", "dev", "ops")
    assert plan.expansion_kb_ids is None


def test_unknown_explicit_scope_is_rejected(tmp_path: Path) -> None:
    """空知识库也有目录记录；不存在的库不能只靠召回为空处理。

    Args:
        tmp_path: 隔离目录。
    """
    with pytest.raises(InvalidKnowledgeScopeError):
        ScopeResolver(_catalog(tmp_path)).resolve(
            KnowledgeScope(mode="explicit", kb_ids=("invented",)),
        )


def test_catalog_persists_metadata_and_does_not_reset_admin_updates(tmp_path: Path) -> None:
    """重建目录对象后保留元数据，也不能把改名的默认库覆盖回初始值。

    Args:
        tmp_path: 隔离目录。
    """
    catalog = _catalog(tmp_path)
    default = catalog.get("default")
    catalog.update(default.model_copy(update={"name": "历史语料"}))
    reopened = SQLiteKnowledgeBaseRepository(tmp_path / "catalog.sqlite3")
    assert reopened.get("default").name == "历史语料"
    assert reopened.get("ops").description == "部署配置"
    assert [kb.id for kb in reopened.list()] == ["default", "dev", "ops"]

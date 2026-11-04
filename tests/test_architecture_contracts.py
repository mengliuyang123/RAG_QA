# -*- coding: utf-8 -*-
"""验证 Clean Architecture 契约和静态依赖方向。"""

import ast
from pathlib import Path

from app.application.ports.embedding import EmbeddingProvider
from app.domain.documents import DocumentChunk, UploadResult
from app.domain.retrieval import RetrievalResult, RetrievalStrategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_LIBRARY_PREFIXES = (
    "chromadb",
    "fastapi",
    "fitz",
    "httpx",
    "langchain_text_splitters",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "multipart",
    "openai",
    "pymupdf",
    "ragas",
    "rank_bm25",
    "uvicorn",
)
LAYER_RULES = {
    "app.domain": ADAPTER_LIBRARY_PREFIXES + (
        "app.application",
        "app.infrastructure",
        "app.interfaces",
        "app.bootstrap",
    ),
    "app.application": ADAPTER_LIBRARY_PREFIXES + (
        "app.infrastructure",
        "app.interfaces",
        "app.bootstrap",
    ),
    "app.interfaces": ("app.infrastructure",),
    "evaluation": ("app.infrastructure",),
}
LEGACY_PACKAGE_NAMES = ("core", "api", "services")


def _python_files(*directories: str) -> list[Path]:
    files: list[Path] = []
    for directory in directories:
        files.extend((PROJECT_ROOT / directory).rglob("*.py"))
    return sorted(files)


def _module_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(relative.parts)


def _imports_from_source(
    source: str,
    module_name: str,
) -> list[tuple[int, str]]:
    tree = ast.parse(source, filename=module_name)
    imports: list[tuple[int, str]] = []
    package_parts = module_name.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base_name = node.module or ""
            else:
                parent_count = node.level - 1
                if parent_count > len(package_parts):
                    base_parts: list[str] = []
                elif parent_count == 0:
                    base_parts = package_parts
                else:
                    base_parts = package_parts[:-parent_count]
                if node.module is not None:
                    base_parts = [*base_parts, *node.module.split(".")]
                base_name = ".".join(base_parts)

            for alias in node.names:
                if alias.name == "*":
                    imported_name = base_name
                elif base_name:
                    imported_name = f"{base_name}.{alias.name}"
                else:
                    imported_name = alias.name
                imports.append((node.lineno, imported_name))
    return imports


def _imports(path: Path) -> list[tuple[int, str]]:
    return _imports_from_source(
        path.read_text(encoding="utf-8"),
        _module_name(path),
    )


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def test_domain_contracts_are_available_without_core() -> None:
    """领域模型和端口不依赖现有 core 实现。"""
    chunk = DocumentChunk(
        id="doc:0",
        doc_id="doc",
        text="正文",
        filename="notes.md",
        page=None,
        chunk_index=0,
    )
    result = RetrievalResult(
        chunk_id=chunk.id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        filename=chunk.filename,
        page=chunk.page,
        chunk_index=chunk.chunk_index,
        score=0.8,
    )
    upload = UploadResult(doc_id="doc", filename="notes.md", chunks_count=1)

    assert result.score == 0.8
    assert upload.chunks_count == 1
    assert RetrievalStrategy.HYBRID_RERANK.value == "hybrid_rerank"
    assert hasattr(EmbeddingProvider, "embed_query")


def test_layer_imports_follow_dependency_direction() -> None:
    """受保护层不得静态导入外层或具体基础设施。"""
    violations: list[str] = []
    source_files = _python_files(
        "app/domain",
        "app/application",
        "app/interfaces",
        "evaluation",
    )
    for path in source_files:
        owner = _module_name(path)
        for source_prefix, forbidden_prefixes in LAYER_RULES.items():
            if not _matches_prefix(owner, source_prefix):
                continue
            for line_number, imported_name in _imports(path):
                for forbidden_prefix in forbidden_prefixes:
                    if _matches_prefix(imported_name, forbidden_prefix):
                        relative = path.relative_to(PROJECT_ROOT)
                        violations.append(
                            f"{relative}:{line_number} {imported_name}"
                        )

    assert violations == [], "\n".join(violations)


def test_import_resolver_covers_absolute_package_and_relative_forms() -> None:
    """静态扫描必须识别三种可绕过浅层规则的导入写法。"""
    imports = _imports_from_source(
        "\n".join(
            [
                "import app.infrastructure.vectorstores",
                "from app import infrastructure",
                "from .. import infrastructure",
            ]
        ),
        "app.application.sample",
    )

    imported_names = [name for _, name in imports]
    assert imported_names == [
        "app.infrastructure.vectorstores",
        "app.infrastructure",
        "app.infrastructure",
    ]


def test_legacy_packages_are_not_imported_or_present() -> None:
    """旧包不得保留源码，也不得被任何 Python 模块导入。"""
    violations: list[str] = []
    for path in _python_files("app", "evaluation", "tests"):
        for line_number, imported_name in _imports(path):
            for package_name in LEGACY_PACKAGE_NAMES:
                legacy_prefix = f"app.{package_name}"
                if _matches_prefix(imported_name, legacy_prefix):
                    relative = path.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"{relative}:{line_number} {imported_name}"
                    )

    for package_name in LEGACY_PACKAGE_NAMES:
        package_path = PROJECT_ROOT / "app" / package_name
        for path in sorted(package_path.rglob("*.py")):
            violations.append(str(path.relative_to(PROJECT_ROOT)))

    assert violations == [], "\n".join(violations)

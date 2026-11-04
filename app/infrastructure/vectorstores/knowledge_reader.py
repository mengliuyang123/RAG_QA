# -*- coding: utf-8 -*-
"""Chroma目录读取辅助：先验证元数据，只对选定的有限块加载正文。"""

from datetime import datetime
from collections.abc import Iterator
from pathlib import PurePath
from pathlib import PureWindowsPath
from typing import Any
from uuid import UUID

from app.application.errors import VectorStoreError
from app.domain.documents import DocumentChunk
from app.domain.knowledge_bases import DEFAULT_KB_ID
from app.domain.knowledge_views import IndexedChunkHeader
from app.domain.knowledge_views import IndexedDocumentHeader


def parse_updated_at(value: object) -> datetime | None:
    """解析明确保存的版本时间，不推测旧索引的时间。

    Args:
        value: 索引中的ISO时间或缺失。

    Returns:
        带时区时间或None。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("文档版本时间无效")
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError("文档版本时间缺少时区")
    return result


def read_document_headers(collection: Any, kb_id: str | None) -> tuple[IndexedDocumentHeader, ...]:
    """召回前按库限定元数据，再按文档聚合，不读取正文。

    Args:
        collection: 当前adapter持有的SDK collection。
        kb_id: 已由用例核对目录的范围，None仅用于管理元数据视图。

    Returns:
        按文档UUID稳定排列的元数据头。
    """
    try:
        groups = {}
        where = {"kb_id": kb_id} if kb_id is not None else None
        for header in _headers(collection, where):
            if kb_id is not None and header.kb_id != kb_id:
                raise ValueError("索引返回了范围外元数据")
            groups.setdefault(header.doc_id, []).append(header)
        documents = []
        for doc_id in sorted(groups, key=str):
            headers = _validate_group(groups[doc_id])
            first = headers[0]
            documents.append(IndexedDocumentHeader(
                doc_id=doc_id, kb_id=first.kb_id, filename=first.filename,
                chunks_count=len(headers), updated_at=first.doc_updated_at,
            ))
        return tuple(documents)
    except Exception as error:
        raise VectorStoreError("知识库文档元数据读取失败") from error


def read_chunk_headers(collection: Any, doc_id: UUID) -> tuple[IndexedChunkHeader, ...]:
    """仅读取指定文档的块元数据，索引不连续时仍按实际顺序选邻居。

    Args:
        collection: 当前SDK collection。
        doc_id: 指定文档UUID。

    Returns:
        已校验同源、同版本且顺序唯一的元数据头。
    """
    try:
        headers = list(_headers(collection, {"doc_id": str(doc_id)}))
        for header in headers:
            if header.doc_id != doc_id:
                raise ValueError("索引返回了其他文档")
        return _validate_group(headers)
    except Exception as error:
        raise VectorStoreError("文档块元数据读取失败") from error


def read_selected_chunks(
    collection: Any, headers: tuple[IndexedChunkHeader, ...],
) -> tuple[DocumentChunk, ...]:
    """仅加载已选定的最多七块，并再次核对文档和知识库。

    Args:
        collection: 当前SDK collection。
        headers: 可信元数据阶段选出的相邻窗口。

    Returns:
        与请求元数据头完全对应的实际正文。
    """
    try:
        if not 1 <= len(headers) <= 7:
            raise ValueError("原文窗口大小无效")
        _validate_group(list(headers))
        first = headers[0]
        expected = {header.chunk_id: header for header in headers}
        if len(expected) != len(headers):
            raise ValueError("原文窗口ID重复")
        result = collection.get(
            ids=list(expected), include=["documents", "metadatas"], limit=len(headers),
            where={"$and": [{"doc_id": str(first.doc_id)}, {"kb_id": first.kb_id}]},
        )
        chunks = {}
        for identifier, text, metadata in zip(
            result["ids"], result["documents"] or [], result["metadatas"] or [], strict=True,
        ):
            header = _decode_header(identifier, metadata)
            if identifier not in expected or header != expected[identifier]:
                raise ValueError("读取期间文档元数据发生变化")
            chunks[identifier] = DocumentChunk(
                id=identifier, doc_id=str(header.doc_id), text=text,
                filename=header.filename, page=header.page, chunk_index=header.chunk_index,
                kb_id=header.kb_id, doc_updated_at=header.doc_updated_at,
            )
        if len(chunks) != len(expected):
            raise ValueError("读取期间文档正文发生变化")
        return tuple(chunks[header.chunk_id] for header in headers)
    except Exception as error:
        raise VectorStoreError("文档原文窗口读取失败") from error


def _headers(collection: Any, where: dict[str, object] | None) -> Iterator[IndexedChunkHeader]:
    offset = 0
    while True:
        result = collection.get(
            where=where, include=["metadatas"], limit=1000, offset=offset,
        )
        ids = result["ids"]
        for identifier, metadata in zip(ids, result["metadatas"] or [], strict=True):
            yield _decode_header(identifier, metadata)
        if len(ids) < 1000:
            break
        offset += len(ids)


def _decode_header(identifier: str, metadata: dict[str, object]) -> IndexedChunkHeader:
    filename = metadata["filename"]
    if not isinstance(filename, str) or filename in {"", ".", ".."}:
        raise ValueError("来源文件名无效")
    if PurePath(filename).name != filename or PureWindowsPath(filename).name != filename:
        raise ValueError("来源文件名不安全")
    return IndexedChunkHeader(
        chunk_id=identifier, doc_id=metadata["doc_id"],
        filename=filename, kb_id=metadata.get("kb_id", DEFAULT_KB_ID),
        chunk_index=_integer(metadata["chunk_index"]),
        page=_integer(metadata["page"]) or None,
        doc_updated_at=parse_updated_at(metadata.get("doc_updated_at")),
    )


def _integer(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    raise ValueError("块序号或页码无效")


def _validate_group(headers: list[IndexedChunkHeader]) -> tuple[IndexedChunkHeader, ...]:
    if not headers:
        return ()
    first = headers[0]
    indexes = set()
    identifiers = set()
    for header in headers:
        if (header.doc_id, header.kb_id, header.filename, header.doc_updated_at) != (
            first.doc_id, first.kb_id, first.filename, first.doc_updated_at,
        ):
            raise ValueError("同一文档的来源或版本元数据不一致")
        if header.chunk_index in indexes or header.chunk_id in identifiers:
            raise ValueError("文档块顺序或ID重复")
        indexes.add(header.chunk_index)
        identifiers.add(header.chunk_id)
    return tuple(sorted(headers, key=lambda header: header.chunk_index))

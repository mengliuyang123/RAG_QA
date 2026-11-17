# -*- coding: utf-8 -*-
"""按 ATX 标题把 Markdown 文本拆成带标题路径的章节；入库解析与评估共用。"""

import re
from collections.abc import Iterator


# CommonMark ATX 标题：最多 3 个前导空格，1-6 个 #，其后是空白或行尾；可选的结尾 # 序列被去掉。
_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


class HeadingTracker:
    """按标题级别维护当前标题路径：进入某级标题时弹出同级及更深的标题。"""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    @property
    def path(self) -> tuple[str, ...]:
        """当前从最外层到最内层的标题路径。"""
        return tuple(title for _, title in self._stack)

    def enter(self, level: int, title: str) -> None:
        """记录一个新标题；空标题只结束同级及更深的章节，不进入路径。

        Args:
            level: 从 1 开始的标题级别，数字越大层级越深。
            title: 标题文字。
        """
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        normalized = " ".join(title.split())
        if normalized:
            self._stack.append((level, normalized))


def split_markdown_sections(text: str) -> Iterator[tuple[tuple[str, ...], str]]:
    """把 Markdown 文本拆成 ``(标题路径, 正文)``，跳过正文为空的章节。

    Args:
        text: 完整 Markdown 文本。

    Returns:
        按文档顺序排列的章节。
    """
    tracker = HeadingTracker()
    lines: list[str] = []
    path: tuple[str, ...] = ()
    fence: str | None = None

    def flush() -> Iterator[tuple[tuple[str, ...], str]]:
        body = "\n".join(lines).strip("\n")
        if body.strip():
            yield path, body
        lines.clear()

    for line in text.splitlines():
        fence_match = _FENCE.match(line)
        if fence is not None:
            if fence_match and fence_match.group(1)[0] == fence[0] \
                    and len(fence_match.group(1)) >= len(fence) \
                    and not line.strip()[len(fence_match.group(1)):].strip():
                fence = None
            lines.append(line)
            continue
        if fence_match:
            fence = fence_match.group(1)
            lines.append(line)
            continue
        heading = _ATX_HEADING.match(line)
        if heading:
            yield from flush()
            tracker.enter(len(heading.group(1)), heading.group(2) or "")
            path = tracker.path
            continue
        lines.append(line)
    yield from flush()

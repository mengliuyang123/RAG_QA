# -*- coding: utf-8 -*-
"""管理员命令行：用已保存的源文件，按当前分块规则原地重建已发布文档。

用法：
    python -m app.interfaces.cli.reindex --dry-run
    python -m app.interfaces.cli.reindex --kb default --kb ops
    python -m app.interfaces.cli.reindex --all --limit 20

默认只处理仍有旧格式文本块的文档；``--all`` 连已是新格式的也重建。
每篇文档的新文本块写入成功前，旧文本块照常可检索。会调用 Embedding 服务。
"""

import argparse
import sys

from app.bootstrap.container import build_application_container
from app.config import Settings
from app.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按当前分块规则原地重建已发布文档")
    parser.add_argument("--kb", action="append", dest="kb_ids", metavar="KB_ID",
                        help="只处理指定知识库，可重复；默认全部")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出将要重建的文档，不读源文件、不调用模型、不写入")
    parser.add_argument("--all", action="store_true", dest="include_current",
                        help="连已是新格式的文档也重建")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="本次最多处理 N 篇，便于分批执行")
    return parser


def main(argv: list[str] | None = None, *, settings: Settings | None = None) -> int:
    """执行重建并逐篇输出结果。

    Returns:
        全部成功或只有跳过时返回 0；有失败时返回 1；参数错误返回 2。
    """
    arguments = build_parser().parse_args(argv)
    if arguments.limit is not None and arguments.limit <= 0:
        print("--limit 必须大于 0", file=sys.stderr)
        return 2
    kb_ids = tuple(arguments.kb_ids) if arguments.kb_ids else None
    container = build_application_container(settings or get_settings())
    try:
        service = container.document_reindex_service
        plan = service.plan(
            kb_ids, include_current=arguments.include_current, limit=arguments.limit,
        )
        print(f"待重建 {len(plan.candidates)} 篇")
        for candidate in plan.candidates:
            print(f"  [{candidate.kb_id}] {candidate.filename}")
        for kb_id, filename in plan.without_source:
            print(f"  跳过 [{kb_id}] {filename}：没有入库记录，找不到源文件")
        if arguments.dry_run:
            print("试运行：未读取源文件，未调用模型，未写入。")
            return 0
        failed = rebuilt = skipped = 0
        for index, candidate in enumerate(plan.candidates, start=1):
            result = service.rebuild(candidate)
            label = {"rebuilt": "完成", "skipped": "跳过", "failed": "失败"}[result.outcome]
            detail = f"{result.chunks} 个子块" if result.outcome == "rebuilt" else result.reason
            print(f"[{index}/{len(plan.candidates)}] {label} [{result.kb_id}] {result.filename}：{detail}",
                  flush=True)
            rebuilt += result.outcome == "rebuilt"
            skipped += result.outcome == "skipped"
            failed += result.outcome == "failed"
        print(f"完成 {rebuilt} 篇，跳过 {skipped + len(plan.without_source)} 篇，失败 {failed} 篇")
        return 1 if failed else 0
    finally:
        container.close()


if __name__ == "__main__":
    raise SystemExit(main())

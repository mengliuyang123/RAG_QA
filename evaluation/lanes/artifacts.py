# -*- coding: utf-8 -*-
"""写入运行专属JSON、Markdown和Schema，不替换已发布latest。"""

import json
import os
import tempfile
from pathlib import Path

from evaluation.lanes.reporting import LaneReport


def write_lane_report(report: LaneReport, destination: Path) -> None:
    """逐文件原子更新运行快照，烟雾和不完整报告不能冒充已发布评估。

    Args:
        report: 已按实际计划分母生成的报告。
        destination: 此运行专属目录，不接受latest文件作为目标。

    Returns:
        None。
    """
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination / "report.json", report.model_dump_json(indent=2) + "\n")
    _atomic_write(destination / "report.md", render_lane_markdown(report))
    schema = json.dumps(LaneReport.model_json_schema(), ensure_ascii=False, indent=2)
    _atomic_write(destination / "schema.json", schema + "\n")


def render_lane_markdown(report: LaneReport) -> str:
    """用明确分母呈现小样本报告，不展示问题或检索正文。

    Args:
        report: 当前运行快照。

    Returns:
        可审阅的中文Markdown报告。
    """
    lines = [
        "# 分通道 RAG 评估", "",
        f"- 用途：`{report.purpose}`；状态：`{report.status}`。",
        f"- 计划 {report.expected_turns} 轮，生成 {report.generated_turns} 轮，"
        f"独立评分 {report.scored_turns} 轮。",
        f"- 检索策略：`{report.manifest.strategy.value}`。",
        f"- 运行指纹：`{report.manifest.fingerprint()}`。",
        "- 烟雾仅证明连通性，不能当作完整能力结论或替换旧latest。",
        "- 每项分数仅对实际分母有效；缺失为未知，不是0分。", "",
    ]
    for policy, summary in report.lanes.items():
        lines.extend([
            f"## {policy}", "",
            f"计划／生成／评分：{summary.expected}／{summary.generated}／{summary.scored}", "",
            "| 指标 | 数值 | 实际分母 |", "| --- | --- | --- |",
        ])
        for name, metric in summary.metrics.items():
            value = "不适用／未知"
            if metric.value is not None:
                value = f"{metric.value:.4f}"
            lines.append(f"| {name} | {value} | {metric.n} |")
        lines.extend([
            "",
            f"生成用量未知调用：{summary.generation.tokens.unknown_input_calls}；"
            f"生成费用状态：`{summary.generation.cost.status}`。",
            f"独立评分费用状态：`{summary.judging.cost.status}`。", "",
        ])
    lines.extend([
        "费用仅汇总成功结果与成功评分，失败尝试需查看隔离追踪库；"
        "费率估计不是账单，未知不表示免费。",
        "知识共同子集和能力标签的交叉统计见JSON；重叠标签行不能相加。", "",
    ])
    if report.issues:
        lines.extend(["## 未完成步骤", ""])
        for issue in report.issues:
            lines.append(
                f"- {issue.episode_id}/{issue.policy}/{issue.turn_index + 1}："
                f"{issue.stage} / {issue.error_category}",
            )
        lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
        prefix=f".{path.name}.", suffix=".tmp",
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

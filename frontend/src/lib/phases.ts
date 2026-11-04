import type { StatusUpdate } from "../api/types";

// 普通用户看到的阶段文案；后端带了 label 时优先使用后端文案。
// 未知阶段返回 null，由调用方忽略，以兼容后端后续新增的阶段。
const PHASE_LABELS: Record<string, string> = {
  queued: "排队中",
  understanding: "正在理解问题",
  deciding: "正在理解问题",
  retrieving: "正在查找资料",
  expanding: "正在扩大查找范围",
  escalating: "这个问题比较复杂，正在深入查找",
  verifying: "正在核对",
  validating: "正在核对",
  generating: "正在整理回答",
};

export function phaseLabel(
  update: StatusUpdate,
  kbName: (kbId: string) => string,
): string | null {
  if (update.phase === "queued") {
    return update.queuePosition
      ? `当前提问人数较多，前面还有 ${update.queuePosition} 人`
      : "排队中";
  }
  if (update.label) return update.label;
  const base = PHASE_LABELS[update.phase];
  if (!base) return null;
  if (update.phase === "retrieving" && update.kbIds.length > 0) {
    return `正在${update.kbIds.map(kbName).join("、")}中查找资料`;
  }
  return base;
}

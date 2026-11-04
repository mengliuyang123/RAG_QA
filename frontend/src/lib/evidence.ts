import type { Turn } from "../api/types";

// 把证据统计翻译成一句用户能看懂的话。新口径缺失（旧记录）时退回
// 只说引用和排除数量的旧文案，不根据缺失字段猜测检索结果。
export function evidenceSummary(turn: Pick<Turn, "outcome" | "evidenceStats">): string | null {
  const stats = turn.evidenceStats;
  if (!stats) return null;
  const { used, excluded, retrieved, relevant } = stats;

  if (retrieved === null || relevant === null) {
    if (used === 0 && excluded === 0) return null;
    return `采用 ${used} 段资料，排除 ${excluded} 段不相关内容`;
  }

  if (turn.outcome === "refused") {
    if (retrieved === 0) return "没有找到相关资料";
    if (relevant === 0) return `找到 ${retrieved} 段资料，但都与问题无关`;
    return `找到 ${relevant} 段相关资料，但不足以回答`;
  }

  if (retrieved === 0) return null;
  return `找到 ${retrieved} 段资料，采用 ${used} 段`;
}

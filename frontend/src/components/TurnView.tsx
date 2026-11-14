import { CheckCircle2Icon, ChevronDownIcon, FileTextIcon, SearchXIcon } from "lucide-react";
import { useState, type ReactNode } from "react";
import type { Citation, KbScope, Turn } from "../api/types";
import { evidenceSummary } from "../lib/evidence";
import { useApp } from "../state/AppContext";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { AnswerText } from "./AnswerText";

// 一轮已完成回答的正文部分（过程摘要、澄清/拒答卡片、正文与引用）。
// 不依赖 assistant-ui 上下文，便于单独测试；问题气泡与操作栏由 chat/Messages 组合。
interface TurnBodyProps {
  turn: Turn;
  activeCitation: Citation | null;
  onCite(citation: Citation): void;
  onAsk(question: string): void;
  onRetryWithScope(scope: KbScope, question: string): void;
  /** 渲染正文；默认使用 AnswerText，assistant-ui 中替换为 MessagePrimitive.Parts。 */
  renderAnswer?: (activePosition: number | null) => ReactNode;
}

function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatDate(value: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleDateString("zh-CN");
}

export const GROUNDING = {
  grounded: { className: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300", text: "有据可查" },
  partial: { className: "bg-amber-500/10 text-amber-700 dark:text-amber-300", text: "部分依据" },
  ungrounded: { className: "bg-muted text-muted-foreground", text: "未参考知识库" },
} as const;

function ProcessSummary({ turn }: { turn: Turn }) {
  const { kbName, user } = useApp();
  const [open, setOpen] = useState(false);
  if (turn.outcome === "clarification") return null;

  const sources = new Set(turn.citations.map((citation) => citation.docId)).size;
  const headline = turn.answerMode === "direct"
    ? "未查询知识库"
    : turn.outcome === "refused"
      ? "已查找，但资料不足"
      : `参考了 ${sources} 份资料`;

  const details: string[] = [];
  if (turn.searchedKbIds.length > 0) {
    details.push(`检索范围：${turn.searchedKbIds.map(kbName).join("、")}`);
  }
  if (turn.scopeExpanded) details.push("所选范围内证据不足，已扩大到全部知识库");
  if (turn.escalated) details.push("问题较复杂，已升级为深度检索");
  if (turn.retrievalCalls > 0) details.push(`共检索 ${turn.retrievalCalls} 次`);
  const evidence = evidenceSummary(turn);
  if (evidence) details.push(evidence);
  if (user?.role === "admin") {
    if (turn.lane) details.push(`执行通道：${turn.lane}`);
    if (turn.traceId) details.push(`追踪 ID：${turn.traceId}`);
  }

  return (
    <div className="mb-2">
      <button
        type="button"
        className="group inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground disabled:hover:text-muted-foreground"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        disabled={details.length === 0}
      >
        <CheckCircle2Icon className="size-4" />
        <span>{headline} · {formatSeconds(turn.latencyMs)}</span>
        {details.length > 0 && (
          <ChevronDownIcon className={cn("size-3.5 transition-transform", open && "rotate-180")} />
        )}
      </button>
      {open && (
        <ul className="mt-2 ml-2 flex flex-col gap-1 border-l pl-4 text-sm text-muted-foreground animate-in fade-in slide-in-from-top-1">
          {details.map((detail) => <li key={detail}>{detail}</li>)}
        </ul>
      )}
    </div>
  );
}

// 指定范围下拒答时，让用户换到范围外的知识库重查。不依赖后端的
// suggested_kb_id：理解层只看库名和描述，答案在别的库时也可能给不出建议。
// 有建议时排在最前并标注"推荐"；点击后才发起新请求，不自动越过原范围。
function RetryScopePicker({ turn, onRetry }: { turn: Turn; onRetry(scope: KbScope): void }) {
  const { knowledgeBases } = useApp();
  if (turn.scopeMode !== "explicit") return null;
  const others = knowledgeBases.filter((kb) => !turn.requestedKbIds.includes(kb.id));
  if (others.length === 0) return null;
  const suggested = turn.refusal?.suggestedKbId ?? null;
  const ordered = [...others].sort((a, b) => Number(b.id === suggested) - Number(a.id === suggested));

  return (
    <div className="mt-3">
      <div className="text-sm text-muted-foreground">在其他知识库中重新查找：</div>
      <div className="mt-2 flex flex-wrap gap-2">
        {ordered.map((kb) => (
          <Button
            key={kb.id}
            type="button"
            size="sm"
            variant={kb.id === suggested ? "default" : "outline"}
            className="rounded-full"
            onClick={() => onRetry({ mode: "explicit", kbIds: [kb.id] })}
          >
            {kb.name}
            {kb.id === suggested && <span className="text-xs opacity-80">推荐</span>}
          </Button>
        ))}
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="rounded-full"
          onClick={() => onRetry({ mode: "auto", kbIds: [] })}
        >
          自动（全部知识库）
        </Button>
      </div>
    </div>
  );
}

export function TurnBody({ turn, activeCitation, onCite, onAsk, onRetryWithScope, renderAnswer }: TurnBodyProps) {
  const { kbName } = useApp();
  // 引用对象在状态更新后会换新引用，按文档、分块和编号判断当前高亮。
  const activePosition = activeCitation && turn.citations.some(
    (citation) => citation.docId === activeCitation.docId && citation.position === activeCitation.position
      && citation.chunkId === activeCitation.chunkId,
  )
    ? activeCitation.position
    : null;

  return (
    <>
      <ProcessSummary turn={turn} />

      {turn.outcome === "clarification" && turn.clarification && (
        <div className="rounded-xl border bg-card p-4">
          <h4 className="m-0 font-medium">{turn.clarification.question}</h4>
          <div className="mt-3 flex flex-wrap gap-2">
            {turn.clarification.options.map((option) => (
              <Button
                key={option}
                type="button"
                size="sm"
                variant="outline"
                className="rounded-full"
                onClick={() => onAsk(option)}
              >
                {option}
              </Button>
            ))}
          </div>
        </div>
      )}

      {turn.outcome === "refused" && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
          <h4 className="m-0 flex items-center gap-2 font-medium">
            <SearchXIcon className="size-4 text-amber-600 dark:text-amber-400" />
            暂时无法从知识库中找到可靠答案
          </h4>
          {(evidenceSummary(turn) ?? turn.refusal?.foundSummary) && (
            <div className="mt-1.5 text-sm text-muted-foreground">
              {evidenceSummary(turn) ?? turn.refusal?.foundSummary}
            </div>
          )}
          {turn.refusal && turn.refusal.missing.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-sm">
              {turn.refusal.missing.map((item) => <li key={item}>缺少：{item}</li>)}
            </ul>
          )}
          <RetryScopePicker turn={turn} onRetry={(scope) => onRetryWithScope(scope, turn.question)} />
        </div>
      )}

      {turn.outcome === "answered" && (
        <>
          {renderAnswer ? renderAnswer(activePosition) : (
            <AnswerText
              answer={turn.answer}
              citations={turn.citations}
              activePosition={activePosition}
              onCite={onCite}
            />
          )}
          {turn.citations.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2" aria-label="引用来源">
              {[...turn.citations].sort((a, b) => a.position - b.position).map((citation) => {
                const updated = formatDate(citation.docUpdatedAt);
                const active = activePosition === citation.position;
                return (
                  <button
                    key={citation.position}
                    type="button"
                    title={updated ? `更新于 ${updated}` : undefined}
                    className={cn(
                      "inline-flex max-w-full items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-muted",
                      active && "border-foreground/30 bg-muted",
                    )}
                    onClick={() => onCite(citation)}
                  >
                    <span className="font-mono text-muted-foreground">[{citation.position}]</span>
                    <FileTextIcon className="size-3.5 shrink-0 text-muted-foreground" />
                    <span className="truncate">
                      {citation.filename}
                      {citation.page !== null && ` · 第 ${citation.page} 页`}
                    </span>
                    {citation.kbId && (
                      <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">
                        {kbName(citation.kbId)}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </>
      )}
    </>
  );
}

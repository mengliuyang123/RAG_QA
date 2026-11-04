import { useState } from "react";
import type { Citation, FeedbackRating, KbScope, Turn } from "../api/types";
import { evidenceSummary } from "../lib/evidence";
import { useApp } from "../state/AppContext";
import { AnswerText } from "./AnswerText";
import { Icon } from "./Icon";

interface TurnViewProps {
  turn: Turn;
  activeCitation: Citation | null;
  onCite(citation: Citation): void;
  onAsk(question: string): void;
  onRetryWithScope(scope: KbScope, question: string): void;
  onFeedback(turnId: string, rating: FeedbackRating | null): void;
}

function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatDate(value: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleDateString("zh-CN");
}

const GROUNDING = {
  grounded: { className: "chip-success", text: "有据可查" },
  partial: { className: "chip-warning", text: "部分依据" },
  ungrounded: { className: "", text: "未参考知识库" },
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
    <div className="process">
      <button
        type="button"
        className="process-head"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        disabled={details.length === 0}
      >
        <span className="step-dot done"><Icon name="check" size={14} /></span>
        <span className="grow">{headline} · {formatSeconds(turn.latencyMs)}</span>
        {details.length > 0 && (
          <span style={{ transform: open ? "rotate(180deg)" : undefined, display: "inline-flex" }}>
            <Icon name="chevron" size={14} />
          </span>
        )}
      </button>
      {open && (
        <ul className="process-steps">
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
    <div style={{ marginTop: 10 }}>
      <div className="secondary" style={{ fontSize: 13 }}>在其他知识库中重新查找：</div>
      <div className="option-row" style={{ marginTop: 6 }}>
        {ordered.map((kb) => (
          <button
            key={kb.id}
            type="button"
            className={`btn ${kb.id === suggested ? "btn-primary" : ""}`}
            onClick={() => onRetry({ mode: "explicit", kbIds: [kb.id] })}
          >
            {kb.name}
            {kb.id === suggested && <span style={{ fontSize: 12, opacity: 0.85 }}>推荐</span>}
          </button>
        ))}
        <button type="button" className="btn" onClick={() => onRetry({ mode: "auto", kbIds: [] })}>
          自动（全部知识库）
        </button>
      </div>
    </div>
  );
}

export function TurnView({ turn, activeCitation, onCite, onAsk, onRetryWithScope, onFeedback }: TurnViewProps) {
  const { kbName } = useApp();
  const [copied, setCopied] = useState(false);
  const grounding = GROUNDING[turn.grounding];
  const activePosition = activeCitation && turn.citations.includes(activeCitation)
    ? activeCitation.position
    : null;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(turn.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  const toggleFeedback = (rating: FeedbackRating) =>
    onFeedback(turn.id, turn.feedback === rating ? null : rating);

  return (
    <article className="turn" aria-label="一轮问答">
      <div className="question">{turn.question}</div>
      <ProcessSummary turn={turn} />

      {turn.outcome === "clarification" && turn.clarification && (
        <div className="notice-card">
          <h4>{turn.clarification.question}</h4>
          <div className="option-row">
            {turn.clarification.options.map((option) => (
              <button key={option} type="button" className="btn" onClick={() => onAsk(option)}>
                {option}
              </button>
            ))}
          </div>
        </div>
      )}

      {turn.outcome === "refused" && (
        <div className="notice-card">
          <h4>暂时无法从知识库中找到可靠答案</h4>
          {(evidenceSummary(turn) ?? turn.refusal?.foundSummary) && (
            <div className="secondary">{evidenceSummary(turn) ?? turn.refusal?.foundSummary}</div>
          )}
          {turn.refusal && turn.refusal.missing.length > 0 && (
            <ul>{turn.refusal.missing.map((item) => <li key={item}>缺少：{item}</li>)}</ul>
          )}
          <RetryScopePicker turn={turn} onRetry={(scope) => onRetryWithScope(scope, turn.question)} />
        </div>
      )}

      {turn.outcome === "answered" && (
        <>
          <AnswerText
            answer={turn.answer}
            citations={turn.citations}
            activePosition={activePosition}
            onCite={onCite}
          />
          {turn.citations.length > 0 && (
            <div className="citations" aria-label="引用来源">
              {[...turn.citations].sort((a, b) => a.position - b.position).map((citation) => {
                const updated = formatDate(citation.docUpdatedAt);
                return (
                  <button
                    key={citation.position}
                    type="button"
                    className={`citation ${activePosition === citation.position ? "active" : ""}`}
                    onClick={() => onCite(citation)}
                  >
                    <span className="pos">[{citation.position}]</span>
                    {citation.kbId && <span className="chip">{kbName(citation.kbId)}</span>}
                    <span className="name">
                      {citation.filename}
                      {citation.page !== null && ` · 第 ${citation.page} 页`}
                    </span>
                    {updated && <span className="muted" style={{ fontSize: 12 }}>更新于 {updated}</span>}
                  </button>
                );
              })}
            </div>
          )}
        </>
      )}

      <div className="turn-actions">
        {turn.outcome === "answered" && (
          <span className={`chip ${grounding.className}`}>{grounding.text}</span>
        )}
        {turn.outcome === "refused" && <span className="chip chip-warning">资料不足</span>}
        <span className="grow" />
        {turn.outcome === "answered" && (
          <button type="button" className="icon-btn" aria-label={copied ? "已复制" : "复制回答"} title={copied ? "已复制" : "复制"} onClick={copy}>
            <Icon name={copied ? "check" : "copy"} />
          </button>
        )}
        <button
          type="button"
          className={`icon-btn ${turn.feedback === "up" ? "on" : ""}`}
          aria-label="有帮助"
          aria-pressed={turn.feedback === "up"}
          onClick={() => toggleFeedback("up")}
        >
          <Icon name="up" />
        </button>
        <button
          type="button"
          className={`icon-btn ${turn.feedback === "down" ? "on" : ""}`}
          aria-label="没帮助"
          aria-pressed={turn.feedback === "down"}
          onClick={() => toggleFeedback("down")}
        >
          <Icon name="down" />
        </button>
      </div>
    </article>
  );
}

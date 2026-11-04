import type { StatusUpdate } from "../api/types";
import { phaseLabel } from "../lib/phases";
import { useApp } from "../state/AppContext";
import { Icon } from "./Icon";

export interface PendingState {
  question: string;
  statuses: StatusUpdate[];
  answer: string;
  error: { code: string; message: string } | null;
}

interface PendingTurnProps {
  pending: PendingState;
  onRetry(): void;
  onCancel(): void;
}

export function PendingTurn({ pending, onRetry, onCancel }: PendingTurnProps) {
  const { kbName } = useApp();
  // 连续相同的文案只显示一次，未知阶段直接忽略。
  const steps: string[] = [];
  for (const status of pending.statuses) {
    const label = phaseLabel(status, kbName);
    if (label && steps.at(-1) !== label) steps.push(label);
  }
  const running = !pending.error;

  return (
    <article className="turn" aria-label="正在回答" aria-busy={running}>
      <div className="question">{pending.question}</div>
      <div className="process" aria-live="polite">
        <ul className="process-steps" style={{ margin: 0 }}>
          {steps.length === 0 && running && (
            <li><span className="step-dot"><span className="spinner" /></span>正在连接</li>
          )}
          {steps.map((step, index) => {
            const last = index === steps.length - 1;
            return (
              <li key={`${index}-${step}`}>
                <span className={`step-dot ${last && running ? "" : "done"}`}>
                  {last && running ? <span className="spinner" /> : <Icon name="check" size={14} />}
                </span>
                {step}
              </li>
            );
          })}
        </ul>
        {running && (
          <div style={{ marginTop: 4 }}>
            <button type="button" className="btn btn-ghost" style={{ padding: "2px 6px", fontSize: 12 }} onClick={onCancel}>
              停止
            </button>
          </div>
        )}
      </div>
      {pending.answer && <div className="answer">{pending.answer}</div>}
      {pending.error && (
        <div className="error-card" role="alert">
          <Icon name="alert" />
          <span className="grow">{pending.error.message}</span>
          <button type="button" className="btn" onClick={onRetry}>重试</button>
        </div>
      )}
    </article>
  );
}

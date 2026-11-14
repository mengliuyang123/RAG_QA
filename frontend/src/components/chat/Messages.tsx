import {
  ActionBarPrimitive,
  MessagePrimitive,
  useAui,
  useAuiState,
  type TextMessagePartProps,
} from "@assistant-ui/react";
import { createContext, useContext } from "react";
import type { Citation, FeedbackRating, Turn } from "../../api/types";
import { phaseLabel } from "../../lib/phases";
import { useRagThread, type AssistantCustom, type PendingState } from "../../runtime/RagRuntime";
import { useApp } from "../../state/AppContext";
import { AnswerText, PendingAnswer } from "../AnswerText";
import { Icon } from "../Icon";
import { GROUNDING, TurnBody } from "../TurnView";

export function UserMessage() {
  return (
    <MessagePrimitive.Root className="question">
      <MessagePrimitive.Parts />
    </MessagePrimitive.Root>
  );
}

// 正文分片由 assistant-ui 的 MessagePrimitive.Parts 渲染；[n] 引用需要本轮的来源列表，
// 通过这个上下文传给 Text 组件。
const AnswerContext = createContext<{
  citations: Citation[];
  activePosition: number | null;
  onCite(citation: Citation): void;
} | null>(null);

function CitedText({ text }: TextMessagePartProps) {
  const context = useContext(AnswerContext);
  if (!context) return <div className="answer"><PendingAnswer text={text} /></div>;
  return (
    <AnswerText
      answer={text}
      citations={context.citations}
      activePosition={context.activePosition}
      onCite={context.onCite}
    />
  );
}

function StreamingText({ text }: TextMessagePartProps) {
  return <div className="answer"><PendingAnswer text={text} /></div>;
}

function FeedbackButtons({ turn }: { turn: Turn }) {
  const { setFeedback } = useRagThread();
  // 后端支持"取消反馈"（rating=null），assistant-ui 内置的 FeedbackPositive/Negative
  // 只能提交不能撤销，所以这里用自定义按钮实现切换。
  const toggle = (rating: FeedbackRating) =>
    void setFeedback(turn.id, turn.feedback === rating ? null : rating);
  return (
    <>
      <button
        type="button"
        className={`icon-btn ${turn.feedback === "up" ? "on" : ""}`}
        aria-label="有帮助"
        aria-pressed={turn.feedback === "up"}
        onClick={() => toggle("up")}
      >
        <Icon name="up" />
      </button>
      <button
        type="button"
        className={`icon-btn ${turn.feedback === "down" ? "on" : ""}`}
        aria-label="没帮助"
        aria-pressed={turn.feedback === "down"}
        onClick={() => toggle("down")}
      >
        <Icon name="down" />
      </button>
    </>
  );
}

function CopyButton() {
  const copied = useAuiState((state) => state.message.isCopied);
  return (
    <ActionBarPrimitive.Copy
      className="icon-btn"
      copiedDuration={1500}
      aria-label={copied ? "已复制" : "复制回答"}
      title={copied ? "已复制" : "复制"}
    >
      <Icon name={copied ? "check" : "copy"} />
    </ActionBarPrimitive.Copy>
  );
}

function AnswerMessage({ turn }: { turn: Turn }) {
  const { activeCitation, setActiveCitation, send } = useRagThread();
  const grounding = GROUNDING[turn.grounding];

  return (
    <>
      <TurnBody
        turn={turn}
        activeCitation={activeCitation}
        onCite={setActiveCitation}
        onAsk={(question) => send(question)}
        onRetryWithScope={(scope, question) => send(question, scope)}
        renderAnswer={(activePosition) => (
          <AnswerContext.Provider value={{ citations: turn.citations, activePosition, onCite: setActiveCitation }}>
            <MessagePrimitive.Parts components={{ Text: CitedText }} />
          </AnswerContext.Provider>
        )}
      />
      <ActionBarPrimitive.Root className="turn-actions">
        {turn.outcome === "answered" && (
          <span className={`chip ${grounding.className}`}>{grounding.text}</span>
        )}
        {turn.outcome === "refused" && <span className="chip chip-warning">资料不足</span>}
        <span className="grow" />
        {turn.outcome === "answered" && <CopyButton />}
        <FeedbackButtons turn={turn} />
      </ActionBarPrimitive.Root>
    </>
  );
}

function PendingMessage({ pending }: { pending: PendingState }) {
  const { kbName } = useApp();
  const aui = useAui();
  // 连续相同的文案只显示一次，未知阶段直接忽略。
  const steps: string[] = [];
  for (const status of pending.statuses) {
    const label = phaseLabel(status, kbName);
    if (label && steps.at(-1) !== label) steps.push(label);
  }
  const running = !pending.error;

  return (
    <>
      <div className="process" aria-live="polite" aria-busy={running}>
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
            <button
              type="button"
              className="btn btn-ghost"
              style={{ padding: "2px 6px", fontSize: 12 }}
              onClick={() => aui.thread().cancelRun()}
            >
              停止
            </button>
          </div>
        )}
      </div>
      <MessagePrimitive.Parts components={{ Text: StreamingText }} />
      {pending.error && (
        <div className="error-card" role="alert">
          <Icon name="alert" />
          <span className="grow">{pending.error.message}</span>
          <ActionBarPrimitive.Root>
            <ActionBarPrimitive.Reload className="btn">重试</ActionBarPrimitive.Reload>
          </ActionBarPrimitive.Root>
        </div>
      )}
    </>
  );
}

export function AssistantMessage() {
  const custom = useAuiState((state) => state.message.metadata.custom) as AssistantCustom | undefined;
  return (
    <MessagePrimitive.Root className="turn" aria-label={custom?.kind === "pending" ? "正在回答" : "回答"}>
      {custom?.kind === "answer" && <AnswerMessage turn={custom.turn} />}
      {custom?.kind === "pending" && <PendingMessage pending={custom.pending} />}
    </MessagePrimitive.Root>
  );
}

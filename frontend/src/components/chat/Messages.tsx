import {
  ActionBarPrimitive,
  AuiIf,
  MessagePrimitive,
  useAui,
  useAuiState,
  type TextMessagePartProps,
} from "@assistant-ui/react";
import {
  CheckIcon,
  CopyIcon,
  Loader2Icon,
  RefreshCwIcon,
  SquareIcon,
  ThumbsDownIcon,
  ThumbsUpIcon,
  TriangleAlertIcon,
} from "lucide-react";
import { createContext, useContext } from "react";
import type { Citation, FeedbackRating, Turn } from "../../api/types";
import { visibleStatuses } from "../../lib/generative";
import { phaseLabel } from "../../lib/phases";
import { useRagThread, type AssistantCustom, type PendingState } from "../../runtime/RagRuntime";
import { useApp } from "../../state/AppContext";
import { AnswerText, PendingAnswer } from "../AnswerText";
import { GROUNDING, TurnBody } from "../TurnView";
import { DATA_UIS, TOOL_UIS } from "./GenerativeParts";
import { Button } from "@/components/ui/button";
import { TooltipIconButton } from "@/components/ui/tooltip-icon-button";
import { cn } from "@/lib/utils";

// 消息样式改写自 assistant-ui 官方 registry 的 thread 组件（shadcn 风格）。

export function UserMessage() {
  return (
    <MessagePrimitive.Root
      data-role="user"
      className="grid auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] content-start gap-y-2 px-2 duration-150 animate-in fade-in slide-in-from-bottom-1"
    >
      <div className="col-start-2 min-w-0 rounded-2xl bg-muted px-4 py-2 break-words whitespace-pre-wrap text-foreground">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
}

// 正文分片由 MessagePrimitive.Parts 渲染；[n] 引用需要本轮的来源列表，通过上下文传给 Text 组件。
const AnswerContext = createContext<{
  citations: Citation[];
  activePosition: number | null;
  onCite(citation: Citation): void;
  /** 拒答和澄清轮次的正文由卡片表达，片段里的文本不再渲染。 */
  showText: boolean;
} | null>(null);

function CitedText({ text }: TextMessagePartProps) {
  const context = useContext(AnswerContext);
  if (!context) return <div className="answer"><PendingAnswer text={text} /></div>;
  if (!context.showText) return null;
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
  // 后端支持"取消反馈"（rating=null），内置的 FeedbackPositive/Negative 只能提交不能撤销，所以用自定义按钮。
  const toggle = (rating: FeedbackRating) =>
    void setFeedback(turn.id, turn.feedback === rating ? null : rating);
  const on = "bg-accent text-accent-foreground";
  return (
    <>
      <TooltipIconButton
        tooltip="有帮助"
        aria-label="有帮助"
        aria-pressed={turn.feedback === "up"}
        className={cn(turn.feedback === "up" && on)}
        onClick={() => toggle("up")}
      >
        <ThumbsUpIcon />
      </TooltipIconButton>
      <TooltipIconButton
        tooltip="没帮助"
        aria-label="没帮助"
        aria-pressed={turn.feedback === "down"}
        className={cn(turn.feedback === "down" && on)}
        onClick={() => toggle("down")}
      >
        <ThumbsDownIcon />
      </TooltipIconButton>
    </>
  );
}

function AnswerMessage({ turn }: { turn: Turn }) {
  const { activeCitation, setActiveCitation, send } = useRagThread();
  const grounding = GROUNDING[turn.grounding];

  return (
    <>
      <div className="px-2 leading-relaxed break-words text-foreground">
        <TurnBody
          turn={turn}
          activeCitation={activeCitation}
          onCite={setActiveCitation}
          onAsk={(question) => send(question)}
          onRetryWithScope={(scope, question) => send(question, scope)}
          renderParts={(activePosition) => (
            <AnswerContext.Provider
              value={{
                citations: turn.citations,
                activePosition,
                onCite: setActiveCitation,
                showText: turn.outcome === "answered",
              }}
            >
              <MessagePrimitive.Parts components={{ Text: CitedText, tools: TOOL_UIS, data: DATA_UIS }} />
            </AnswerContext.Provider>
          )}
        />
      </div>
      <div className="ms-2 mt-2 flex min-h-7 items-center gap-2">
        {turn.outcome === "answered" && (
          <span className={cn("rounded-md px-1.5 py-0.5 text-xs", grounding.className)}>{grounding.text}</span>
        )}
        {turn.outcome === "refused" && (
          <span className="rounded-md bg-amber-500/10 px-1.5 py-0.5 text-xs text-amber-700 dark:text-amber-300">
            资料不足
          </span>
        )}
        <ActionBarPrimitive.Root
          hideWhenRunning
          autohide="not-last"
          className="flex gap-1 text-muted-foreground duration-200 animate-in fade-in"
        >
          {turn.outcome === "answered" && (
            <ActionBarPrimitive.Copy asChild copiedDuration={1500}>
              <TooltipIconButton tooltip="复制" aria-label="复制回答">
                <AuiIf condition={(state) => state.message.isCopied}>
                  <CheckIcon className="duration-200 animate-in fade-in zoom-in-50" />
                </AuiIf>
                <AuiIf condition={(state) => !state.message.isCopied}>
                  <CopyIcon className="duration-150 animate-in fade-in zoom-in-75" />
                </AuiIf>
              </TooltipIconButton>
            </ActionBarPrimitive.Copy>
          )}
          <FeedbackButtons turn={turn} />
        </ActionBarPrimitive.Root>
      </div>
    </>
  );
}

function PendingMessage({ pending }: { pending: PendingState }) {
  const { kbName } = useApp();
  const aui = useAui();
  // 连续相同的文案只显示一次，未知阶段直接忽略。
  const steps: string[] = [];
  // 本轮有检索卡片时，检索类阶段由卡片表达，这里只保留排队、理解、核对、整理回答等阶段。
  for (const status of visibleStatuses(pending.statuses, pending.toolCalls)) {
    const label = phaseLabel(status, kbName);
    if (label && steps.at(-1) !== label) steps.push(label);
  }
  const running = !pending.error;

  return (
    <div className="px-2 leading-relaxed break-words text-foreground">
      <div className="mb-2 flex flex-col gap-1 text-sm text-muted-foreground" aria-live="polite" aria-busy={running}>
        {steps.length === 0 && running && pending.toolCalls.length === 0 && (
          <div className="flex items-center gap-2">
            <Loader2Icon className="size-4 animate-spin" />
            <span className="animate-pulse">正在连接</span>
          </div>
        )}
        {steps.map((step, index) => {
          const current = index === steps.length - 1 && running;
          return (
            <div key={`${index}-${step}`} className="flex items-center gap-2 duration-200 animate-in fade-in">
              {current
                ? <Loader2Icon className="size-4 animate-spin" />
                : <CheckIcon className="size-4" />}
              <span className={cn(current && "text-foreground")}>{step}</span>
            </div>
          );
        })}
        {running && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 w-fit gap-1.5 px-2 text-xs"
            onClick={() => aui.thread().cancelRun()}
          >
            <SquareIcon className="size-3 fill-current" />
            停止
          </Button>
        )}
      </div>
      <MessagePrimitive.Parts components={{ Text: StreamingText, tools: TOOL_UIS, data: DATA_UIS }} />
      {running && pending.answer === "" && steps.length > 0 && (
        <span className="animate-pulse font-sans" aria-hidden>●</span>
      )}
      {pending.error && (
        <div
          role="alert"
          className={cn(
            "mt-2 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md border p-3 text-sm",
            pending.error.code === "cancelled"
              ? "border-border bg-muted/40 text-muted-foreground"
              : "border-destructive bg-destructive/10 text-destructive dark:bg-destructive/5 dark:text-red-200",
          )}
        >
          <TriangleAlertIcon className="size-4 shrink-0" />
          <span className="min-w-0 flex-1">{pending.error.message}</span>
          <ActionBarPrimitive.Root>
            <ActionBarPrimitive.Reload asChild>
              <Button variant="outline" size="sm" className="h-7 gap-1.5 bg-transparent">
                <RefreshCwIcon className="size-3.5" />
                重试
              </Button>
            </ActionBarPrimitive.Reload>
          </ActionBarPrimitive.Root>
        </div>
      )}
    </div>
  );
}

export function AssistantMessage() {
  const custom = useAuiState((state) => state.message.metadata.custom) as AssistantCustom | undefined;
  return (
    <MessagePrimitive.Root
      data-role="assistant"
      className="relative duration-150 animate-in fade-in slide-in-from-bottom-1"
      aria-label={custom?.kind === "pending" ? "正在回答" : "回答"}
    >
      {custom?.kind === "answer" && <AnswerMessage turn={custom.turn} />}
      {custom?.kind === "pending" && <PendingMessage pending={custom.pending} />}
    </MessagePrimitive.Root>
  );
}

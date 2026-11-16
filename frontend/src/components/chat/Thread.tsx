import { AuiIf, ThreadPrimitive, useAuiState } from "@assistant-ui/react";
import { ArrowDownIcon, BookOpenIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../../api";
import type { SuggestedQuestion } from "../../api/types";
import { useRagThread } from "../../runtime/RagRuntime";
import { useApp } from "../../state/AppContext";
import { Skeleton } from "@/components/ui/skeleton";
import { TooltipIconButton } from "@/components/ui/tooltip-icon-button";
import { cn } from "@/lib/utils";
import { Composer } from "./Composer";
import { AssistantMessage, UserMessage } from "./Messages";

// 布局改写自 assistant-ui 官方 registry 的 Thread：新对话时输入框居中，
// 有消息后输入框吸底，消息列宽 44rem。

function Welcome() {
  return (
    <div className="mb-6 flex flex-col px-2">
      <p className="m-0 text-2xl font-medium tracking-tight duration-200 animate-in fade-in slide-in-from-bottom-1 fill-mode-both">
        想了解什么？
      </p>
      <p className="m-0 mt-1 text-muted-foreground duration-200 animate-in fade-in slide-in-from-bottom-1 fill-mode-both [animation-delay:50ms]">
        回答都来自团队知识库，并标注出处
      </p>
    </div>
  );
}

// 欢迎页推荐问题：按知识库分组；知识库卡片下的问题用自动范围提问，与输入框所选范围无关。
function Suggestions() {
  const { knowledgeBases } = useApp();
  const { send } = useRagThread();
  const [suggestions, setSuggestions] = useState<SuggestedQuestion[]>([]);

  useEffect(() => {
    api.suggestedQuestions().then(setSuggestions).catch(() => setSuggestions([]));
  }, []);

  const ungrouped = suggestions.filter((item) => !item.kbId || !knowledgeBases.some((kb) => kb.id === item.kbId));
  const item = "group flex w-full items-baseline gap-2.5 rounded-md px-2 py-1.5 text-start text-sm transition-colors outline-none hover:bg-foreground/[0.03] focus-visible:ring-1 focus-visible:ring-ring/50";
  const marker = (
    <span aria-hidden className="font-mono text-xs text-muted-foreground/60 transition-colors group-hover:text-foreground">
      {">"}
    </span>
  );

  return (
    <div className="grid w-full gap-3 sm:grid-cols-2">
      {knowledgeBases.map((kb, index) => {
        const questions = suggestions.filter((entry) => entry.kbId === kb.id).slice(0, 3);
        return (
          <section
            key={kb.id}
            aria-label={kb.name}
            className="rounded-xl border p-3 duration-200 animate-in fade-in slide-in-from-bottom-2 fill-mode-both"
            style={{ animationDelay: `${index * 40}ms` }}
          >
            <div className="flex items-center gap-2 px-2">
              <BookOpenIcon className="size-4 text-muted-foreground" />
              <h3 className="m-0 text-sm font-medium">{kb.name}</h3>
              <span className="ms-auto text-xs text-muted-foreground">{kb.documentsCount} 篇文档</span>
            </div>
            {kb.description && (
              <p className="m-0 mt-1 line-clamp-2 px-2 text-xs text-muted-foreground">{kb.description}</p>
            )}
            {questions.length > 0 && (
              <div className="mt-2 flex flex-col">
                {questions.map((entry) => (
                  <button
                    key={entry.id}
                    type="button"
                    className={item}
                    onClick={() => send(entry.text, { mode: "auto", kbIds: [] })}
                  >
                    {marker}
                    <span className="min-w-0 flex-1 truncate">{entry.text}</span>
                  </button>
                ))}
              </div>
            )}
          </section>
        );
      })}
      {ungrouped.length > 0 && (
        <section aria-label="试试这些问题" className="rounded-xl border p-3">
          <h3 className="m-0 px-2 text-sm font-medium">试试这些问题</h3>
          <div className="mt-2 flex flex-col">
            {ungrouped.slice(0, 4).map((entry) => (
              <ThreadPrimitive.Suggestion key={entry.id} prompt={entry.text} send className={item}>
                {marker}
                <span className="min-w-0 flex-1 truncate">{entry.text}</span>
              </ThreadPrimitive.Suggestion>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function HistorySkeleton() {
  return (
    <div role="status" className="flex flex-col gap-y-6 duration-200 animate-in fade-in fill-mode-both [animation-delay:150ms]">
      <span className="sr-only">正在加载对话</span>
      <Skeleton className="ml-auto h-9 w-2/5 rounded-xl" />
      <div className="flex flex-col gap-y-2">
        <Skeleton className="h-4 w-11/12" />
        <Skeleton className="h-4 w-4/5" />
        <Skeleton className="h-4 w-3/5" />
      </div>
    </div>
  );
}

export function Thread() {
  const { conversationId, conversation } = useRagThread();
  const hasMessages = useAuiState((state) => state.thread.messages.length > 0);
  const isNewChat = conversationId === null && !hasMessages;
  const isLoading = conversationId !== null && !conversation;

  return (
    <ThreadPrimitive.Root
      className="flex h-full min-h-0 flex-col bg-background"
      style={{ ["--thread-max-width" as string]: "44rem" }}
    >
      <ThreadPrimitive.Viewport
        turnAnchor="top"
        className="relative flex flex-1 flex-col overflow-x-auto overflow-y-scroll scroll-smooth"
      >
        <div
          className={cn(
            "mx-auto flex w-full max-w-(--thread-max-width) flex-1 flex-col px-4 pt-4",
            isNewChat && "justify-center",
          )}
        >
          {isNewChat && <Welcome />}
          {isLoading && <HistorySkeleton />}
          {conversation && conversation.turns.length === 0 && !hasMessages && (
            <div className="py-10 text-center text-muted-foreground">输入问题开始对话</div>
          )}

          <div className="mb-14 flex flex-col gap-y-6 empty:hidden">
            <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
          </div>

          <ThreadPrimitive.ViewportFooter
            className={cn(
              "flex flex-col gap-4 overflow-visible bg-background pb-4 md:pb-6",
              !isNewChat && "sticky bottom-0 mt-auto rounded-t-2xl",
            )}
          >
            <ThreadPrimitive.ScrollToBottom asChild>
              <TooltipIconButton
                tooltip="回到底部"
                variant="outline"
                className="absolute -top-12 z-10 self-center rounded-full p-4 disabled:invisible dark:border-border dark:bg-background dark:hover:bg-accent"
              >
                <ArrowDownIcon />
              </TooltipIconButton>
            </ThreadPrimitive.ScrollToBottom>
            <Composer
              placeholder={isNewChat ? "比如：服务启动报数据库锁定怎么办" : "继续追问"}
              autoFocus={isNewChat}
            />
            <AuiIf condition={(state) => state.thread.isRunning}>
              <div className="-mt-2 px-2 text-xs text-muted-foreground">正在回答，完成后可以继续提问</div>
            </AuiIf>
            {isNewChat && <Suggestions />}
          </ThreadPrimitive.ViewportFooter>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}

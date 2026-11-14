import { AuiIf, ComposerPrimitive, useAuiState } from "@assistant-ui/react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { api } from "../../api";
import type { KbScope } from "../../api/types";
import { MAX_QUESTION_LENGTH, useRagThread } from "../../runtime/RagRuntime";
import { useApp } from "../../state/AppContext";
import { Button } from "@/components/ui/button";
import { TooltipIconButton } from "@/components/ui/tooltip-icon-button";
import { cn } from "@/lib/utils";

function ScopePicker({ scope, onChange }: { scope: KbScope; onChange(scope: KbScope): void }) {
  const { knowledgeBases } = useApp();
  if (!api.supports.kbScope || knowledgeBases.length <= 1) return null;

  const toggleKb = (kbId: string) => {
    const selected = scope.mode === "explicit" ? scope.kbIds : [];
    const next = selected.includes(kbId)
      ? selected.filter((id) => id !== kbId)
      : [...selected, kbId];
    onChange(next.length === 0 ? { mode: "auto", kbIds: [] } : { mode: "explicit", kbIds: next });
  };

  const chip = (on: boolean) => cn(
    "h-7 rounded-full border px-2.5 text-xs transition-colors",
    on
      ? "border-foreground/20 bg-foreground text-background"
      : "border-foreground/10 text-muted-foreground hover:bg-muted hover:text-foreground",
  );

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-1.5" role="group" aria-label="查询范围">
      <button
        type="button"
        className={chip(scope.mode === "auto")}
        aria-pressed={scope.mode === "auto"}
        onClick={() => onChange({ mode: "auto", kbIds: [] })}
      >
        自动
      </button>
      {knowledgeBases.map((kb) => {
        const on = scope.mode === "explicit" && scope.kbIds.includes(kb.id);
        return (
          <button key={kb.id} type="button" className={chip(on)} aria-pressed={on} onClick={() => toggleKb(kb.id)}>
            {kb.name}
          </button>
        );
      })}
    </div>
  );
}

// 外观改写自 assistant-ui 官方 registry 的 Composer。回车发送、Shift+回车换行、
// 输入法组字、自动增高由 ComposerPrimitive 处理；超长在运行时 onNew 中拦截并保留草稿。
export function Composer({ placeholder, autoFocus }: { placeholder: string; autoFocus?: boolean }) {
  const { scope, setScope, composerError, clearComposerError } = useRagThread();
  const length = useAuiState((state) => state.composer.text.length);
  const over = length > MAX_QUESTION_LENGTH;
  const message = composerError ?? (over ? `问题最多 ${MAX_QUESTION_LENGTH} 字` : null);

  return (
    <div className="flex w-full flex-col gap-1.5">
      <ComposerPrimitive.Root
        className="relative flex w-full flex-col"
        onChange={() => composerError && clearComposerError()}
      >
        <div className="flex w-full cursor-text flex-col gap-2 rounded-2xl border border-foreground/10 bg-[color-mix(in_oklab,var(--color-muted)_30%,transparent)] p-2 transition-[border-color] focus-within:border-foreground/25">
          <ComposerPrimitive.Input
            id="question-input"
            aria-label="问题"
            rows={1}
            placeholder={placeholder}
            autoFocus={autoFocus}
            submitMode="enter"
            enterKeyHint="send"
            className="max-h-48 min-h-10 w-full resize-none bg-transparent px-2.5 py-1 text-base leading-6 caret-primary outline-none placeholder:text-muted-foreground/60"
          />
          <div className="flex items-end justify-between gap-2">
            <ScopePicker scope={scope} onChange={setScope} />
            <div className="ms-auto flex shrink-0 items-center gap-1.5">
              {length > MAX_QUESTION_LENGTH * 0.8 && (
                <span className={cn("text-xs text-muted-foreground", over && "text-destructive")}>
                  {length}/{MAX_QUESTION_LENGTH}
                </span>
              )}
              <AuiIf condition={(state) => !state.thread.isRunning}>
                <ComposerPrimitive.Send asChild disabled={over}>
                  <TooltipIconButton
                    tooltip="发送"
                    side="bottom"
                    type="button"
                    variant="default"
                    size="icon"
                    className="size-7 rounded-full"
                    aria-label="发送"
                  >
                    <ArrowUpIcon className="size-4" />
                  </TooltipIconButton>
                </ComposerPrimitive.Send>
              </AuiIf>
              <AuiIf condition={(state) => state.thread.isRunning}>
                <ComposerPrimitive.Cancel asChild>
                  <Button type="button" variant="default" size="icon" className="size-7 rounded-full" aria-label="停止回答">
                    <SquareIcon className="size-3.5 fill-current" />
                  </Button>
                </ComposerPrimitive.Cancel>
              </AuiIf>
            </div>
          </div>
        </div>
      </ComposerPrimitive.Root>
      {message && <div role="alert" className="px-2 text-sm text-destructive">{message}</div>}
    </div>
  );
}

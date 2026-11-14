import { FileTextIcon, XIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { ChunkContext, Citation } from "../api/types";
import { useApp } from "../state/AppContext";
import { TooltipIconButton } from "@/components/ui/tooltip-icon-button";
import { cn } from "@/lib/utils";

interface SourcePanelProps {
  citation: Citation;
  onClose(): void;
}

export function SourcePanel({ citation, onClose }: SourcePanelProps) {
  const { kbName } = useApp();
  const [context, setContext] = useState<ChunkContext | null>(null);
  const [error, setError] = useState<string | null>(null);
  const target = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setContext(null);
    setError(null);
    api.getChunkContext(citation)
      .then((result) => { if (!cancelled) setContext(result); })
      .catch((reason: Error) => {
        if (cancelled) return;
        // 文档已删除等情况下，退回回答里保存的引用正文。
        setError(reason.message);
        setContext({
          docId: citation.docId,
          filename: citation.filename,
          kbId: citation.kbId,
          docUpdatedAt: citation.docUpdatedAt,
          chunks: [{ chunkId: citation.chunkId ?? "", page: citation.page, text: citation.chunkText, isTarget: true }],
        });
      });
    return () => { cancelled = true; };
  }, [citation]);

  useEffect(() => {
    target.current?.scrollIntoView({ block: "center" });
  }, [context]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const updated = context?.docUpdatedAt ? new Date(context.docUpdatedAt).toLocaleDateString("zh-CN") : null;
  const meta = [
    citation.kbId ? kbName(citation.kbId) : null,
    citation.page !== null ? `第 ${citation.page} 页` : null,
    updated ? `更新于 ${updated}` : null,
  ].filter(Boolean).join(" · ");

  return (
    <aside
      aria-label={`来源 ${citation.position}`}
      className="fixed inset-0 z-30 flex min-h-0 flex-col bg-background duration-200 animate-in fade-in slide-in-from-right-4 lg:static lg:z-auto lg:border-l"
    >
      <div className="flex h-12 shrink-0 items-center gap-2 border-b px-4">
        <span className="text-sm font-medium">来源 [{citation.position}]</span>
        <TooltipIconButton tooltip="关闭" aria-label="关闭来源面板" className="ms-auto size-7" onClick={onClose}>
          <XIcon />
        </TooltipIconButton>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        <div className="flex items-start gap-2">
          <FileTextIcon className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
          <div className="min-w-0">
            <div className="font-medium break-all">{citation.filename}</div>
            {meta && <div className="mt-0.5 text-xs text-muted-foreground">{meta}</div>}
          </div>
        </div>
        {error && (
          <div className="mt-3 rounded-md border border-amber-500/30 bg-amber-500/5 p-2.5 text-xs text-amber-800 dark:text-amber-200">
            {error}，以下为回答时保存的原文片段。
          </div>
        )}
        {!context && <div className="mt-4 text-sm text-muted-foreground">正在加载原文…</div>}
        <div className="mt-4 flex flex-col gap-2">
          {context?.chunks.map((chunk) => (
            <div
              key={chunk.chunkId}
              ref={chunk.isTarget ? target : undefined}
              className={cn(
                "rounded-lg p-3 text-sm leading-relaxed whitespace-pre-wrap",
                chunk.isTarget
                  ? "border border-amber-500/40 bg-amber-500/10 text-foreground"
                  : "text-muted-foreground",
              )}
            >
              {chunk.text}
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

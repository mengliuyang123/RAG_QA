import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { ChunkContext, Citation } from "../api/types";
import { useApp } from "../state/AppContext";
import { Icon } from "./Icon";

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

  return (
    <aside className="source-panel" aria-label={`来源 ${citation.position}`}>
      <div className="source-head">
        <span className="title">来源 [{citation.position}]</span>
        <button type="button" className="icon-btn" aria-label="关闭来源面板" onClick={onClose}>
          <Icon name="x" />
        </button>
      </div>
      <div className="source-body">
        <div style={{ fontWeight: 500, wordBreak: "break-all" }}>{citation.filename}</div>
        <div className="source-meta">
          {[
            citation.kbId ? kbName(citation.kbId) : null,
            citation.page !== null ? `第 ${citation.page} 页` : null,
            updated ? `更新于 ${updated}` : null,
          ].filter(Boolean).join(" · ")}
        </div>
        {error && <div className="banner" style={{ marginBottom: 10 }}>{error}，以下为回答时保存的原文片段。</div>}
        {!context && <div className="muted">正在加载原文…</div>}
        {context?.chunks.map((chunk) => (
          <div
            key={chunk.chunkId}
            ref={chunk.isTarget ? target : undefined}
            className={`source-chunk ${chunk.isTarget ? "target" : ""}`}
          >
            {chunk.text}
          </div>
        ))}
      </div>
    </aside>
  );
}

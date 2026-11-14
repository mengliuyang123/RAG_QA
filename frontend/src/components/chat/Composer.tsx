import { AuiIf, ComposerPrimitive, useAuiState } from "@assistant-ui/react";
import { api } from "../../api";
import type { KbScope } from "../../api/types";
import { MAX_QUESTION_LENGTH, useRagThread } from "../../runtime/RagRuntime";
import { useApp } from "../../state/AppContext";
import { Icon } from "../Icon";

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

  return (
    <>
      <span className="muted" style={{ fontSize: 12 }}>查询范围</span>
      <button
        type="button"
        className={`scope-chip ${scope.mode === "auto" ? "on" : ""}`}
        aria-pressed={scope.mode === "auto"}
        onClick={() => onChange({ mode: "auto", kbIds: [] })}
      >
        自动
      </button>
      {knowledgeBases.map((kb) => {
        const on = scope.mode === "explicit" && scope.kbIds.includes(kb.id);
        return (
          <button
            key={kb.id}
            type="button"
            className={`scope-chip ${on ? "on" : ""}`}
            aria-pressed={on}
            onClick={() => toggleKb(kb.id)}
          >
            {kb.name}
          </button>
        );
      })}
    </>
  );
}

// 基于 assistant-ui ComposerPrimitive：回车发送、Shift+回车换行、输入法组字时不发送、
// 自动增高都由框架处理；超长校验在运行时 onNew 中拦截并保留草稿。
export function Composer({ placeholder, autoFocus }: { placeholder: string; autoFocus?: boolean }) {
  const { scope, setScope, composerError, clearComposerError } = useRagThread();
  const length = useAuiState((state) => state.composer.text.length);
  const over = length > MAX_QUESTION_LENGTH;

  return (
    <div>
      <ComposerPrimitive.Root className="composer" onChange={() => composerError && clearComposerError()}>
        <label className="sr-only" htmlFor="question-input">问题</label>
        <ComposerPrimitive.Input
          id="question-input"
          rows={1}
          maxRows={8}
          placeholder={placeholder}
          autoFocus={autoFocus}
          submitMode="enter"
        />
        <div className="composer-bar">
          <ScopePicker scope={scope} onChange={setScope} />
          <span className="spacer" />
          {length > MAX_QUESTION_LENGTH * 0.8 && (
            <span className={`char-count ${over ? "over" : ""}`}>
              {length}/{MAX_QUESTION_LENGTH}
            </span>
          )}
          <AuiIf condition={(state) => !state.thread.isRunning}>
            <ComposerPrimitive.Send className="btn btn-primary btn-icon" aria-label="发送" disabled={over}>
              <Icon name="send" />
            </ComposerPrimitive.Send>
          </AuiIf>
          <AuiIf condition={(state) => state.thread.isRunning}>
            <ComposerPrimitive.Cancel className="btn btn-icon" aria-label="停止回答" title="停止回答">
              <Icon name="x" />
            </ComposerPrimitive.Cancel>
          </AuiIf>
        </div>
      </ComposerPrimitive.Root>
      {(composerError || over) && (
        <div role="alert" style={{ color: "var(--danger-text)", fontSize: 13, marginTop: 6 }}>
          {composerError ?? `问题最多 ${MAX_QUESTION_LENGTH} 字`}
        </div>
      )}
      <AuiIf condition={(state) => state.thread.isRunning}>
        <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>正在回答，完成后可以继续提问</div>
      </AuiIf>
    </div>
  );
}

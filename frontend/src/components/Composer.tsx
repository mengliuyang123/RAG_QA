import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { KbScope, KnowledgeBase } from "../api/types";
import { Icon } from "./Icon";

export const MAX_QUESTION_LENGTH = 8000;

interface ComposerProps {
  onSubmit(question: string): void;
  busy: boolean;
  scope: KbScope;
  onScopeChange(scope: KbScope): void;
  knowledgeBases: KnowledgeBase[];
  showScope: boolean;
  placeholder: string;
  autoFocus?: boolean;
}

export function Composer({
  onSubmit,
  busy,
  scope,
  onScopeChange,
  knowledgeBases,
  showScope,
  placeholder,
  autoFocus,
}: ComposerProps) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [text]);

  const submit = () => {
    const question = text.trim();
    if (!question) {
      setError("先输入问题");
      return;
    }
    if (question.length > MAX_QUESTION_LENGTH) {
      setError(`问题最多 ${MAX_QUESTION_LENGTH} 字`);
      return;
    }
    if (busy) return;
    onSubmit(question);
    setText("");
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // 输入法组字时回车用于选词，不能触发发送。
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  };

  const toggleKb = (kbId: string) => {
    const selected = scope.mode === "explicit" ? scope.kbIds : [];
    const next = selected.includes(kbId)
      ? selected.filter((id) => id !== kbId)
      : [...selected, kbId];
    onScopeChange(next.length === 0 ? { mode: "auto", kbIds: [] } : { mode: "explicit", kbIds: next });
  };

  const over = text.length > MAX_QUESTION_LENGTH;

  return (
    <div>
      <div className="composer">
        <label className="sr-only" htmlFor="question-input">问题</label>
        <textarea
          id="question-input"
          ref={textarea}
          rows={1}
          value={text}
          placeholder={placeholder}
          autoFocus={autoFocus}
          onChange={(event) => {
            setText(event.target.value);
            if (error) setError(null);
          }}
          onKeyDown={onKeyDown}
        />
        <div className="composer-bar">
          {showScope && knowledgeBases.length > 1 && (
            <>
              <span className="muted" style={{ fontSize: 12 }}>查询范围</span>
              <button
                type="button"
                className={`scope-chip ${scope.mode === "auto" ? "on" : ""}`}
                aria-pressed={scope.mode === "auto"}
                onClick={() => onScopeChange({ mode: "auto", kbIds: [] })}
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
          )}
          <span className="spacer" />
          {text.length > MAX_QUESTION_LENGTH * 0.8 && (
            <span className={`char-count ${over ? "over" : ""}`}>
              {text.length}/{MAX_QUESTION_LENGTH}
            </span>
          )}
          <button
            type="button"
            className="btn btn-primary btn-icon"
            aria-label="发送"
            disabled={busy || !text.trim() || over}
            onClick={submit}
          >
            <Icon name="send" />
          </button>
        </div>
      </div>
      {error && <div role="alert" style={{ color: "var(--danger-text)", fontSize: 13, marginTop: 6 }}>{error}</div>}
      {busy && <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>正在回答，完成后可以继续提问</div>}
    </div>
  );
}

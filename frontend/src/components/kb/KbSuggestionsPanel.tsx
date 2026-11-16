import { useEffect, useState } from "react";
import { api } from "../../api";

// 后端读取接口单次最多返回 20 条，编辑器也限制为 20 条，避免保存时把多出的悄悄删掉。
const MAX_QUESTIONS = 20;

// 推荐问题显示在聊天首页，帮助用户知道这个库能问什么。
export function KbSuggestionsPanel({ kbId }: { kbId: string }) {
  const [text, setText] = useState<string | null>(null);
  const [saved, setSaved] = useState("");
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setText(null);
    setMessage(null);
    api.admin.getSuggestedQuestions(kbId)
      .then((questions) => {
        setText(questions.join("\n"));
        setSaved(questions.join("\n"));
      })
      .catch((error: Error) => setMessage({ kind: "error", text: error.message }));
  }, [kbId]);

  const lines = (text ?? "").split("\n").map((line) => line.trim()).filter(Boolean);
  const tooMany = lines.length > MAX_QUESTIONS;
  const tooLong = lines.some((line) => line.length > 500);

  const save = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const questions = await api.admin.setSuggestedQuestions(kbId, [...new Set(lines)]);
      setText(questions.join("\n"));
      setSaved(questions.join("\n"));
      setMessage({ kind: "ok", text: questions.length ? `已发布 ${questions.length} 个推荐问题` : "已清空推荐问题" });
    } catch (error) {
      setMessage({ kind: "error", text: (error as Error).message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section aria-label="推荐问题">
      <p className="secondary" style={{ marginTop: 0, fontSize: 13 }}>
        每行一个问题，最多 {MAX_QUESTIONS} 个。会显示在聊天首页，提示用户这个知识库能回答什么。
      </p>
      {text === null && !message && <div className="muted">正在加载…</div>}
      {text !== null && (
        <>
          <textarea
            aria-label="推荐问题列表"
            className="field"
            style={{ minHeight: 160 }}
            value={text}
            onChange={(event) => { setText(event.target.value); setMessage(null); }}
          />
          {tooMany && <div role="alert" className="form-error">最多 {MAX_QUESTIONS} 个问题，当前 {lines.length} 个</div>}
          {tooLong && <div role="alert" className="form-error">每个问题不超过 500 个字符</div>}
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
            <button type="button" className="btn btn-primary" disabled={busy || tooMany || tooLong || text === saved} onClick={save}>
              {busy ? "正在保存…" : "发布"}
            </button>
            <span className="muted" style={{ fontSize: 12 }}>{lines.length} 个问题</span>
          </div>
        </>
      )}
      {message && (
        <div role={message.kind === "error" ? "alert" : "status"} className={message.kind === "error" ? "banner" : "chip chip-success"} style={{ marginTop: 10, display: "block", padding: "6px 10px" }}>
          {message.text}
        </div>
      )}
    </section>
  );
}

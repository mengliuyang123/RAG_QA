import { useEffect, useState } from "react";
import { api } from "../../api";
import type { FaqEntry, FaqState } from "../../api/types";

const PUBLICATION_TEXT: Record<NonNullable<FaqState["publication"]>["status"], string> = {
  queued: "已提交发布，入库完成后问答即可使用",
  unchanged: "内容没有变化，无需重新发布",
  removed: "标准答案已清空，对应文档已移除",
  none: "还没有标准答案",
  busy: "上一版仍在入库，请稍后点击“重新发布”",
};

function EntryForm({
  initial, busy, onSubmit, onCancel,
}: {
  initial?: { question: string; answer: string };
  busy: boolean;
  onSubmit(entry: { question: string; answer: string }): void;
  onCancel(): void;
}) {
  const [question, setQuestion] = useState(initial?.question ?? "");
  const [answer, setAnswer] = useState(initial?.answer ?? "");
  const valid = question.trim().length > 0 && answer.trim().length > 0;
  return (
    <div className="card" style={{ marginBottom: 10, padding: 12 }}>
      <label className="field-label" htmlFor="faq-question" style={{ marginTop: 0 }}>问题</label>
      <input id="faq-question" className="text-input" maxLength={500} value={question} onChange={(event) => setQuestion(event.target.value)} />
      <label className="field-label" htmlFor="faq-answer">标准答案</label>
      <textarea id="faq-answer" className="field" maxLength={5000} style={{ minHeight: 100 }} value={answer} onChange={(event) => setAnswer(event.target.value)} />
      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
        <button type="button" className="btn btn-primary" disabled={busy || !valid} onClick={() => onSubmit({ question, answer })}>保存并发布</button>
        <button type="button" className="btn" onClick={onCancel}>取消</button>
      </div>
    </div>
  );
}

// 标准答案发布为该知识库里的一篇「常见问题」文档，问答时作为资料被检索和引用。
export function KbFaqPanel({ kbId }: { kbId: string }) {
  const [state, setState] = useState<FaqState | null>(null);
  const [editing, setEditing] = useState<string | "new" | null>(null);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setState(null);
    setEditing(null);
    setMessage(null);
    api.admin.listFaq(kbId).then(setState).catch((error: Error) => setMessage({ kind: "error", text: error.message }));
  }, [kbId]);

  const run = async (action: () => Promise<FaqState>) => {
    setBusy(true);
    setMessage(null);
    try {
      const next = await action();
      setState(next);
      setEditing(null);
      if (next.publication) setMessage({ kind: next.publication.status === "busy" ? "error" : "ok", text: PUBLICATION_TEXT[next.publication.status] });
    } catch (error) {
      setMessage({ kind: "error", text: (error as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const remove = (entry: FaqEntry) => {
    if (!window.confirm(`删除标准答案“${entry.question}”？`)) return;
    void run(() => api.admin.deleteFaq(kbId, entry.id));
  };

  return (
    <section aria-label="标准答案">
      <p className="secondary" style={{ marginTop: 0, fontSize: 13 }}>
        为高频问题写一个确定的答案。所有标准答案会合成一篇
        {state ? `“${state.filename}”` : "「常见问题」"}
        文档发布到本知识库，问答时被检索和引用；每次修改都会重新发布。
      </p>
      {message && (
        <div role={message.kind === "error" ? "alert" : "status"} className={message.kind === "error" ? "banner" : "chip chip-success"} style={{ marginBottom: 10, display: "block", padding: "6px 10px" }}>
          {message.text}
        </div>
      )}
      {!state && !message && <div className="muted">正在加载…</div>}
      {state && (
        <>
          <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
            {editing === null && <button type="button" className="btn btn-primary" onClick={() => setEditing("new")}>新增标准答案</button>}
            {state.entries.length > 0 && (
              <button type="button" className="btn" disabled={busy} onClick={() => void run(() => api.admin.publishFaq(kbId))}>重新发布</button>
            )}
          </div>
          {editing === "new" && (
            <EntryForm busy={busy} onCancel={() => setEditing(null)} onSubmit={(entry) => void run(() => api.admin.createFaq(kbId, entry))} />
          )}
          {state.entries.length === 0 && editing !== "new" && <div className="empty" style={{ padding: 30 }}>还没有标准答案</div>}
          {state.entries.map((entry) => (
            editing === entry.id ? (
              <EntryForm
                key={entry.id}
                initial={entry}
                busy={busy}
                onCancel={() => setEditing(null)}
                onSubmit={(next) => void run(() => api.admin.updateFaq(kbId, entry.id, next))}
              />
            ) : (
              <article key={entry.id} className="card" aria-label={entry.question} style={{ marginBottom: 8, padding: 12 }}>
                <div style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
                  <strong style={{ flex: 1 }}>{entry.question}</strong>
                  <button type="button" className="btn" disabled={busy || editing !== null} onClick={() => setEditing(entry.id)}>编辑</button>
                  <button type="button" className="btn btn-danger" disabled={busy || editing !== null} onClick={() => remove(entry)}>删除</button>
                </div>
                <div className="secondary" style={{ fontSize: 13, marginTop: 6, whiteSpace: "pre-wrap" }}>{entry.answer}</div>
              </article>
            )
          ))}
        </>
      )}
    </section>
  );
}

import { useEffect, useState } from "react";
import { api } from "../api";
import type { FeedbackItem, FeedbackReason, FeedbackTriage, RefusalRecord } from "../api/types";
import { Topbar } from "../components/Layout";
import { useApp } from "../state/AppContext";
import { formatDate } from "./KnowledgePage";

const TRIAGE_LABELS: Record<FeedbackTriage, string> = {
  retrieval_error: "检索问题",
  generation_error: "生成问题",
  missing_content: "缺内容",
  not_an_issue: "不是问题",
};

const REASON_LABELS: Record<FeedbackReason, string> = {
  incorrect: "不正确",
  incomplete: "不完整",
  irrelevant: "答非所问",
  outdated: "内容过时",
  citation_wrong: "引用有误",
};

type Tab = "refusals" | "feedback";
type TriageFilter = "untriaged" | "all" | FeedbackTriage;

function RefusalList() {
  const { kbName } = useApp();
  const [refusals, setRefusals] = useState<RefusalRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.admin.refusals().then(setRefusals).catch((reason: Error) => setError(reason.message));
  }, []);

  return (
    <>
      <p className="secondary" style={{ marginTop: -6 }}>
        这些问题在知识库里找不到可靠答案，通常意味着需要补充文档。
      </p>
      {error && <div className="banner" role="alert">{error}</div>}
      {!refusals && !error && <div className="muted">正在加载…</div>}
      {refusals && refusals.length === 0 && <div className="empty">最近没有拒答问题</div>}
      {refusals && refusals.length > 0 && (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr><th>问题</th><th>缺少的信息</th><th>知识库</th><th>时间</th></tr>
            </thead>
            <tbody>
              {refusals.map((item) => (
                <tr key={item.id}>
                  <td>{item.question}</td>
                  <td className="secondary">{item.missing.join("；") || "—"}</td>
                  <td>{item.kbIds.map((id) => <span key={id} className="chip" style={{ marginRight: 4 }}>{kbName(id)}</span>)}</td>
                  <td>{formatDate(item.askedAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function FeedbackCard({ item, onChange }: { item: FeedbackItem; onChange(item: FeedbackItem, message: string): void }) {
  const { kbName } = useApp();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [marked, setMarked] = useState<Set<string>>(() => new Set());

  const triage = async (value: FeedbackTriage | null) => {
    setBusy(true);
    setError(null);
    try {
      const saved = await api.admin.triageFeedback(item.id, value, item.revision);
      onChange(saved, value ? `已归类为「${TRIAGE_LABELS[value]}」` : "已清除分类");
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // 用户说内容过时时，可以直接把被引用的文档标记为已过时，它会立即停止参与检索。
  const markOutdated = async (kbId: string, docId: string, filename: string) => {
    setBusy(true);
    setError(null);
    try {
      await api.admin.saveDocumentReview(kbId, docId, {
        reviewDueOn: null, outdated: true, note: item.comment ? `用户反馈：${item.comment}` : "用户反馈内容过时",
      });
      setMarked((current) => new Set(current).add(docId));
      onChange(item, `已将「${filename}」标记为已过时，它不再参与检索`);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const outdated = item.reasons.includes("outdated");
  const docs = [...new Map(item.citations.filter((c) => c.kbId).map((c) => [c.docId, c])).values()];

  return (
    <article className="card" aria-label={`点踩：${item.question}`} style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "baseline", flexWrap: "wrap" }}>
        <strong style={{ flex: 1, minWidth: 200 }}>{item.question}</strong>
        <span className="muted" style={{ fontSize: 12 }}>{formatDate(item.createdAt)}</span>
      </div>
      <div className="secondary" style={{ fontSize: 13, margin: "6px 0", whiteSpace: "pre-wrap" }}>
        {item.answer.length > 300 ? `${item.answer.slice(0, 300)}…` : item.answer}
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center", fontSize: 13 }}>
        {item.reasons.map((reason) => <span key={reason} className={reason === "outdated" ? "chip chip-warning" : "chip"}>{REASON_LABELS[reason]}</span>)}
        {item.comment && <span>“{item.comment}”</span>}
      </div>
      {docs.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 13 }}>
          <span className="muted">引用：</span>
          {docs.map((citation) => (
            <span key={citation.docId} style={{ marginRight: 10 }}>
              {citation.filename}<span className="muted">（{kbName(citation.kbId ?? "")}）</span>
              {outdated && (
                marked.has(citation.docId)
                  ? <span className="chip chip-success" style={{ marginLeft: 4 }}>已标记过时</span>
                  : (
                    <button type="button" className="btn" style={{ marginLeft: 4, padding: "2px 8px" }} disabled={busy}
                      onClick={() => void markOutdated(citation.kbId ?? "", citation.docId, citation.filename)}>
                      标记过时
                    </button>
                  )
              )}
            </span>
          ))}
        </div>
      )}
      <div role="group" aria-label="分类" style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 10 }}>
        {(Object.keys(TRIAGE_LABELS) as FeedbackTriage[]).map((value) => (
          <button
            key={value}
            type="button"
            aria-pressed={item.triage === value}
            className={item.triage === value ? "btn btn-primary" : "btn"}
            disabled={busy}
            onClick={() => void triage(item.triage === value ? null : value)}
          >
            {TRIAGE_LABELS[value]}
          </button>
        ))}
      </div>
      {error && <div role="alert" className="form-error">{error}</div>}
    </article>
  );
}

function FeedbackList() {
  const [items, setItems] = useState<FeedbackItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [filter, setFilter] = useState<TriageFilter>("untriaged");

  useEffect(() => {
    api.admin.listFeedback().then(setItems).catch((reason: Error) => setError(reason.message));
  }, []);

  const matches = (item: FeedbackItem, value: TriageFilter) =>
    value === "all" || (value === "untriaged" ? item.triage === null : item.triage === value);
  const visible = items?.filter((item) => matches(item, filter)) ?? [];
  const filters: TriageFilter[] = ["untriaged", "all", ...(Object.keys(TRIAGE_LABELS) as FeedbackTriage[])];

  return (
    <>
      <p className="secondary" style={{ marginTop: -6 }}>
        给每条点踩归类：检索问题和生成问题需要调整系统，缺内容需要补文档（可到“知识缺口”处理），内容过时可以直接标记被引用的文档。
      </p>
      {error && <div className="banner" role="alert">{error}</div>}
      {message && <div role="status" className="chip chip-success" style={{ marginBottom: 10, display: "block", padding: "6px 10px" }}>{message}</div>}
      <div className="tabs" role="tablist" aria-label="按分类筛选">
        {filters.map((value) => (
          <button key={value} type="button" role="tab" aria-selected={filter === value} className={filter === value ? "btn btn-primary" : "btn"} onClick={() => setFilter(value)}>
            {value === "untriaged" ? "未分类" : value === "all" ? "全部" : TRIAGE_LABELS[value]} {items?.filter((item) => matches(item, value)).length ?? 0}
          </button>
        ))}
      </div>
      {!items && !error && <div className="muted">正在加载…</div>}
      {items && visible.length === 0 && <div className="empty">没有这类点踩</div>}
      {visible.map((item) => (
        <FeedbackCard
          key={item.id}
          item={item}
          onChange={(saved, text) => {
            setMessage(text);
            setItems((current) => current?.map((entry) => (entry.id === saved.id ? saved : entry)) ?? null);
          }}
        />
      ))}
    </>
  );
}

export function AdminInsightsPage() {
  const [tab, setTab] = useState<Tab>("refusals");

  return (
    <>
      <Topbar title="问题洞察" />
      <div className="page">
        <div className="page-inner">
          {!api.supports.insights && (
            <div className="banner" style={{ marginBottom: 12 }}>
              后端尚未提供洞察接口，以下为示例数据。
            </div>
          )}
          <div className="tabs" role="tablist" aria-label="洞察类型">
            <button type="button" role="tab" aria-selected={tab === "refusals"} className={tab === "refusals" ? "btn btn-primary" : "btn"} onClick={() => setTab("refusals")}>拒答问题</button>
            <button type="button" role="tab" aria-selected={tab === "feedback"} className={tab === "feedback" ? "btn btn-primary" : "btn"} onClick={() => setTab("feedback")}>点踩反馈</button>
          </div>
          {tab === "refusals" ? <RefusalList /> : <FeedbackList />}
        </div>
      </div>
    </>
  );
}

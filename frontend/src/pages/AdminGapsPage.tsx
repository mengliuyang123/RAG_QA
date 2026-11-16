import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import type {
  GapCandidate,
  GapCheck,
  GapCheckResult,
  GapInput,
  GapStatus,
  KnowledgeGap,
} from "../api/types";
import { Topbar } from "../components/Layout";
import { useApp } from "../state/AppContext";
import { formatDate } from "./KnowledgePage";

export interface GapRow extends GapInput {
  count: number;
  lastAskedAt: string | null;
  tracked: boolean;
}

const STATUS_LABELS: Record<GapStatus, string> = {
  open: "待处理",
  document_added: "已补文档",
  verified: "已验证",
};

const STATUS_CHIPS: Record<GapStatus, string> = {
  open: "chip chip-warning",
  document_added: "chip chip-accent",
  verified: "chip chip-success",
};

const STATUS_ORDER: Record<GapStatus, number> = { open: 0, document_added: 1, verified: 2 };

type Filter = "all" | GapStatus;

/** 已跟踪的缺口优先；候选与已跟踪缺口按键或任一相同问题合并，避免聚类ID变化后重复出现。 */
export function mergeGaps(candidates: GapCandidate[], gaps: KnowledgeGap[]): GapRow[] {
  const rows: GapRow[] = gaps.map(({ updatedAt: _updatedAt, ...gap }) => ({
    ...gap, count: 0, lastAskedAt: null, tracked: true,
  }));
  for (const candidate of candidates) {
    const match = rows.find((row) => row.key === candidate.key)
      ?? rows.find((row) => row.tracked && row.questions.some((question) => candidate.questions.includes(question)));
    if (match) {
      match.count = Math.max(match.count, candidate.count);
      if (!match.lastAskedAt || (candidate.lastAskedAt && candidate.lastAskedAt > match.lastAskedAt)) {
        match.lastAskedAt = candidate.lastAskedAt;
      }
      continue;
    }
    rows.push({
      ...candidate, status: "open", docId: null, lastCheck: null, tracked: false,
    });
  }
  return rows.sort((a, b) => STATUS_ORDER[a.status] - STATUS_ORDER[b.status] || b.count - a.count);
}

function toInput(row: GapRow, patch: Partial<GapInput> = {}): GapInput {
  const { count: _count, lastAskedAt: _lastAskedAt, tracked: _tracked, ...gap } = row;
  return { ...gap, ...patch };
}

/** 已验证后又出现了新的拒答或点踩，提示可能需要重新处理。 */
function askedAfterVerify(row: GapRow): boolean {
  return row.status === "verified" && !!row.lastAskedAt && !!row.lastCheck && row.lastAskedAt > row.lastCheck.checkedAt;
}

function CheckResults({ check }: { check: GapCheck }) {
  return (
    <ul aria-label="复测结果" style={{ listStyle: "none", padding: 0, margin: "8px 0 0" }}>
      {check.results.map((result) => (
        <li key={result.question} style={{ display: "flex", gap: 8, alignItems: "baseline", padding: "3px 0" }}>
          <span className={result.answered ? "chip chip-success" : "chip chip-danger"}>
            {result.answered ? "已回答" : result.outcome === "error" ? "失败" : result.outcome === "clarification" ? "需澄清" : "仍拒答"}
          </span>
          <span>{result.question}</span>
        </li>
      ))}
    </ul>
  );
}

function GapDetail({ row, onSaved }: { row: GapRow; onSaved(gap: KnowledgeGap, message: string): void }) {
  const { knowledgeBases, kbName } = useApp();
  const navigate = useNavigate();
  const [kbId, setKbId] = useState<string | null>(row.kbId ?? knowledgeBases[0]?.id ?? null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<GapCheckResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setKbId(row.kbId ?? knowledgeBases[0]?.id ?? null);
    setProgress(null);
    setError(null);
  }, [row.key]); // eslint-disable-line react-hooks/exhaustive-deps

  const save = async (patch: Partial<GapInput>, message: string) => {
    const saved = await api.admin.saveGap(toInput(row, { kbId, ...patch }));
    onSaved(saved, message);
    return saved;
  };

  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const goUpload = () => run(async () => {
    if (!kbId) throw new Error("请先选择要补充文档的知识库");
    await save({}, "已开始跟踪此缺口");
    navigate(`/admin/kb?kb=${encodeURIComponent(kbId)}&gap=${encodeURIComponent(row.key)}`);
  });

  const verify = () => run(async () => {
    const results: GapCheckResult[] = [];
    setProgress(results);
    for (const question of row.questions) {
      results.push(await api.admin.checkQuestion(question, kbId));
      setProgress([...results]);
    }
    const lastCheck: GapCheck = { checkedAt: new Date().toISOString(), kbId, results };
    const passed = results.every((result) => result.answered);
    const status: GapStatus = passed ? "verified" : row.docId || row.status !== "open" ? "document_added" : "open";
    const answered = results.filter((result) => result.answered).length;
    await save(
      { status, lastCheck },
      passed ? "复测全部通过，已标记为已验证" : `复测 ${answered}/${results.length} 个问题已能回答，暂不标记为已验证`,
    );
    setProgress(null);
  });

  const check = progress ? { checkedAt: "", kbId, results: progress } : row.lastCheck;

  return (
    <section className="card" aria-label="缺口详情" style={{ marginBottom: 16 }}>
      <div className="toolbar" style={{ marginBottom: 6 }}>
        <h2 style={{ fontSize: 16 }}>{row.title}</h2>
        <span className={STATUS_CHIPS[row.status]}>{STATUS_LABELS[row.status]}</span>
      </div>
      <div className="secondary" style={{ fontSize: 13, marginBottom: 8 }}>
        {row.source === "refusal_cluster" ? `拒答聚类 · ${row.count || "—"} 次` : "用户点踩"}
        {row.lastAskedAt && ` · 最近 ${formatDate(row.lastAskedAt)}`}
        {row.docId && " · 已补充文档"}
      </div>
      <div className="field-label">样例问题（复测时逐条重新提问）</div>
      <ul style={{ margin: "0 0 8px", paddingLeft: 18 }}>
        {row.questions.map((question) => <li key={question}>{question}</li>)}
      </ul>
      <label className="field-label" htmlFor="gap-kb">补充到知识库 / 复测范围</label>
      <select
        id="gap-kb"
        className="text-input"
        style={{ maxWidth: 280 }}
        value={kbId ?? ""}
        onChange={(event) => setKbId(event.target.value || null)}
      >
        {knowledgeBases.map((kb) => <option key={kb.id} value={kb.id}>{kbName(kb.id)}</option>)}
      </select>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
        <button type="button" className="btn btn-primary" disabled={busy} onClick={goUpload}>去上传文档</button>
        <button type="button" className="btn" disabled={busy} onClick={verify}>
          {progress ? `复测中 ${progress.length}/${row.questions.length}…` : "复测"}
        </button>
        {row.status === "open" && (
          <button type="button" className="btn" disabled={busy} onClick={() => run(() => save({ status: "document_added" }, "已标记为已补充文档"))}>
            标记为已补文档
          </button>
        )}
        {row.status !== "open" && (
          <button type="button" className="btn" disabled={busy} onClick={() => run(() => save({ status: "open" }, "已重新打开"))}>
            重新打开
          </button>
        )}
      </div>
      {error && <div role="alert" className="form-error">{error}</div>}
      {check && check.results.length > 0 && (
        <>
          <div className="field-label">
            {progress ? "本次复测" : `最近复测${row.lastCheck?.checkedAt ? `（${formatDate(row.lastCheck.checkedAt)}）` : ""}`}
          </div>
          <CheckResults check={check} />
        </>
      )}
    </section>
  );
}

export function AdminGapsPage() {
  const { kbName } = useApp();
  const [rows, setRows] = useState<GapRow[] | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [candidates, gaps] = await Promise.all([api.admin.gapCandidates(), api.admin.listGaps()]);
    setWarnings(candidates.warnings);
    setRows(mergeGaps(candidates.candidates, gaps));
  }, []);

  useEffect(() => {
    load().catch((reason: Error) => setError(reason.message));
  }, [load]);

  const onSaved = (gap: KnowledgeGap, text: string) => {
    setMessage(text);
    setRows((current) => current?.map((row) => (row.key === gap.key
      ? { ...row, ...gap, tracked: true }
      : row)) ?? null);
  };

  const counts = (status: Filter) => rows?.filter((row) => status === "all" || row.status === status).length ?? 0;
  const visible = rows?.filter((row) => filter === "all" || row.status === filter) ?? [];
  const current = rows?.find((row) => row.key === selected) ?? null;

  return (
    <>
      <Topbar title="知识缺口" />
      <div className="page">
        <div className="page-inner">
          <p className="secondary" style={{ marginTop: 0 }}>
            来自近 30 天的拒答聚类和用户点踩。打开一个缺口，补充文档后复测样例问题，全部能回答即为已验证。
          </p>
          {warnings.map((text) => <div key={text} className="banner" style={{ marginBottom: 8 }}>{text}</div>)}
          {error && <div className="banner" role="alert">{error}</div>}
          {message && (
            <div role="status" className="chip chip-success" style={{ marginBottom: 10, display: "block", padding: "6px 10px" }}>{message}</div>
          )}
          {current && <GapDetail row={current} onSaved={onSaved} />}
          <div className="tabs" role="tablist" aria-label="按状态筛选">
            {(["all", "open", "document_added", "verified"] as const).map((value) => (
              <button
                key={value}
                type="button"
                role="tab"
                aria-selected={filter === value}
                className={filter === value ? "btn btn-primary" : "btn"}
                onClick={() => setFilter(value)}
              >
                {value === "all" ? "全部" : STATUS_LABELS[value]} {counts(value)}
              </button>
            ))}
          </div>
          {!rows && !error && <div className="muted">正在加载…</div>}
          {rows && visible.length === 0 && <div className="empty">没有这类缺口</div>}
          {visible.length > 0 && (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr><th>问题</th><th>来源</th><th className="num">次数</th><th>知识库</th><th>状态</th><th /></tr>
                </thead>
                <tbody>
                  {visible.map((row) => (
                    <tr key={row.key} aria-selected={row.key === selected}>
                      <td>
                        {row.title}
                        {row.questions.length > 1 && <span className="muted"> 等 {row.questions.length} 个问题</span>}
                      </td>
                      <td>{row.source === "refusal_cluster" ? "拒答" : "点踩"}</td>
                      <td className="num">{row.count || "—"}</td>
                      <td>{row.kbId ? <span className="chip">{kbName(row.kbId)}</span> : "—"}</td>
                      <td>
                        <span className={STATUS_CHIPS[row.status]}>{STATUS_LABELS[row.status]}</span>
                        {askedAfterVerify(row) && <span className="chip chip-warning" style={{ marginLeft: 4 }}>验证后又出现</span>}
                      </td>
                      <td>
                        <button type="button" className="btn" onClick={() => { setSelected(row.key); setMessage(null); }}>打开</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

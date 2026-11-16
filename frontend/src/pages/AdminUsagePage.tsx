import { useEffect, useState } from "react";
import { api } from "../api";
import type { CostEstimate, TraceSummary, UsageGroup, UsageReport } from "../api/types";
import { Topbar } from "../components/Layout";
import { useApp } from "../state/AppContext";
import { formatDate } from "./KnowledgePage";

const GROUP_LABELS: Record<UsageGroup, string> = { day: "按日期", user: "按用户", kb: "按知识库" };
const OUTCOME_LABELS = { answered: "已回答", refused: "拒答", clarification: "追问" } as const;

function formatCost(costs: CostEstimate[]): string {
  if (costs.length === 0) return "—";
  return costs.map((cost) => `${cost.currency} ${cost.amount.toFixed(cost.amount < 1 ? 3 : 2)}`).join(" + ");
}

function formatTokens(value: number): string {
  return value >= 10_000 ? `${(value / 10_000).toFixed(1)} 万` : String(value);
}

function costValue(costs: CostEstimate[]): number {
  return costs.reduce((total, cost) => total + cost.amount, 0);
}

export function AdminUsagePage() {
  const { kbName } = useApp();
  const [days, setDays] = useState(7);
  const [groupBy, setGroupBy] = useState<UsageGroup>("day");
  const [report, setReport] = useState<UsageReport | null>(null);
  const [traces, setTraces] = useState<TraceSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setReport(null);
    api.admin.usage(days, groupBy).then(setReport).catch((reason: Error) => setError(reason.message));
  }, [days, groupBy]);

  useEffect(() => {
    api.admin.recentTraces(100).then(setTraces).catch((reason: Error) => setError(reason.message));
  }, []);

  // 最近请求按估算费用排序，找出最耗费的问题。
  const expensive = traces ? [...traces].sort((a, b) => costValue(b.costs) - costValue(a.costs) || b.inputTokens - a.inputTokens).slice(0, 20) : null;

  return (
    <>
      <Topbar title="用量与费用" />
      <div className="page">
        <div className="page-inner">
          <p className="secondary" style={{ marginTop: 0 }}>
            费用按配置的模型费率估算，不是实际账单；只统计仍在保留期内的请求记录。
          </p>
          {error && <div className="banner" role="alert">{error}</div>}
          <div className="toolbar">
            <div className="tabs" role="tablist" aria-label="统计窗口" style={{ marginBottom: 0 }}>
              {[7, 30, 90].map((value) => (
                <button key={value} type="button" role="tab" aria-selected={days === value} className={days === value ? "btn btn-primary" : "btn"} onClick={() => setDays(value)}>
                  近 {value} 天
                </button>
              ))}
            </div>
            <div className="tabs" role="tablist" aria-label="分组方式" style={{ marginBottom: 0 }}>
              {(Object.keys(GROUP_LABELS) as UsageGroup[]).map((value) => (
                <button key={value} type="button" role="tab" aria-selected={groupBy === value} className={groupBy === value ? "btn btn-primary" : "btn"} onClick={() => setGroupBy(value)}>
                  {GROUP_LABELS[value]}
                </button>
              ))}
            </div>
          </div>
          {!report && !error && <div className="muted">正在加载…</div>}
          {report && (
            <>
              <div className="metrics">
                <div className="card"><div className="secondary">提问</div><div style={{ fontSize: 22, fontWeight: 600 }}>{report.totals.questions}</div></div>
                <div className="card"><div className="secondary">模型调用</div><div style={{ fontSize: 22, fontWeight: 600 }}>{report.totals.modelCalls}</div></div>
                <div className="card"><div className="secondary">Token（输入 / 输出）</div><div style={{ fontSize: 22, fontWeight: 600 }}>{formatTokens(report.totals.inputTokens)} / {formatTokens(report.totals.outputTokens)}</div></div>
                <div className="card">
                  <div className="secondary">估算费用</div>
                  <div style={{ fontSize: 22, fontWeight: 600 }}>{formatCost(report.totals.costs)}</div>
                  {report.totals.costStatus !== "estimated" && <div className="muted" style={{ fontSize: 12 }}>部分调用未配置费率</div>}
                </div>
              </div>
              {report.retentionLimited && <div className="banner" style={{ marginBottom: 12 }}>窗口超过记录保留期，较早的用量未计入。</div>}
              {groupBy === "kb" && <p className="muted" style={{ fontSize: 12 }}>一次提问可能检索多个知识库，按知识库分组的行相加会大于合计。</p>}
              <div className="table-wrap" style={{ marginBottom: 20 }}>
                <table className="table" aria-label="分组用量">
                  <thead>
                    <tr><th>{GROUP_LABELS[groupBy].slice(1)}</th><th className="num">提问</th><th className="num">模型调用</th><th className="num">输入 Token</th><th className="num">输出 Token</th><th className="num">估算费用</th></tr>
                  </thead>
                  <tbody>
                    {report.rows.length === 0 && <tr><td colSpan={6} className="muted">没有记录</td></tr>}
                    {report.rows.map((row) => (
                      <tr key={row.key}>
                        <td>{groupBy === "kb" ? kbName(row.key) : row.label}</td>
                        <td className="num">{row.questions}</td>
                        <td className="num">{row.modelCalls}</td>
                        <td className="num">{formatTokens(row.inputTokens)}</td>
                        <td className="num">{formatTokens(row.outputTokens)}</td>
                        <td className="num">{formatCost(row.costs)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          <div className="toolbar"><h2>最耗费的请求</h2></div>
          <p className="muted" style={{ fontSize: 12, marginTop: -8 }}>最近 100 次请求中按估算费用排序。</p>
          {expensive && (
            <div className="table-wrap">
              <table className="table" aria-label="最耗费的请求">
                <thead>
                  <tr><th>问题</th><th>结果</th><th className="num">模型调用</th><th className="num">Token</th><th className="num">耗时</th><th className="num">估算费用</th><th>时间</th></tr>
                </thead>
                <tbody>
                  {expensive.length === 0 && <tr><td colSpan={7} className="muted">没有记录</td></tr>}
                  {expensive.map((trace) => (
                    <tr key={trace.id}>
                      <td>{trace.question ?? <span className="muted">（未保存问题原文）</span>}</td>
                      <td>{trace.outcome ? OUTCOME_LABELS[trace.outcome] : "—"}</td>
                      <td className="num">{trace.modelCalls}</td>
                      <td className="num">{formatTokens(trace.inputTokens + trace.outputTokens)}</td>
                      <td className="num">{trace.latencyMs === null ? "—" : `${(trace.latencyMs / 1000).toFixed(1)}s`}</td>
                      <td className="num">{formatCost(trace.costs)}</td>
                      <td>{formatDate(trace.createdAt)}</td>
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

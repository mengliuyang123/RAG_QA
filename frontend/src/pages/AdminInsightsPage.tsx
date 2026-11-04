import { useEffect, useState } from "react";
import { api } from "../api";
import type { RefusalRecord } from "../api/types";
import { Topbar } from "../components/Layout";
import { useApp } from "../state/AppContext";
import { formatDate } from "./KnowledgePage";

export function AdminInsightsPage() {
  const { kbName } = useApp();
  const [refusals, setRefusals] = useState<RefusalRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.admin.refusals().then(setRefusals).catch((reason: Error) => setError(reason.message));
  }, []);

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
          <div className="toolbar"><h2>拒答问题</h2></div>
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
        </div>
      </div>
    </>
  );
}

import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import type { KbDocument } from "../api/types";
import { Topbar } from "../components/Layout";
import { useApp } from "../state/AppContext";

export function formatDate(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleDateString("zh-CN");
}

export function KnowledgePage() {
  const { knowledgeBases } = useApp();
  const [params, setParams] = useSearchParams();
  const kbId = params.get("kb") ?? knowledgeBases[0]?.id ?? null;
  const kb = knowledgeBases.find((item) => item.id === kbId) ?? null;
  const [documents, setDocuments] = useState<KbDocument[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!kbId) return;
    let cancelled = false;
    setDocuments(null);
    setError(null);
    api.listKbDocuments(kbId)
      .then((result) => { if (!cancelled) setDocuments(result); })
      .catch((reason: Error) => { if (!cancelled) setError(reason.message); });
    return () => { cancelled = true; };
  }, [kbId]);

  const visible = (documents ?? []).filter((doc) =>
    doc.filename.toLowerCase().includes(query.trim().toLowerCase()),
  );

  return (
    <>
      <Topbar title="知识库" />
      <div className="page">
        <div className="page-inner">
          <div className="tabs" role="tablist" aria-label="知识库">
            {knowledgeBases.map((item) => (
              <button
                key={item.id}
                type="button"
                role="tab"
                aria-selected={item.id === kbId}
                className={`scope-chip ${item.id === kbId ? "on" : ""}`}
                onClick={() => setParams({ kb: item.id })}
              >
                {item.name} · {item.documentsCount}
              </button>
            ))}
          </div>
          {kb && <p className="secondary" style={{ marginTop: 0 }}>{kb.description}</p>}
          <div className="toolbar">
            <h2>{kb?.name ?? "知识库"}</h2>
            <label className="sr-only" htmlFor="doc-search">搜索文件名</label>
            <input
              id="doc-search"
              className="search-input"
              placeholder="搜索文件名"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
          {error && <div className="banner" role="alert">{error}</div>}
          {!documents && !error && <div className="muted">正在加载…</div>}
          {documents && visible.length === 0 && (
            <div className="empty">{query ? "没有匹配的文件" : "这个知识库还没有文档"}</div>
          )}
          {visible.length > 0 && (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr><th>文件</th><th>类型</th><th className="num">文本块</th><th>更新时间</th></tr>
                </thead>
                <tbody>
                  {visible.map((doc) => (
                    <tr key={doc.docId}>
                      <td style={{ wordBreak: "break-all" }}>{doc.filename}</td>
                      <td><span className="chip">{doc.fileType.toUpperCase() || "—"}</span></td>
                      <td className="num">{doc.chunksCount}</td>
                      <td>{formatDate(doc.updatedAt)}</td>
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

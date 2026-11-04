import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { KbDocument, KbInsight, UploadProgress } from "../api/types";
import { Icon } from "../components/Icon";
import { Topbar } from "../components/Layout";
import { useApp } from "../state/AppContext";
import { formatDate } from "./KnowledgePage";

const ACCEPTED = ".pdf,.md,.markdown,.txt";
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

const STAGE_LABELS: Record<UploadProgress["stage"], string> = {
  uploading: "上传中",
  parsing: "解析中",
  chunking: "分块中",
  embedding: "向量化",
  indexing: "建立索引",
  done: "完成",
};

interface UploadRow {
  key: string;
  filename: string;
  progress: UploadProgress;
  error: string | null;
}

// 没有样本时后端返回 null，显示为"—"，不能当成 0%。
function percent(value: number | null): string {
  return value === null ? "—" : `${Math.round(value * 100)}%`;
}

function above(value: number | null, threshold: number): boolean {
  return value !== null && value > threshold;
}

const KB_ID_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/;

function NewKnowledgeBaseForm({ onCreated, onCancel }: { onCreated(id: string): void; onCancel(): void }) {
  const [id, setId] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!KB_ID_PATTERN.test(id)) {
      setError("ID 只能使用字母、数字、下划线和连字符，且以字母或数字开头");
      return;
    }
    if (!name.trim()) {
      setError("请输入名称");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.admin.createKnowledgeBase({ id, name: name.trim(), description: description.trim() });
      onCreated(id);
    } catch (reason) {
      setError((reason as Error).message);
      setBusy(false);
    }
  };

  return (
    <section className="card" aria-label="新建知识库" style={{ marginBottom: 16 }}>
      <div className="toolbar" style={{ marginBottom: 4 }}><h2 style={{ fontSize: 16 }}>新建知识库</h2></div>
      <label className="field-label" htmlFor="new-kb-id">ID（创建后不可修改，用于接口和引用）</label>
      <input id="new-kb-id" className="text-input" placeholder="ops" value={id} onChange={(event) => { setId(event.target.value.trim()); setError(null); }} />
      <label className="field-label" htmlFor="new-kb-name">名称</label>
      <input id="new-kb-name" className="text-input" placeholder="运维手册" value={name} onChange={(event) => { setName(event.target.value); setError(null); }} />
      <label className="field-label" htmlFor="new-kb-description">描述（用于自动路由）</label>
      <textarea id="new-kb-description" className="field" value={description} onChange={(event) => setDescription(event.target.value)} />
      {error && <div role="alert" className="form-error">{error}</div>}
      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <button type="button" className="btn btn-primary" disabled={busy} onClick={submit}>{busy ? "正在创建…" : "创建"}</button>
        <button type="button" className="btn" onClick={onCancel}>取消</button>
      </div>
    </section>
  );
}

export function AdminKbPage() {
  const { knowledgeBases, refreshKnowledgeBases } = useApp();
  const [selected, setSelected] = useState<string | null>(null);
  const [insights, setInsights] = useState<KbInsight[]>([]);
  const [documents, setDocuments] = useState<KbDocument[] | null>(null);
  const [uploads, setUploads] = useState<UploadRow[]>([]);
  const [description, setDescription] = useState("");
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [creating, setCreating] = useState(false);

  const kbId = selected ?? knowledgeBases[0]?.id ?? null;
  const kb = knowledgeBases.find((item) => item.id === kbId) ?? null;

  // 知识库增删后重新统计，新建的知识库也能显示指标行。
  useEffect(() => {
    api.admin.insights().then(setInsights).catch(() => setInsights([]));
  }, [knowledgeBases.length]);

  useEffect(() => { setDescription(kb?.description ?? ""); }, [kb?.id, kb?.description]);

  const loadDocuments = useCallback(async () => {
    if (!kbId) return;
    setDocuments(await api.admin.listDocuments(kbId));
  }, [kbId]);

  useEffect(() => {
    setDocuments(null);
    loadDocuments().catch((error: Error) => setMessage({ kind: "error", text: error.message }));
  }, [loadDocuments]);

  const upload = async (files: FileList | null) => {
    if (!files || !kbId) return;
    for (const file of Array.from(files)) {
      const key = `${file.name}-${Date.now()}-${Math.random()}`;
      if (file.size > MAX_UPLOAD_BYTES) {
        setUploads((rows) => [...rows, { key, filename: file.name, progress: { stage: "uploading", progress: 0 }, error: "文件超过 20 MB" }]);
        continue;
      }
      setUploads((rows) => [...rows, { key, filename: file.name, progress: { stage: "uploading", progress: 0 }, error: null }]);
      const update = (patch: Partial<UploadRow>) =>
        setUploads((rows) => rows.map((row) => (row.key === key ? { ...row, ...patch } : row)));
      try {
        await api.admin.uploadDocument(kbId, file, (progress) => update({ progress }));
        setUploads((rows) => rows.filter((row) => row.key !== key));
        await Promise.all([loadDocuments(), refreshKnowledgeBases()]);
      } catch (error) {
        update({ error: (error as Error).message });
      }
    }
    if (fileInput.current) fileInput.current.value = "";
  };

  const remove = async (doc: KbDocument) => {
    if (!window.confirm(`删除"${doc.filename}"？删除后它将不再出现在任何回答中。`)) return;
    try {
      await api.admin.deleteDocument(doc.docId);
      setMessage({ kind: "ok", text: `已删除 ${doc.filename}` });
      await Promise.all([loadDocuments(), refreshKnowledgeBases()]);
    } catch (error) {
      setMessage({ kind: "error", text: (error as Error).message });
    }
  };

  const saveDescription = async () => {
    if (!kbId) return;
    try {
      await api.admin.updateKnowledgeBase(kbId, description.trim());
      await refreshKnowledgeBases();
      setMessage({ kind: "ok", text: "描述已保存，自动路由会立即使用新描述" });
    } catch (error) {
      setMessage({ kind: "error", text: (error as Error).message });
    }
  };

  const insightFor = (id: string) => insights.find((item) => item.kbId === id);

  return (
    <>
      <Topbar title="知识库管理">
        {!creating && (
          <button type="button" className="btn" onClick={() => setCreating(true)}>
            <Icon name="plus" /> 新建知识库
          </button>
        )}
      </Topbar>
      <div className="page">
        <div className="page-inner">
          {creating && (
            <NewKnowledgeBaseForm
              onCancel={() => setCreating(false)}
              onCreated={(id) => {
                setCreating(false);
                setSelected(id);
                setMessage({ kind: "ok", text: "知识库已创建，可以上传文档了" });
                void refreshKnowledgeBases();
              }}
            />
          )}
          {!api.supports.insights && (
            <div className="banner" style={{ marginBottom: 12 }}>
              后端尚未提供统计接口，提问量、拒答率等指标为示例数据。
            </div>
          )}
          <div className="metrics">
            {knowledgeBases.map((item) => {
              const insight = insightFor(item.id);
              return (
                <button
                  key={item.id}
                  type="button"
                  className={`card metric-card ${item.id === kbId ? "selected" : ""}`}
                  aria-pressed={item.id === kbId}
                  onClick={() => setSelected(item.id)}
                >
                  <div style={{ fontWeight: 600, marginBottom: 6 }}>{item.name}</div>
                  <div className="metric-row"><span>文档</span><span>{item.documentsCount}</span></div>
                  {insight && (
                    <>
                      <div className="metric-row"><span>近 7 天提问</span><span>{insight.questions}</span></div>
                      <div className="metric-row">
                        <span>拒答率</span>
                        <span style={{ color: above(insight.refusalRate, 0.1) ? "var(--danger-text)" : undefined }}>{percent(insight.refusalRate)}</span>
                      </div>
                      <div className="metric-row">
                        <span>扩大范围比例</span>
                        <span style={{ color: above(insight.expandedRate, 0.15) ? "var(--warning-text)" : undefined }}>{percent(insight.expandedRate)}</span>
                      </div>
                    </>
                  )}
                </button>
              );
            })}
          </div>

          {kb && (
            <section className="card" aria-label={`${kb.name}详情`}>
              <div className="toolbar">
                <h2 style={{ fontSize: 16 }}>{kb.name}</h2>
                <input
                  ref={fileInput}
                  type="file"
                  accept={ACCEPTED}
                  multiple
                  className="sr-only"
                  id="upload-input"
                  onChange={(event) => upload(event.target.files)}
                />
                <label htmlFor="upload-input" className="btn btn-primary" style={{ cursor: "pointer" }}>
                  <Icon name="upload" /> 上传文档
                </label>
              </div>

              <label htmlFor="kb-description" className="muted" style={{ fontSize: 12 }}>
                描述（用于自动路由，写清楚这个库能回答哪类问题）
              </label>
              <textarea
                id="kb-description"
                className="field"
                value={description}
                disabled={!api.supports.editKnowledgeBase}
                onChange={(event) => setDescription(event.target.value)}
              />
              <div style={{ display: "flex", gap: 8, alignItems: "center", margin: "8px 0 12px" }}>
                <button
                  type="button"
                  className="btn"
                  disabled={!api.supports.editKnowledgeBase || description.trim() === kb.description}
                  onClick={saveDescription}
                >
                  保存描述
                </button>
                {!api.supports.editKnowledgeBase && <span className="muted" style={{ fontSize: 12 }}>后端尚未支持编辑知识库</span>}
                {api.supports.editKnowledgeBase && <span className="muted" style={{ fontSize: 12 }}>保存后立即用于自动路由</span>}
              </div>

              {above(insightFor(kb.id)?.expandedRate ?? null, 0.15) && (
                <div className="banner" style={{ marginBottom: 12 }}>
                  {percent(insightFor(kb.id)?.expandedRate ?? null)} 的问题路由到这里后因证据不足扩大了范围，建议在描述中划清边界，或补充相关文档。
                </div>
              )}

              {message && (
                <div
                  role={message.kind === "error" ? "alert" : "status"}
                  className={message.kind === "error" ? "banner" : "chip chip-success"}
                  style={{ marginBottom: 10, display: "block", padding: "6px 10px" }}
                >
                  {message.text}
                </div>
              )}

              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr><th>文件</th><th className="num">文本块</th><th>状态</th><th>更新时间</th><th aria-label="操作" /></tr>
                  </thead>
                  <tbody>
                    {uploads.map((row) => (
                      <tr key={row.key}>
                        <td style={{ wordBreak: "break-all" }}>{row.filename}</td>
                        <td className="num">—</td>
                        <td>
                          {row.error ? (
                            <span className="chip chip-danger">{row.error}</span>
                          ) : (
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                              <span className="chip chip-accent">
                                {STAGE_LABELS[row.progress.stage]}
                                {row.progress.progress !== null && row.progress.stage !== "done" && ` ${percent(row.progress.progress)}`}
                              </span>
                              {row.progress.progress === null && <span className="spinner" aria-hidden="true" />}
                            </div>
                          )}
                        </td>
                        <td>—</td>
                        <td>
                          {row.error && (
                            <button type="button" className="icon-btn" aria-label="移除失败记录" onClick={() => setUploads((rows) => rows.filter((item) => item.key !== row.key))}>
                              <Icon name="x" />
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                    {documents?.map((doc) => (
                      <tr key={doc.docId}>
                        <td style={{ wordBreak: "break-all" }}>{doc.filename}</td>
                        <td className="num">{doc.chunksCount}</td>
                        <td>
                          {doc.fileStatus === "available"
                            ? <span className="chip chip-success">可用</span>
                            : <span className="chip chip-danger" title="索引仍在，但源文件已丢失，建议重新上传">源文件缺失</span>}
                        </td>
                        <td>{formatDate(doc.updatedAt)}</td>
                        <td>
                          <button type="button" className="icon-btn" aria-label={`删除 ${doc.filename}`} onClick={() => remove(doc)}>
                            <Icon name="trash" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {documents && documents.length === 0 && uploads.length === 0 && (
                <div className="empty">还没有文档，点击"上传文档"添加 PDF、Markdown 或 TXT 文件</div>
              )}
              {!documents && <div className="muted" style={{ padding: 12 }}>正在加载…</div>}
            </section>
          )}
        </div>
      </div>
    </>
  );
}

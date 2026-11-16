import { useEffect, useState } from "react";
import { api } from "../../api";
import type { KbAccess, KbMemberRole, ManagedUser } from "../../api/types";

const ROLE_LABELS: Record<KbMemberRole, string> = { viewer: "可查看", editor: "可管理" };

// 管理员设置知识库是否仅成员可见，以及谁能查看、谁能管理。
export function KbAccessPanel({ kbId }: { kbId: string }) {
  const [access, setAccess] = useState<KbAccess | null>(null);
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [draft, setDraft] = useState<KbAccess | null>(null);
  const [adding, setAdding] = useState("");
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setAccess(null);
    setDraft(null);
    setMessage(null);
    Promise.all([api.admin.getKbAccess(kbId), api.admin.listUsers()])
      .then(([loaded, list]) => {
        setAccess(loaded);
        setDraft(loaded);
        setUsers(list);
      })
      .catch((error: Error) => setMessage({ kind: "error", text: error.message }));
  }, [kbId]);

  if (!draft || !access) {
    return message ? <div className="banner" role="alert">{message.text}</div> : <div className="muted">正在加载…</div>;
  }

  // 管理员本来就能看到和管理所有知识库，不需要加为成员。
  const candidates = users.filter((user) => user.role === "member" && !draft.members.some((member) => member.userId === user.id));
  const dirty = JSON.stringify(draft) !== JSON.stringify(access);

  const update = (patch: Partial<KbAccess>) => {
    setDraft({ ...draft, ...patch });
    setMessage(null);
  };

  const add = () => {
    const user = users.find((item) => item.id === adding);
    if (!user) return;
    update({ members: [...draft.members, { userId: user.id, role: "viewer", username: user.username, name: user.name }] });
    setAdding("");
  };

  const save = async () => {
    setBusy(true);
    try {
      const saved = await api.admin.saveKbAccess(kbId, {
        restricted: draft.restricted,
        members: draft.members.map((member) => ({ userId: member.userId, role: member.role })),
      });
      setAccess(saved);
      setDraft(saved);
      setMessage({ kind: "ok", text: "访问设置已保存，成员的下一次请求即生效" });
    } catch (error) {
      setMessage({ kind: "error", text: (error as Error).message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section aria-label="访问权限">
      <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 14 }}>
        <input type="checkbox" checked={draft.restricted} onChange={(event) => update({ restricted: event.target.checked })} />
        仅成员可见
      </label>
      <p className="secondary" style={{ fontSize: 13, margin: "4px 0 12px" }}>
        {draft.restricted
          ? "只有下面的成员和管理员能看到这个知识库、在问答中检索它。"
          : "所有登录用户都能看到并提问。下面的“可管理”成员可以维护文档、描述、推荐问题和标准答案。"}
      </p>
      <div className="table-wrap">
        <table className="table" aria-label="成员">
          <thead><tr><th>成员</th><th>权限</th><th aria-label="操作" /></tr></thead>
          <tbody>
            {draft.members.length === 0 && <tr><td colSpan={3} className="muted">还没有成员</td></tr>}
            {draft.members.map((member) => (
              <tr key={member.userId}>
                <td>{member.name ?? member.username}<span className="muted">（{member.username}）</span></td>
                <td>
                  <select
                    aria-label={`${member.username} 的权限`}
                    className="text-input"
                    style={{ maxWidth: 140 }}
                    value={member.role}
                    onChange={(event) => update({
                      members: draft.members.map((item) => (item.userId === member.userId ? { ...item, role: event.target.value as KbMemberRole } : item)),
                    })}
                  >
                    {(Object.keys(ROLE_LABELS) as KbMemberRole[]).map((role) => <option key={role} value={role}>{ROLE_LABELS[role]}</option>)}
                  </select>
                </td>
                <td>
                  <button type="button" className="btn" onClick={() => update({ members: draft.members.filter((item) => item.userId !== member.userId) })}>移除</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 10, flexWrap: "wrap" }}>
        <select aria-label="添加成员" className="text-input" style={{ maxWidth: 240 }} value={adding} onChange={(event) => setAdding(event.target.value)}>
          <option value="">选择用户…</option>
          {candidates.map((user) => <option key={user.id} value={user.id}>{user.name}（{user.username}）{user.disabled ? " · 已禁用" : ""}</option>)}
        </select>
        <button type="button" className="btn" disabled={!adding} onClick={add}>添加</button>
        <span style={{ flex: 1 }} />
        <button type="button" className="btn btn-primary" disabled={!dirty || busy} onClick={save}>{busy ? "正在保存…" : "保存"}</button>
      </div>
      {message && (
        <div role={message.kind === "error" ? "alert" : "status"} className={message.kind === "error" ? "banner" : "chip chip-success"} style={{ marginTop: 10, display: "block", padding: "6px 10px" }}>
          {message.text}
        </div>
      )}
    </section>
  );
}

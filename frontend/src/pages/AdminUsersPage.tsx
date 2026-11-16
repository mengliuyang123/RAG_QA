import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { ManagedUser, Role } from "../api/types";
import { Topbar } from "../components/Layout";
import { useApp } from "../state/AppContext";
import { formatDate } from "./KnowledgePage";

const USERNAME_PATTERN = /^[a-z0-9][a-z0-9_.-]{2,63}$/;

function passwordError(password: string): string | null {
  return password.length < 15 || password.length > 128 ? "口令长度必须为 15–128 个字符" : null;
}

function NewUserForm({ onCreated, onCancel }: { onCreated(user: ManagedUser): void; onCancel(): void }) {
  const [username, setUsername] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("member");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    const normalized = username.trim().toLowerCase();
    if (!USERNAME_PATTERN.test(normalized)) {
      setError("登录名需 3–64 位，只能包含小写字母、数字、点、下划线和连字符");
      return;
    }
    if (!name.trim()) {
      setError("请输入展示名");
      return;
    }
    const invalid = passwordError(password);
    if (invalid) {
      setError(invalid);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const created = await api.admin.createUser({ username: normalized, name: name.trim(), password, role });
      setPassword("");
      onCreated(created);
    } catch (reason) {
      setError((reason as Error).message);
      setBusy(false);
    }
  };

  return (
    <section className="card" aria-label="新建用户" style={{ marginBottom: 16 }}>
      <div className="toolbar" style={{ marginBottom: 4 }}><h2 style={{ fontSize: 16 }}>新建用户</h2></div>
      <label className="field-label" htmlFor="new-user-username">登录名</label>
      <input id="new-user-username" className="text-input" autoComplete="off" value={username} onChange={(event) => { setUsername(event.target.value); setError(null); }} />
      <label className="field-label" htmlFor="new-user-name">展示名</label>
      <input id="new-user-name" className="text-input" value={name} onChange={(event) => { setName(event.target.value); setError(null); }} />
      <label className="field-label" htmlFor="new-user-password">初始口令（15–128 个字符，请通过安全渠道告知对方）</label>
      <input id="new-user-password" className="text-input" type="password" autoComplete="new-password" value={password} onChange={(event) => { setPassword(event.target.value); setError(null); }} />
      <label className="field-label" htmlFor="new-user-role">角色</label>
      <select id="new-user-role" className="text-input" value={role} onChange={(event) => setRole(event.target.value as Role)}>
        <option value="member">成员</option>
        <option value="admin">管理员</option>
      </select>
      {error && <div role="alert" className="form-error">{error}</div>}
      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <button type="button" className="btn btn-primary" disabled={busy} onClick={submit}>{busy ? "正在创建…" : "创建"}</button>
        <button type="button" className="btn" onClick={onCancel}>取消</button>
      </div>
    </section>
  );
}

function ResetPasswordForm({ user, onDone }: { user: ManagedUser; onDone(message: string | null): void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    const invalid = passwordError(password);
    if (invalid) {
      setError(invalid);
      return;
    }
    setBusy(true);
    try {
      await api.admin.resetPassword(user.id, password);
      setPassword("");
      onDone(`已重置 ${user.username} 的口令，该账号所有已登录的浏览器需要重新登录`);
    } catch (reason) {
      setError((reason as Error).message);
      setBusy(false);
    }
  };

  return (
    <div role="group" aria-label={`重置 ${user.username} 的口令`} style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginTop: 6 }}>
      <input
        aria-label="新口令"
        className="text-input"
        type="password"
        autoComplete="new-password"
        style={{ maxWidth: 260 }}
        value={password}
        onChange={(event) => { setPassword(event.target.value); setError(null); }}
      />
      <button type="button" className="btn btn-primary" disabled={busy} onClick={submit}>确认重置</button>
      <button type="button" className="btn" onClick={() => onDone(null)}>取消</button>
      {error && <span role="alert" className="form-error" style={{ marginTop: 0 }}>{error}</span>}
    </div>
  );
}

export function AdminUsersPage() {
  const { user: me } = useApp();
  const [users, setUsers] = useState<ManagedUser[] | null>(null);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [creating, setCreating] = useState(false);
  const [resetting, setResetting] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);

  const load = useCallback(async () => {
    setUsers(await api.admin.listUsers());
  }, []);

  useEffect(() => {
    load().catch((error: Error) => setMessage({ kind: "error", text: error.message }));
  }, [load]);

  const activeAdmins = users?.filter((item) => item.role === "admin" && !item.disabled).length ?? 0;

  const update = async (target: ManagedUser, changes: { role?: Role; disabled?: boolean }, done: string) => {
    setPending(target.id);
    try {
      const updated = await api.admin.updateUser(target.id, changes);
      setUsers((current) => current?.map((item) => (item.id === updated.id ? updated : item)) ?? null);
      setMessage({ kind: "ok", text: done });
    } catch (error) {
      setMessage({ kind: "error", text: (error as Error).message });
    } finally {
      setPending(null);
    }
  };

  const toggleRole = (target: ManagedUser) => {
    const role: Role = target.role === "admin" ? "member" : "admin";
    void update(target, { role }, role === "admin" ? `已将 ${target.username} 设为管理员` : `已将 ${target.username} 改为成员`);
  };

  const toggleDisabled = (target: ManagedUser) => {
    if (!target.disabled && !window.confirm(`禁用 ${target.username}？该账号会立即退出所有已登录的浏览器，并且无法再登录。`)) return;
    void update(
      target,
      { disabled: !target.disabled },
      target.disabled ? `已启用 ${target.username}，需要重新登录` : `已禁用 ${target.username}`,
    );
  };

  return (
    <>
      <Topbar title="用户管理">
        {!creating && (
          <button type="button" className="btn" onClick={() => setCreating(true)}>新建用户</button>
        )}
      </Topbar>
      <div className="page">
        <div className="page-inner">
          {creating && (
            <NewUserForm
              onCancel={() => setCreating(false)}
              onCreated={(created) => {
                setCreating(false);
                setUsers((current) => [...(current ?? []), created]);
                setMessage({ kind: "ok", text: `已创建 ${created.username}` });
              }}
            />
          )}
          <p className="secondary" style={{ marginTop: 0 }}>
            禁用或重置口令会让该账号立即退出所有已登录的浏览器；修改角色在对方的下一次请求生效。
            不能降级或禁用自己，也不能移除最后一名启用的管理员。
          </p>
          {message && (
            <div
              role={message.kind === "error" ? "alert" : "status"}
              className={message.kind === "error" ? "banner" : "chip chip-success"}
              style={{ marginBottom: 10, display: "block", padding: "6px 10px" }}
            >
              {message.text}
            </div>
          )}
          {!users && !message && <div className="muted">正在加载…</div>}
          {users && (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr><th>登录名</th><th>展示名</th><th>角色</th><th>状态</th><th>创建时间</th><th>操作</th></tr>
                </thead>
                <tbody>
                  {users.map((item) => {
                    const self = item.id === me?.id;
                    const lastAdmin = item.role === "admin" && !item.disabled && activeAdmins <= 1;
                    const locked = self || lastAdmin;
                    const lockReason = self ? "不能修改自己的角色或禁用自己" : lastAdmin ? "这是最后一名启用的管理员" : undefined;
                    return (
                      <tr key={item.id}>
                        <td>
                          {item.username}
                          {self && <span className="chip chip-accent" style={{ marginLeft: 6 }}>我</span>}
                          {resetting === item.id && (
                            <ResetPasswordForm
                              user={item}
                              onDone={(text) => {
                                setResetting(null);
                                if (text) setMessage({ kind: "ok", text });
                              }}
                            />
                          )}
                        </td>
                        <td>{item.name}</td>
                        <td>{item.role === "admin" ? <span className="chip chip-accent">管理员</span> : <span className="chip">成员</span>}</td>
                        <td>{item.disabled ? <span className="chip chip-danger">已禁用</span> : <span className="chip chip-success">正常</span>}</td>
                        <td>{formatDate(item.createdAt)}</td>
                        <td>
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                            <button
                              type="button"
                              className="btn"
                              disabled={pending === item.id || (item.role === "admin" && locked)}
                              title={item.role === "admin" ? lockReason : undefined}
                              onClick={() => toggleRole(item)}
                            >
                              {item.role === "admin" ? "设为成员" : "设为管理员"}
                            </button>
                            <button
                              type="button"
                              className={item.disabled ? "btn" : "btn btn-danger"}
                              disabled={pending === item.id || (!item.disabled && locked)}
                              title={!item.disabled ? lockReason : undefined}
                              onClick={() => toggleDisabled(item)}
                            >
                              {item.disabled ? "启用" : "禁用"}
                            </button>
                            <button type="button" className="btn" onClick={() => setResetting(item.id)}>重置口令</button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

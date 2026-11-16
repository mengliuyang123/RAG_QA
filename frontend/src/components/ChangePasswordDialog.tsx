import { useState } from "react";
import { api } from "../api";

// 用户自己修改口令。成功后其他浏览器的登录会被撤销，当前浏览器保持登录。
export function ChangePasswordDialog({ onClose }: { onClose(): void }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (next.length < 15 || next.length > 128) {
      setError("新口令长度必须为 15–128 个字符");
      return;
    }
    if (next !== confirm) {
      setError("两次输入的新口令不一致");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.changePassword(current, next);
      setCurrent("");
      setNext("");
      setConfirm("");
      setDone(true);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 50, display: "flex", alignItems: "center",
        justifyContent: "center", padding: 16, background: "color-mix(in oklab, black 40%, transparent)",
      }}
    >
      <div role="dialog" aria-modal="true" aria-labelledby="password-title" className="card" style={{ maxWidth: 400, width: "100%" }}>
        <h2 id="password-title" style={{ fontSize: 16, marginBottom: 4 }}>修改口令</h2>
        {done ? (
          <>
            <p role="status" style={{ margin: "8px 0 12px" }}>口令已修改。其他浏览器上的登录已退出，当前浏览器保持登录。</p>
            <button type="button" className="btn btn-primary" onClick={onClose}>完成</button>
          </>
        ) : (
          <>
            <label className="field-label" htmlFor="pw-current">当前口令</label>
            <input id="pw-current" className="text-input" type="password" autoComplete="current-password" value={current} onChange={(event) => { setCurrent(event.target.value); setError(null); }} />
            <label className="field-label" htmlFor="pw-new">新口令（15–128 个字符）</label>
            <input id="pw-new" className="text-input" type="password" autoComplete="new-password" value={next} onChange={(event) => { setNext(event.target.value); setError(null); }} />
            <label className="field-label" htmlFor="pw-confirm">再次输入新口令</label>
            <input id="pw-confirm" className="text-input" type="password" autoComplete="new-password" value={confirm} onChange={(event) => { setConfirm(event.target.value); setError(null); }} />
            {error && <div role="alert" className="form-error">{error}</div>}
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <button type="button" className="btn btn-primary" disabled={busy || !current} onClick={submit}>{busy ? "正在保存…" : "保存"}</button>
              <button type="button" className="btn" onClick={onClose}>取消</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

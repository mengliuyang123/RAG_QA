import { useState, type FormEvent } from "react";
import { ApiError } from "../api";
import { useApp } from "../state/AppContext";

function loginErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "账号或密码不正确";
    if (error.status === 429) return "尝试次数过多，请稍后再试";
    if (error.status === 403) return "请求来源不受信任，请从正式地址访问";
    return error.message;
  }
  return "登录失败，请稍后再试";
}

export function LoginPage() {
  const { login, loadError } = useApp();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!username.trim() || !password) {
      setError("请输入账号和密码");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await login(username.trim(), password);
    } catch (reason) {
      setError(loginErrorMessage(reason));
      setPassword("");
      setBusy(false);
    }
  };

  return (
    <main className="login">
      <form className="card login-card" onSubmit={submit} noValidate>
        <div className="brand" style={{ justifyContent: "center", paddingBottom: 4 }}>
          <span className="brand-mark">R</span>RAG 知识库
        </div>
        <p className="secondary" style={{ textAlign: "center", margin: "0 0 18px" }}>登录后提问和查看你的对话</p>
        {loadError && <div className="banner" role="alert" style={{ marginBottom: 12 }}>{loadError}</div>}
        <label className="field-label" htmlFor="login-username">账号</label>
        <input
          id="login-username"
          className="text-input"
          autoComplete="username"
          autoFocus
          value={username}
          onChange={(event) => { setUsername(event.target.value); setError(null); }}
        />
        <label className="field-label" htmlFor="login-password">密码</label>
        <input
          id="login-password"
          className="text-input"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(event) => { setPassword(event.target.value); setError(null); }}
        />
        {error && <div role="alert" className="form-error">{error}</div>}
        <button type="submit" className="btn btn-primary" disabled={busy} style={{ width: "100%", justifyContent: "center", marginTop: 16 }}>
          {busy ? "正在登录…" : "登录"}
        </button>
      </form>
    </main>
  );
}

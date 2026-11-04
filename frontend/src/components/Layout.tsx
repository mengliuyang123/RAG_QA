import { createContext, useContext, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation, useMatch, useNavigate } from "react-router-dom";
import { api } from "../api";
import { useApp } from "../state/AppContext";
import { Icon } from "./Icon";

const MenuContext = createContext<() => void>(() => undefined);

function groupLabel(updatedAt: string): string {
  const date = new Date(updatedAt);
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (date.getTime() >= startOfToday) return "今天";
  if (date.getTime() >= startOfToday - 6 * 86400000) return "最近 7 天";
  return "更早";
}

function Sidebar({ open, onNavigate }: { open: boolean; onNavigate(): void }) {
  const { user, conversations, refreshConversations, logout } = useApp();
  const navigate = useNavigate();
  // 侧栏位于父级布局路由中，读取不到子路由参数，用 useMatch 获取当前会话。
  const conversationId = useMatch("/c/:conversationId")?.params.conversationId;
  const [actionError, setActionError] = useState<string | null>(null);

  const go = (path: string) => {
    navigate(path);
    onNavigate();
  };

  const rename = async (id: string, current: string) => {
    const title = window.prompt("重命名对话", current)?.trim();
    if (!title || title === current) return;
    await api.renameConversation(id, title.slice(0, 50));
    await refreshConversations();
  };

  const remove = async (id: string, title: string) => {
    if (!window.confirm(`删除对话"${title}"？删除后无法恢复。`)) return;
    try {
      await api.deleteConversation(id);
      setActionError(null);
      if (id === conversationId) navigate("/");
      await refreshConversations();
    } catch (error) {
      setActionError((error as Error).message);
    }
  };

  const groups = new Map<string, typeof conversations>();
  for (const conversation of conversations) {
    const label = groupLabel(conversation.updatedAt);
    groups.set(label, [...(groups.get(label) ?? []), conversation]);
  }

  return (
    <nav className={`sidebar ${open ? "open" : ""}`} aria-label="主导航">
      <div className="brand"><span className="brand-mark">R</span>RAG 知识库</div>
      <button type="button" className="btn btn-primary" style={{ justifyContent: "center" }} onClick={() => go("/")}>
        <Icon name="plus" /> 新对话
      </button>

      <div className="sidebar-scroll" style={{ marginTop: 8 }}>
        {conversations.length === 0 && <div className="sidebar-section">还没有对话</div>}
        {[...groups.entries()].map(([label, items]) => (
          <div key={label}>
            <div className="sidebar-section">{label}</div>
            {items.map((conversation) => (
              <div
                key={conversation.id}
                className={`nav-item ${conversation.id === conversationId ? "active" : ""}`}
              >
                <button
                  type="button"
                  className="label"
                  style={{ border: "none", background: "none", padding: 0, textAlign: "left", cursor: "pointer", color: "inherit" }}
                  onClick={() => go(`/c/${conversation.id}`)}
                >
                  {conversation.title}
                </button>
                <span className="actions">
                  <button type="button" aria-label={`重命名 ${conversation.title}`} onClick={() => rename(conversation.id, conversation.title)}>
                    <Icon name="edit" size={14} />
                  </button>
                  <button type="button" aria-label={`删除 ${conversation.title}`} onClick={() => remove(conversation.id, conversation.title)}>
                    <Icon name="trash" size={14} />
                  </button>
                </span>
              </div>
            ))}
          </div>
        ))}
        {actionError && <div className="banner" style={{ margin: "8px 4px" }}>{actionError}</div>}
      </div>

      <div className="sidebar-footer">
        <NavLink to="/kb" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`} onClick={onNavigate}>
          <Icon name="book" /> <span className="label">知识库</span>
        </NavLink>
        {user?.role === "admin" && (
          <>
            <div className="sidebar-section">管理后台</div>
            <NavLink to="/admin/kb" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`} onClick={onNavigate}>
              <Icon name="settings" /> <span className="label">知识库管理</span>
            </NavLink>
            <NavLink to="/admin/insights" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`} onClick={onNavigate}>
              <Icon name="chart" /> <span className="label">问题洞察</span>
            </NavLink>
          </>
        )}
        <div className="nav-item" style={{ cursor: "default", marginTop: 6 }}>
          <span className="avatar">{user?.name.slice(0, 1) ?? "?"}</span>
          <span className="label">{user?.name ?? "加载中"}</span>
          {api.mode === "mock" && <span className="chip">演示</span>}
          {api.mode === "live" && (
            <button type="button" className="btn btn-ghost" style={{ padding: "2px 6px", fontSize: 12 }} onClick={() => void logout()}>
              退出
            </button>
          )}
        </div>
      </div>
    </nav>
  );
}

export function Topbar({ title, children }: { title: string; children?: ReactNode }) {
  const openMenu = useContext(MenuContext);
  return (
    <header className="topbar">
      <button type="button" className="icon-btn menu-toggle" aria-label="打开导航" onClick={openMenu}>
        <Icon name="menu" size={18} />
      </button>
      <h1>{title}</h1>
      {children}
    </header>
  );
}

export function Layout() {
  const { loadError } = useApp();
  const [open, setOpen] = useState(false);
  const location = useLocation();

  return (
    <MenuContext.Provider value={() => setOpen(true)}>
      <div className="app" onKeyDown={(event) => { if (event.key === "Escape") setOpen(false); }}>
        <Sidebar open={open} onNavigate={() => setOpen(false)} />
        {open && (
          <button
            type="button"
            aria-label="关闭导航"
            onClick={() => setOpen(false)}
            style={{ position: "fixed", inset: 0, zIndex: 19, background: "transparent", border: "none" }}
          />
        )}
        <main className="main" key={location.pathname.startsWith("/c/") ? "chat" : location.pathname}>
          {loadError && <div className="banner" role="alert" style={{ margin: 12 }}>加载失败：{loadError}</div>}
          <Outlet />
        </main>
      </div>
    </MenuContext.Provider>
  );
}

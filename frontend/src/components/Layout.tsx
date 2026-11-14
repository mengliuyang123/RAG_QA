import {
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "../api";
import { RagRuntimeProvider, useRagThread } from "../runtime/RagRuntime";
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

function ConversationItem() {
  const aui = useAui();
  const title = useAuiState((state) => state.threadListItem.title) ?? "新对话";

  const rename = () => {
    const next = window.prompt("重命名对话", title)?.trim();
    if (next && next !== title) aui.threadListItem().rename(next);
  };

  const remove = () => {
    if (window.confirm(`删除对话"${title}"？删除后无法恢复。`)) aui.threadListItem().delete();
  };

  return (
    <ThreadListItemPrimitive.Root className="nav-item">
      <ThreadListItemPrimitive.Trigger className="label nav-trigger">
        <ThreadListItemPrimitive.Title fallback="新对话" />
      </ThreadListItemPrimitive.Trigger>
      <span className="actions">
        <button type="button" aria-label={`重命名 ${title}`} onClick={rename}>
          <Icon name="edit" size={14} />
        </button>
        <button type="button" aria-label={`删除 ${title}`} onClick={remove}>
          <Icon name="trash" size={14} />
        </button>
      </span>
    </ThreadListItemPrimitive.Root>
  );
}

// 会话列表由 assistant-ui ThreadList 渲染：切换、新建、重命名、删除都经过运行时的
// threadList 适配器（见 runtime/RagRuntime.tsx），这里只负责按更新时间分组。
function Sidebar({ open, onNavigate }: { open: boolean; onNavigate(): void }) {
  const { user, conversations, logout } = useApp();
  const { listError } = useRagThread();

  const groups = new Map<string, number[]>();
  conversations.forEach((conversation, index) => {
    const label = groupLabel(conversation.updatedAt);
    groups.set(label, [...(groups.get(label) ?? []), index]);
  });

  return (
    <nav className={`sidebar ${open ? "open" : ""}`} aria-label="主导航">
      <div className="brand"><span className="brand-mark">R</span>RAG 知识库</div>
      <ThreadListPrimitive.Root className="thread-list">
        <ThreadListPrimitive.New className="btn btn-primary" style={{ justifyContent: "center" }}>
          <Icon name="plus" /> 新对话
        </ThreadListPrimitive.New>

        <div className="sidebar-scroll" style={{ marginTop: 8 }}>
          {conversations.length === 0 && <div className="sidebar-section">还没有对话</div>}
          {[...groups.entries()].map(([label, indexes]) => (
            <div key={label}>
              <div className="sidebar-section">{label}</div>
              {indexes.map((index) => (
                <ThreadListPrimitive.ItemByIndex
                  key={conversations[index].id}
                  index={index}
                  components={{ ThreadListItem: ConversationItem }}
                />
              ))}
            </div>
          ))}
          {listError && <div className="banner" style={{ margin: "8px 4px" }}>{listError}</div>}
        </div>
      </ThreadListPrimitive.Root>

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
  // 移动端抽屉在路由变化（切换会话、新建对话）后自动收起。
  useEffect(() => setOpen(false), [location.pathname]);
  // 首页与会话页是同一个聊天视图，切换时不重新挂载，避免首问跳转时闪烁。
  const isChat = location.pathname === "/" || location.pathname.startsWith("/c/");

  return (
    <RagRuntimeProvider>
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
          <main className="main" key={isChat ? "chat" : location.pathname}>
            {loadError && <div className="banner" role="alert" style={{ margin: 12 }}>加载失败：{loadError}</div>}
            <Outlet />
          </main>
        </div>
      </MenuContext.Provider>
    </RagRuntimeProvider>
  );
}

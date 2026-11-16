import type { ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Layout, canManageKnowledge } from "./components/Layout";
import { AdminGapsPage } from "./pages/AdminGapsPage";
import { AdminInsightsPage } from "./pages/AdminInsightsPage";
import { AdminKbPage } from "./pages/AdminKbPage";
import { AdminSettingsPage } from "./pages/AdminSettingsPage";
import { AdminUsagePage } from "./pages/AdminUsagePage";
import { AdminUsersPage } from "./pages/AdminUsersPage";
import { SharePage } from "./pages/SharePage";
import { ChatPage } from "./pages/ChatPage";
import { KnowledgePage } from "./pages/KnowledgePage";
import { LoginPage } from "./pages/LoginPage";
import { useApp } from "./state/AppContext";

// 前端只做展示层面的拦截；真正的权限校验必须由后端完成。
function AdminOnly({ children }: { children: ReactNode }) {
  const { user } = useApp();
  if (!user) return null;
  return user.role === "admin" ? children : <Navigate to="/" replace />;
}

// 知识库管理页对管理员和库 editor 开放；editor 只看到自己负责的库。
function KbManagerOnly({ children }: { children: ReactNode }) {
  const { user } = useApp();
  if (!user) return null;
  return canManageKnowledge(user) ? children : <Navigate to="/" replace />;
}

export function App() {
  const { authStatus } = useApp();
  if (authStatus === "loading") return <div className="full-center">正在加载…</div>;
  if (authStatus === "anonymous") return <LoginPage />;
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<ChatPage />} />
        <Route path="c/:conversationId" element={<ChatPage />} />
        <Route path="kb" element={<KnowledgePage />} />
        <Route path="s/:token" element={<SharePage />} />
        <Route path="admin/kb" element={<KbManagerOnly><AdminKbPage /></KbManagerOnly>} />
        <Route path="admin/usage" element={<AdminOnly><AdminUsagePage /></AdminOnly>} />
        <Route path="admin/insights" element={<AdminOnly><AdminInsightsPage /></AdminOnly>} />
        <Route path="admin/gaps" element={<AdminOnly><AdminGapsPage /></AdminOnly>} />
        <Route path="admin/users" element={<AdminOnly><AdminUsersPage /></AdminOnly>} />
        <Route path="admin/settings" element={<AdminOnly><AdminSettingsPage /></AdminOnly>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

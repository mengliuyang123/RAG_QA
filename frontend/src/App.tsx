import type { ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AdminInsightsPage } from "./pages/AdminInsightsPage";
import { AdminKbPage } from "./pages/AdminKbPage";
import { ChatPage } from "./pages/ChatPage";
import { HomePage } from "./pages/HomePage";
import { KnowledgePage } from "./pages/KnowledgePage";
import { LoginPage } from "./pages/LoginPage";
import { useApp } from "./state/AppContext";

// 前端只做展示层面的拦截；真正的权限校验必须由后端完成。
function AdminOnly({ children }: { children: ReactNode }) {
  const { user } = useApp();
  if (!user) return null;
  return user.role === "admin" ? children : <Navigate to="/" replace />;
}

export function App() {
  const { authStatus } = useApp();
  if (authStatus === "loading") return <div className="full-center">正在加载…</div>;
  if (authStatus === "anonymous") return <LoginPage />;
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="c/:conversationId" element={<ChatPage />} />
        <Route path="kb" element={<KnowledgePage />} />
        <Route path="admin/kb" element={<AdminOnly><AdminKbPage /></AdminOnly>} />
        <Route path="admin/insights" element={<AdminOnly><AdminInsightsPage /></AdminOnly>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

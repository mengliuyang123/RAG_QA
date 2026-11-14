import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { Thread } from "../components/chat/Thread";
import { Topbar } from "../components/Layout";
import { SourcePanel } from "../components/SourcePanel";
import { useRagThread } from "../runtime/RagRuntime";
import { useApp } from "../state/AppContext";

// 首页（新对话）和会话页共用一个 assistant-ui 线程；会话状态在布局层的 RagRuntimeProvider 中。
export function ChatPage() {
  const navigate = useNavigate();
  const { knowledgeBases, conversations } = useApp();
  const { conversationId, conversation, loadError, scope, activeCitation, setActiveCitation } = useRagThread();

  if (loadError) {
    return (
      <>
        <Topbar title="对话" />
        <div className="empty">
          <p>{loadError}</p>
          <button type="button" className="btn" onClick={() => navigate("/")}>开始新对话</button>
        </div>
      </>
    );
  }

  // 标题以会话列表为准：首轮回答后后端会改名，列表刷新后这里同步更新。
  const title = conversationId === null
    ? "新对话"
    : conversations.find((item) => item.id === conversationId)?.title ?? conversation?.title ?? "加载中…";

  return (
    <>
      <Topbar title={title}>
        {conversationId !== null && api.supports.kbScope && scope.mode === "explicit" && (
          <span className="chip chip-accent">
            范围：{scope.kbIds.map((id) => knowledgeBases.find((kb) => kb.id === id)?.name ?? id).join("、")}
          </span>
        )}
      </Topbar>
      <div className={`chat ${activeCitation ? "with-source" : ""}`} style={{ flex: 1, minHeight: 0 }}>
        <Thread />
        {activeCitation && (
          <SourcePanel citation={activeCitation} onClose={() => setActiveCitation(null)} />
        )}
      </div>
    </>
  );
}

import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { Thread } from "../components/chat/Thread";
import { Topbar } from "../components/Layout";
import { SourcePanel } from "../components/SourcePanel";
import { useRagThread } from "../runtime/RagRuntime";
import { useApp } from "../state/AppContext";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// 首页（新对话）和会话页共用一个 assistant-ui 线程；会话状态在布局层的 RagRuntimeProvider 中。
export function ChatPage() {
  const navigate = useNavigate();
  const { knowledgeBases, conversations } = useApp();
  const { conversationId, conversation, loadError, scope, activeCitation, setActiveCitation } = useRagThread();

  if (loadError) {
    return (
      <>
        <Topbar title="对话" />
        <div className="flex flex-1 flex-col items-center justify-center gap-4 p-6 text-muted-foreground">
          <p className="m-0">{loadError}</p>
          <Button variant="outline" onClick={() => navigate("/")}>开始新对话</Button>
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
          <span className="shrink-0 rounded-full border px-2.5 py-0.5 text-xs text-muted-foreground">
            范围：{scope.kbIds.map((id) => knowledgeBases.find((kb) => kb.id === id)?.name ?? id).join("、")}
          </span>
        )}
      </Topbar>
      <div
        className={cn(
          "grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)]",
          activeCitation && "lg:grid-cols-[minmax(0,1fr)_400px]",
        )}
      >
        <div className="min-h-0 min-w-0">
          <Thread />
        </div>
        {activeCitation && (
          <SourcePanel citation={activeCitation} onClose={() => setActiveCitation(null)} />
        )}
      </div>
    </>
  );
}

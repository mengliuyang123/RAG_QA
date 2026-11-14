import { ThreadPrimitive } from "@assistant-ui/react";
import { useEffect, useState } from "react";
import { api } from "../../api";
import type { SuggestedQuestion } from "../../api/types";
import { useRagThread } from "../../runtime/RagRuntime";
import { useApp } from "../../state/AppContext";
import { Icon } from "../Icon";
import { Composer } from "./Composer";
import { AssistantMessage, UserMessage } from "./Messages";

// 新对话的欢迎页：输入框 + 各知识库推荐问题。点击推荐问题直接发送。
function Welcome() {
  const { knowledgeBases } = useApp();
  const { send } = useRagThread();
  const [suggestions, setSuggestions] = useState<SuggestedQuestion[]>([]);

  useEffect(() => {
    api.suggestedQuestions().then(setSuggestions).catch(() => setSuggestions([]));
  }, []);

  const ungrouped = suggestions.filter((item) => !item.kbId || !knowledgeBases.some((kb) => kb.id === item.kbId));

  return (
    <div className="page">
      <div className="home">
        <h2>想了解什么？</h2>
        <p className="subtitle">回答都来自团队知识库，并标注出处</p>
        <Composer placeholder="比如：服务启动报数据库锁定怎么办" autoFocus />

        <div className="kb-grid">
          {knowledgeBases.map((kb) => {
            const questions = suggestions.filter((item) => item.kbId === kb.id).slice(0, 3);
            return (
              <section key={kb.id} className="card kb-card" aria-label={kb.name}>
                <h3>{kb.name}</h3>
                <div className="meta">{kb.documentsCount} 篇文档</div>
                <div className="secondary" style={{ fontSize: 12, marginBottom: 6 }}>{kb.description}</div>
                {questions.map((item) => (
                  // 知识库卡片下的推荐问题用自动范围提问，与输入框当前所选范围无关。
                  <button
                    key={item.id}
                    type="button"
                    className="suggestion"
                    onClick={() => send(item.text, { mode: "auto", kbIds: [] })}
                  >
                    {item.text}
                  </button>
                ))}
              </section>
            );
          })}
          {ungrouped.length > 0 && (
            <section className="card kb-card" aria-label="试试这些问题">
              <h3>试试这些问题</h3>
              {ungrouped.slice(0, 4).map((item) => (
                <ThreadPrimitive.Suggestion key={item.id} className="suggestion" prompt={item.text} send>
                  {item.text}
                </ThreadPrimitive.Suggestion>
              ))}
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

function Conversation() {
  const { conversation } = useRagThread();
  return (
    <div className="thread-column">
      <ThreadPrimitive.Viewport className="thread">
        <div className="thread-inner">
          {!conversation && <div className="muted">正在加载对话…</div>}
          {conversation && (
            <ThreadPrimitive.Empty>
              <div className="empty">输入问题开始对话</div>
            </ThreadPrimitive.Empty>
          )}
          <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
        </div>
        <ThreadPrimitive.ScrollToBottom className="icon-btn scroll-to-bottom" aria-label="回到底部">
          <Icon name="chevron" />
        </ThreadPrimitive.ScrollToBottom>
      </ThreadPrimitive.Viewport>
      <div className="composer-dock">
        <Composer placeholder="继续追问" />
      </div>
    </div>
  );
}

export function Thread() {
  const { conversationId } = useRagThread();
  return (
    <ThreadPrimitive.Root className="thread-root">
      {/* 首页（还没有会话）且没有正在进行的提问时显示欢迎页；首个问题发出后立即切到会话视图。 */}
      {conversationId === null ? (
        <>
          <ThreadPrimitive.Empty><Welcome /></ThreadPrimitive.Empty>
          <ThreadPrimitive.If empty={false}><Conversation /></ThreadPrimitive.If>
        </>
      ) : (
        <Conversation />
      )}
    </ThreadPrimitive.Root>
  );
}

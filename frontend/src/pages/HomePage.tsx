import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import type { KbScope, SuggestedQuestion } from "../api/types";
import { Composer } from "../components/Composer";
import { Topbar } from "../components/Layout";
import { useApp } from "../state/AppContext";

export interface PendingQuestion {
  question: string;
  scope: KbScope;
}

export function HomePage() {
  const { knowledgeBases, refreshConversations } = useApp();
  const navigate = useNavigate();
  const [scope, setScope] = useState<KbScope>({ mode: "auto", kbIds: [] });
  const [suggestions, setSuggestions] = useState<SuggestedQuestion[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.suggestedQuestions().then(setSuggestions).catch(() => setSuggestions([]));
  }, []);

  const ask = async (question: string, askScope: KbScope = scope) => {
    setBusy(true);
    setError(null);
    try {
      // 首页选择的范围同时作为会话默认范围保存到后端，刷新后仍然生效。
      const conversation = await api.createConversation(askScope);
      await refreshConversations();
      const state: PendingQuestion = { question, scope: askScope };
      navigate(`/c/${conversation.id}`, { state });
    } catch (reason) {
      setError((reason as Error).message);
      setBusy(false);
    }
  };

  const ungrouped = suggestions.filter((item) => !item.kbId || !knowledgeBases.some((kb) => kb.id === item.kbId));

  return (
    <>
      <Topbar title="新对话" />
      <div className="page">
        <div className="home">
          <h2>想了解什么？</h2>
          <p className="subtitle">回答都来自团队知识库，并标注出处</p>
          <Composer
            onSubmit={(question) => ask(question)}
            busy={busy}
            scope={scope}
            onScopeChange={setScope}
            knowledgeBases={knowledgeBases}
            showScope={api.supports.kbScope}
            placeholder="比如：服务启动报数据库锁定怎么办"
            autoFocus
          />
          {error && <div className="banner" role="alert" style={{ marginTop: 10 }}>{error}</div>}

          <div className="kb-grid">
            {knowledgeBases.map((kb) => {
              const questions = suggestions.filter((item) => item.kbId === kb.id).slice(0, 3);
              return (
                <section key={kb.id} className="card kb-card" aria-label={kb.name}>
                  <h3>{kb.name}</h3>
                  <div className="meta">{kb.documentsCount} 篇文档</div>
                  <div className="secondary" style={{ fontSize: 12, marginBottom: 6 }}>{kb.description}</div>
                  {questions.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      className="suggestion"
                      disabled={busy}
                      onClick={() => ask(item.text, { mode: "auto", kbIds: [] })}
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
                  <button key={item.id} type="button" className="suggestion" disabled={busy} onClick={() => ask(item.text)}>
                    {item.text}
                  </button>
                ))}
              </section>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

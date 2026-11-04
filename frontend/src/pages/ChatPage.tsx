import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api";
import type { Citation, Conversation, FeedbackRating, KbScope } from "../api/types";
import { Composer } from "../components/Composer";
import { Topbar } from "../components/Layout";
import { PendingTurn, type PendingState } from "../components/PendingTurn";
import { SourcePanel } from "../components/SourcePanel";
import { TurnView } from "../components/TurnView";
import { useApp } from "../state/AppContext";
import type { PendingQuestion } from "./HomePage";

export function ChatPage() {
  const { conversationId = "" } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const { knowledgeBases, conversations, refreshConversations } = useApp();

  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [pending, setPending] = useState<PendingState | null>(null);
  const [scope, setScope] = useState<KbScope>({ mode: "auto", kbIds: [] });
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const lastAskRef = useRef<{ question: string; scope: KbScope } | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setConversation(null);
    setLoadError(null);
    setPending(null);
    setActiveCitation(null);
    api.getConversation(conversationId)
      .then((result) => {
        if (cancelled) return;
        setConversation(result);
        setScope(result.kbScope);
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setLoadError(error instanceof ApiError && error.status === 404 ? "这个对话不存在或已被删除" : error.message);
      });
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
  }, [conversationId]);

  useEffect(() => {
    const element = threadRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [conversation?.turns.length, pending?.statuses.length, pending?.answer, pending?.error]);

  const ask = useCallback(async (question: string, askScope: KbScope) => {
    lastAskRef.current = { question, scope: askScope };
    const controller = new AbortController();
    abortRef.current = controller;
    setPending({ question, statuses: [], answer: "", error: null });

    const fail = (code: string, message: string) =>
      setPending((current) => current && { ...current, error: { code, message } });

    try {
      await api.streamQuery(conversationId, question, askScope, {
        onStatus: (update) =>
          setPending((current) => current && { ...current, statuses: [...current.statuses, update] }),
        onToken: (token) =>
          setPending((current) => current && { ...current, answer: current.answer + token }),
        onComplete: (turn) => {
          setConversation((current) => current && { ...current, turns: [...current.turns, turn] });
          setPending(null);
          void refreshConversations();
        },
        onError: fail,
      }, controller.signal);
    } catch (error) {
      if ((error as Error).name === "AbortError") {
        fail("cancelled", "已停止回答");
        return;
      }
      const status = error instanceof ApiError ? error.status : 0;
      fail(String(status), status === 409 ? "上一个问题还在回答中，请稍后再试" : (error as Error).message);
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, [conversationId, refreshConversations]);

  // 首页创建会话后带着问题跳转过来，只发送一次，并清掉路由状态防止刷新重复提问。
  useEffect(() => {
    const state = location.state as PendingQuestion | null;
    if (!conversation || !state?.question) return;
    navigate(location.pathname, { replace: true, state: null });
    setScope(state.scope);
    void ask(state.question, state.scope);
  }, [conversation, location.state, location.pathname, navigate, ask]);

  const onFeedback = async (turnId: string, rating: FeedbackRating | null) => {
    const previous = conversation?.turns.find((turn) => turn.id === turnId)?.feedback ?? null;
    const apply = (value: FeedbackRating | null) =>
      setConversation((current) => current && {
        ...current,
        turns: current.turns.map((turn) => (turn.id === turnId ? { ...turn, feedback: value } : turn)),
      });
    apply(rating);
    try {
      await api.setFeedback(turnId, rating);
    } catch {
      // 保存失败时恢复原状态，避免界面显示一个服务端没有记录的反馈。
      apply(previous);
    }
  };

  const busy = pending !== null && pending.error === null;
  const send = (question: string, askScope: KbScope = scope) => {
    if (busy) return;
    void ask(question, askScope);
  };

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

  return (
    <>
      {/* 标题以会话列表为准：首轮回答后后端会改名，列表刷新后这里同步更新。 */}
      <Topbar title={conversations.find((item) => item.id === conversationId)?.title ?? conversation?.title ?? "加载中…"}>
        {api.supports.kbScope && scope.mode === "explicit" && (
          <span className="chip chip-accent">
            范围：{scope.kbIds.map((id) => knowledgeBases.find((kb) => kb.id === id)?.name ?? id).join("、")}
          </span>
        )}
      </Topbar>
      <div className={`chat ${activeCitation ? "with-source" : ""}`} style={{ flex: 1, minHeight: 0 }}>
        <div className="thread-column">
          <div className="thread" ref={threadRef}>
            <div className="thread-inner">
              {!conversation && <div className="muted">正在加载对话…</div>}
              {conversation?.turns.length === 0 && !pending && (
                <div className="empty">输入问题开始对话</div>
              )}
              {conversation?.turns.map((turn) => (
                <TurnView
                  key={turn.id}
                  turn={turn}
                  activeCitation={activeCitation}
                  onCite={setActiveCitation}
                  onAsk={(question) => send(question)}
                  onRetryWithScope={(next, question) => {
                    setScope(next);
                    send(question, next);
                  }}
                  onFeedback={onFeedback}
                />
              ))}
              {pending && (
                <PendingTurn
                  pending={pending}
                  onCancel={() => abortRef.current?.abort()}
                  onRetry={() => {
                    const last = lastAskRef.current;
                    if (last) void ask(last.question, last.scope);
                  }}
                />
              )}
            </div>
          </div>
          <div className="composer-dock">
            <Composer
              onSubmit={(question) => send(question)}
              busy={busy || !conversation}
              scope={scope}
              onScopeChange={setScope}
              knowledgeBases={knowledgeBases}
              showScope={api.supports.kbScope}
              placeholder="继续追问"
            />
          </div>
        </div>
        {activeCitation && (
          <SourcePanel citation={activeCitation} onClose={() => setActiveCitation(null)} />
        )}
      </div>
    </>
  );
}

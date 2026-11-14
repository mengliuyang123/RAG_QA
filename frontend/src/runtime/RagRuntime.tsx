import {
  AssistantRuntimeProvider,
  MessageNotSentError,
  useExternalStoreRuntime,
  type AppendMessage,
  type ExternalStoreThreadListAdapter,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useMatch, useNavigate } from "react-router-dom";
import { api, ApiError } from "../api";
import type {
  Citation,
  Conversation,
  FeedbackRating,
  KbScope,
  StatusUpdate,
  Turn,
} from "../api/types";
import { useApp } from "../state/AppContext";

export const MAX_QUESTION_LENGTH = 8000;
const AUTO_SCOPE: KbScope = { mode: "auto", kbIds: [] };

export interface PendingState {
  question: string;
  statuses: StatusUpdate[];
  answer: string;
  error: { code: string; message: string } | null;
}

// 线程里的消息由正式轮次和正在进行的一轮派生，assistant-ui 只负责渲染与交互。
// 一轮问答拆成用户消息 + 助手消息，助手消息的 metadata.custom 携带完整领域数据。
export type RagMessage =
  | { kind: "question"; id: string; text: string; createdAt: string }
  | { kind: "answer"; id: string; turn: Turn }
  | { kind: "pending"; id: string; pending: PendingState };

export type AssistantCustom =
  | { kind: "answer"; turn: Turn }
  | { kind: "pending"; pending: PendingState };

export function convertMessage(message: RagMessage): ThreadMessageLike {
  if (message.kind === "question") {
    return {
      id: message.id,
      role: "user",
      content: message.text,
      createdAt: new Date(message.createdAt),
    };
  }
  if (message.kind === "answer") {
    return {
      id: message.id,
      role: "assistant",
      content: [{ type: "text", text: message.turn.answer }],
      createdAt: new Date(message.turn.createdAt),
      status: { type: "complete", reason: "stop" },
      metadata: { custom: { kind: "answer", turn: message.turn } },
    };
  }
  const { pending } = message;
  return {
    id: message.id,
    role: "assistant",
    content: pending.answer ? [{ type: "text", text: pending.answer }] : [],
    status: pending.error
      ? pending.error.code === "cancelled"
        ? { type: "incomplete", reason: "cancelled" }
        : { type: "incomplete", reason: "error", error: pending.error.message }
      : { type: "running" },
    metadata: { custom: { kind: "pending", pending } },
  };
}

function textOf(message: AppendMessage): string {
  return message.content
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("")
    .trim();
}

interface RagThreadState {
  /** 当前路由对应的会话；新对话（首页）为 null。 */
  conversationId: string | null;
  conversation: Conversation | null;
  loadError: string | null;
  scope: KbScope;
  setScope(scope: KbScope): void;
  /** 绕过输入框直接提问，可指定本次范围（澄清选项、换库重查、推荐问题）。 */
  send(question: string, scope?: KbScope): void;
  composerError: string | null;
  clearComposerError(): void;
  activeCitation: Citation | null;
  setActiveCitation(citation: Citation | null): void;
  setFeedback(turnId: string, rating: FeedbackRating | null): Promise<void>;
  /** 会话列表操作（重命名、删除）的最近一次错误。 */
  listError: string | null;
}

const RagThreadContext = createContext<RagThreadState | null>(null);

export function RagRuntimeProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const { conversations, refreshConversations } = useApp();
  // 运行时挂在布局层，侧栏和主区共享；用 useMatch 读取当前会话。
  const routeId = useMatch("/c/:conversationId")?.params.conversationId ?? null;

  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [pending, setPending] = useState<PendingState | null>(null);
  const [scope, setScope] = useState<KbScope>(AUTO_SCOPE);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [composerError, setComposerError] = useState<string | null>(null);
  const [listError, setListError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const lastAskRef = useRef<{ question: string; scope: KbScope } | null>(null);
  // 正在展示的会话 ID；首页提问时本地创建会话后再跳转，路由变化不应重新加载。
  const currentIdRef = useRef<string | null>(null);

  useEffect(() => {
    // 结果是否过期只看 currentIdRef，不用 effect 的 cancelled 标记：
    // StrictMode 下 effect 会立即重跑一次，重跑时这里已经提前返回。
    if (routeId === currentIdRef.current) return;
    abortRef.current?.abort();
    abortRef.current = null;
    currentIdRef.current = routeId;
    setPending(null);
    setActiveCitation(null);
    setLoadError(null);
    setComposerError(null);
    setConversation(null);
    if (!routeId) {
      setScope(AUTO_SCOPE);
      return;
    }
    api.getConversation(routeId)
      .then((result) => {
        if (currentIdRef.current !== routeId) return;
        setConversation(result);
        setScope(result.kbScope);
      })
      .catch((error: Error) => {
        if (currentIdRef.current !== routeId) return;
        setLoadError(
          error instanceof ApiError && error.status === 404 ? "这个对话不存在或已被删除" : error.message,
        );
      });
  }, [routeId]);

  // 离开页面时停止仍在进行的回答。
  useEffect(() => () => abortRef.current?.abort(), []);

  const busy = pending !== null && pending.error === null;

  const ask = useCallback(async (question: string, askScope: KbScope) => {
    lastAskRef.current = { question, scope: askScope };
    setPending({ question, statuses: [], answer: "", error: null });

    let id = currentIdRef.current;
    if (!id) {
      // 首页的第一个问题：先创建会话（保存所选范围为默认范围），再跳到会话地址。
      try {
        const summary = await api.createConversation(askScope);
        id = summary.id;
        currentIdRef.current = id;
        setConversation({ ...summary, turns: [] });
        navigate(`/c/${id}`);
        void refreshConversations();
      } catch (error) {
        setPending((current) => current && {
          ...current,
          error: { code: "create_failed", message: (error as Error).message },
        });
        return;
      }
    }

    const conversationId = id;
    const controller = new AbortController();
    abortRef.current = controller;
    const stillCurrent = () => currentIdRef.current === conversationId;
    const fail = (code: string, message: string) => {
      if (stillCurrent()) setPending((current) => current && { ...current, error: { code, message } });
    };

    try {
      await api.streamQuery(conversationId, question, askScope, {
        onStatus: (update) => {
          if (stillCurrent()) {
            setPending((current) => current && { ...current, statuses: [...current.statuses, update] });
          }
        },
        onToken: (token) => {
          if (stillCurrent()) setPending((current) => current && { ...current, answer: current.answer + token });
        },
        onComplete: (turn) => {
          if (!stillCurrent()) return;
          setConversation((current) => current && { ...current, turns: [...current.turns, turn] });
          setPending(null);
          // 首轮回答后后端会自动改名，刷新列表同步标题。
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
  }, [navigate, refreshConversations]);

  const send = useCallback((question: string, askScope?: KbScope) => {
    if (busy) return;
    const next = askScope ?? scope;
    if (askScope) setScope(askScope);
    void ask(question, next);
  }, [ask, busy, scope]);

  const setFeedback = useCallback(async (turnId: string, rating: FeedbackRating | null) => {
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
  }, [conversation]);

  const messages = useMemo<RagMessage[]>(() => {
    const list: RagMessage[] = [];
    for (const turn of conversation?.turns ?? []) {
      list.push({ kind: "question", id: `${turn.id}:q`, text: turn.question, createdAt: turn.createdAt });
      list.push({ kind: "answer", id: turn.id, turn });
    }
    if (pending) {
      list.push({ kind: "question", id: "pending:q", text: pending.question, createdAt: new Date().toISOString() });
      list.push({ kind: "pending", id: "pending:a", pending });
    }
    return list;
  }, [conversation, pending]);

  const threadList = useMemo<ExternalStoreThreadListAdapter>(() => {
    const threads = conversations.map((item) => ({ status: "regular" as const, id: item.id, title: item.title }));
    // 首问刚创建的会话在列表刷新前就成为当前线程；assistant-ui 要求当前线程必须在列表里。
    if (routeId && !threads.some((item) => item.id === routeId)) {
      threads.unshift({ status: "regular", id: routeId, title: conversation?.title ?? "新对话" });
    }
    return {
      threadId: routeId ?? undefined,
      threads,
      onSwitchToThread: (threadId) => navigate(`/c/${threadId}`),
      onSwitchToNewThread: () => navigate("/"),
      onRename: async (threadId, title) => {
        try {
          await api.renameConversation(threadId, title.slice(0, 50));
          setListError(null);
        } catch (error) {
          setListError((error as Error).message);
        }
        await refreshConversations();
      },
      onDelete: async (threadId) => {
        try {
          await api.deleteConversation(threadId);
          setListError(null);
          if (threadId === currentIdRef.current) navigate("/");
        } catch (error) {
          setListError((error as Error).message);
        }
        await refreshConversations();
      },
    };
  }, [routeId, conversations, conversation?.title, navigate, refreshConversations]);

  const runtime = useExternalStoreRuntime<RagMessage>({
    messages,
    convertMessage,
    isRunning: busy,
    // 会话加载中或加载失败时禁止输入。
    isDisabled: routeId !== null && !conversation,
    onNew: async (message) => {
      const question = textOf(message);
      if (!question) throw new MessageNotSentError("empty");
      if (question.length > MAX_QUESTION_LENGTH) {
        setComposerError(`问题最多 ${MAX_QUESTION_LENGTH} 字`);
        throw new MessageNotSentError("too_long");
      }
      if (busy) throw new MessageNotSentError("busy");
      setComposerError(null);
      void ask(question, scope);
    },
    onCancel: async () => {
      abortRef.current?.abort();
    },
    // 失败或被停止的一轮重新提问（同一问题、同一范围）。
    onReload: async () => {
      const last = lastAskRef.current;
      if (last && !busy) void ask(last.question, last.scope);
    },
    unstable_capabilities: { copy: true },
    adapters: { threadList },
  });

  const value = useMemo<RagThreadState>(() => ({
    conversationId: routeId,
    conversation,
    loadError,
    scope,
    setScope,
    send,
    composerError,
    clearComposerError: () => setComposerError(null),
    activeCitation,
    setActiveCitation,
    setFeedback,
    listError,
  }), [routeId, conversation, loadError, scope, send, composerError, activeCitation, setFeedback, listError]);

  return (
    <RagThreadContext.Provider value={value}>
      <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
    </RagThreadContext.Provider>
  );
}

export function useRagThread(): RagThreadState {
  const value = useContext(RagThreadContext);
  if (!value) throw new Error("useRagThread 必须在 RagRuntimeProvider 内使用");
  return value;
}

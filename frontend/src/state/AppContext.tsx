import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api, ApiError } from "../api";
import { AUTH_EXPIRED_EVENT } from "../api/live";
import type { ConversationSummary, KnowledgeBase, User } from "../api/types";

export type AuthStatus = "loading" | "anonymous" | "signed-in";

interface AppState {
  authStatus: AuthStatus;
  user: User | null;
  knowledgeBases: KnowledgeBase[];
  conversations: ConversationSummary[];
  kbName(kbId: string): string;
  login(username: string, password: string): Promise<void>;
  logout(): Promise<void>;
  refreshConversations(): Promise<void>;
  refreshKnowledgeBases(): Promise<void>;
  loadError: string | null;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [authStatus, setAuthStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<User | null>(null);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const signOutLocally = useCallback(() => {
    setUser(null);
    setKnowledgeBases([]);
    setConversations([]);
    setAuthStatus("anonymous");
  }, []);

  const refreshConversations = useCallback(async () => {
    setConversations(await api.listConversations());
  }, []);

  const refreshKnowledgeBases = useCallback(async () => {
    setKnowledgeBases(await api.listKnowledgeBases());
  }, []);

  // 身份确认后再加载业务数据，避免未登录时连发一串 401 请求。
  const loadSession = useCallback(async (me: User) => {
    setUser(me);
    setAuthStatus("signed-in");
    setLoadError(null);
    try {
      const [kbs, convs] = await Promise.all([api.listKnowledgeBases(), api.listConversations()]);
      setKnowledgeBases(kbs);
      setConversations(convs);
    } catch (error) {
      if (!(error instanceof ApiError && error.status === 401)) setLoadError((error as Error).message);
    }
  }, []);

  useEffect(() => {
    api.me()
      .then(loadSession)
      .catch((error: Error) => {
        if (error instanceof ApiError && error.status === 401) signOutLocally();
        else {
          setLoadError(error.message);
          setAuthStatus("anonymous");
        }
      });
  }, [loadSession, signOutLocally]);

  // 会话过期或在别处注销后，任何接口返回 401 都回到登录页。
  useEffect(() => {
    window.addEventListener(AUTH_EXPIRED_EVENT, signOutLocally);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, signOutLocally);
  }, [signOutLocally]);

  const login = useCallback(async (username: string, password: string) => {
    await loadSession(await api.login(username, password));
  }, [loadSession]);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      signOutLocally();
    }
  }, [signOutLocally]);

  const kbName = useCallback(
    (kbId: string) => knowledgeBases.find((kb) => kb.id === kbId)?.name ?? kbId,
    [knowledgeBases],
  );

  const value = useMemo(
    () => ({
      authStatus,
      user,
      knowledgeBases,
      conversations,
      kbName,
      login,
      logout,
      refreshConversations,
      refreshKnowledgeBases,
      loadError,
    }),
    [authStatus, user, knowledgeBases, conversations, kbName, login, logout,
      refreshConversations, refreshKnowledgeBases, loadError],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppState {
  const value = useContext(AppContext);
  if (!value) throw new Error("useApp 必须在 AppProvider 内使用");
  return value;
}

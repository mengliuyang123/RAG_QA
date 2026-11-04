// 前端内部统一使用的领域模型。后端两种回答外壳（查询结果与历史轮次）
// 都在 adapters.ts 中转换成这里的结构，组件只依赖这些类型。

export type Role = "member" | "admin";

export interface QuestionQuota {
  limit: number;
  remaining: number;
  windowSeconds: number;
  retryAfter: number;
}

export interface User {
  id: string;
  name: string;
  role: Role;
  quota: QuestionQuota | null;
}

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  icon: string | null;
  documentsCount: number;
  updatedAt: string | null;
}

export interface KbDocument {
  docId: string;
  kbId: string | null;
  filename: string;
  fileType: string;
  chunksCount: number;
  updatedAt: string | null;
  fileStatus: "available" | "missing";
}

export interface SuggestedQuestion {
  id: string;
  kbId: string | null;
  text: string;
}

export interface KbScope {
  mode: "auto" | "explicit";
  kbIds: string[];
}

export type Lane = "direct" | "clarify" | "fast" | "agent";
export type Grounding = "grounded" | "partial" | "ungrounded";
export type Outcome = "answered" | "refused" | "clarification";
export type FeedbackRating = "up" | "down";

export interface Citation {
  position: number;
  kbId: string | null;
  docId: string;
  chunkId: string | null;
  filename: string;
  page: number | null;
  chunkText: string;
  docUpdatedAt: string | null;
}

// retrieved = relevant + excluded；相关但不足以回答的块只计入 relevant。
// 旧记录没有 retrieved/relevant，两者为 null。
export interface EvidenceStats {
  used: number;
  excluded: number;
  retrieved: number | null;
  relevant: number | null;
}

export interface Refusal {
  reason: "no_evidence" | "insufficient_evidence";
  foundSummary: string | null;
  missing: string[];
  suggestedKbId: string | null;
}

export interface Clarification {
  question: string;
  options: string[];
}

export interface Turn {
  id: string;
  conversationId: string;
  question: string;
  answer: string;
  answerMode: "direct" | "retrieval";
  outcome: Outcome;
  grounding: Grounding;
  lane: Lane | null;
  scopeExpanded: boolean;
  escalated: boolean;
  searchedKbIds: string[];
  /** 这一轮请求的范围；拒答时据此判断能否改到其他知识库重查。 */
  scopeMode: "auto" | "explicit";
  requestedKbIds: string[];
  retrievalCalls: number;
  citations: Citation[];
  evidenceStats: EvidenceStats | null;
  refusal: Refusal | null;
  clarification: Clarification | null;
  latencyMs: number;
  traceId: string | null;
  feedback: FeedbackRating | null;
  createdAt: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: string;
  turnsCount: number;
  kbScope: KbScope;
}

export interface Conversation extends ConversationSummary {
  turns: Turn[];
}

export interface StatusUpdate {
  phase: string;
  attempt: number | null;
  kbIds: string[];
  label: string | null;
  queuePosition: number | null;
}

export interface StreamHandlers {
  onStatus(update: StatusUpdate): void;
  onToken(token: string): void;
  onComplete(turn: Turn): void;
  onError(code: string, message: string): void;
}

export interface ChunkContext {
  docId: string;
  filename: string;
  kbId: string | null;
  docUpdatedAt: string | null;
  chunks: { chunkId: string; page: number | null; text: string; isTarget: boolean }[];
}

export interface UploadProgress {
  stage: "uploading" | "parsing" | "chunking" | "embedding" | "indexing" | "done";
  progress: number | null;
}

// 比例在样本为零时由后端返回 null，前端显示为"—"，不能当成 0%。
export interface KbInsight {
  kbId: string;
  questions: number;
  refusalRate: number | null;
  thumbsDownRate: number | null;
  expandedRate: number | null;
}

export interface RefusalRecord {
  id: string;
  question: string;
  kbIds: string[];
  missing: string[];
  askedAt: string;
}

// 后端尚未实现的能力。live 模式下为 false 的功能会退回本地数据，
// 页面据此显示"示例数据"提示，避免把假数据当成真实结果。
export type Feature =
  | "auth"
  | "knowledgeBases"
  | "kbScope"
  | "rename"
  | "feedback"
  | "chunkContext"
  | "insights"
  | "editKnowledgeBase";

export interface Api {
  readonly mode: "mock" | "live";
  readonly supports: Record<Feature, boolean>;
  /** 未登录时以 status 401 的 ApiError 拒绝。 */
  me(): Promise<User>;
  login(username: string, password: string): Promise<User>;
  logout(): Promise<void>;
  listKnowledgeBases(): Promise<KnowledgeBase[]>;
  listKbDocuments(kbId: string): Promise<KbDocument[]>;
  suggestedQuestions(): Promise<SuggestedQuestion[]>;
  listConversations(): Promise<ConversationSummary[]>;
  createConversation(scope: KbScope): Promise<ConversationSummary>;
  getConversation(id: string): Promise<Conversation>;
  renameConversation(id: string, title: string): Promise<void>;
  deleteConversation(id: string): Promise<void>;
  streamQuery(
    conversationId: string,
    question: string,
    scope: KbScope,
    handlers: StreamHandlers,
    signal: AbortSignal,
  ): Promise<void>;
  getChunkContext(citation: Citation): Promise<ChunkContext>;
  setFeedback(turnId: string, rating: FeedbackRating | null): Promise<void>;
  admin: {
    listDocuments(kbId: string | null): Promise<KbDocument[]>;
    uploadDocument(
      kbId: string,
      file: File,
      onProgress: (progress: UploadProgress) => void,
    ): Promise<KbDocument>;
    deleteDocument(docId: string): Promise<void>;
    createKnowledgeBase(kb: { id: string; name: string; description: string }): Promise<void>;
    updateKnowledgeBase(kbId: string, description: string): Promise<void>;
    insights(): Promise<KbInsight[]>;
    refusals(): Promise<RefusalRecord[]>;
  };
}

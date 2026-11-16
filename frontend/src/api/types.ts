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
  /** 入库失败（没有文本块）的文档仍会列出，便于删除。 */
  ingestionFailed?: boolean;
}

/** 同名不同内容时用户的选择：作为新版本替换，或另存为一篇。 */
export type UploadConflictChoice = "replace" | "keep_both";

/** 后端 documents 登记中已有的文档。 */
export interface ExistingDocument {
  docId: string;
  kbId: string;
  filename: string;
  status: "pending" | "ready" | "failed";
  uploadedAt: string;
}

export interface UploadOptions {
  onConflict?: UploadConflictChoice;
  replaceDocId?: string;
}

/** uploaded：新文档入库完成；exists：内容完全相同的文档已存在，未重复入库。 */
export type UploadResult =
  | { kind: "uploaded"; document: KbDocument }
  | { kind: "exists"; existing: ExistingDocument };

export interface BatchDeleteResult {
  deleted: string[];
  notFound: string[];
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
  /** 本轮每次工具调用的最终状态；旧记录或后端未提供时为空数组。 */
  toolCalls: ToolCall[];
  /** 本轮附带的结构化数据块（如表格）；缺失时为空数组。 */
  dataBlocks: DataBlock[];
}

// 接口约定 v1：SSE `event: tool` 与轮次字段 `tool_calls`。args / result 保留原始 JSON，
// 由各工具的渲染组件自行解析，未知工具不会因为字段不同而报错。
export type ToolCallStatus = "running" | "complete" | "error";

export interface ToolCall {
  id: string;
  name: string;
  status: ToolCallStatus;
  args: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: { code: string; message: string } | null;
}

// 接口约定 v1：SSE `event: data` 与轮次字段 `data_blocks`。type 未知时前端忽略。
export interface DataBlock {
  id: string;
  type: string;
  data: unknown;
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
  /** 工具调用的状态变化；同一 id 先 running，再推唯一终态。 */
  onTool(call: ToolCall): void;
  /** 结构化数据块（二期，后端可能晚到）。 */
  onData(block: DataBlock): void;
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
// 管理员在线修改的模型配置。secret 为 API Key，按产品要求后端返回明文。
export type ModelSettingKind = "secret" | "url" | "text" | "int";

export interface ModelSettingField {
  name: string;
  envKey: string;
  label: string;
  kind: ModelSettingKind;
  hint: string;
  value: string | number | null;
  defaultValue: string | number | null;
  /** 进程环境变量里也有同名键：重启后会以环境变量为准，.env 里的值不生效。 */
  overriddenByEnvironment: boolean;
}

export interface ModelSettingGroup {
  id: string;
  title: string;
  description: string;
  fields: ModelSettingField[];
}

export interface ModelSettings {
  groups: ModelSettingGroup[];
  /** 向量索引已锁定的维度；还没有入库任何向量时为 null。 */
  storedEmbeddingDimensions: number | null;
}

// 管理员账号管理。后端从不返回口令或哈希。
export interface ManagedUser {
  id: string;
  username: string;
  name: string;
  role: Role;
  disabled: boolean;
  createdAt: string;
}

export interface NewUser {
  username: string;
  name: string;
  password: string;
  role: Role;
}

// 知识缺口工作台：缺口来自拒答聚类与点踩反馈，后端只保存处理状态。
export type GapStatus = "open" | "document_added" | "verified";
export type GapSource = "refusal_cluster" | "feedback";

export interface GapCheckResult {
  question: string;
  outcome: Outcome | "error";
  answered: boolean;
}

export interface GapCheck {
  checkedAt: string;
  kbId: string | null;
  results: GapCheckResult[];
}

/** 从洞察接口得到、尚未必然被跟踪的缺口候选。 */
export interface GapCandidate {
  key: string;
  source: GapSource;
  title: string;
  questions: string[];
  kbId: string | null;
  /** 聚类内拒答次数；点踩为 1。 */
  count: number;
  lastAskedAt: string | null;
}

export interface GapCandidates {
  candidates: GapCandidate[];
  /** 某个来源加载失败时的提示，其余来源照常显示。 */
  warnings: string[];
}

/** 已保存处理状态的缺口。 */
export interface KnowledgeGap {
  key: string;
  source: GapSource;
  title: string;
  questions: string[];
  kbId: string | null;
  status: GapStatus;
  docId: string | null;
  lastCheck: GapCheck | null;
  updatedAt: string | null;
}

export type GapInput = Omit<KnowledgeGap, "updatedAt">;

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
      options?: UploadOptions,
    ): Promise<UploadResult>;
    deleteDocument(docId: string): Promise<void>;
    deleteDocuments(docIds: string[]): Promise<BatchDeleteResult>;
    createKnowledgeBase(kb: { id: string; name: string; description: string }): Promise<void>;
    updateKnowledgeBase(kbId: string, description: string): Promise<void>;
    insights(): Promise<KbInsight[]>;
    refusals(): Promise<RefusalRecord[]>;
    getModelSettings(): Promise<ModelSettings>;
    /** 只提交改动的字段；null 表示清空（恢复默认）。保存成功后立即生效。 */
    updateModelSettings(values: Record<string, string | number | null>): Promise<ModelSettings>;
    listUsers(): Promise<ManagedUser[]>;
    createUser(user: NewUser): Promise<ManagedUser>;
    updateUser(userId: string, changes: { role?: Role; disabled?: boolean }): Promise<ManagedUser>;
    /** 重置后该账号所有已登录的浏览器都需重新登录。 */
    resetPassword(userId: string, password: string): Promise<void>;
    gapCandidates(): Promise<GapCandidates>;
    listGaps(): Promise<KnowledgeGap[]>;
    /** 整体保存一个缺口的状态；verified 必须附带全部通过的复测。 */
    saveGap(gap: GapInput): Promise<KnowledgeGap>;
    /** 通过问答接口复测一个问题，限定到 kbId（null 表示自动路由）。 */
    checkQuestion(question: string, kbId: string | null): Promise<GapCheckResult>;
  };
}

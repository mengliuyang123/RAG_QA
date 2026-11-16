import { normalizeDocument, normalizeToolCall, normalizeTurn } from "./adapters";
import {
  MOCK_DOCUMENTS,
  MOCK_GAP_CANDIDATES,
  MOCK_INSIGHTS,
  MOCK_KNOWLEDGE_BASES,
  MOCK_REFUSALS,
  MOCK_SUGGESTED_QUESTIONS,
  MOCK_USERS,
  type MockDocument,
} from "./mockData";
import type {
  Api,
  Citation,
  ExistingDocument,
  ChunkContext,
  Conversation,
  ConversationSummary,
  FeedbackRating,
  KbDocument,
  KbScope,
  KnowledgeBase,
  KnowledgeGap,
  ManagedUser,
  ModelSettings,
  StreamHandlers,
  Turn,
  User,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, message: string, code = "request_failed") {
    super(message);
    this.status = status;
    this.code = code;
  }
}

/** 同名不同内容：前端据此弹出"替换 / 另存为一篇"选择框。 */
export class UploadConflictError extends ApiError {
  readonly existing: ExistingDocument;

  constructor(message: string, existing: ExistingDocument) {
    super(409, message, "document_name_conflict");
    this.existing = existing;
  }
}

function toExisting(doc: MockDocument): ExistingDocument {
  return { docId: doc.docId, kbId: doc.kbId, filename: doc.filename, status: "ready", uploadedAt: doc.updatedAt };
}

async function readText(file: File): Promise<string> {
  if (typeof file.text === "function") return file.text();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });
}

function uuid(): string {
  return crypto.randomUUID();
}

type RawTurn = Record<string, unknown>;

interface Scenario {
  statuses: Record<string, unknown>[];
  turn: RawTurn;
  /** 接口约定 v1 的 data_blocks（二期）。 */
  dataBlocks?: Record<string, unknown>[];
  /** 演示检索失败：第几次检索返回 error 终态（从 1 开始）。 */
  failAttempt?: number;
}

const RETRIEVAL_PHASES = new Set(["retrieving", "expanding"]);

// 按接口约定 v1 把场景里的检索阶段转成 retrieve_knowledge 调用：
// 首次为 initial，expanding 为 expand_scope，escalating 之后的检索为 escalate。
// 命中片段取自对应知识库的演示文档，被正式引用的分块标记 used。
export function mockToolCalls(question: string, scenario: Scenario): Record<string, unknown>[] {
  const citations = (scenario.turn.citations as { doc_id: string; chunk_id: string }[] | undefined) ?? [];
  const used = new Set(citations.map((item) => item.chunk_id));
  const calls: Record<string, unknown>[] = [];
  let escalated = false;
  let attempt = 0;
  for (const status of scenario.statuses) {
    if (status.phase === "escalating") escalated = true;
    if (!RETRIEVAL_PHASES.has(String(status.phase))) continue;
    attempt += 1;
    const kbIds = Array.isArray(status.kb_ids) ? (status.kb_ids as string[]) : [];
    const label = typeof status.label === "string" ? status.label.replace(/^正在补充查找[:：]s*/, "") : null;
    const reason = status.phase === "expanding" ? "expand_scope" : escalated && attempt > 1 ? "escalate" : "initial";
    const args = { query: label ?? question, kb_ids: kbIds, attempt, reason };
    const id = `tc_${String(attempt).padStart(2, "0")}`;
    if (scenario.failAttempt === attempt) {
      calls.push({ tool_call_id: id, tool_name: "retrieve_knowledge", status: "error", args,
        error: { code: "retrieval_failed", message: "检索服务暂时不可用" } });
      continue;
    }
    const hits = MOCK_DOCUMENTS
      .filter((doc) => kbIds.length === 0 || kbIds.includes(doc.kbId))
      .flatMap((doc) => doc.chunks.map((chunk, index) => ({ doc, chunk, index })))
      // 被引用的分块排在前面，模拟真实检索里高分片段被采用。
      .sort((x, y) => Number(used.has(`${y.doc.docId}:${y.index}`)) - Number(used.has(`${x.doc.docId}:${x.index}`)))
      .slice(0, 4)
      .map(({ doc, chunk, index }, order) => {
        const chunkId = `${doc.docId}:${index}`;
        return {
          doc_id: doc.docId, chunk_id: chunkId, kb_id: doc.kbId, filename: doc.filename,
          page: chunk.page, score: Number((0.86 - order * 0.07).toFixed(2)),
          snippet: chunk.text.slice(0, 200), used: used.has(chunkId),
        };
      });
    const relevant = hits.filter((hit) => hit.used).length || Math.min(1, hits.length);
    calls.push({
      tool_call_id: id, tool_name: "retrieve_knowledge", status: "complete", args,
      result: { retrieved: hits.length * 2, relevant, hits },
    });
  }
  return calls;
}

function cite(doc: MockDocument, chunkIndex: number, position: number) {
  return {
    position,
    kb_id: doc.kbId,
    doc_id: doc.docId,
    chunk_id: `${doc.docId}:${chunkIndex}`,
    filename: doc.filename,
    page: doc.chunks[chunkIndex].page,
    chunk_text: doc.chunks[chunkIndex].text,
    doc_updated_at: doc.updatedAt,
  };
}

const [persistenceDoc, retrievalDoc, deployDoc, , releaseDoc] = MOCK_DOCUMENTS;

// 根据问题关键词挑选一个演示场景，覆盖直接回答、澄清、快速通道、
// 扩大范围、升级 Agent 和拒答六种前端需要渲染的状态。
export function pickScenario(question: string, scope: KbScope): Scenario {
  const explicit = scope.mode === "explicit";
  const searched = (ids: string[]) => (explicit ? scope.kbIds : ids);
  const base = { answer_mode: "retrieval", latency_ms: 0, trace_id: `tr_${uuid().slice(0, 8)}` };

  if (/^(你好|谢谢|感谢|hi|hello)/i.test(question.trim())) {
    return {
      statuses: [{ phase: "understanding" }],
      turn: {
        ...base,
        answer: "不客气！有关于研发、运维或产品的问题，随时问我。",
        answer_mode: "direct",
        outcome: "answered",
        grounding: "ungrounded",
        lane: "direct",
        retrieval_calls: 0,
        citations: [],
        kb_scope: { mode: scope.mode, requested: scope.kbIds, searched: [] },
      },
    };
  }

  if (question.includes("那个") && question.length < 14) {
    return {
      statuses: [{ phase: "understanding" }],
      turn: {
        ...base,
        answer: "你指的是哪一类配置？",
        answer_mode: "direct",
        outcome: "clarification",
        grounding: "ungrounded",
        lane: "clarify",
        retrieval_calls: 0,
        citations: [],
        clarification: {
          question: "你指的是哪一类配置？",
          options: ["checkpoint 存储配置", "检索策略配置", "部署环境配置"],
        },
        kb_scope: { mode: scope.mode, requested: scope.kbIds, searched: [] },
      },
    };
  }

  if (/检索失败|故障演练/.test(question)) {
    return {
      statuses: [{ phase: "understanding" }, { phase: "retrieving", attempt: 1, kb_ids: searched(["dev"]) }, { phase: "verifying" }],
      failAttempt: 1,
      turn: {
        ...base,
        answer: "根据现有知识库，我无法可靠回答这个问题",
        outcome: "refused",
        grounding: "ungrounded",
        lane: "fast",
        retrieval_calls: 1,
        citations: [],
        evidence_stats: { used: 0, excluded: 0, retrieved: 0, relevant: 0 },
        refusal: { reason: "no_evidence", found_summary: null, missing: [], suggested_kb_id: null },
        kb_scope: { mode: scope.mode, requested: scope.kbIds, searched: searched(["dev"]) },
      },
    };
  }

  if (/redis/i.test(question)) {
    const kbs = searched(["dev", "ops", "product"]);
    return {
      statuses: [
        { phase: "understanding" },
        { phase: "retrieving", attempt: 1, kb_ids: explicit ? kbs : ["dev"] },
        ...(explicit ? [] : [{ phase: "expanding", attempt: 2, kb_ids: ["dev", "ops", "product"] }]),
        { phase: "verifying" },
      ],
      turn: {
        ...base,
        answer: "根据现有知识库，我无法可靠回答这个问题",
        outcome: "refused",
        grounding: "ungrounded",
        lane: "fast",
        scope_expanded: !explicit,
        retrieval_calls: explicit ? 1 : 2,
        citations: [],
        evidence_stats: { used: 0, excluded: 5, retrieved: 6, relevant: 1 },
        refusal: {
          reason: "insufficient_evidence",
          found_summary: "知识库中有 checkpoint 存储的介绍，但没有提到 Redis。",
          missing: ["Redis 作为缓存或 checkpoint 存储的配置方法"],
          // 与真实后端一致：理解层不一定给出建议，前端仍会列出其他知识库。
          suggested_kb_id: explicit && !scope.kbIds.includes("dev") ? "dev" : null,
        },
        kb_scope: { mode: scope.mode, requested: scope.kbIds, searched: kbs },
      },
    };
  }

  if (/(区别|对比|比较|哪个好)/.test(question)) {
    return {
      statuses: [
        { phase: "understanding" },
        { phase: "retrieving", attempt: 1, kb_ids: searched(["dev"]) },
        { phase: "escalating" },
        { phase: "retrieving", attempt: 2, kb_ids: searched(["dev"]), label: "正在补充查找：hybrid_rerank 的延迟" },
        { phase: "verifying" },
      ],
      turn: {
        ...base,
        answer:
          "两者的差别在于是否多一步重排：\n" +
          "• hybrid 同时做向量检索和 BM25 检索，再用 RRF 融合排名 [1]。\n" +
          "• hybrid_rerank 会把前 20 个候选再交给重排模型精排，准确率更高 [2]，但平均多出约 0.6 秒延迟 [3]。\n" +
          "对准确率要求高的场景建议用 hybrid_rerank。",
        outcome: "answered",
        grounding: "grounded",
        lane: "agent",
        escalated: true,
        retrieval_calls: 2,
        citations: [cite(retrievalDoc, 0, 1), cite(retrievalDoc, 1, 2), cite(retrievalDoc, 2, 3)],
        evidence_stats: { used: 3, excluded: 5, retrieved: 8, relevant: 3 },
        kb_scope: { mode: scope.mode, requested: scope.kbIds, searched: searched(["dev"]) },
      },
    };
  }

  if (/(锁定|锁住|locked|启动)/i.test(question)) {
    const expanded = !explicit;
    return {
      statuses: [
        { phase: "understanding" },
        { phase: "retrieving", attempt: 1, kb_ids: searched(["ops"]) },
        ...(expanded ? [{ phase: "expanding", attempt: 2, kb_ids: ["dev", "ops", "product"] }] : []),
        { phase: "verifying" },
      ],
      turn: {
        ...base,
        answer: expanded
          ? "这通常是多个进程同时写同一个 SQLite 文件导致的。\n1. 确认生产环境只启动了一个 worker [1]；\n2. 开启 WAL 模式并设置 busy_timeout [2]；\n3. 需要多 worker 部署时，改用 PostgresSaver [3]。"
          : "这通常是多个进程同时写同一个 SQLite 文件导致的，请确认生产环境只启动了一个 worker [1]。",
        outcome: "answered",
        grounding: expanded ? "grounded" : "partial",
        lane: "fast",
        scope_expanded: expanded,
        retrieval_calls: expanded ? 2 : 1,
        citations: expanded
          ? [cite(deployDoc, 1, 1), cite(persistenceDoc, 1, 2), cite(persistenceDoc, 2, 3)]
          : [cite(deployDoc, 1, 1)],
        evidence_stats: expanded
          ? { used: 3, excluded: 4, retrieved: 7, relevant: 3 }
          : { used: 1, excluded: 2, retrieved: 3, relevant: 1 },
        kb_scope: {
          mode: scope.mode,
          requested: scope.kbIds,
          searched: expanded ? ["ops", "dev", "product"] : searched(["ops"]),
        },
      },
    };
  }

  if (/(版本|更新|新功能)/.test(question)) {
    return {
      statuses: [
        { phase: "understanding" },
        { phase: "retrieving", attempt: 1, kb_ids: searched(["product"]) },
        { phase: "verifying" },
      ],
      turn: {
        ...base,
        answer: "0.2 版本主要有两项更新：新增多轮会话并标注可点击的引用来源 [1]；新增文档管理页面，可以查看入库状态和删除文档 [2]。",
        outcome: "answered",
        grounding: "grounded",
        lane: "fast",
        retrieval_calls: 1,
        citations: [cite(releaseDoc, 0, 1), cite(releaseDoc, 1, 2)],
        evidence_stats: { used: 2, excluded: 3, retrieved: 5, relevant: 2 },
        kb_scope: { mode: scope.mode, requested: scope.kbIds, searched: searched(["product"]) },
      },
      dataBlocks: [{
        data_id: "d_01",
        type: "table",
        data: {
          caption: "0.2 版本更新一览",
          columns: ["版本", "更新内容", "涉及页面"],
          rows: [["0.2", "多轮会话与可点击的引用来源", "对话页"], ["0.2", "文档管理：查看入库状态、删除文档", "知识库管理"]],
          source_positions: [1, 2],
        },
      }],
    };
  }

  return {
    statuses: [
      { phase: "understanding" },
      { phase: "retrieving", attempt: 1, kb_ids: searched(["dev"]) },
      { phase: "verifying" },
    ],
    turn: {
      ...base,
      answer: "Checkpointer 会在图的每个超步保存状态，让对话可以中断后恢复 [1]。本地开发可以用 SQLite 版本 [2]，生产环境推荐 PostgresSaver [3]。",
      outcome: "answered",
      grounding: "grounded",
      lane: "fast",
      retrieval_calls: 1,
      citations: [cite(persistenceDoc, 0, 1), cite(persistenceDoc, 1, 2), cite(persistenceDoc, 2, 3)],
      evidence_stats: { used: 3, excluded: 2, retrieved: 5, relevant: 3 },
      kb_scope: { mode: scope.mode, requested: scope.kbIds, searched: searched(["dev"]) },
    },
  };
}

interface StoredConversation {
  id: string;
  title: string;
  updatedAt: string;
  kbScope: KbScope;
  turns: Turn[];
}

export function createMockApi(): Api {
  const conversations = new Map<string, StoredConversation>();
  const documents = [...MOCK_DOCUMENTS];
  const knowledgeBases = MOCK_KNOWLEDGE_BASES.map((kb) => ({ ...kb }));
  const busy = new Set<string>();
  const users: ManagedUser[] = MOCK_USERS.map((item) => ({ ...item }));
  const gaps = new Map<string, KnowledgeGap>();
  const user: User = {
    id: "u_demo",
    name: "小刘",
    role: "admin",
    quota: { limit: 10, remaining: 10, windowSeconds: 60, retryAfter: 0 },
  };

  const toDocument = (doc: MockDocument): KbDocument =>
    normalizeDocument({
      doc_id: doc.docId,
      kb_id: doc.kbId,
      filename: doc.filename,
      chunks_count: doc.chunks.length,
      updated_at: doc.updatedAt,
      file_status: doc.fileStatus,
    });

  const summary = (conversation: StoredConversation): ConversationSummary => ({
    id: conversation.id,
    title: conversation.title,
    updatedAt: conversation.updatedAt,
    turnsCount: conversation.turns.length,
    kbScope: conversation.kbScope,
  });

  const find = (id: string): StoredConversation => {
    const conversation = conversations.get(id);
    if (!conversation) throw new ApiError(404, "会话不存在", "not_found");
    return conversation;
  };

  return {
    mode: "mock",
    supports: {
      auth: true,
      knowledgeBases: true,
      kbScope: true,
      rename: true,
      feedback: true,
      chunkContext: true,
      insights: true,
      editKnowledgeBase: true,
    },

    async me() {
      await sleep(80);
      return user;
    },

    async login() {
      await sleep(80);
      return user;
    },

    async logout() {
      await sleep(40);
    },

    async listKnowledgeBases(): Promise<KnowledgeBase[]> {
      await sleep(120);
      return knowledgeBases.map((kb) => ({
        ...kb,
        documentsCount: documents.filter((doc) => doc.kbId === kb.id).length,
        updatedAt: documents
          .filter((doc) => doc.kbId === kb.id)
          .map((doc) => doc.updatedAt)
          .sort()
          .at(-1) ?? null,
      }));
    },

    async listKbDocuments(kbId) {
      await sleep(120);
      return documents.filter((doc) => doc.kbId === kbId).map(toDocument);
    },

    async suggestedQuestions() {
      await sleep(80);
      return MOCK_SUGGESTED_QUESTIONS;
    },

    async listConversations() {
      await sleep(80);
      return [...conversations.values()]
        .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
        .map(summary);
    },

    async createConversation(scope) {
      await sleep(60);
      const now = new Date().toISOString();
      const conversation: StoredConversation = { id: uuid(), title: "新对话", updatedAt: now, kbScope: scope, turns: [] };
      conversations.set(conversation.id, conversation);
      return summary(conversation);
    },

    async getConversation(id): Promise<Conversation> {
      await sleep(80);
      const conversation = find(id);
      return { ...summary(conversation), turns: [...conversation.turns] };
    },

    async renameConversation(id, title) {
      await sleep(60);
      find(id).title = title;
    },

    async deleteConversation(id) {
      await sleep(60);
      if (busy.has(id)) throw new ApiError(409, "上一个问题还在回答中", "conversation_busy");
      find(id);
      conversations.delete(id);
    },

    async streamQuery(conversationId, question, scope, handlers: StreamHandlers, signal) {
      const conversation = find(conversationId);
      if (busy.has(conversationId)) {
        throw new ApiError(409, "上一个问题还在回答中", "conversation_busy");
      }
      busy.add(conversationId);
      const started = performance.now();
      try {
        const scenario = pickScenario(question, scope);
        const toolCalls = mockToolCalls(question, scenario);
        let attempt = 0;
        for (const status of scenario.statuses) {
          handlers.onStatus({
            phase: String(status.phase),
            attempt: typeof status.attempt === "number" ? status.attempt : null,
            kbIds: Array.isArray(status.kb_ids) ? (status.kb_ids as string[]) : [],
            label: typeof status.label === "string" ? status.label : null,
            queuePosition: null,
          });
          // 检索阶段同时按约定推送 tool 事件：先 running，再唯一终态，全部在第一个 token 之前。
          const call = RETRIEVAL_PHASES.has(String(status.phase)) ? toolCalls[attempt++] : undefined;
          if (call) {
            const running = normalizeToolCall({ ...call, status: "running", result: undefined, error: undefined });
            if (running) handlers.onTool(running);
            await sleep(900, signal);
            const final = normalizeToolCall(call);
            if (final) handlers.onTool(final);
            await sleep(250, signal);
          } else {
            await sleep(550, signal);
          }
        }
        const turn = normalizeTurn(
          {
            ...scenario.turn,
            tool_calls: toolCalls,
            data_blocks: scenario.dataBlocks ?? [],
            turn_id: uuid(),
            conversation_id: conversationId,
            latency_ms: Math.round(performance.now() - started),
            created_at: new Date().toISOString(),
          },
          { conversationId, question },
        );
        // 与后端约定一致：正文通过校验后才分片发出。
        for (let offset = 0; offset < turn.answer.length; offset += 6) {
          handlers.onToken(turn.answer.slice(offset, offset + 6));
          await sleep(25, signal);
        }
        for (const block of turn.dataBlocks) handlers.onData(block);
        conversation.turns.push(turn);
        conversation.updatedAt = turn.createdAt;
        if (conversation.turns.length === 1) conversation.title = question.slice(0, 50);
        handlers.onComplete(turn);
      } finally {
        busy.delete(conversationId);
      }
    },

    async getChunkContext(citation: Citation): Promise<ChunkContext> {
      await sleep(120);
      const doc = documents.find((item) => item.docId === citation.docId);
      if (!doc) throw new ApiError(410, "文档已被删除", "document_deleted");
      const target = Number(citation.chunkId?.split(":").pop());
      const from = Math.max(0, target - 1);
      const to = Math.min(doc.chunks.length - 1, target + 1);
      return {
        docId: doc.docId,
        filename: doc.filename,
        kbId: doc.kbId,
        docUpdatedAt: doc.updatedAt,
        chunks: doc.chunks.slice(from, to + 1).map((chunk, offset) => ({
          chunkId: `${doc.docId}:${from + offset}`,
          page: chunk.page,
          text: chunk.text,
          isTarget: from + offset === target,
        })),
      };
    },

    async setFeedback(turnId, rating: FeedbackRating | null) {
      await sleep(60);
      for (const conversation of conversations.values()) {
        const turn = conversation.turns.find((item) => item.id === turnId);
        if (turn) {
          const index = conversation.turns.indexOf(turn);
          conversation.turns[index] = { ...turn, feedback: rating };
        }
      }
    },

    admin: {
      async listDocuments(kbId) {
        await sleep(120);
        return documents.filter((doc) => !kbId || doc.kbId === kbId).map(toDocument);
      },

      async uploadDocument(kbId, file, onProgress, options = {}) {
        const content = await readText(file);
        const live = documents.filter((doc) => doc.kbId === kbId);
        const same = live.find((doc) => doc.content !== undefined && doc.content === content);
        if (same) return { kind: "exists", existing: toExisting(same) };
        const named = options.replaceDocId
          ? live.find((doc) => doc.docId === options.replaceDocId)
          : live.find((doc) => doc.filename === file.name);
        if (named && !options.onConflict) {
          throw new UploadConflictError("知识库中已有同名文档，请选择替换或另存为一篇", toExisting(named));
        }
        const stages = ["uploading", "parsing", "chunking", "embedding", "indexing"] as const;
        for (const stage of stages) {
          if (stage === "embedding") {
            for (let step = 1; step <= 5; step += 1) {
              onProgress({ stage, progress: step / 5 });
              await sleep(220);
            }
          } else {
            onProgress({ stage, progress: null });
            await sleep(350);
          }
        }
        const doc: MockDocument = {
          docId: uuid(),
          kbId,
          filename: file.name,
          updatedAt: new Date().toISOString(),
          fileStatus: "available",
          chunks: [{ text: `${file.name} 的示例内容。`, page: null }],
          content,
        };
        documents.unshift(doc);
        // 新版本入库完成后才移除旧版本，与后端替换语义一致。
        if (named && options.onConflict === "replace") {
          documents.splice(documents.indexOf(named), 1);
        }
        onProgress({ stage: "done", progress: 1 });
        return { kind: "uploaded", document: toDocument(doc) };
      },

      async deleteDocument(docId) {
        await sleep(150);
        const index = documents.findIndex((doc) => doc.docId === docId);
        if (index === -1) throw new ApiError(404, "文档不存在", "not_found");
        documents.splice(index, 1);
      },

      async deleteDocuments(docIds) {
        await sleep(150);
        const deleted: string[] = [];
        const notFound: string[] = [];
        for (const docId of new Set(docIds)) {
          const index = documents.findIndex((doc) => doc.docId === docId);
          if (index === -1) {
            notFound.push(docId);
          } else {
            documents.splice(index, 1);
            deleted.push(docId);
          }
        }
        return { deleted, notFound };
      },

      async createKnowledgeBase(kb) {
        await sleep(150);
        if (knowledgeBases.some((item) => item.id === kb.id)) {
          throw new ApiError(409, "知识库 ID 已存在", "conflict");
        }
        knowledgeBases.push({ ...kb, icon: "books" });
      },

      async updateKnowledgeBase(kbId, description) {
        await sleep(150);
        const kb = knowledgeBases.find((item) => item.id === kbId);
        if (!kb) throw new ApiError(404, "知识库不存在", "not_found");
        kb.description = description;
      },

      async insights() {
        await sleep(120);
        return MOCK_INSIGHTS;
      },

      async refusals() {
        await sleep(120);
        return MOCK_REFUSALS;
      },

      async listUsers() {
        await sleep(100);
        return users.map((item) => ({ ...item }));
      },

      async createUser(input) {
        await sleep(150);
        const username = input.username.trim().toLowerCase();
        if (!/^[a-z0-9][a-z0-9_.-]{2,63}$/.test(username)) {
          throw new ApiError(422, "登录名需 3–64 位，只能包含小写字母、数字、点、下划线和连字符", "validation_error");
        }
        if (input.password.length < 15 || input.password.length > 128) {
          throw new ApiError(422, "口令长度必须为15–128字符", "validation_error");
        }
        if (users.some((item) => item.username === username)) {
          throw new ApiError(409, "登录名已被使用", "user_exists");
        }
        const created: ManagedUser = {
          id: uuid(), username, name: input.name.trim(), role: input.role, disabled: false,
          createdAt: new Date().toISOString(),
        };
        users.push(created);
        return { ...created };
      },

      async updateUser(userId, changes) {
        await sleep(120);
        const target = users.find((item) => item.id === userId);
        if (!target) throw new ApiError(404, "账号不存在", "user_not_found");
        if (userId === user.id && (changes.role === "member" || changes.disabled === true)) {
          throw new ApiError(409, "不能降级或禁用自己的账号", "self_lockout");
        }
        const role = changes.role ?? target.role;
        const disabled = changes.disabled ?? target.disabled;
        const wasAdmin = target.role === "admin" && !target.disabled;
        const otherAdmins = users.filter((item) => item.id !== userId && item.role === "admin" && !item.disabled);
        if (wasAdmin && !(role === "admin" && !disabled) && otherAdmins.length === 0) {
          throw new ApiError(409, "至少需要保留一名启用的管理员", "last_admin");
        }
        Object.assign(target, { role, disabled });
        return { ...target };
      },

      async resetPassword(userId, password) {
        await sleep(120);
        if (!users.some((item) => item.id === userId)) throw new ApiError(404, "账号不存在", "user_not_found");
        if (password.length < 15 || password.length > 128) {
          throw new ApiError(422, "口令长度必须为15–128字符", "validation_error");
        }
      },

      async gapCandidates() {
        await sleep(150);
        return { candidates: structuredClone(MOCK_GAP_CANDIDATES), warnings: [] };
      },

      async listGaps() {
        await sleep(80);
        return [...gaps.values()]
          .sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)))
          .map((gap) => structuredClone(gap));
      },

      async saveGap(input) {
        await sleep(100);
        if (input.status === "verified" && !(input.lastCheck && input.lastCheck.results.every((item) => item.answered))) {
          throw new ApiError(422, "只有全部样例问题复测通过后才能标记为已验证", "gap_not_verified");
        }
        const saved: KnowledgeGap = { ...structuredClone(input), updatedAt: new Date().toISOString() };
        gaps.set(input.key, saved);
        return structuredClone(saved);
      },

      async checkQuestion(question, kbId) {
        await sleep(300);
        // 演示规则：本次会话里向该知识库上传过文档后，复测视为能回答。
        const answered = documents.some((doc) => doc.content !== undefined && (!kbId || doc.kbId === kbId));
        return { question, outcome: answered ? "answered" : "refused", answered };
      },

      async getModelSettings() {
        await sleep(120);
        return structuredClone(modelSettings);
      },

      async updateModelSettings(values) {
        await sleep(300);
        const fields = new Map(modelSettings.groups.flatMap((group) => group.fields).map((field) => [field.name, field]));
        for (const name of Object.keys(values)) {
          if (!fields.has(name)) throw new ApiError(422, `不支持在线修改的配置项：${name}`);
        }
        // 与后端一致：已入库向量为 1536 维，改成其他维度会被拒绝。
        const dimensions = values.embedding_dimensions;
        if (dimensions !== undefined && dimensions !== null && dimensions !== "" && Number(dimensions) !== 1536) {
          throw new ApiError(409, `新向量模型输出 ${dimensions} 维，已入库的向量是 1536 维，两者不兼容。请保持原维度，或清空知识库后重新入库。`);
        }
        for (const [name, value] of Object.entries(values)) {
          const field = fields.get(name)!;
          field.value = value === null || value === "" ? field.defaultValue : field.kind === "int" ? Number(value) : value;
        }
        return structuredClone(modelSettings);
      },
    },
  };
}

const field = (
  name: string, label: string, kind: "secret" | "url" | "text" | "int",
  value: string | number | null, defaultValue: string | number | null = null, hint = "",
) => ({ name, envKey: name.toUpperCase(), label, kind, hint, value, defaultValue, overriddenByEnvironment: false });

const INHERIT = "留空时沿用生成模型的配置";

// 演示数据：结构与后端 GET /api/admin/settings/models 一致，Key 为假值。
const modelSettings: ModelSettings = {
  storedEmbeddingDimensions: 1536,
  groups: [
    { id: "llm", title: "生成模型", description: "Agent 规划、检索决策和最终回答使用的主模型。", fields: [
      field("llm_api_key", "API Key", "secret", "sk-demo-llm-key", null, "留空且地址为 OpenAI 官方时，使用下方的 OpenAI 通用 Key"),
      field("llm_base_url", "API 地址", "url", "https://api.openai.com/v1", "https://api.openai.com/v1", "OpenAI 兼容接口的根地址"),
      field("llm_model", "模型名", "text", "gpt-4o-mini", "gpt-4o-mini"),
    ] },
    { id: "understanding", title: "问题理解模型", description: "改写问题、判断走哪条处理通道。可以单独配一个更快更便宜的模型。", fields: [
      field("understanding_api_key", "API Key", "secret", null, null, INHERIT),
      field("understanding_base_url", "API 地址", "url", null, null, INHERIT),
      field("understanding_model", "模型名", "text", null, null, INHERIT),
    ] },
    { id: "judge", title: "质量审核模型", description: "判断检索证据是否充分、审核回答质量。", fields: [
      field("judge_api_key", "API Key", "secret", null, null, INHERIT),
      field("judge_base_url", "API 地址", "url", null, null, INHERIT),
      field("judge_model", "模型名", "text", null, null, INHERIT),
    ] },
    { id: "embedding", title: "向量模型", description: "文档入库和提问检索时把文本转成向量。更换模型或维度后，已有向量不再兼容。", fields: [
      field("embedding_api_key", "API Key", "secret", "sk-demo-embed-key", null, "留空且地址为 OpenAI 官方时，使用下方的 OpenAI 通用 Key"),
      field("embedding_base_url", "API 地址", "url", "https://api.openai.com/v1", "https://api.openai.com/v1"),
      field("embedding_model", "模型名", "text", "text-embedding-3-small", "text-embedding-3-small"),
      field("embedding_dimensions", "向量维度", "int", null, null, "留空使用模型默认维度；只有支持 dimensions 参数的模型（如 text-embedding-3）才能指定"),
    ] },
    { id: "reranker", title: "重排序模型", description: "hybrid_rerank 策略下对召回结果精排。", fields: [
      field("reranker_api_key", "API Key", "secret", null, null, "不配置时不启用重排序"),
      field("reranker_base_url", "API 地址", "url", null),
      field("reranker_model", "模型名", "text", "qwen3-rerank", "qwen3-rerank"),
    ] },
    { id: "shared", title: "OpenAI 通用 Key", description: "各模型地址为 OpenAI 官方且没有单独配置 Key 时使用。", fields: [
      field("openai_api_key", "API Key", "secret", null),
    ] },
  ],
};

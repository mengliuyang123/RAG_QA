import { normalizeDocument, normalizeTurn } from "./adapters";
import {
  MOCK_DOCUMENTS,
  MOCK_INSIGHTS,
  MOCK_KNOWLEDGE_BASES,
  MOCK_REFUSALS,
  MOCK_SUGGESTED_QUESTIONS,
  type MockDocument,
} from "./mockData";
import type {
  Api,
  Citation,
  ChunkContext,
  Conversation,
  ConversationSummary,
  FeedbackRating,
  KbDocument,
  KbScope,
  KnowledgeBase,
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
        for (const status of scenario.statuses) {
          handlers.onStatus({
            phase: String(status.phase),
            attempt: typeof status.attempt === "number" ? status.attempt : null,
            kbIds: Array.isArray(status.kb_ids) ? (status.kb_ids as string[]) : [],
            label: typeof status.label === "string" ? status.label : null,
            queuePosition: null,
          });
          await sleep(550, signal);
        }
        const turn = normalizeTurn(
          {
            ...scenario.turn,
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

      async uploadDocument(kbId, file, onProgress) {
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
        };
        documents.unshift(doc);
        onProgress({ stage: "done", progress: 1 });
        return toDocument(doc);
      },

      async deleteDocument(docId) {
        await sleep(150);
        const index = documents.findIndex((doc) => doc.docId === docId);
        if (index === -1) throw new ApiError(404, "文档不存在", "not_found");
        documents.splice(index, 1);
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
    },
  };
}

import {
  errorMessage,
  normalizeDocument,
  normalizeStatus,
  normalizeTurn,
} from "./adapters";
import { ApiError } from "./mock";
import type {
  Api,
  Citation,
  ConversationSummary,
  KbInsight,
  KbScope,
  KnowledgeBase,
  RefusalRecord,
  User,
} from "./types";
import { SseParser } from "../lib/sse";

type Raw = Record<string, unknown>;

// 后端用两个 Cookie 认证：会话 Cookie 为 HttpOnly，前端读不到；
// CSRF Cookie 可读，所有写请求都要把它原样放进请求头。
const CSRF_COOKIE = "rag_csrf";
const CSRF_HEADER = "X-CSRF-Token";
export const AUTH_EXPIRED_EVENT = "rag:auth-expired";

const INGESTION_POLL_MS = 1000;

function readCookie(name: string): string | null {
  for (const part of document.cookie.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return decodeURIComponent(rest.join("="));
  }
  return null;
}

function errorCode(body: unknown): string | undefined {
  if (!body || typeof body !== "object") return undefined;
  const error = (body as Raw).error;
  return error && typeof error === "object" && typeof (error as Raw).code === "string"
    ? String((error as Raw).code)
    : undefined;
}

function toApiError(status: number, body: unknown): ApiError {
  if (status === 401) window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
  return new ApiError(status, errorMessage(body, `请求失败（${status}）`), errorCode(body));
}

function writeHeaders(method: string): Record<string, string> {
  if (method === "GET" || method === "HEAD") return {};
  const token = readCookie(CSRF_COOKIE);
  return token ? { [CSRF_HEADER]: token } : {};
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = (init.method ?? "GET").toUpperCase();
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init.body && !(init.body instanceof FormData)
          ? { "Content-Type": "application/json" }
          : {}),
        ...writeHeaders(method),
        ...init.headers,
      },
      credentials: "same-origin",
    });
  } catch {
    throw new ApiError(0, "无法连接到服务，请检查网络或后端是否已启动", "network_error");
  }
  if (response.status === 204) return undefined as T;
  const body = await response.json().catch(() => null);
  if (!response.ok) throw toApiError(response.status, body);
  return body as T;
}

function toUser(raw: Raw): User {
  const quota = raw.quota as Raw | undefined;
  return {
    id: String(raw.id),
    name: String(raw.name ?? "用户"),
    role: raw.role === "admin" ? "admin" : "member",
    quota: quota && typeof quota === "object"
      ? {
          limit: Number(quota.limit ?? 0),
          remaining: Number(quota.remaining ?? 0),
          windowSeconds: Number(quota.window_seconds ?? 60),
          retryAfter: Number(quota.retry_after ?? 0),
        }
      : null,
  };
}

function toScope(raw: unknown): KbScope {
  if (!raw || typeof raw !== "object") return { mode: "auto", kbIds: [] };
  const value = raw as Raw;
  const kbIds = Array.isArray(value.kb_ids) ? value.kb_ids.map(String) : [];
  return value.mode === "explicit" && kbIds.length > 0
    ? { mode: "explicit", kbIds }
    : { mode: "auto", kbIds: [] };
}

function scopeBody(scope: KbScope) {
  return scope.mode === "explicit"
    ? { mode: "explicit", kb_ids: scope.kbIds }
    : { mode: "auto", kb_ids: [] };
}

function toSummary(raw: Raw): ConversationSummary {
  return {
    id: String(raw.id),
    title: String(raw.title ?? "新对话"),
    updatedAt: String(raw.updated_at ?? raw.created_at ?? new Date().toISOString()),
    turnsCount: Number(raw.turns_count ?? 0),
    kbScope: toScope(raw.kb_scope),
  };
}

function toKnowledgeBase(raw: Raw): KnowledgeBase {
  return {
    id: String(raw.id),
    name: String(raw.name ?? raw.id),
    description: String(raw.description ?? ""),
    icon: typeof raw.icon === "string" ? raw.icon : null,
    documentsCount: Number(raw.documents_count ?? 0),
    updatedAt: typeof raw.updated_at === "string" ? raw.updated_at : null,
  };
}

function rate(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function createLiveApi(): Api {
  return {
    mode: "live",
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
      return toUser(await request<Raw>("/api/me"));
    },

    async login(username, password) {
      const body = await request<{ user: Raw }>("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
      // 登录响应不含额度，再读一次 /me 拿到完整身份。
      return body.user ? this.me() : Promise.reject(new ApiError(500, "登录响应无效"));
    },

    async logout() {
      await request("/api/auth/logout", { method: "POST" });
    },

    async listKnowledgeBases() {
      const body = await request<{ knowledge_bases: Raw[] }>("/api/knowledge-bases");
      return body.knowledge_bases.map(toKnowledgeBase);
    },

    async listKbDocuments(kbId) {
      const body = await request<{ documents: Raw[] }>(
        `/api/knowledge-bases/${encodeURIComponent(kbId)}/documents?limit=100`,
      );
      return body.documents.map((raw) => normalizeDocument({ ...raw, kb_id: kbId }));
    },

    async suggestedQuestions() {
      const body = await request<{ questions: Raw[] }>("/api/suggested-questions?limit=20");
      return body.questions.map((raw) => ({
        id: String(raw.id),
        kbId: typeof raw.kb_id === "string" ? raw.kb_id : null,
        text: String(raw.text),
      }));
    },

    async listConversations() {
      const body = await request<{ conversations: Raw[] }>("/api/conversations");
      return body.conversations.map(toSummary);
    },

    async createConversation(scope) {
      return toSummary(await request<Raw>("/api/conversations", {
        method: "POST",
        body: JSON.stringify({ kb_scope: scopeBody(scope) }),
      }));
    },

    async getConversation(id) {
      const raw = await request<Raw & { turns?: Raw[] }>(`/api/conversations/${id}`);
      return {
        ...toSummary(raw),
        turns: (raw.turns ?? []).map((turn) => normalizeTurn(turn, { conversationId: id })),
      };
    },

    async renameConversation(id, title) {
      await request(`/api/conversations/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ title }),
      });
    },

    async deleteConversation(id) {
      await request(`/api/conversations/${id}`, { method: "DELETE" });
    },

    async streamQuery(conversationId, question, scope, handlers, signal) {
      let response: Response;
      try {
        response = await fetch(`/api/conversations/${conversationId}/query/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
            ...writeHeaders("POST"),
          },
          body: JSON.stringify({ question, kb_scope: scopeBody(scope) }),
          credentials: "same-origin",
          signal,
        });
      } catch (error) {
        if ((error as Error).name === "AbortError") throw error;
        throw new ApiError(0, "无法连接到服务，请检查网络或后端是否已启动", "network_error");
      }
      if (!response.ok || !response.body) {
        const body = await response.json().catch(() => null);
        throw toApiError(response.status, body);
      }

      const parser = new SseParser();
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let finished = false;
      const dispatch = (event: string, data: string) => {
        let payload: Raw;
        try {
          payload = JSON.parse(data) as Raw;
        } catch {
          return;
        }
        if (event === "status") handlers.onStatus(normalizeStatus(payload));
        else if (event === "token" && typeof payload.token === "string") handlers.onToken(payload.token);
        else if (event === "complete") {
          finished = true;
          handlers.onComplete(normalizeTurn(payload, { conversationId, question }));
        } else if (event === "error") {
          finished = true;
          handlers.onError(String(payload.code ?? "error"), String(payload.message ?? "回答失败"));
        }
      };

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        for (const message of parser.push(value)) dispatch(message.event, message.data);
      }
      for (const message of parser.flush()) dispatch(message.event, message.data);
      if (!finished) handlers.onError("stream_interrupted", "连接中断，回答未完成");
    },

    async getChunkContext(citation: Citation) {
      if (!citation.chunkId) throw new ApiError(404, "这条引用没有原文位置信息", "not_found");
      const raw = await request<Raw & { chunks: Raw[] }>(
        `/api/documents/${citation.docId}/chunks/${encodeURIComponent(citation.chunkId)}?context=1`,
      );
      return {
        docId: String(raw.doc_id),
        filename: String(raw.filename),
        kbId: typeof raw.kb_id === "string" ? raw.kb_id : null,
        docUpdatedAt: typeof raw.doc_updated_at === "string" ? raw.doc_updated_at : null,
        chunks: raw.chunks.map((chunk) => ({
          chunkId: String(chunk.chunk_id),
          page: typeof chunk.page === "number" ? chunk.page : null,
          text: String(chunk.text),
          isTarget: chunk.is_target === true,
        })),
      };
    },

    async setFeedback(turnId, rating) {
      await request(`/api/turns/${turnId}/feedback`, {
        method: "PUT",
        body: JSON.stringify({ rating }),
      });
    },

    admin: {
      async listDocuments(kbId) {
        const query = kbId ? `?kb_id=${encodeURIComponent(kbId)}` : "";
        const body = await request<{ documents: Raw[] }>(`/api/documents${query}`);
        return body.documents.map(normalizeDocument);
      },

      async uploadDocument(kbId, file, onProgress) {
        const form = new FormData();
        form.append("file", file);
        onProgress({ stage: "uploading", progress: null });
        const queued = await request<Raw>(
          `/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
          { method: "POST", body: form },
        );
        const jobId = String(queued.job_id);
        // 入库在后端异步执行，轮询任务直到成功或失败。
        for (;;) {
          await sleep(INGESTION_POLL_MS);
          const job = await request<Raw>(`/api/admin/ingestion-jobs/${encodeURIComponent(jobId)}`);
          if (job.status === "failed") {
            const failure = job.error as Raw | null;
            throw new ApiError(422, String(failure?.message ?? "入库失败"), String(failure?.code ?? "ingestion_failed"));
          }
          if (job.status === "succeeded") {
            onProgress({ stage: "done", progress: 1 });
            return normalizeDocument({
              doc_id: job.doc_id,
              kb_id: job.kb_id,
              filename: job.filename,
              chunks_count: job.chunks_count,
              updated_at: job.updated_at,
              file_status: "available",
            });
          }
          const stage = job.status === "queued" ? "uploading" : job.stage;
          onProgress({
            stage: stage === "parsing" || stage === "chunking" || stage === "embedding" || stage === "indexing"
              ? stage
              : "uploading",
            progress: typeof job.progress === "number" && job.status === "running" ? job.progress : null,
          });
        }
      },

      async deleteDocument(docId) {
        await request(`/api/documents/${docId}`, { method: "DELETE" });
      },

      async createKnowledgeBase(kb) {
        await request("/api/admin/knowledge-bases", {
          method: "POST",
          body: JSON.stringify(kb),
        });
      },

      async updateKnowledgeBase(kbId, description) {
        await request(`/api/admin/knowledge-bases/${encodeURIComponent(kbId)}`, {
          method: "PATCH",
          body: JSON.stringify({ description }),
        });
      },

      async insights(): Promise<KbInsight[]> {
        const body = await request<{ knowledge_bases: Raw[] }>("/api/admin/insights/overview?days=7");
        return body.knowledge_bases.map((raw) => ({
          kbId: String(raw.kb_id),
          questions: Number(raw.questions ?? 0),
          refusalRate: rate(raw.refusal_rate),
          thumbsDownRate: rate(raw.thumbs_down_rate),
          expandedRate: rate(raw.expanded_rate),
        }));
      },

      async refusals(): Promise<RefusalRecord[]> {
        // 使用列表模式：聚类模式需要调用 Embedding 服务，会产生费用。
        const body = await request<{ refusals: Raw[] }>("/api/admin/insights/refusals?limit=50");
        return body.refusals.map((raw) => {
          const refusal = (raw.refusal ?? {}) as Raw;
          return {
            id: String(raw.turn_id),
            question: String(raw.question),
            kbIds: Array.isArray(raw.kb_ids) ? raw.kb_ids.map(String) : [],
            missing: Array.isArray(refusal.missing) ? refusal.missing.map(String) : [],
            askedAt: String(raw.created_at),
          };
        });
      },
    },
  };
}

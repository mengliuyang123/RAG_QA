import {
  errorMessage,
  normalizeCitation,
  normalizeDataBlock,
  normalizeDocument,
  normalizeExistingDocument,
  normalizeToolCall,
  normalizeStatus,
  normalizeTurn,
} from "./adapters";
import { ApiError, UploadConflictError } from "./mock";
import type {
  Api,
  Citation,
  ConversationSummary,
  KbInsight,
  KbScope,
  KnowledgeBase,
  GapCandidate,
  GapCheck,
  GapCheckResult,
  GapInput,
  CostEstimate,
  DocumentReview,
  FaqState,
  FeedbackItem,
  KbAccess,
  KnowledgeGap,
  ManagedUser,
  TraceSummary,
  UsageRow,
  ModelSettings,
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
  const message = errorMessage(body, `请求失败（${status}）`);
  const code = errorCode(body);
  if (status === 409 && code === "document_name_conflict") {
    const details = ((body as Raw).error as Raw).details as Raw | undefined;
    const existing = details?.existing;
    if (existing && typeof existing === "object") {
      return new UploadConflictError(message, normalizeExistingDocument(existing as Raw));
    }
  }
  return new ApiError(status, message, code);
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
    editableKbIds: Array.isArray(raw.editable_kb_ids) ? raw.editable_kb_ids.map(String) : null,
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

    async changePassword(currentPassword, newPassword) {
      await request("/api/me/password", {
        method: "POST",
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
      });
    },

    async createShare(turnId) {
      const body = await request<Raw>(`/api/turns/${encodeURIComponent(turnId)}/shares`, {
        method: "POST",
        body: JSON.stringify({ expires_in_days: 30 }),
      });
      return { url: String(body.url), expiresAt: String(body.expires_at) };
    },

    async getShare(token) {
      const body = await request<Raw>(`/api/shares/${encodeURIComponent(token)}`);
      return {
        question: String(body.question),
        answer: String(body.answer),
        citations: (Array.isArray(body.citations) ? (body.citations as Raw[]) : []).map(normalizeCitation),
        sharedBy: String(body.shared_by ?? ""),
        createdAt: String(body.created_at ?? ""),
      };
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
        else if (event === "tool") {
          const call = normalizeToolCall(payload);
          if (call) handlers.onTool(call);
        } else if (event === "data") {
          const block = normalizeDataBlock(payload);
          if (block) handlers.onData(block);
        }
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

      async uploadDocument(kbId, file, onProgress, options = {}) {
        const form = new FormData();
        form.append("file", file);
        if (options.onConflict) form.append("on_conflict", options.onConflict);
        if (options.replaceDocId) form.append("replace_doc_id", options.replaceDocId);
        onProgress({ stage: "uploading", progress: null });
        const queued = await request<Raw>(
          `/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
          { method: "POST", body: form },
        );
        // 内容完全相同：后端直接返回已有文档，不再入库。
        if (queued.status === "exists" && queued.existing && typeof queued.existing === "object") {
          return { kind: "exists", existing: normalizeExistingDocument(queued.existing as Raw) };
        }
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
            return {
              kind: "uploaded",
              document: normalizeDocument({
                doc_id: job.doc_id,
                kb_id: job.kb_id,
                filename: job.filename,
                chunks_count: job.chunks_count,
                updated_at: job.updated_at,
                file_status: "available",
              }),
            };
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

      async deleteDocuments(docIds) {
        const body = await request<{ deleted: unknown[]; not_found: unknown[] }>("/api/documents/batch-delete", {
          method: "POST",
          body: JSON.stringify({ doc_ids: docIds }),
        });
        return { deleted: body.deleted.map(String), notFound: body.not_found.map(String) };
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

      async listUsers() {
        const body = await request<{ users: Raw[] }>("/api/admin/users");
        return body.users.map(toManagedUser);
      },

      async createUser(user) {
        return toManagedUser(await request<Raw>("/api/admin/users", {
          method: "POST",
          body: JSON.stringify(user),
        }));
      },

      async updateUser(userId, changes) {
        return toManagedUser(await request<Raw>(`/api/admin/users/${encodeURIComponent(userId)}`, {
          method: "PATCH",
          body: JSON.stringify(changes),
        }));
      },

      async resetPassword(userId, password) {
        await request(`/api/admin/users/${encodeURIComponent(userId)}/password`, {
          method: "POST",
          body: JSON.stringify({ password }),
        });
      },

      async gapCandidates() {
        // 聚类会调用 Embedding（有内存缓存）；列表模式只用来补全样例问题原文。
        const [clusters, refusals, feedback] = await Promise.allSettled([
          request<{ clusters: Raw[] }>("/api/admin/insights/refusals?group=cluster&days=30&limit=50"),
          request<{ refusals: Raw[] }>("/api/admin/insights/refusals?limit=100"),
          request<{ feedback: Raw[] }>("/api/admin/insights/feedback?limit=50"),
        ]);
        const warnings: string[] = [];
        const candidates: GapCandidate[] = [];
        const questionByTurn = new Map<string, string>();
        if (refusals.status === "fulfilled") {
          for (const raw of refusals.value.refusals) questionByTurn.set(String(raw.turn_id), String(raw.question));
        }
        if (clusters.status === "fulfilled") {
          for (const raw of clusters.value.clusters) {
            const title = String(raw.representative_question);
            const samples = (Array.isArray(raw.sample_turn_ids) ? raw.sample_turn_ids : [])
              .map((id) => questionByTurn.get(String(id)))
              .filter((text): text is string => !!text);
            const kbIds = Array.isArray(raw.kb_ids) ? raw.kb_ids.map(String) : [];
            candidates.push({
              key: `cluster:${String(raw.cluster_id)}`,
              source: "refusal_cluster",
              title,
              questions: [...new Set([title, ...samples])].slice(0, 10),
              kbId: kbIds[0] ?? null,
              count: Number(raw.count ?? 1),
              lastAskedAt: typeof raw.last_asked_at === "string" ? raw.last_asked_at : null,
            });
          }
        } else {
          warnings.push(`拒答聚类加载失败：${(clusters.reason as Error).message}`);
        }
        if (feedback.status === "fulfilled") {
          for (const raw of feedback.value.feedback) {
            // 已判定"不是问题"的点踩不算知识缺口。
            if (raw.triage === "not_an_issue") continue;
            const citations = Array.isArray(raw.citations) ? (raw.citations as Raw[]) : [];
            const kbId = citations.find((item) => typeof item.kb_id === "string")?.kb_id;
            candidates.push({
              key: `feedback:${String(raw.feedback_id)}`,
              source: "feedback",
              title: String(raw.question),
              questions: [String(raw.question)],
              kbId: typeof kbId === "string" ? kbId : null,
              count: 1,
              lastAskedAt: typeof raw.created_at === "string" ? raw.created_at : null,
            });
          }
        } else {
          warnings.push(`点踩反馈加载失败：${(feedback.reason as Error).message}`);
        }
        return { candidates, warnings };
      },

      async listGaps() {
        const body = await request<{ gaps: Raw[] }>("/api/admin/gaps");
        return body.gaps.map(toGap);
      },

      async saveGap(gap: GapInput) {
        return toGap(await request<Raw>(`/api/admin/gaps/${encodeURIComponent(gap.key)}`, {
          method: "PUT",
          body: JSON.stringify({
            source: gap.source,
            title: gap.title,
            questions: gap.questions,
            kb_id: gap.kbId,
            status: gap.status,
            doc_id: gap.docId,
            last_check: gap.lastCheck && {
              checked_at: gap.lastCheck.checkedAt,
              kb_id: gap.lastCheck.kbId,
              results: gap.lastCheck.results,
            },
          }),
        }));
      },

      async checkQuestion(question, kbId): Promise<GapCheckResult> {
        try {
          const raw = await request<Raw>("/api/query", {
            method: "POST",
            body: JSON.stringify({
              question,
              kb_scope: kbId ? { mode: "explicit", kb_ids: [kbId] } : { mode: "auto", kb_ids: [] },
            }),
          });
          const turn = normalizeTurn(raw, { question });
          return { question, outcome: turn.outcome, answered: turn.outcome === "answered" && turn.grounding !== "ungrounded" };
        } catch {
          return { question, outcome: "error", answered: false };
        }
      },

      async getSuggestedQuestions(kbId) {
        const body = await request<{ questions: Raw[] }>(
          `/api/suggested-questions?kb_id=${encodeURIComponent(kbId)}&limit=20`,
        );
        return body.questions.map((raw) => String(raw.text));
      },

      async setSuggestedQuestions(kbId, questions) {
        const body = await request<{ questions: Raw[] }>(
          `/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/suggested-questions`,
          { method: "PUT", body: JSON.stringify({ questions }) },
        );
        return body.questions.map((raw) => String(raw.text));
      },

      async listFeedback() {
        const body = await request<{ feedback: Raw[] }>("/api/admin/insights/feedback?limit=100");
        return body.feedback.map(toFeedback);
      },

      async triageFeedback(id, triage, revision) {
        return toFeedback(await request<Raw>(`/api/admin/insights/feedback/${encodeURIComponent(id)}`, {
          method: "PATCH",
          body: JSON.stringify({ triage, expected_revision: revision }),
        }));
      },

      async usage(days, groupBy) {
        const body = await request<Raw>(`/api/admin/usage?days=${days}&group_by=${groupBy}`);
        return {
          days: Number(body.days ?? days),
          groupBy,
          totals: toUsageRow({ key: "total", label: "合计", ...(body.totals as Raw) }),
          rows: (Array.isArray(body.rows) ? (body.rows as Raw[]) : []).map(toUsageRow),
          retentionLimited: body.retention_limited === true,
        };
      },

      async recentTraces(limit) {
        const body = await request<{ traces: Raw[] }>(`/api/admin/traces?limit=${limit}`);
        return body.traces.map(toTraceSummary);
      },

      async getKbAccess(kbId) {
        return toKbAccess(await request<Raw>(`/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/access`));
      },

      async saveKbAccess(kbId, access) {
        return toKbAccess(await request<Raw>(`/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/access`, {
          method: "PUT",
          body: JSON.stringify({
            restricted: access.restricted,
            members: access.members.map((member) => ({ user_id: member.userId, role: member.role })),
          }),
        }));
      },

      async listDocumentReviews(kbId) {
        const body = await request<{ reviews: Raw[] }>(
          `/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/document-reviews`,
        );
        return body.reviews.map(toReview);
      },

      async saveDocumentReview(kbId, docId, review) {
        return toReview(await request<Raw>(
          `/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/document-reviews/${encodeURIComponent(docId)}`,
          {
            method: "PUT",
            body: JSON.stringify({ review_due_on: review.reviewDueOn, outdated: review.outdated, note: review.note }),
          },
        ));
      },

      async listFaq(kbId) {
        return toFaqState(await request<Raw>(`/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/faq`));
      },

      async createFaq(kbId, entry) {
        return toFaqState(await request<Raw>(`/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/faq`, {
          method: "POST",
          body: JSON.stringify(entry),
        }));
      },

      async updateFaq(kbId, id, entry) {
        return toFaqState(await request<Raw>(
          `/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/faq/${encodeURIComponent(id)}`,
          { method: "PUT", body: JSON.stringify(entry) },
        ));
      },

      async deleteFaq(kbId, id) {
        return toFaqState(await request<Raw>(
          `/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/faq/${encodeURIComponent(id)}`,
          { method: "DELETE" },
        ));
      },

      async publishFaq(kbId) {
        return toFaqState(await request<Raw>(
          `/api/admin/knowledge-bases/${encodeURIComponent(kbId)}/faq/publish`,
          { method: "POST" },
        ));
      },

      async getModelSettings() {
        return toModelSettings(await request<Raw>("/api/admin/settings/models"));
      },

      async updateModelSettings(values) {
        return toModelSettings(await request<Raw>("/api/admin/settings/models", {
          method: "PUT",
          body: JSON.stringify({ values }),
        }));
      },
    },
  };
}

function toFeedback(raw: Raw): FeedbackItem {
  const triage = raw.triage;
  return {
    id: String(raw.feedback_id),
    turnId: String(raw.turn_id),
    question: String(raw.question ?? ""),
    answer: String(raw.answer ?? ""),
    citations: (Array.isArray(raw.citations) ? (raw.citations as Raw[]) : []).map(normalizeCitation),
    reasons: (Array.isArray(raw.reasons) ? raw.reasons.map(String) : []) as FeedbackItem["reasons"],
    comment: typeof raw.comment === "string" ? raw.comment : null,
    revision: Number(raw.revision ?? 1),
    triage: triage === "retrieval_error" || triage === "generation_error" || triage === "missing_content" || triage === "not_an_issue"
      ? triage
      : null,
    createdAt: String(raw.created_at ?? ""),
  };
}

function toCosts(raw: unknown): { costs: CostEstimate[]; status: UsageRow["costStatus"] } {
  const cost = (raw && typeof raw === "object" ? raw : {}) as Raw;
  const byCurrency = new Map<string, number>();
  for (const estimate of Array.isArray(cost.estimates) ? (cost.estimates as Raw[]) : []) {
    const currency = String(estimate.currency);
    byCurrency.set(currency, (byCurrency.get(currency) ?? 0) + Number(estimate.amount ?? 0));
  }
  const status = cost.status === "estimated" || cost.status === "partial" ? cost.status : "unknown";
  return { costs: [...byCurrency].map(([currency, amount]) => ({ currency, amount })), status };
}

function tokenCounts(raw: unknown): { inputTokens: number; outputTokens: number } {
  const tokens = (raw && typeof raw === "object" ? raw : {}) as Raw;
  return { inputTokens: Number(tokens.known_input ?? 0), outputTokens: Number(tokens.known_output ?? 0) };
}

function toUsageRow(raw: Raw): UsageRow {
  const { costs, status } = toCosts(raw.cost);
  return {
    key: String(raw.key ?? ""),
    label: String(raw.label ?? raw.key ?? ""),
    questions: Number(raw.questions ?? 0),
    modelCalls: Number(raw.model_calls ?? 0),
    ...tokenCounts(raw.tokens),
    costs,
    costStatus: status,
  };
}

function toTraceSummary(raw: Raw): TraceSummary {
  const outcome = raw.outcome;
  const lane = raw.lane;
  return {
    id: String(raw.trace_id),
    question: typeof raw.question === "string" ? raw.question : null,
    createdAt: String(raw.created_at ?? ""),
    outcome: outcome === "answered" || outcome === "refused" || outcome === "clarification" ? outcome : null,
    lane: lane === "direct" || lane === "clarify" || lane === "fast" || lane === "agent" ? lane : null,
    latencyMs: typeof raw.latency_ms === "number" ? raw.latency_ms : null,
    modelCalls: Number(raw.model_calls ?? 0),
    ...tokenCounts(raw.tokens),
    costs: toCosts(raw.cost).costs,
  };
}

function toKbAccess(raw: Raw): KbAccess {
  return {
    kbId: String(raw.kb_id),
    restricted: raw.restricted === true,
    members: (Array.isArray(raw.members) ? (raw.members as Raw[]) : []).map((member) => ({
      userId: String(member.user_id),
      role: member.role === "editor" ? "editor" : "viewer",
      username: typeof member.username === "string" ? member.username : null,
      name: typeof member.name === "string" ? member.name : null,
    })),
  };
}

function toReview(raw: Raw): DocumentReview {
  return {
    docId: String(raw.doc_id),
    kbId: String(raw.kb_id),
    reviewDueOn: typeof raw.review_due_on === "string" ? raw.review_due_on : null,
    outdated: raw.outdated === true,
    note: typeof raw.note === "string" ? raw.note : null,
  };
}

function toFaqState(raw: Raw): FaqState {
  const publication = raw.publication as Raw | null | undefined;
  const status = publication?.status;
  return {
    entries: (Array.isArray(raw.entries) ? (raw.entries as Raw[]) : []).map((entry) => ({
      id: String(entry.id),
      kbId: String(entry.kb_id),
      question: String(entry.question),
      answer: String(entry.answer),
      updatedAt: String(entry.updated_at ?? ""),
    })),
    filename: String(raw.filename ?? "常见问题（标准答案）.md"),
    publication: publication && (status === "queued" || status === "unchanged" || status === "removed" || status === "none" || status === "busy")
      ? { status, jobId: typeof publication.job_id === "string" ? publication.job_id : null }
      : null,
  };
}

function toManagedUser(raw: Raw): ManagedUser {
  return {
    id: String(raw.id),
    username: String(raw.username),
    name: String(raw.name ?? raw.username),
    role: raw.role === "admin" ? "admin" : "member",
    disabled: raw.disabled === true,
    createdAt: String(raw.created_at ?? ""),
  };
}

function toGapCheck(raw: unknown): GapCheck | null {
  if (!raw || typeof raw !== "object") return null;
  const value = raw as Raw;
  const results = Array.isArray(value.results) ? (value.results as Raw[]) : [];
  return {
    checkedAt: String(value.checked_at ?? ""),
    kbId: typeof value.kb_id === "string" ? value.kb_id : null,
    results: results.map((item) => ({
      question: String(item.question),
      outcome: item.outcome === "answered" || item.outcome === "refused" || item.outcome === "clarification"
        ? item.outcome
        : "error",
      answered: item.answered === true,
    })),
  };
}

function toGap(raw: Raw): KnowledgeGap {
  return {
    key: String(raw.gap_key),
    source: raw.source === "feedback" ? "feedback" : "refusal_cluster",
    title: String(raw.title),
    questions: Array.isArray(raw.questions) ? raw.questions.map(String) : [],
    kbId: typeof raw.kb_id === "string" ? raw.kb_id : null,
    status: raw.status === "document_added" || raw.status === "verified" ? raw.status : "open",
    docId: typeof raw.doc_id === "string" ? raw.doc_id : null,
    lastCheck: toGapCheck(raw.last_check),
    updatedAt: typeof raw.updated_at === "string" ? raw.updated_at : null,
  };
}

function settingValue(value: unknown): string | number | null {
  return typeof value === "string" || typeof value === "number" ? value : null;
}

function toModelSettings(raw: Raw): ModelSettings {
  const groups = Array.isArray(raw.groups) ? (raw.groups as Raw[]) : [];
  return {
    storedEmbeddingDimensions:
      typeof raw.stored_embedding_dimensions === "number" ? raw.stored_embedding_dimensions : null,
    groups: groups.map((group) => ({
      id: String(group.id),
      title: String(group.title ?? ""),
      description: String(group.description ?? ""),
      fields: (Array.isArray(group.fields) ? (group.fields as Raw[]) : []).map((field) => ({
        name: String(field.name),
        envKey: String(field.env_key ?? field.name),
        label: String(field.label ?? field.name),
        kind: field.kind === "secret" || field.kind === "url" || field.kind === "int" ? field.kind : "text",
        hint: String(field.hint ?? ""),
        value: settingValue(field.value),
        defaultValue: settingValue(field.default),
        overriddenByEnvironment: field.overridden_by_environment === true,
      })),
    })),
  };
}

import type {
  Citation,
  Clarification,
  Grounding,
  KbDocument,
  Lane,
  Outcome,
  Refusal,
  StatusUpdate,
  Turn,
} from "./types";

// 后端拒答时返回的固定文案前缀（Agent 与单轮查询两种措辞相同开头）。
const REFUSAL_PREFIX = "根据现有知识库，我无法";

type Raw = Record<string, unknown>;

function str(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function strOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function num(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function numOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function bool(value: unknown): boolean {
  return value === true;
}

function list(value: unknown): Raw[] {
  return Array.isArray(value) ? (value as Raw[]) : [];
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function oneOf<T extends string>(value: unknown, options: readonly T[]): T | null {
  return typeof value === "string" && (options as readonly string[]).includes(value)
    ? (value as T)
    : null;
}

export function isRefusalText(answer: string): boolean {
  return answer.trim().startsWith(REFUSAL_PREFIX);
}

export function normalizeCitation(raw: Raw, index: number): Citation {
  return {
    // 旧后端不返回 position 时退回列表顺序，保证正文 [n] 仍尽量可定位。
    position: num(raw.position, index + 1),
    kbId: strOrNull(raw.kb_id),
    docId: str(raw.doc_id),
    chunkId: strOrNull(raw.chunk_id),
    filename: str(raw.filename, "未知文件"),
    page: numOrNull(raw.page),
    chunkText: str(raw.chunk_text),
    docUpdatedAt: strOrNull(raw.doc_updated_at),
  };
}

function normalizeRefusal(raw: unknown): Refusal | null {
  if (!raw || typeof raw !== "object") return null;
  const value = raw as Raw;
  return {
    reason: oneOf(value.reason, ["no_evidence", "insufficient_evidence"] as const)
      ?? "insufficient_evidence",
    foundSummary: strOrNull(value.found_summary),
    missing: strings(value.missing),
    suggestedKbId: strOrNull(value.suggested_kb_id),
  };
}

function normalizeClarification(raw: unknown): Clarification | null {
  if (!raw || typeof raw !== "object") return null;
  const value = raw as Raw;
  const question = str(value.question);
  return question ? { question, options: strings(value.options) } : null;
}

// 同时接受"查询结果"（turn_id/answer）和"历史轮次"（id/assistant_message）
// 两种外壳；后端尚未提供的新字段按现有字段推导出保守的默认值。
export function normalizeTurn(
  raw: Raw,
  context: { conversationId?: string; question?: string } = {},
): Turn {
  const answer = str(raw.answer ?? raw.assistant_message);
  const answerMode = raw.answer_mode === "direct" ? "direct" : "retrieval";
  const citations = list(raw.citations).map(normalizeCitation);

  const derivedOutcome: Outcome = isRefusalText(answer) ? "refused" : "answered";
  const outcome = oneOf(raw.outcome, ["answered", "refused", "clarification"] as const)
    ?? derivedOutcome;

  let derivedGrounding: Grounding = "partial";
  if (answerMode === "direct" || outcome !== "answered") derivedGrounding = "ungrounded";
  else if (citations.length > 0) derivedGrounding = "grounded";
  const grounding = oneOf(raw.grounding, ["grounded", "partial", "ungrounded"] as const)
    ?? derivedGrounding;

  const lane: Lane | null = oneOf(raw.lane, ["direct", "clarify", "fast", "agent"] as const)
    ?? (answerMode === "direct" ? "direct" : null);

  let refusal = normalizeRefusal(raw.refusal);
  if (!refusal && outcome === "refused") {
    refusal = { reason: "insufficient_evidence", foundSummary: null, missing: [], suggestedKbId: null };
  }

  const scope = (raw.kb_scope && typeof raw.kb_scope === "object") ? (raw.kb_scope as Raw) : {};
  const stats = raw.evidence_stats as Raw | undefined;

  return {
    id: str(raw.turn_id ?? raw.id),
    conversationId: str(raw.conversation_id, context.conversationId ?? ""),
    question: str(raw.user_message, context.question ?? ""),
    answer,
    answerMode,
    outcome,
    grounding,
    lane,
    scopeExpanded: bool(raw.scope_expanded),
    escalated: bool(raw.escalated),
    searchedKbIds: strings(scope.searched),
    scopeMode: scope.mode === "explicit" ? "explicit" : "auto",
    requestedKbIds: strings(scope.requested),
    retrievalCalls: num(raw.retrieval_calls),
    citations,
    evidenceStats: stats && typeof stats === "object"
      ? {
          used: num(stats.used),
          excluded: num(stats.excluded),
          retrieved: numOrNull(stats.retrieved),
          relevant: numOrNull(stats.relevant),
        }
      : null,
    refusal,
    clarification: normalizeClarification(raw.clarification),
    latencyMs: num(raw.latency_ms),
    traceId: strOrNull(raw.trace_id),
    feedback: oneOf(raw.feedback, ["up", "down"] as const),
    createdAt: str(raw.created_at, new Date().toISOString()),
  };
}

export function normalizeStatus(raw: Raw): StatusUpdate {
  return {
    phase: str(raw.phase, "working"),
    attempt: numOrNull(raw.attempt),
    kbIds: strings(raw.kb_ids),
    label: strOrNull(raw.label),
    queuePosition: numOrNull(raw.position),
  };
}

export function normalizeDocument(raw: Raw): KbDocument {
  const filename = str(raw.filename, "未知文件");
  const extension = filename.includes(".") ? filename.split(".").pop()! : "";
  return {
    docId: str(raw.doc_id),
    kbId: strOrNull(raw.kb_id),
    filename,
    fileType: str(raw.file_type, extension.toLowerCase()),
    chunksCount: num(raw.chunks_count),
    updatedAt: strOrNull(raw.updated_at ?? raw.created_at),
    fileStatus: raw.file_status === "missing" ? "missing" : "available",
  };
}

// 统一提取后端错误文案：兼容 {detail: str}、{detail: {message}}、
// {error: {message}} 以及 FastAPI 422 的 {detail: [{msg}]}。
export function errorMessage(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object") return fallback;
  const value = body as Raw;
  const error = value.error as Raw | undefined;
  if (error && typeof error.message === "string") return error.message;
  const detail = value.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0] && typeof (detail[0] as Raw).msg === "string") {
    return str((detail[0] as Raw).msg);
  }
  if (detail && typeof detail === "object" && typeof (detail as Raw).message === "string") {
    return str((detail as Raw).message);
  }
  return fallback;
}

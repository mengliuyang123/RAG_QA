import type { Citation, StatusUpdate, ToolCall } from "../api/types";

// 接口约定 v1 中各工具 / 数据块的字段解析。后端字段缺失或类型不对时给保守值，不抛错。

export const RETRIEVAL_TOOL = "retrieve_knowledge";

export type RetrievalReason = "initial" | "expand_scope" | "escalate";

export interface RetrievalArgs {
  query: string;
  kbIds: string[];
  attempt: number | null;
  reason: RetrievalReason | null;
}

export interface RetrievalHit {
  docId: string;
  chunkId: string | null;
  kbId: string | null;
  filename: string;
  page: number | null;
  score: number | null;
  snippet: string;
  used: boolean;
}

export interface RetrievalResult {
  retrieved: number | null;
  relevant: number | null;
  hits: RetrievalHit[];
}

export interface TableData {
  caption: string | null;
  columns: string[];
  rows: string[][];
  sourcePositions: number[];
}

type Raw = Record<string, unknown>;

function asRecord(value: unknown): Raw {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Raw) : {};
}

function str(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

export function parseRetrievalArgs(args: unknown): RetrievalArgs {
  const value = asRecord(args);
  const reason = value.reason === "initial" || value.reason === "expand_scope" || value.reason === "escalate"
    ? value.reason
    : null;
  return {
    query: str(value.query) ?? "",
    kbIds: strings(value.kb_ids),
    attempt: num(value.attempt),
    reason,
  };
}

export function parseRetrievalResult(result: unknown): RetrievalResult {
  const value = asRecord(result);
  const hits = Array.isArray(value.hits) ? value.hits : [];
  return {
    retrieved: num(value.retrieved),
    relevant: num(value.relevant),
    hits: hits.flatMap((item) => {
      const hit = asRecord(item);
      const docId = str(hit.doc_id);
      if (!docId) return [];
      return [{
        docId,
        chunkId: str(hit.chunk_id),
        kbId: str(hit.kb_id),
        filename: str(hit.filename) ?? "未知文件",
        page: num(hit.page),
        score: num(hit.score),
        snippet: str(hit.snippet) ?? "",
        used: hit.used === true,
      }];
    }),
  };
}

// 协议只支持字符串和有限数字；布尔值、对象、数组、null 等不受支持的值显示为空单元格。
function cell(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return "";
}

// columns 和 rows 必须存在；行比表头短时补空，长时截断，避免表格错位。
export function parseTable(data: unknown): TableData | null {
  const value = asRecord(data);
  if (!Array.isArray(value.columns) || !Array.isArray(value.rows)) return null;
  const columns = value.columns.map(cell);
  if (columns.length === 0) return null;
  const rows = value.rows
    .filter((row): row is unknown[] => Array.isArray(row))
    .map((row) => columns.map((_, index) => cell(row[index])));
  const positions = Array.isArray(value.source_positions)
    ? value.source_positions.filter((item): item is number => Number.isInteger(item) && (item as number) > 0)
    : [];
  return { caption: str(value.caption) || null, columns, rows, sourcePositions: positions };
}

// 命中片段按 doc_id + chunk_id 对应本轮正式引用并沿用引用编号；任一侧 chunk_id 为 null
// （如旧引用）时不强行匹配。匹配不上时 position 为 0，来源面板显示为"检索片段"。
export function hitToCitation(hit: RetrievalHit, citations: readonly Citation[]): Citation {
  const match = hit.chunkId === null
    ? undefined
    : citations.find((citation) => citation.docId === hit.docId && citation.chunkId === hit.chunkId);
  return match ?? {
    position: 0,
    kbId: hit.kbId,
    docId: hit.docId,
    chunkId: hit.chunkId,
    filename: hit.filename,
    page: hit.page,
    chunkText: hit.snippet,
    docUpdatedAt: null,
  };
}

// 检索类阶段已经由工具卡片表达；其余阶段（排队、理解、核对、整理回答）照常显示。
const RETRIEVAL_PHASES = new Set(["retrieving", "expanding", "escalating"]);

export function visibleStatuses(statuses: readonly StatusUpdate[], toolCalls: readonly ToolCall[]): StatusUpdate[] {
  if (!toolCalls.some((call) => call.name === RETRIEVAL_TOOL)) return [...statuses];
  return statuses.filter((status) => !RETRIEVAL_PHASES.has(status.phase));
}

export function retrievalReasons(toolCalls: readonly ToolCall[]): Set<RetrievalReason> {
  const reasons = new Set<RetrievalReason>();
  for (const call of toolCalls) {
    if (call.name !== RETRIEVAL_TOOL) continue;
    const { reason } = parseRetrievalArgs(call.args);
    if (reason) reasons.add(reason);
  }
  return reasons;
}

import {
  useAuiState,
  type DataMessagePartComponent,
  type ToolCallMessagePartComponent,
} from "@assistant-ui/react";
import {
  ChevronDownIcon,
  CircleSlashIcon,
  FileTextIcon,
  Loader2Icon,
  SearchIcon,
  TriangleAlertIcon,
} from "lucide-react";
import { useState } from "react";
import type { Citation } from "../../api/types";
import {
  hitToCitation,
  parseRetrievalArgs,
  parseRetrievalResult,
  parseTable,
  RETRIEVAL_TOOL,
  type RetrievalArgs,
  type RetrievalResult,
  type TableData,
} from "../../lib/generative";
import { useRagThread, type AssistantCustom } from "../../runtime/RagRuntime";
import { useApp } from "../../state/AppContext";
import { cn } from "@/lib/utils";

// assistant-ui Generative UI：按名字注册工具调用片段（tools.by_name）和数据片段（data.by_name）的渲染组件，
// 通过 MessagePrimitive.Parts 的 components 参数传入。未注册的工具名和数据类型不渲染。

export type RetrievalState = "running" | "complete" | "error" | "stopped";

const REASON_LABEL = { expand_scope: "扩大范围", escalate: "深度检索" } as const;

function sameChunk(a: Citation | null, b: Citation): boolean {
  return !!a && a.docId === b.docId && a.chunkId === b.chunkId;
}

interface RetrievalCardViewProps {
  args: RetrievalArgs;
  result: RetrievalResult | null;
  state: RetrievalState;
  errorMessage?: string | null;
  kbName(kbId: string): string;
  /** 已知知识库总数；kb_ids 枚举了全部可用库时标题显示"全部知识库"。 */
  totalKbCount?: number;
  citations: readonly Citation[];
  activeCitation: Citation | null;
  onOpen(citation: Citation): void;
}

export function RetrievalCardView({
  args, result, state, errorMessage, kbName, totalKbCount = 0, citations, activeCitation, onOpen,
}: RetrievalCardViewProps) {
  const [open, setOpen] = useState(false);
  // 后端约定：kb_ids 为空表示范围未明确，不代表全部知识库，此时标题不写范围；
  // 搜索全部可用库时后端会枚举实际 ID，全部覆盖才显示"全部知识库"。
  const coversAll = totalKbCount > 1 && new Set(args.kbIds).size >= totalKbCount;
  const scope = args.kbIds.length === 0 ? null : coversAll ? "全部知识库" : args.kbIds.map(kbName).join("、");
  const where = scope ? `在 ${scope} 中` : "";
  const query = args.query ? `「${args.query}」` : "";
  const hits = result?.hits ?? [];

  let title: string;
  if (state === "running") title = scope ? `正在 ${scope} 中检索${query}` : `正在检索${query}`;
  else if (state === "error") title = `${where}检索${query}失败`;
  else if (state === "stopped") title = `${where}检索${query}（已停止）`;
  else {
    const stats = [
      result?.retrieved !== null && result?.retrieved !== undefined ? `找到 ${result.retrieved} 段` : null,
      result?.relevant !== null && result?.relevant !== undefined ? `相关 ${result.relevant} 段` : null,
    ].filter(Boolean).join("，");
    title = `${where}检索${query}${stats ? ` · ${stats}` : ""}`;
  }

  const icon = {
    running: <Loader2Icon className="size-4 shrink-0 animate-spin" aria-label="检索中" />,
    complete: <SearchIcon className="size-4 shrink-0" />,
    error: <TriangleAlertIcon className="size-4 shrink-0 text-destructive" />,
    stopped: <CircleSlashIcon className="size-4 shrink-0" />,
  }[state];

  return (
    <div
      data-slot="retrieval-card"
      data-state={state}
      className={cn(
        "mb-2 rounded-lg border text-sm duration-200 animate-in fade-in",
        state === "error" && "border-destructive/40",
      )}
    >
      <button
        type="button"
        aria-expanded={open}
        disabled={hits.length === 0}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-muted-foreground transition-colors hover:text-foreground disabled:hover:text-muted-foreground"
      >
        {icon}
        <span className={cn("min-w-0 flex-1 truncate", state === "running" && "text-foreground")}>{title}</span>
        {args.reason && args.reason !== "initial" && (
          <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">
            {REASON_LABEL[args.reason]}
          </span>
        )}
        {hits.length > 0 && (
          <ChevronDownIcon className={cn("size-3.5 shrink-0 transition-transform", open && "rotate-180")} />
        )}
      </button>
      {state === "error" && (
        <div role="alert" className="mx-3 mb-2 rounded-md bg-destructive/10 px-2.5 py-1.5 text-xs text-destructive">
          {errorMessage || "检索失败"}
        </div>
      )}
      {open && hits.length > 0 && (
        <ul className="m-0 flex list-none flex-col gap-1.5 border-t p-2 duration-150 animate-in fade-in slide-in-from-top-1">
          {result?.retrieved !== null && result?.retrieved !== undefined && result.retrieved > hits.length && (
            <li className="px-1 text-xs text-muted-foreground">
              按相关度列出前 {hits.length} 段，共找到 {result.retrieved} 段
            </li>
          )}
          {hits.map((hit, index) => {
            const citation = hitToCitation(hit, citations);
            const active = sameChunk(activeCitation, citation);
            return (
              <li key={`${hit.docId}:${hit.chunkId ?? index}`}>
                <button
                  type="button"
                  aria-label={`查看检索片段：${hit.filename}${hit.used ? "（已采用）" : ""}`}
                  onClick={() => onOpen(citation)}
                  className={cn(
                    "flex w-full flex-col gap-1 rounded-md border px-2.5 py-2 text-left transition-colors hover:bg-muted",
                    hit.used ? "border-emerald-500/30 bg-emerald-500/5" : "border-transparent",
                    active && "ring-1 ring-foreground/30",
                  )}
                >
                  <span className="flex min-w-0 items-center gap-1.5 text-xs">
                    {citation.position > 0 && (
                      <span className="font-mono text-muted-foreground">[{citation.position}]</span>
                    )}
                    <FileTextIcon className="size-3.5 shrink-0 text-muted-foreground" />
                    <span className="truncate font-medium">{hit.filename}</span>
                    {hit.page !== null && <span className="shrink-0 text-muted-foreground">第 {hit.page} 页</span>}
                    {hit.kbId && (
                      <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">
                        {kbName(hit.kbId)}
                      </span>
                    )}
                    {hit.used && (
                      <span className="ms-auto shrink-0 rounded bg-emerald-500/10 px-1.5 py-0.5 text-[11px] text-emerald-700 dark:text-emerald-300">
                        已采用
                      </span>
                    )}
                  </span>
                  {hit.snippet && (
                    <span className="line-clamp-2 text-xs leading-relaxed text-muted-foreground">{hit.snippet}</span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

interface DataTableViewProps {
  table: TableData;
  citations: readonly Citation[];
  activeCitation: Citation | null;
  onCite(citation: Citation): void;
}

export function DataTableView({ table, citations, activeCitation, onCite }: DataTableViewProps) {
  const byPosition = new Map(citations.map((citation) => [citation.position, citation]));
  return (
    <figure data-slot="data-table" className="my-3 duration-200 animate-in fade-in">
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full border-collapse text-sm">
          {table.caption && (
            <caption className="border-b px-3 py-2 text-left text-xs font-medium text-muted-foreground">
              {table.caption}
            </caption>
          )}
          <thead>
            <tr className="bg-muted/50">
              {table.columns.map((column, index) => (
                <th key={index} scope="col" className="border-b px-3 py-2 text-left font-medium whitespace-nowrap">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-b last:border-b-0">
                {row.map((value, cellIndex) => (
                  <td key={cellIndex} className="px-3 py-2 align-top">{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.sourcePositions.length > 0 && (
        <figcaption className="mt-1.5 flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
          来源
          {table.sourcePositions.map((position) => {
            const citation = byPosition.get(position);
            if (!citation) {
              return (
                <span key={position} className="cite-btn missing" title="没有找到这条引用的来源">{position}</span>
              );
            }
            return (
              <button
                key={position}
                type="button"
                className={cn("cite-btn", sameChunk(activeCitation, citation) && "active")}
                aria-label={`查看来源 ${position}：${citation.filename}`}
                onClick={() => onCite(citation)}
              >
                {position}
              </button>
            );
          })}
        </figcaption>
      )}
    </figure>
  );
}

// ---- assistant-ui 包装层：从消息与运行时上下文取数据 ----

const NO_CITATIONS: readonly Citation[] = [];

function useTurnCitations(): readonly Citation[] {
  return useAuiState((state) => {
    const custom = state.message.metadata.custom as AssistantCustom | undefined;
    return custom?.kind === "answer" ? custom.turn.citations : NO_CITATIONS;
  });
}

const RetrievalToolUI: ToolCallMessagePartComponent = ({ args, result, isError, status }) => {
  const { kbName, knowledgeBases } = useApp();
  const { activeCitation, setActiveCitation } = useRagThread();
  const citations = useTurnCitations();

  let state: RetrievalState = "complete";
  if (isError) state = "error";
  else if (status.type === "running") state = "running";
  else if (status.type === "incomplete") state = "stopped";
  const failure = isError ? (result as { error?: { message?: string } } | undefined)?.error : undefined;

  return (
    <RetrievalCardView
      args={parseRetrievalArgs(args)}
      result={state === "complete" ? parseRetrievalResult(result) : null}
      state={state}
      errorMessage={failure?.message ?? null}
      kbName={kbName}
      totalKbCount={knowledgeBases.length}
      citations={citations}
      activeCitation={activeCitation}
      onOpen={setActiveCitation}
    />
  );
};

const TableDataUI: DataMessagePartComponent = ({ data }) => {
  const { activeCitation, setActiveCitation } = useRagThread();
  const citations = useTurnCitations();
  const table = parseTable(data);
  if (!table) return null;
  return (
    <DataTableView table={table} citations={citations} activeCitation={activeCitation} onCite={setActiveCitation} />
  );
};

// 未注册的工具和数据类型返回 null，界面上不出现。
const Ignore = () => null;

export const TOOL_UIS = {
  by_name: { [RETRIEVAL_TOOL]: RetrievalToolUI },
  Fallback: Ignore,
};

export const DATA_UIS = {
  by_name: { table: TableDataUI },
  Fallback: Ignore,
};

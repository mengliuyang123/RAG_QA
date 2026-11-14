import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Citation } from "../../api/types";
import { hitToCitation, parseRetrievalArgs, parseRetrievalResult, parseTable, visibleStatuses } from "../../lib/generative";
import { DataTableView, RetrievalCardView } from "./GenerativeParts";

vi.mock("../../runtime/RagRuntime", () => ({ useRagThread: () => ({}) }));
vi.mock("../../state/AppContext", () => ({ useApp: () => ({}) }));

afterEach(cleanup);

const kbName = (id: string) => ({ ops: "运维手册", dev: "研发文档" } as Record<string, string>)[id] ?? id;

const citation: Citation = {
  position: 1, kbId: "ops", docId: "d1", chunkId: "d1:0", filename: "deploy.md",
  page: 3, chunkText: "只启动一个 worker", docUpdatedAt: null,
};

const result = parseRetrievalResult({
  retrieved: 8,
  relevant: 3,
  hits: [
    { doc_id: "d1", chunk_id: "d1:0", kb_id: "ops", filename: "deploy.md", page: 3, score: 0.82, snippet: "只启动一个 worker", used: true },
    { doc_id: "d2", chunk_id: "d2:4", kb_id: "dev", filename: "persistence.md", page: null, score: 0.5, snippet: "SQLite 适合本地", used: false },
  ],
});

function renderCard(overrides: Partial<Parameters<typeof RetrievalCardView>[0]> = {}) {
  const onOpen = vi.fn();
  render(
    <RetrievalCardView
      args={parseRetrievalArgs({ query: "多 worker 用什么存储", kb_ids: ["ops"], attempt: 1, reason: "initial" })}
      result={result}
      state="complete"
      kbName={kbName}
      citations={[citation]}
      activeCitation={null}
      onOpen={onOpen}
      {...overrides}
    />,
  );
  return onOpen;
}

describe("RetrievalCardView", () => {
  it("shows a collapsed summary title with counts", () => {
    renderCard();
    expect(screen.getByText("在 运维手册 中检索「多 worker 用什么存储」 · 找到 8 段，相关 3 段")).toBeTruthy();
    expect(screen.getByRole("button", { expanded: false })).toBeTruthy();
    expect(screen.queryByText("deploy.md")).toBeNull();
  });

  it("expands to list hits, marks used ones and opens the source panel on click", () => {
    const onOpen = renderCard();
    fireEvent.click(screen.getByRole("button", { expanded: false }));
    expect(screen.getByText("已采用")).toBeTruthy();
    expect(screen.getByText("[1]")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "查看检索片段：deploy.md（已采用）" }));
    // 与正式引用相同的分块沿用引用对象。
    expect(onOpen).toHaveBeenLastCalledWith(citation);
    fireEvent.click(screen.getByRole("button", { name: "查看检索片段：persistence.md" }));
    expect(onOpen).toHaveBeenLastCalledWith(expect.objectContaining({
      position: 0, docId: "d2", chunkId: "d2:4", chunkText: "SQLite 适合本地",
    }));
  });

  it("labels expanded and escalated retrievals", () => {
    renderCard({ args: parseRetrievalArgs({ query: "q", kb_ids: ["ops", "dev"], reason: "expand_scope" }) });
    expect(screen.getByText("扩大范围")).toBeTruthy();
    expect(screen.getByText(/在 运维手册、研发文档 中检索/)).toBeTruthy();
    cleanup();
    renderCard({ args: parseRetrievalArgs({ query: "q", reason: "escalate" }) });
    expect(screen.getByText("深度检索")).toBeTruthy();
  });

  it("shows a spinner while running and a red alert on error", () => {
    renderCard({ state: "running", result: null });
    expect(screen.getByLabelText("检索中")).toBeTruthy();
    expect(screen.getByText("正在 运维手册 中检索「多 worker 用什么存储」")).toBeTruthy();
    cleanup();
    renderCard({ state: "error", result: null, errorMessage: "检索服务暂时不可用" });
    expect(screen.getByRole("alert").textContent).toBe("检索服务暂时不可用");
  });
});

describe("DataTableView", () => {
  it("renders caption, rows and clickable sources", () => {
    const onCite = vi.fn();
    const table = parseTable({ caption: "版本一览", columns: ["版本", "日期"], rows: [["1.2", "2026-10-01"], ["1.3"]], source_positions: [1, 9] })!;
    render(<DataTableView table={table} citations={[citation]} activeCitation={null} onCite={onCite} />);
    expect(screen.getByText("版本一览")).toBeTruthy();
    expect(screen.getAllByRole("row")).toHaveLength(3);
    expect(screen.getByRole("columnheader", { name: "日期" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "查看来源 1：deploy.md" }));
    expect(onCite).toHaveBeenCalledWith(citation);
    // 找不到的编号显示为不可点击的灰色。
    expect(screen.getByTitle("没有找到这条引用的来源").textContent).toBe("9");
  });

  it("rejects malformed table data", () => {
    expect(parseTable(null)).toBeNull();
    expect(parseTable({ columns: [], rows: [] })).toBeNull();
    expect(parseTable({ columns: ["a"] })).toBeNull();
  });
});

describe("visibleStatuses", () => {
  const statuses = ["queued", "understanding", "retrieving", "expanding", "escalating", "verifying", "generating"]
    .map((phase) => ({ phase, attempt: null, kbIds: [], label: null, queuePosition: null }));

  it("keeps every phase when the turn has no retrieval cards", () => {
    expect(visibleStatuses(statuses, [])).toHaveLength(7);
  });

  it("hides retrieval phases once retrieval cards exist", () => {
    const call = { id: "tc", name: "retrieve_knowledge", status: "running" as const, args: {}, result: null, error: null };
    expect(visibleStatuses(statuses, [call]).map((status) => status.phase))
      .toEqual(["queued", "understanding", "verifying", "generating"]);
  });
});

describe("RetrievalCardView scope label", () => {
  it("says 全部知识库 when the expanded scope covers every knowledge base", () => {
    renderCard({
      args: parseRetrievalArgs({ query: "q", kb_ids: ["ops", "dev", "product"], reason: "expand_scope" }),
      totalKbCount: 3,
    });
    expect(screen.getByText(/在 全部知识库 中检索「q」/)).toBeTruthy();
  });
});

describe("backend contract clarifications", () => {
  // 约定第 3 点：空 kb_ids 表示范围未明确，不能显示成"全部知识库"。
  it("omits the scope for empty kb_ids instead of claiming every knowledge base", () => {
    renderCard({ args: parseRetrievalArgs({ query: "q", kb_ids: [] }), totalKbCount: 3 });
    expect(screen.getByText("检索「q」 · 找到 8 段，相关 3 段")).toBeTruthy();
    expect(screen.queryByText(/全部知识库/)).toBeNull();
    cleanup();
    renderCard({ args: parseRetrievalArgs({ query: "q" }), state: "running", result: null });
    expect(screen.getByText("正在检索「q」")).toBeTruthy();
  });

  // 约定第 4 点：hits 最多 10 条，统计数量可以大于列表长度。
  it("notes when the hit list is shorter than the retrieved count", () => {
    renderCard();
    fireEvent.click(screen.getByRole("button", { expanded: false }));
    expect(screen.getByText("按相关度列出前 2 段，共找到 8 段")).toBeTruthy();
  });

  // 约定第 2 点：chunk_id 为 null 时不强行匹配引用编号。
  it("never matches a citation through a null chunk_id", () => {
    const hit = result.hits[0];
    const legacy: Citation = { ...citation, chunkId: null };
    expect(hitToCitation({ ...hit, chunkId: null }, [citation]).position).toBe(0);
    expect(hitToCitation(hit, [legacy]).position).toBe(0);
    expect(hitToCitation(hit, [citation])).toBe(citation);
  });

  // 约定第 7 点：单元格仅支持字符串和有限数字。
  it("blanks unsupported table cell values", () => {
    const table = parseTable({
      columns: ["文本", "数字", "布尔", "对象", "空", "无穷"],
      rows: [["a", 1.5, true, { x: 1 }, null, Infinity]],
    })!;
    expect(table.rows[0]).toEqual(["a", "1.5", "", "", "", ""]);
  });
});

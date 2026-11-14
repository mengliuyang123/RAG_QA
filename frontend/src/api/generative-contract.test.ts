import { afterEach, describe, expect, it, vi } from "vitest";
import { mergeToolCall, normalizeDataBlock, normalizeToolCall, normalizeTurn } from "./adapters";
import { createLiveApi } from "./live";
import { createMockApi } from "./mock";
import type { DataBlock, ToolCall, Turn } from "./types";

const runningRaw = {
  tool_call_id: "tc_01",
  tool_name: "retrieve_knowledge",
  status: "running",
  args: { query: "多 worker 用什么存储", kb_ids: ["ops"], attempt: 1, reason: "initial" },
};

const completeRaw = {
  ...runningRaw,
  status: "complete",
  result: {
    retrieved: 8,
    relevant: 3,
    hits: [{ doc_id: "d1", chunk_id: "d1:0", kb_id: "ops", filename: "deploy.md", page: 3, score: 0.82, snippet: "只启动一个 worker", used: true }],
  },
};

const tableRaw = {
  data_id: "d_01",
  type: "table",
  data: { caption: "版本", columns: ["版本", "发布日期"], rows: [["1.2", "2026-10-01"]], source_positions: [1] },
};

describe("tool / data contract parsing", () => {
  it("treats missing tool_calls and data_blocks as empty (old turns)", () => {
    const turn = normalizeTurn({ id: "t1", user_message: "q", assistant_message: "a", citations: [] });
    expect(turn.toolCalls).toEqual([]);
    expect(turn.dataBlocks).toEqual([]);
  });

  it("reads persisted tool_calls and data_blocks", () => {
    const turn = normalizeTurn({
      id: "t1", user_message: "q", assistant_message: "a", citations: [],
      tool_calls: [completeRaw, { tool_call_id: "tc_02", tool_name: "future_tool", status: "complete", args: {} }],
      data_blocks: [tableRaw, { data_id: "d_02", type: "chart", data: { points: [] } }],
    });
    expect(turn.toolCalls.map((call) => [call.id, call.name, call.status])).toEqual([
      ["tc_01", "retrieve_knowledge", "complete"],
      ["tc_02", "future_tool", "complete"],
    ]);
    expect(turn.toolCalls[0].result).toMatchObject({ retrieved: 8, relevant: 3 });
    expect(turn.dataBlocks.map((block) => block.type)).toEqual(["table", "chart"]);
  });

  it("drops malformed entries instead of throwing", () => {
    expect(normalizeToolCall(null)).toBeNull();
    expect(normalizeToolCall({ tool_name: "retrieve_knowledge" })).toBeNull();
    expect(normalizeToolCall({ tool_call_id: "x" })).toBeNull();
    expect(normalizeDataBlock({ type: "table" })).toBeNull();
    expect(normalizeDataBlock("table")).toBeNull();
    const turn = normalizeTurn({ id: "t1", tool_calls: [42, "x", {}], data_blocks: "nope" });
    expect(turn.toolCalls).toEqual([]);
    expect(turn.dataBlocks).toEqual([]);
  });

  it("parses the error terminal state", () => {
    const call = normalizeToolCall({ ...runningRaw, status: "error", error: { code: "retrieval_failed", message: "检索失败" } });
    expect(call).toMatchObject({ status: "error", error: { code: "retrieval_failed", message: "检索失败" }, result: null });
  });

  it("merges streaming updates by tool_call_id", () => {
    const running = normalizeToolCall(runningRaw)!;
    const complete = normalizeToolCall(completeRaw)!;
    const second = normalizeToolCall({ ...runningRaw, tool_call_id: "tc_02" })!;
    let calls: ToolCall[] = mergeToolCall([], running);
    calls = mergeToolCall(calls, second);
    calls = mergeToolCall(calls, complete);
    expect(calls.map((call) => [call.id, call.status])).toEqual([["tc_01", "complete"], ["tc_02", "running"]]);
  });
});

function sse(events: [string, unknown][]): Response {
  const body = events.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join("");
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

describe("live SSE dispatch", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("forwards tool and data events and ignores malformed ones", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => sse([
      ["status", { phase: "retrieving", kb_ids: ["ops"] }],
      ["tool", runningRaw],
      ["tool", { tool_name: "missing_id" }],
      ["tool", completeRaw],
      ["token", { token: "答案" }],
      ["data", tableRaw],
      ["data", { type: "table" }],
      ["complete", { turn_id: "t1", answer: "答案", answer_mode: "retrieval", citations: [], tool_calls: [completeRaw], data_blocks: [tableRaw] }],
    ])));
    const tools: ToolCall[] = [];
    const blocks: DataBlock[] = [];
    let completed: Turn | null = null;
    await createLiveApi().streamQuery("c1", "q", { mode: "auto", kbIds: [] }, {
      onStatus: () => undefined,
      onToken: () => undefined,
      onTool: (call) => tools.push(call),
      onData: (block) => blocks.push(block),
      onComplete: (turn) => { completed = turn; },
      onError: (code) => { throw new Error(code); },
    }, new AbortController().signal);
    expect(tools.map((call) => call.status)).toEqual(["running", "complete"]);
    expect(blocks.map((block) => block.id)).toEqual(["d_01"]);
    expect(completed!.toolCalls).toHaveLength(1);
    expect(completed!.dataBlocks).toHaveLength(1);
  });
});

describe("mock streaming", () => {
  it("emits running then terminal tool events before tokens, with two calls when the scope expands", async () => {
    const api = createMockApi();
    const conversation = await api.createConversation({ mode: "auto", kbIds: [] });
    const order: string[] = [];
    let completed: Turn | null = null;
    await api.streamQuery(conversation.id, "服务启动报数据库锁定怎么办", { mode: "auto", kbIds: [] }, {
      onStatus: () => undefined,
      onToken: () => order.push("token"),
      onTool: (call) => order.push(`${call.id}:${call.status}`),
      onData: () => order.push("data"),
      onComplete: (turn) => { completed = turn; },
      onError: () => undefined,
    }, new AbortController().signal);
    const tools = order.filter((item) => item !== "token");
    expect(tools).toEqual(["tc_01:running", "tc_01:complete", "tc_02:running", "tc_02:complete"]);
    expect(order.indexOf("token")).toBeGreaterThan(order.lastIndexOf("tc_02:complete"));
    expect(completed!.toolCalls.map((call) => call.args.reason)).toEqual(["initial", "expand_scope"]);
    const stored = await api.getConversation(conversation.id);
    expect(stored.turns[0].toolCalls).toHaveLength(2);
  }, 15000);

  it("attaches a table data block to the version scenario", async () => {
    const api = createMockApi();
    const conversation = await api.createConversation({ mode: "auto", kbIds: [] });
    const blocks: DataBlock[] = [];
    let completed: Turn | null = null;
    await api.streamQuery(conversation.id, "最新版本更新了什么", { mode: "auto", kbIds: [] }, {
      onStatus: () => undefined, onToken: () => undefined, onTool: () => undefined,
      onData: (block) => blocks.push(block),
      onComplete: (turn) => { completed = turn; },
      onError: () => undefined,
    }, new AbortController().signal);
    expect(blocks.map((block) => block.type)).toEqual(["table"]);
    expect(completed!.dataBlocks).toHaveLength(1);
  }, 15000);
});

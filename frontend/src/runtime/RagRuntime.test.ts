import { describe, expect, it } from "vitest";
import { convertMessage, type PendingState } from "./RagRuntime";

function pending(overrides: Partial<PendingState> = {}): PendingState {
  return { question: "q", statuses: [], answer: "", error: null, toolCalls: [], dataBlocks: [], ...overrides };
}

describe("convertMessage", () => {
  it("maps a question to a user message", () => {
    const message = convertMessage({ kind: "question", id: "t1:q", text: "你好", createdAt: "2026-11-01T00:00:00Z" });
    expect(message).toMatchObject({ id: "t1:q", role: "user", content: "你好" });
  });

  it("keeps a streaming answer running and carries the pending state", () => {
    const state = pending({ answer: "部分" });
    const message = convertMessage({ kind: "pending", id: "pending:a", pending: state });
    expect(message.status).toEqual({ type: "running" });
    expect(message.content).toEqual([{ type: "text", text: "部分" }]);
    expect(message.metadata?.custom).toEqual({ kind: "pending", pending: state });
  });

  it("distinguishes a user cancel from a failure", () => {
    const cancelled = convertMessage({
      kind: "pending", id: "p", pending: pending({ error: { code: "cancelled", message: "已停止回答" } }),
    });
    expect(cancelled.status).toEqual({ type: "incomplete", reason: "cancelled" });
    const failed = convertMessage({
      kind: "pending", id: "p", pending: pending({ error: { code: "500", message: "回答失败" } }),
    });
    expect(failed.status).toEqual({ type: "incomplete", reason: "error", error: "回答失败" });
  });
});

describe("convertMessage parts (generative UI)", () => {
  const running = { id: "tc_01", name: "retrieve_knowledge", status: "running" as const, args: { query: "q" }, result: null, error: null };
  const complete = { ...running, id: "tc_02", status: "complete" as const, result: { retrieved: 3 } };
  const failed = {
    ...running, id: "tc_03", status: "error" as const, error: { code: "retrieval_failed", message: "检索失败" },
  };
  const table = { id: "d_01", type: "table", data: { columns: ["a"], rows: [] } };

  it("orders tool calls, then text, then data blocks", () => {
    const message = convertMessage({
      kind: "pending", id: "p",
      pending: pending({ answer: "正文", toolCalls: [running, complete, failed], dataBlocks: [table] }),
    });
    const content = message.content as readonly { type: string }[];
    expect(content.map((part) => part.type)).toEqual(["tool-call", "tool-call", "tool-call", "text", "data"]);
  });

  it("leaves running calls without a result and marks failures as errors", () => {
    const message = convertMessage({ kind: "pending", id: "p", pending: pending({ toolCalls: [running, complete, failed] }) });
    const [first, second, third] = message.content as readonly Record<string, unknown>[];
    expect(first).toMatchObject({ toolCallId: "tc_01", toolName: "retrieve_knowledge" });
    expect("result" in first).toBe(false);
    expect(second).toMatchObject({ result: { retrieved: 3 } });
    expect(third).toMatchObject({ isError: true, result: { error: { message: "检索失败" } } });
  });

  it("keeps old turns as a single text part", () => {
    const message = convertMessage({
      kind: "answer", id: "t1",
      turn: {
        id: "t1", conversationId: "c", question: "q", answer: "旧回答", answerMode: "retrieval", outcome: "answered",
        grounding: "grounded", lane: null, scopeExpanded: false, escalated: false, searchedKbIds: [], scopeMode: "auto",
        requestedKbIds: [], retrievalCalls: 1, citations: [], evidenceStats: null, refusal: null, clarification: null,
        latencyMs: 0, traceId: null, feedback: null, createdAt: "2026-11-01T00:00:00Z", toolCalls: [], dataBlocks: [],
      },
    });
    expect(message.content).toEqual([{ type: "text", text: "旧回答" }]);
  });
});

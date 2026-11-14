import { describe, expect, it } from "vitest";
import { convertMessage, type PendingState } from "./RagRuntime";

function pending(overrides: Partial<PendingState> = {}): PendingState {
  return { question: "q", statuses: [], answer: "", error: null, ...overrides };
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

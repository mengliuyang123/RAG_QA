import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Turn } from "../api/types";
import { TurnView } from "./TurnView";

vi.mock("../state/AppContext", () => ({
  useApp: () => ({
    user: { id: "u", name: "测试", role: "member", quota: null },
    knowledgeBases: [
      { id: "default", name: "默认知识库", description: "", icon: null, documentsCount: 1, updatedAt: null },
      { id: "ops", name: "运维手册", description: "", icon: null, documentsCount: 1, updatedAt: null },
      { id: "dev", name: "研发文档", description: "", icon: null, documentsCount: 1, updatedAt: null },
    ],
    kbName: (id: string) => ({ default: "默认知识库", ops: "运维手册", dev: "研发文档" } as Record<string, string>)[id] ?? id,
  }),
  AppProvider: ({ children }: { children: ReactNode }) => children,
}));

afterEach(cleanup);

function refusedTurn(overrides: Partial<Turn> = {}): Turn {
  return {
    id: "t1", conversationId: "c1", question: "多个 worker 用什么存储？",
    answer: "根据现有知识库，我无法可靠回答这个问题", answerMode: "retrieval",
    outcome: "refused", grounding: "ungrounded", lane: "agent",
    scopeExpanded: false, escalated: true, searchedKbIds: ["ops"],
    scopeMode: "explicit", requestedKbIds: ["ops"], retrievalCalls: 1, citations: [],
    evidenceStats: { used: 0, excluded: 0, retrieved: 1, relevant: 1 },
    refusal: { reason: "insufficient_evidence", foundSummary: "检索到1段资料，1段相关。", missing: ["存储方案"], suggestedKbId: null },
    clarification: null, latencyMs: 1000, traceId: null, feedback: null, createdAt: "2026-10-28T00:00:00Z",
    ...overrides,
  };
}

function renderTurn(turn: Turn, onRetryWithScope = vi.fn()) {
  render(
    <TurnView
      turn={turn} activeCitation={null} onCite={() => undefined} onAsk={() => undefined}
      onRetryWithScope={onRetryWithScope} onFeedback={() => undefined}
    />,
  );
  return onRetryWithScope;
}

describe("TurnView refusal", () => {
  it("offers every out-of-scope knowledge base even without a backend suggestion", () => {
    const onRetry = renderTurn(refusedTurn());
    expect(screen.getByText("找到 1 段相关资料，但不足以回答")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "运维手册" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "默认知识库" }));
    expect(onRetry).toHaveBeenCalledWith({ mode: "explicit", kbIds: ["default"] }, "多个 worker 用什么存储？");
    fireEvent.click(screen.getByRole("button", { name: "自动（全部知识库）" }));
    expect(onRetry).toHaveBeenLastCalledWith({ mode: "auto", kbIds: [] }, "多个 worker 用什么存储？");
  });

  it("puts the suggested knowledge base first and marks it", () => {
    renderTurn(refusedTurn({
      refusal: { reason: "insufficient_evidence", foundSummary: null, missing: [], suggestedKbId: "dev" },
    }));
    const buttons = screen.getAllByRole("button").filter((button) => /知识库|文档/.test(button.textContent ?? ""));
    expect(buttons[0].textContent).toBe("研发文档推荐");
  });

  it("does not offer a retry picker for automatic scope", () => {
    renderTurn(refusedTurn({ scopeMode: "auto", requestedKbIds: [] }));
    expect(screen.queryByText("在其他知识库中重新查找：")).toBeNull();
  });
});

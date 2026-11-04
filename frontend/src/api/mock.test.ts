import { describe, expect, it } from "vitest";
import { createMockApi, pickScenario } from "./mock";
import type { StatusUpdate, Turn } from "./types";

describe("pickScenario", () => {
  it("only expands scope in auto mode", () => {
    const auto = pickScenario("服务启动报数据库锁定", { mode: "auto", kbIds: [] });
    const explicit = pickScenario("服务启动报数据库锁定", { mode: "explicit", kbIds: ["ops"] });
    expect(auto.turn.scope_expanded).toBe(true);
    expect(explicit.turn.scope_expanded).toBe(false);
    expect(explicit.statuses.some((status) => status.phase === "expanding")).toBe(false);
    expect((explicit.turn.kb_scope as { searched: string[] }).searched).toEqual(["ops"]);
  });

  it("keeps every answer marker backed by a citation", () => {
    const questions = ["checkpoint 怎么配置", "hybrid 和 hybrid_rerank 的区别", "服务启动报数据库锁定", "最新版本更新了什么"];
    for (const question of questions) {
      const { turn } = pickScenario(question, { mode: "auto", kbIds: [] });
      const markers = [...String(turn.answer).matchAll(/\[(\d+)\]/g)].map((match) => Number(match[1]));
      const positions = (turn.citations as { position: number }[]).map((citation) => citation.position);
      expect(markers.every((marker) => positions.includes(marker))).toBe(true);
    }
  });
});

describe("createMockApi streamQuery", () => {
  it("emits statuses, tokens, then one complete event and stores the turn", async () => {
    const api = createMockApi();
    const conversation = await api.createConversation({ mode: "auto", kbIds: [] });
    const statuses: StatusUpdate[] = [];
    let text = "";
    let completed: Turn | null = null;
    await api.streamQuery(conversation.id, "你好", { mode: "auto", kbIds: [] }, {
      onStatus: (update) => statuses.push(update),
      onToken: (token) => { text += token; },
      onComplete: (turn) => { completed = turn; },
      onError: () => { throw new Error("unexpected error event"); },
    }, new AbortController().signal);

    expect(statuses.length).toBeGreaterThan(0);
    expect(completed).not.toBeNull();
    expect(text).toBe(completed!.answer);
    const stored = await api.getConversation(conversation.id);
    expect(stored.turns).toHaveLength(1);
    expect(stored.title).toBe("你好");
  });

  it("rejects a second concurrent question on the same conversation", async () => {
    const api = createMockApi();
    const conversation = await api.createConversation({ mode: "auto", kbIds: [] });
    const noop = { onStatus() {}, onToken() {}, onComplete() {}, onError() {} };
    const controller = new AbortController();
    const first = api.streamQuery(conversation.id, "checkpoint", { mode: "auto", kbIds: [] }, noop, controller.signal);
    await expect(
      api.streamQuery(conversation.id, "again", { mode: "auto", kbIds: [] }, noop, new AbortController().signal),
    ).rejects.toMatchObject({ status: 409 });
    controller.abort();
    await expect(first).rejects.toMatchObject({ name: "AbortError" });
  });
});

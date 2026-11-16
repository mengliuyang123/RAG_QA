import { describe, expect, it } from "vitest";
import { UploadConflictError, createMockApi, pickScenario } from "./mock";
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
      onTool: () => undefined,
      onData: () => undefined,
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
    const noop = { onStatus() {}, onToken() {}, onTool() {}, onData() {}, onComplete() {}, onError() {} };
    const controller = new AbortController();
    const first = api.streamQuery(conversation.id, "checkpoint", { mode: "auto", kbIds: [] }, noop, controller.signal);
    await expect(
      api.streamQuery(conversation.id, "again", { mode: "auto", kbIds: [] }, noop, new AbortController().signal),
    ).rejects.toMatchObject({ status: 409 });
    controller.abort();
    await expect(first).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("createMockApi uploadDocument", () => {
  const noop = () => undefined;

  it("returns the existing document instead of ingesting identical content again", async () => {
    const api = createMockApi();
    const first = await api.admin.uploadDocument("ops", new File(["same body"], "a.md"), noop);
    const before = (await api.admin.listDocuments("ops")).length;
    const second = await api.admin.uploadDocument("ops", new File(["same body"], "renamed.md"), noop);
    expect(first.kind).toBe("uploaded");
    expect(second).toMatchObject({ kind: "exists", existing: { filename: "a.md" } });
    expect((await api.admin.listDocuments("ops")).length).toBe(before);
  }, 20000);

  it("asks for a choice on a name conflict, then replaces or keeps both", async () => {
    const api = createMockApi();
    const original = await api.admin.uploadDocument("ops", new File(["v1"], "guide.md"), noop);
    const originalId = original.kind === "uploaded" ? original.document.docId : "";
    const conflict = api.admin.uploadDocument("ops", new File(["v2"], "guide.md"), noop);
    await expect(conflict).rejects.toBeInstanceOf(UploadConflictError);
    await expect(conflict).rejects.toMatchObject({ existing: { docId: originalId } });

    await api.admin.uploadDocument("ops", new File(["v2"], "guide.md"), noop, { onConflict: "replace", replaceDocId: originalId });
    let named = (await api.admin.listDocuments("ops")).filter((doc) => doc.filename === "guide.md");
    expect(named).toHaveLength(1);
    expect(named[0].docId).not.toBe(originalId);

    await api.admin.uploadDocument("ops", new File(["v3"], "guide.md"), noop, { onConflict: "keep_both" });
    named = (await api.admin.listDocuments("ops")).filter((doc) => doc.filename === "guide.md");
    expect(named).toHaveLength(2);
  }, 20000);

  it("deletes documents in a batch and reports unknown ids", async () => {
    const api = createMockApi();
    const docs = await api.admin.listDocuments(null);
    const ids = docs.slice(0, 2).map((doc) => doc.docId);
    const result = await api.admin.deleteDocuments([...ids, "missing"]);
    expect(result).toEqual({ deleted: ids, notFound: ["missing"] });
    expect((await api.admin.listDocuments(null)).length).toBe(docs.length - 2);
  });
});

describe("mock user management rules", () => {
  it("rejects self lockout and removing the last enabled admin, like the backend", async () => {
    const api = createMockApi();
    await expect(api.admin.updateUser("u_demo", { disabled: true })).rejects.toMatchObject({ code: "self_lockout" });
    await expect(api.admin.updateUser("u_demo", { role: "member" })).rejects.toMatchObject({ code: "self_lockout" });
    await api.admin.updateUser("u_ops", { role: "admin" });
    await expect(api.admin.updateUser("u_ops", { role: "member" })).resolves.toMatchObject({ role: "member" });
    await expect(api.admin.createUser({ username: "OPS.WANG", name: "x", password: "replace-me-demo-passphrase", role: "member" }))
      .rejects.toMatchObject({ code: "user_exists" });
    await expect(api.admin.resetPassword("u_ops", "short")).rejects.toMatchObject({ status: 422 });
    const users = await api.admin.listUsers();
    expect(users.every((user) => !("password" in user))).toBe(true);
  });

  it("only accepts a verified gap with a fully answered re-check", async () => {
    const api = createMockApi();
    const gap = {
      key: "cluster:x", source: "refusal_cluster" as const, title: "q", questions: ["q"],
      kbId: "dev", status: "verified" as const, docId: null, lastCheck: null,
    };
    await expect(api.admin.saveGap(gap)).rejects.toMatchObject({ code: "gap_not_verified" });
    const lastCheck = { checkedAt: "2026-11-17T00:00:00Z", kbId: "dev", results: [{ question: "q", outcome: "answered" as const, answered: true }] };
    await expect(api.admin.saveGap({ ...gap, lastCheck })).resolves.toMatchObject({ status: "verified" });
    expect(await api.admin.listGaps()).toHaveLength(1);
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AUTH_EXPIRED_EVENT, createLiveApi } from "./live";

type Call = { url: string; init: RequestInit };

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function mockFetch(responses: Response[]): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit = {}) => {
    calls.push({ url, init });
    const next = responses.shift();
    if (!next) throw new Error(`unexpected request ${url}`);
    return next;
  }));
  return calls;
}

function header(call: Call, name: string): string | undefined {
  return (call.init.headers as Record<string, string> | undefined)?.[name];
}

beforeEach(() => {
  document.cookie = "rag_csrf=csrf-token-value; path=/";
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
  document.cookie = "rag_csrf=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
});

describe("live api", () => {
  it("sends the CSRF cookie value on writes but not on reads", async () => {
    const calls = mockFetch([jsonResponse({ conversations: [] }), jsonResponse(null, 204)]);
    const api = createLiveApi();
    await api.listConversations();
    await api.setFeedback("t1", "up");
    expect(header(calls[0], "X-CSRF-Token")).toBeUndefined();
    expect(header(calls[1], "X-CSRF-Token")).toBe("csrf-token-value");
    expect(calls[1].init.method).toBe("PUT");
    expect(JSON.parse(String(calls[1].init.body))).toEqual({ rating: "up" });
  });

  it("announces an expired session on 401 and exposes the error code", async () => {
    mockFetch([jsonResponse({ detail: "请先登录", error: { code: "unauthenticated", message: "请先登录" } }, 401)]);
    const listener = vi.fn();
    window.addEventListener(AUTH_EXPIRED_EVENT, listener);
    await expect(createLiveApi().me()).rejects.toMatchObject({ status: 401, code: "unauthenticated", message: "请先登录" });
    expect(listener).toHaveBeenCalledTimes(1);
    window.removeEventListener(AUTH_EXPIRED_EVENT, listener);
  });

  it("serializes explicit and auto scopes in the backend shape", async () => {
    const summary = { id: "c1", title: "新对话", updated_at: "2026-10-28T00:00:00Z", turns_count: 0 };
    const calls = mockFetch([
      jsonResponse({ ...summary, kb_scope: { mode: "explicit", kb_ids: ["ops"] } }, 201),
      jsonResponse(summary, 201),
    ]);
    const api = createLiveApi();
    const explicit = await api.createConversation({ mode: "explicit", kbIds: ["ops"] });
    const auto = await api.createConversation({ mode: "auto", kbIds: [] });
    expect(JSON.parse(String(calls[0].init.body))).toEqual({ kb_scope: { mode: "explicit", kb_ids: ["ops"] } });
    expect(JSON.parse(String(calls[1].init.body))).toEqual({ kb_scope: { mode: "auto", kb_ids: [] } });
    expect(explicit.kbScope).toEqual({ mode: "explicit", kbIds: ["ops"] });
    expect(auto.kbScope).toEqual({ mode: "auto", kbIds: [] });
  });

  it("polls an ingestion job until it succeeds", async () => {
    vi.useFakeTimers();
    const calls = mockFetch([
      jsonResponse({ job_id: "job_1", doc_id: "d1", status: "queued" }, 202),
      jsonResponse({ job_id: "job_1", status: "running", stage: "embedding", progress: 0.5 }),
      jsonResponse({
        job_id: "job_1", doc_id: "d1", kb_id: "ops", filename: "a.md", status: "succeeded",
        stage: "indexing", progress: 1, chunks_count: 4, updated_at: "2026-10-28T00:00:00Z",
      }),
    ]);
    const stages: string[] = [];
    const upload = createLiveApi().admin.uploadDocument(
      "ops", new File(["x"], "a.md"), (progress) => stages.push(progress.stage),
    );
    await vi.advanceTimersByTimeAsync(2500);
    const doc = await upload;
    expect(calls[0].url).toBe("/api/admin/knowledge-bases/ops/documents");
    expect(header(calls[0], "X-CSRF-Token")).toBe("csrf-token-value");
    expect(calls[1].url).toBe("/api/admin/ingestion-jobs/job_1");
    expect(stages).toEqual(["uploading", "embedding", "done"]);
    expect(doc).toMatchObject({ docId: "d1", kbId: "ops", chunksCount: 4 });
  });

  it("rejects with the backend failure when an ingestion job fails", async () => {
    vi.useFakeTimers();
    mockFetch([
      jsonResponse({ job_id: "job_2", doc_id: "d2", status: "queued" }, 202),
      jsonResponse({ job_id: "job_2", status: "failed", stage: "parsing", progress: 0, error: { code: "empty_document", message: "文档中没有可提取的文本" } }),
    ]);
    const upload = createLiveApi().admin.uploadDocument("ops", new File(["x"], "b.md"), () => undefined);
    const assertion = expect(upload).rejects.toMatchObject({ code: "empty_document", message: "文档中没有可提取的文本" });
    await vi.advanceTimersByTimeAsync(1500);
    await assertion;
  });

  it("keeps null insight rates instead of turning them into zero", async () => {
    mockFetch([jsonResponse({ knowledge_bases: [{ kb_id: "dev", questions: 0, refusal_rate: null, thumbs_down_rate: null, expanded_rate: null }] })]);
    const [insight] = await createLiveApi().admin.insights();
    expect(insight).toEqual({ kbId: "dev", questions: 0, refusalRate: null, thumbsDownRate: null, expandedRate: null });
  });
});

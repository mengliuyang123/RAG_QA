import { describe, expect, it } from "vitest";
import { errorMessage, normalizeDocument, normalizeTurn } from "./adapters";

const citation = {
  position: 5,
  doc_id: "doc-1",
  filename: "deploy-guide.md",
  page: null,
  chunk_text: "只启动一个 worker",
};

describe("normalizeTurn", () => {
  it("reads the query-result shell", () => {
    const turn = normalizeTurn(
      {
        conversation_id: "c1",
        turn_id: "t1",
        answer: "只启动一个 worker [5]",
        answer_mode: "retrieval",
        retrieval_calls: 1,
        citations: [citation],
        latency_ms: 1200,
      },
      { question: "怎么办" },
    );
    expect(turn.id).toBe("t1");
    expect(turn.conversationId).toBe("c1");
    expect(turn.question).toBe("怎么办");
    expect(turn.answer).toBe("只启动一个 worker [5]");
    expect(turn.citations[0].position).toBe(5);
    expect(turn.outcome).toBe("answered");
    expect(turn.grounding).toBe("grounded");
  });

  it("reads the history-turn shell without renaming fields", () => {
    const turn = normalizeTurn(
      {
        id: "t2",
        user_message: "你好",
        assistant_message: "你好！",
        answer_mode: "direct",
        retrieval_calls: 0,
        citations: [],
        latency_ms: 300,
        created_at: "2026-10-23T08:00:00Z",
      },
      { conversationId: "c9" },
    );
    expect(turn.id).toBe("t2");
    expect(turn.question).toBe("你好");
    expect(turn.answer).toBe("你好！");
    expect(turn.conversationId).toBe("c9");
    expect(turn.lane).toBe("direct");
    expect(turn.grounding).toBe("ungrounded");
  });

  it("derives a refusal from the backend refusal text", () => {
    const turn = normalizeTurn({
      turn_id: "t3",
      answer: "根据现有知识库，我无法可靠回答这个问题",
      answer_mode: "retrieval",
      retrieval_calls: 3,
      citations: [],
    });
    expect(turn.outcome).toBe("refused");
    expect(turn.grounding).toBe("ungrounded");
    expect(turn.refusal?.reason).toBe("insufficient_evidence");
  });

  it("falls back to list order when position is missing", () => {
    const { position: _ignored, ...withoutPosition } = citation;
    const turn = normalizeTurn({ turn_id: "t4", answer: "a [1] b [2]", citations: [withoutPosition, withoutPosition] });
    expect(turn.citations.map((item) => item.position)).toEqual([1, 2]);
  });

  it("prefers explicit new fields over derived defaults", () => {
    const turn = normalizeTurn({
      turn_id: "t5",
      answer: "x [1]",
      answer_mode: "retrieval",
      outcome: "answered",
      grounding: "partial",
      lane: "agent",
      scope_expanded: true,
      escalated: true,
      kb_scope: { mode: "auto", requested: [], searched: ["ops", "dev"] },
      citations: [{ ...citation, position: 1, kb_id: "ops" }],
    });
    expect(turn.grounding).toBe("partial");
    expect(turn.lane).toBe("agent");
    expect(turn.scopeExpanded).toBe(true);
    expect(turn.escalated).toBe(true);
    expect(turn.searchedKbIds).toEqual(["ops", "dev"]);
    expect(turn.citations[0].kbId).toBe("ops");
  });

  it("maps scope mode and the four evidence counts, keeping legacy counts null", () => {
    const fresh = normalizeTurn({
      turn_id: "t7", answer: "根据现有知识库，我无法可靠回答这个问题", answer_mode: "retrieval",
      kb_scope: { mode: "explicit", requested: ["ops"], searched: ["ops"] },
      evidence_stats: { used: 0, excluded: 0, retrieved: 1, relevant: 1 },
    });
    expect(fresh.scopeMode).toBe("explicit");
    expect(fresh.requestedKbIds).toEqual(["ops"]);
    expect(fresh.evidenceStats).toEqual({ used: 0, excluded: 0, retrieved: 1, relevant: 1 });
    const legacy = normalizeTurn({ turn_id: "t8", answer: "x", evidence_stats: { used: 1, excluded: 2 } });
    expect(legacy.scopeMode).toBe("auto");
    expect(legacy.evidenceStats).toEqual({ used: 1, excluded: 2, retrieved: null, relevant: null });
  });

  it("ignores unknown enum values", () => {
    const turn = normalizeTurn({ turn_id: "t6", answer: "x", answer_mode: "direct", lane: "expanded", grounding: 0.8 });
    expect(turn.lane).toBe("direct");
    expect(turn.grounding).toBe("ungrounded");
  });
});

describe("normalizeDocument", () => {
  it("maps the current /api/documents shape", () => {
    const doc = normalizeDocument({
      doc_id: "d1",
      filename: "notes.PDF",
      chunks_count: 3,
      created_at: null,
      file_status: "missing",
    });
    expect(doc).toMatchObject({ docId: "d1", fileType: "pdf", chunksCount: 3, updatedAt: null, fileStatus: "missing" });
  });
});

describe("errorMessage", () => {
  it.each([
    [{ detail: "会话不存在" }, "会话不存在"],
    [{ detail: { code: "document_delete_partial", message: "文档索引已删除" } }, "文档索引已删除"],
    [{ detail: [{ msg: "问题不能为空" }] }, "问题不能为空"],
    [{ error: { code: "rate_limited", message: "每分钟最多提问 10 次" } }, "每分钟最多提问 10 次"],
    [null, "默认"],
  ])("extracts %j", (body, expected) => {
    expect(errorMessage(body, "默认")).toBe(expected);
  });
});

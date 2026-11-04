import { describe, expect, it } from "vitest";
import { evidenceSummary } from "./evidence";

const stats = (used: number, excluded: number, retrieved: number | null, relevant: number | null) =>
  ({ used, excluded, retrieved, relevant });

describe("evidenceSummary", () => {
  it("explains relevant-but-insufficient evidence instead of 0 used / 0 excluded", () => {
    expect(evidenceSummary({ outcome: "refused", evidenceStats: stats(0, 0, 1, 1) }))
      .toBe("找到 1 段相关资料，但不足以回答");
  });

  it("distinguishes irrelevant results from finding nothing", () => {
    expect(evidenceSummary({ outcome: "refused", evidenceStats: stats(0, 3, 3, 0) }))
      .toBe("找到 3 段资料，但都与问题无关");
    expect(evidenceSummary({ outcome: "refused", evidenceStats: stats(0, 0, 0, 0) }))
      .toBe("没有找到相关资料");
  });

  it("summarizes answered turns by retrieved and used", () => {
    expect(evidenceSummary({ outcome: "answered", evidenceStats: stats(2, 4, 6, 2) }))
      .toBe("找到 6 段资料，采用 2 段");
  });

  it("falls back to the old wording for legacy records", () => {
    expect(evidenceSummary({ outcome: "answered", evidenceStats: stats(1, 2, null, null) }))
      .toBe("采用 1 段资料，排除 2 段不相关内容");
    expect(evidenceSummary({ outcome: "refused", evidenceStats: stats(0, 0, null, null) })).toBeNull();
    expect(evidenceSummary({ outcome: "answered", evidenceStats: null })).toBeNull();
  });
});

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Api, GapCandidate, KnowledgeGap } from "../api/types";

const KBS = [
  { id: "dev", name: "研发文档", description: "", icon: null, documentsCount: 1, updatedAt: null },
  { id: "ops", name: "运维手册", description: "", icon: null, documentsCount: 1, updatedAt: null },
];

vi.mock("../state/AppContext", () => ({
  useApp: () => ({
    knowledgeBases: KBS,
    kbName: (id: string) => KBS.find((kb) => kb.id === id)?.name ?? id,
  }),
}));

vi.mock("../components/Layout", () => ({
  Topbar: ({ title }: { title: string }) => <h1>{title}</h1>,
}));

vi.mock("../api", () => ({ api: {} }));

const { api } = (await import("../api")) as unknown as { api: Api };
const { createMockApi } = await import("../api/mock");
const { AdminGapsPage, mergeGaps } = await import("./AdminGapsPage");

const WAIT = { timeout: 3000 };

function KbStub() {
  const location = useLocation();
  return <div data-testid="kb-page">{location.search}</div>;
}

function renderPage() {
  render(
    <MemoryRouter initialEntries={["/admin/gaps"]}>
      <Routes>
        <Route path="/admin/gaps" element={<AdminGapsPage />} />
        <Route path="/admin/kb" element={<KbStub />} />
      </Routes>
    </MemoryRouter>,
  );
}

async function openGap(title: string) {
  const cell = await screen.findByText(title, {}, WAIT);
  fireEvent.click(within(cell.closest("tr") as HTMLElement).getByRole("button", { name: "打开" }));
  return screen.getByRole("region", { name: "缺口详情" });
}

beforeEach(() => {
  Object.assign(api, createMockApi());
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("AdminGapsPage", () => {
  it("lists gaps from refusal clusters and negative feedback", async () => {
    renderPage();
    await screen.findByText("Redis 缓存怎么配置", {}, WAIT);
    expect(screen.getByText("企业版支持私有化部署吗")).toBeTruthy();
    expect(screen.getByRole("tab", { name: "全部 3" })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "待处理 3" })).toBeTruthy();
    expect(screen.getAllByText("拒答")).toHaveLength(2);
    expect(screen.getAllByText("点踩")).toHaveLength(1);
  });

  it("shows sample questions and hands off to the upload page for the chosen knowledge base", async () => {
    renderPage();
    const detail = await openGap("Redis 缓存怎么配置");
    expect(within(detail).getByText("Redis 的过期时间在哪里设置")).toBeTruthy();
    fireEvent.change(within(detail).getByLabelText(/补充到知识库/), { target: { value: "ops" } });
    fireEvent.click(within(detail).getByRole("button", { name: "去上传文档" }));
    const kbPage = await screen.findByTestId("kb-page", {}, WAIT);
    const params = new URLSearchParams(kbPage.textContent ?? "");
    expect(params.get("kb")).toBe("ops");
    expect(params.get("gap")).toBe("cluster:cl_redis0000000000000000000");
    const [tracked] = await api.admin.listGaps();
    expect(tracked).toMatchObject({ key: "cluster:cl_redis0000000000000000000", kbId: "ops", status: "open" });
  });

  it("re-runs the sample questions and only marks verified when all are answered", async () => {
    renderPage();
    let detail = await openGap("Redis 缓存怎么配置");
    fireEvent.click(within(detail).getByRole("button", { name: "复测" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("0/3"), WAIT);
    detail = screen.getByRole("region", { name: "缺口详情" });
    expect(within(detail).getAllByText("仍拒答")).toHaveLength(3);
    expect(within(detail).getByText("待处理")).toBeTruthy();

    const check = vi.spyOn(api.admin, "checkQuestion").mockImplementation(async (question, kbId) => {
      expect(kbId).toBe("dev");
      return { question, outcome: "answered", answered: true };
    });
    fireEvent.click(within(detail).getByRole("button", { name: "复测" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("全部通过"), WAIT);
    expect(check).toHaveBeenCalledTimes(3);
    expect(within(screen.getByRole("region", { name: "缺口详情" })).getByText("已验证")).toBeTruthy();
    expect(screen.getByRole("tab", { name: "已验证 1" })).toBeTruthy();
    const [saved] = await api.admin.listGaps();
    expect(saved.status).toBe("verified");
    expect(saved.lastCheck?.results.map((item) => item.answered)).toEqual([true, true, true]);
  });

  it("resolves a gap by writing a standard answer instead of uploading", async () => {
    const create = vi.spyOn(api.admin, "createFaq");
    renderPage();
    const detail = await openGap("企业版支持私有化部署吗");
    fireEvent.click(within(detail).getByRole("button", { name: "写标准答案" }));
    fireEvent.change(within(detail).getByLabelText(/标准答案（问题/), { target: { value: "支持，需单独签署部署协议。" } });
    fireEvent.click(within(detail).getByRole("button", { name: "保存并发布" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("已写入标准答案"), WAIT);
    expect(create).toHaveBeenCalledWith("product", { question: "企业版支持私有化部署吗", answer: "支持，需单独签署部署协议。" });
    expect(screen.getByRole("tab", { name: "已补文档 1" })).toBeTruthy();
  });

  it("lets the admin mark a document as added and reopen the gap", async () => {
    renderPage();
    const detail = await openGap("灰度发布的流程是什么");
    fireEvent.click(within(detail).getByRole("button", { name: "标记为已补文档" }));
    await waitFor(() => expect(screen.getByRole("tab", { name: "已补文档 1" })).toBeTruthy(), WAIT);
    fireEvent.click(within(screen.getByRole("region", { name: "缺口详情" })).getByRole("button", { name: "重新打开" }));
    await waitFor(() => expect(screen.getByRole("tab", { name: "待处理 3" })).toBeTruthy(), WAIT);
  });

  it("still shows the other source when one fails to load", async () => {
    vi.spyOn(api.admin, "gapCandidates").mockResolvedValue({
      candidates: [{
        key: "feedback:f1", source: "feedback", title: "点踩的问题", questions: ["点踩的问题"],
        kbId: null, count: 1, lastAskedAt: null,
      }],
      warnings: ["拒答聚类加载失败：向量服务不可用"],
    });
    renderPage();
    await screen.findByText("点踩的问题", {}, WAIT);
    expect(screen.getByText("拒答聚类加载失败：向量服务不可用")).toBeTruthy();
  });
});

describe("mergeGaps", () => {
  const candidate = (overrides: Partial<GapCandidate>): GapCandidate => ({
    key: "cluster:new", source: "refusal_cluster", title: "Redis 缓存怎么配置",
    questions: ["Redis 缓存怎么配置"], kbId: "dev", count: 4, lastAskedAt: "2026-11-10T00:00:00Z", ...overrides,
  });
  const gap = (overrides: Partial<KnowledgeGap>): KnowledgeGap => ({
    key: "cluster:old", source: "refusal_cluster", title: "Redis 缓存怎么配置",
    questions: ["Redis 缓存怎么配置", "Redis 过期时间"], kbId: "dev", status: "verified", docId: "d1",
    lastCheck: { checkedAt: "2026-11-01T00:00:00Z", kbId: "dev", results: [] }, updatedAt: "2026-11-01T00:00:00Z", ...overrides,
  });

  it("folds a re-clustered candidate into the tracked gap that shares a question", () => {
    const rows = mergeGaps([candidate({}), candidate({ key: "cluster:other", title: "别的问题", questions: ["别的问题"] })], [gap({})]);
    expect(rows.map((row) => [row.key, row.status, row.count])).toEqual([
      ["cluster:other", "open", 4],
      ["cluster:old", "verified", 4],
    ]);
    expect(rows[1].lastAskedAt).toBe("2026-11-10T00:00:00Z");
    expect(rows[1].tracked).toBe(true);
  });

  it("keeps tracked gaps whose source no longer appears", () => {
    expect(mergeGaps([], [gap({ status: "open" })])).toHaveLength(1);
  });
});

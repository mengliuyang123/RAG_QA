import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Api } from "../api/types";

const KBS = [
  { id: "dev", name: "研发文档", description: "", icon: null, documentsCount: 2, updatedAt: null },
  { id: "ops", name: "运维手册", description: "", icon: null, documentsCount: 1, updatedAt: null },
];

vi.mock("../state/AppContext", () => ({
  useApp: () => ({
    user: { id: "u_demo", name: "小刘", role: "admin", quota: null, editableKbIds: null },
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
const { AdminInsightsPage } = await import("./AdminInsightsPage");
const { AdminUsagePage } = await import("./AdminUsagePage");
const { SharePage } = await import("./SharePage");
const { KbFaqPanel } = await import("../components/kb/KbFaqPanel");
const { KbSuggestionsPanel } = await import("../components/kb/KbSuggestionsPanel");
const { KbAccessPanel } = await import("../components/kb/KbAccessPanel");
const { ChangePasswordDialog } = await import("../components/ChangePasswordDialog");

const WAIT = { timeout: 3000 };

beforeEach(() => {
  Object.assign(api, createMockApi());
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("feedback triage", () => {
  it("classifies a thumbs-down and marks an outdated cited document", async () => {
    const review = vi.spyOn(api.admin, "saveDocumentReview");
    render(<AdminInsightsPage />);
    fireEvent.click(screen.getByRole("tab", { name: "点踩反馈" }));
    const card = await screen.findByRole("article", { name: "点踩：生产环境数据库连接池上限是多少" }, WAIT);
    expect(screen.getByRole("tab", { name: "未分类 1" })).toBeTruthy();
    expect(screen.queryByRole("article", { name: "点踩：专业版和企业版的区别" })).toBeNull();

    fireEvent.click(within(card).getByRole("button", { name: "标记过时" }));
    await waitFor(() => expect(within(card).getByText("已标记过时")).toBeTruthy(), WAIT);
    expect(review).toHaveBeenCalledWith("ops", "d1f0c2a4-0000-4000-8000-000000000003", {
      reviewDueOn: null, outdated: true, note: "用户反馈：已经调整为 100 了",
    });

    fireEvent.click(within(card).getByRole("button", { name: "缺内容" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("缺内容"), WAIT);
    expect(screen.getByRole("tab", { name: "未分类 0" })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "缺内容 2" })).toBeTruthy();
  });

  it("shows a conflict when the user changed the feedback meanwhile", async () => {
    vi.spyOn(api.admin, "triageFeedback").mockRejectedValue(new Error("用户已修改反馈，请刷新后重试"));
    render(<AdminInsightsPage />);
    fireEvent.click(screen.getByRole("tab", { name: "点踩反馈" }));
    const card = await screen.findByRole("article", { name: "点踩：生产环境数据库连接池上限是多少" }, WAIT);
    fireEvent.click(within(card).getByRole("button", { name: "检索问题" }));
    await waitFor(() => expect(within(card).getByRole("alert").textContent).toContain("刷新"), WAIT);
  });
});

describe("usage page", () => {
  it("shows totals, grouped rows and the most expensive requests", async () => {
    const usage = vi.spyOn(api.admin, "usage");
    render(<AdminUsagePage />);
    const grouped = await screen.findByRole("table", { name: "分组用量" }, WAIT);
    expect(within(grouped).getByText("2026-11-16")).toBeTruthy();
    expect(screen.getByText("CNY 8.85")).toBeTruthy();
    const expensive = await screen.findByRole("table", { name: "最耗费的请求" }, WAIT);
    const rows = within(expensive).getAllByRole("row").slice(1);
    expect(rows[0].textContent).toContain("LangGraph 检查点怎么持久化");

    fireEvent.click(screen.getByRole("tab", { name: "按知识库" }));
    await waitFor(() => expect(within(screen.getByRole("table", { name: "分组用量" })).getByText("研发文档")).toBeTruthy(), WAIT);
    fireEvent.click(screen.getByRole("tab", { name: "近 30 天" }));
    await waitFor(() => expect(usage).toHaveBeenLastCalledWith(30, "kb"), WAIT);
  });
});

describe("share page", () => {
  it("renders a shared answer with its citations", async () => {
    vi.spyOn(api, "getShare").mockResolvedValue({
      question: "怎么回滚？", answer: "执行 rollback 脚本[1]。", sharedBy: "小刘", createdAt: "2026-11-17T00:00:00Z",
      citations: [{ position: 1, kbId: "ops", docId: "d", chunkId: "d:0", filename: "deploy-guide.md", page: 3, chunkText: "rollback.sh", docUpdatedAt: null }],
    });
    render(
      <MemoryRouter initialEntries={["/s/shr_abc"]}>
        <Routes><Route path="/s/:token" element={<SharePage />} /></Routes>
      </MemoryRouter>,
    );
    expect(await screen.findByText("怎么回滚？", {}, WAIT)).toBeTruthy();
    expect(screen.getByText(/deploy-guide.md/)).toBeTruthy();
    expect(screen.getByText(/小刘 分享/)).toBeTruthy();
    expect(api.getShare).toHaveBeenCalledWith("shr_abc");
  });

  it("explains an expired or missing share", async () => {
    vi.spyOn(api, "getShare").mockRejectedValue(new Error("分享已过期"));
    render(
      <MemoryRouter initialEntries={["/s/shr_old"]}>
        <Routes><Route path="/s/:token" element={<SharePage />} /></Routes>
      </MemoryRouter>,
    );
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("分享已过期"), WAIT);
  });
});

describe("knowledge base panels", () => {
  it("edits suggested questions and enforces the limit", async () => {
    render(<KbSuggestionsPanel kbId="dev" />);
    const box = await screen.findByLabelText("推荐问题列表", {}, WAIT) as HTMLTextAreaElement;
    expect(box.value).toContain("checkpoint 怎么配置");
    fireEvent.change(box, { target: { value: Array.from({ length: 21 }, (_, i) => `问题${i}`).join("\n") } });
    expect(screen.getByRole("alert").textContent).toContain("最多 20");
    expect((screen.getByRole("button", { name: "发布" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(box, { target: { value: "新问题一\n新问题二\n新问题一" } });
    fireEvent.click(screen.getByRole("button", { name: "发布" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("已发布 2 个"), WAIT);
    expect(await api.admin.getSuggestedQuestions("dev")).toEqual(["新问题一", "新问题二"]);
  });

  it("adds, edits and deletes standard answers, publishing each time", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<KbFaqPanel kbId="ops" />);
    fireEvent.click(await screen.findByRole("button", { name: "新增标准答案" }, WAIT));
    fireEvent.change(screen.getByLabelText("问题"), { target: { value: "怎么回滚？" } });
    fireEvent.change(screen.getByLabelText("标准答案", { selector: "textarea" }), { target: { value: "执行 rollback 脚本。" } });
    fireEvent.click(screen.getByRole("button", { name: "保存并发布" }));
    const entry = await screen.findByRole("article", { name: "怎么回滚？" }, WAIT);
    expect(screen.getByRole("status").textContent).toContain("已提交发布");
    expect((await api.listKbDocuments("ops")).some((doc) => doc.filename === "常见问题（标准答案）.md")).toBe(true);

    fireEvent.click(within(entry).getByRole("button", { name: "编辑" }));
    fireEvent.change(screen.getByLabelText("标准答案", { selector: "textarea" }), { target: { value: "先切流量再回滚。" } });
    fireEvent.click(screen.getByRole("button", { name: "保存并发布" }));
    await waitFor(() => expect(within(screen.getByRole("article", { name: "怎么回滚？" })).getByText("先切流量再回滚。")).toBeTruthy(), WAIT);

    fireEvent.click(within(screen.getByRole("article", { name: "怎么回滚？" })).getByRole("button", { name: "删除" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("已清空"), WAIT);
    expect((await api.listKbDocuments("ops")).some((doc) => doc.filename === "常见问题（标准答案）.md")).toBe(false);
  });

  it("restricts a knowledge base and adds an editor", async () => {
    const save = vi.spyOn(api.admin, "saveKbAccess");
    render(<KbAccessPanel kbId="ops" />);
    fireEvent.click(await screen.findByLabelText("仅成员可见", {}, WAIT));
    fireEvent.change(screen.getByLabelText("添加成员"), { target: { value: "u_ops" } });
    fireEvent.click(screen.getByRole("button", { name: "添加" }));
    fireEvent.change(screen.getByLabelText("ops.wang 的权限"), { target: { value: "editor" } });
    // 管理员不出现在候选里。
    expect(within(screen.getByLabelText("添加成员")).queryByText(/xiaoliu/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "保存" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("已保存"), WAIT);
    expect(save).toHaveBeenCalledWith("ops", { restricted: true, members: [{ userId: "u_ops", role: "editor" }] });
    expect((screen.getByRole("button", { name: "保存" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("change password dialog", () => {
  it("validates locally, reports a wrong current password, then succeeds", async () => {
    const onClose = vi.fn();
    render(<ChangePasswordDialog onClose={onClose} />);
    fireEvent.change(screen.getByLabelText("当前口令"), { target: { value: "wrong-password" } });
    fireEvent.change(screen.getByLabelText(/^新口令/), { target: { value: "short" } });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));
    expect(screen.getByRole("alert").textContent).toContain("15–128");
    fireEvent.change(screen.getByLabelText(/^新口令/), { target: { value: "replace-me-demo-passphrase" } });
    fireEvent.change(screen.getByLabelText("再次输入新口令"), { target: { value: "replace-me-demo-passphrase-x" } });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));
    expect(screen.getByRole("alert").textContent).toContain("不一致");
    fireEvent.change(screen.getByLabelText("再次输入新口令"), { target: { value: "replace-me-demo-passphrase" } });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("当前口令不正确"), WAIT);
    fireEvent.change(screen.getByLabelText("当前口令"), { target: { value: "demo-password-123" } });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("其他浏览器上的登录已退出"), WAIT);
    fireEvent.click(screen.getByRole("button", { name: "完成" }));
    expect(onClose).toHaveBeenCalled();
  });
});

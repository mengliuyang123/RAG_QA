import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ConversationSummary } from "../api/types";
import { Layout } from "./Layout";

let conversations: ConversationSummary[] = [];
let currentUser: { id: string; name: string; role: string; quota: null; editableKbIds: string[] | null } = {
  id: "u", name: "测试", role: "member", quota: null, editableKbIds: [],
};

vi.mock("../state/AppContext", () => ({
  useApp: () => ({
    user: currentUser,
    knowledgeBases: [],
    conversations,
    kbName: (id: string) => id,
    logout: async () => undefined,
    refreshConversations: async () => undefined,
    loadError: null,
  }),
}));

afterEach(() => {
  cleanup();
  conversations = [];
  currentUser = { id: "u", name: "测试", role: "member", quota: null, editableKbIds: [] };
});

function summary(id: string, title: string): ConversationSummary {
  return {
    id, title, updatedAt: new Date().toISOString(), turnsCount: 1, kbScope: { mode: "auto", kbIds: [] },
  };
}

function renderLayout() {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Layout />
    </MemoryRouter>,
  );
}

describe("Layout conversation list", () => {
  // 登录后会话列表异步加载：列表变长的那次渲染里，assistant-ui 的线程列表还没同步，
  // 侧栏若按会话列表的下标渲染会越界，抛出 useClientLookup: key "undefined" not found。
  it("renders conversations that arrive after the first render", async () => {
    const view = renderLayout();
    expect(screen.getByText("还没有对话")).toBeTruthy();

    conversations = [summary("a", "第一个对话"), summary("b", "第二个对话")];
    view.rerender(
      <MemoryRouter initialEntries={["/"]}>
        <Layout />
      </MemoryRouter>,
    );

    expect(await screen.findByText("第一个对话")).toBeTruthy();
    expect(screen.getByText("第二个对话")).toBeTruthy();
  });
});

describe("Layout navigation by role", () => {
  it("hides management links from a plain member", () => {
    renderLayout();
    expect(screen.queryByRole("link", { name: "知识库管理" })).toBeNull();
    expect(screen.getByRole("button", { name: "修改口令" })).toBeTruthy();
  });

  it("shows a KB editor only the knowledge-base management link", () => {
    currentUser = { ...currentUser, editableKbIds: ["hr"] };
    renderLayout();
    expect(screen.getByText("我负责的知识库")).toBeTruthy();
    expect(screen.getByRole("link", { name: "知识库管理" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: "用户管理" })).toBeNull();
    expect(screen.queryByRole("link", { name: "用量与费用" })).toBeNull();
  });

  it("shows admins every admin page", () => {
    currentUser = { ...currentUser, role: "admin", editableKbIds: null };
    renderLayout();
    for (const name of ["知识库管理", "问题洞察", "用量与费用", "知识缺口", "用户管理", "模型配置"]) {
      expect(screen.getByRole("link", { name })).toBeTruthy();
    }
  });
});

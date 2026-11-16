import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Api } from "../api/types";

vi.mock("../state/AppContext", () => ({
  useApp: () => ({ user: { id: "u_demo", name: "小刘", role: "admin", quota: null } }),
}));

vi.mock("../components/Layout", () => ({
  Topbar: ({ title, children }: { title: string; children?: unknown }) => <header><h1>{title}</h1>{children as never}</header>,
}));

vi.mock("../api", () => ({ api: {} }));

const { api } = (await import("../api")) as unknown as { api: Api };
const { createMockApi } = await import("../api/mock");
const { AdminUsersPage } = await import("./AdminUsersPage");

const VALID_PASSWORD = "replace-me-demo-passphrase";

beforeEach(() => {
  // 每个用例使用全新的 mock 后端状态。
  Object.assign(api, createMockApi());
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

async function rowOf(username: string): Promise<HTMLElement> {
  const cell = await screen.findByText(username, {}, { timeout: 2000 });
  return cell.closest("tr") as HTMLElement;
}

describe("AdminUsersPage", () => {
  it("lists users and locks the current admin out of demoting or disabling themself", async () => {
    render(<AdminUsersPage />);
    const self = await rowOf("xiaoliu");
    expect(within(self).getByText("我")).toBeTruthy();
    expect((within(self).getByRole("button", { name: "设为成员" }) as HTMLButtonElement).disabled).toBe(true);
    expect((within(self).getByRole("button", { name: "禁用" }) as HTMLButtonElement).disabled).toBe(true);
    expect((within(self).getByRole("button", { name: "重置口令" }) as HTMLButtonElement).disabled).toBe(false);
    const disabled = await rowOf("intern");
    expect(within(disabled).getByText("已禁用")).toBeTruthy();
  });

  it("creates a user with a role and validates the password locally first", async () => {
    const create = vi.spyOn(api.admin, "createUser");
    render(<AdminUsersPage />);
    await rowOf("xiaoliu");
    fireEvent.click(screen.getByRole("button", { name: "新建用户" }));
    fireEvent.change(screen.getByLabelText("登录名"), { target: { value: "Zhao.Li" } });
    fireEvent.change(screen.getByLabelText("展示名"), { target: { value: "赵丽" } });
    fireEvent.change(screen.getByLabelText(/初始口令/), { target: { value: "short" } });
    fireEvent.change(screen.getByLabelText("角色"), { target: { value: "admin" } });
    fireEvent.click(screen.getByRole("button", { name: "创建" }));
    expect(screen.getByRole("alert").textContent).toContain("15–128");
    expect(create).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText(/初始口令/), { target: { value: VALID_PASSWORD } });
    fireEvent.click(screen.getByRole("button", { name: "创建" }));
    const created = await rowOf("zhao.li");
    expect(within(created).getByText("管理员")).toBeTruthy();
    expect(create).toHaveBeenCalledWith({ username: "zhao.li", name: "赵丽", password: VALID_PASSWORD, role: "admin" });
    expect(screen.getByRole("status").textContent).toContain("已创建 zhao.li");
    expect(document.body.textContent).not.toContain(VALID_PASSWORD);

    fireEvent.click(screen.getByRole("button", { name: "新建用户" }));
    fireEvent.change(screen.getByLabelText("登录名"), { target: { value: "ops.wang" } });
    fireEvent.change(screen.getByLabelText("展示名"), { target: { value: "重复" } });
    fireEvent.change(screen.getByLabelText(/初始口令/), { target: { value: VALID_PASSWORD } });
    fireEvent.click(screen.getByRole("button", { name: "创建" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("登录名已被使用"));
  });

  it("changes roles, and disables only after confirmation", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<AdminUsersPage />);
    let row = await rowOf("ops.wang");
    fireEvent.click(within(row).getByRole("button", { name: "设为管理员" }));
    await waitFor(() => expect(within(row).getByText("管理员")).toBeTruthy());
    // 有了第二名管理员后，自己的行仍然锁定，对方可以被降级。
    fireEvent.click(within(row).getByRole("button", { name: "设为成员" }));
    await waitFor(() => expect(within(row).getByText("成员")).toBeTruthy());

    fireEvent.click(within(row).getByRole("button", { name: "禁用" }));
    expect(confirm).toHaveBeenCalled();
    expect(within(row).getByText("正常")).toBeTruthy();
    confirm.mockReturnValue(true);
    fireEvent.click(within(row).getByRole("button", { name: "禁用" }));
    row = await rowOf("ops.wang");
    await waitFor(() => expect(within(row).getByText("已禁用")).toBeTruthy());
    fireEvent.click(within(row).getByRole("button", { name: "启用" }));
    await waitFor(() => expect(within(row).getByText("正常")).toBeTruthy());
  });

  it("shows the backend's refusal when an update breaks a safety rule", async () => {
    vi.spyOn(api.admin, "updateUser").mockRejectedValue(new Error("至少需要保留一名启用的管理员"));
    render(<AdminUsersPage />);
    const row = await rowOf("ops.wang");
    fireEvent.click(within(row).getByRole("button", { name: "设为管理员" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("至少需要保留一名启用的管理员"));
  });

  it("resets a password and explains that sessions are revoked", async () => {
    const reset = vi.spyOn(api.admin, "resetPassword");
    render(<AdminUsersPage />);
    const row = await rowOf("ops.wang");
    fireEvent.click(within(row).getByRole("button", { name: "重置口令" }));
    const form = screen.getByRole("group", { name: "重置 ops.wang 的口令" });
    fireEvent.change(within(form).getByLabelText("新口令"), { target: { value: "short" } });
    fireEvent.click(within(form).getByRole("button", { name: "确认重置" }));
    expect(within(form).getByRole("alert").textContent).toContain("15–128");
    expect(reset).not.toHaveBeenCalled();
    fireEvent.change(within(form).getByLabelText("新口令"), { target: { value: VALID_PASSWORD } });
    fireEvent.click(within(form).getByRole("button", { name: "确认重置" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("需要重新登录"));
    expect(reset).toHaveBeenCalledWith("u_ops", VALID_PASSWORD);
    expect(screen.queryByRole("group", { name: "重置 ops.wang 的口令" })).toBeNull();
  });
});

import {
  ThreadListItemMorePrimitive,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import {
  BarChart3Icon,
  BookOpenIcon,
  LogOutIcon,
  MenuIcon,
  MoreHorizontalIcon,
  PencilIcon,
  PlusIcon,
  SearchIcon,
  SettingsIcon,
  SlidersHorizontalIcon,
  TrashIcon,
} from "lucide-react";
import { createContext, Fragment, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "../api";
import { RagRuntimeProvider, useRagThread } from "../runtime/RagRuntime";
import { useApp } from "../state/AppContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { TooltipProvider } from "@/components/ui/tooltip";
import { TooltipIconButton } from "@/components/ui/tooltip-icon-button";
import { cn } from "@/lib/utils";

const MenuContext = createContext<() => void>(() => undefined);

function groupLabel(updatedAt: string): string {
  const date = new Date(updatedAt);
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (date.getTime() >= startOfToday) return "今天";
  if (date.getTime() >= startOfToday - 6 * 86400000) return "最近 7 天";
  return "更早";
}

const menuItem = "flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm outline-none select-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground";

function RenameInput({ onDone }: { onDone(): void }) {
  const aui = useAui();
  const title = useAuiState((state) => state.threadListItem.title) ?? "";
  const [value, setValue] = useState(title);
  const settled = useRef(false);

  const commit = () => {
    if (settled.current) return;
    settled.current = true;
    const next = value.trim();
    if (next && next !== title) aui.threadListItem().rename(next);
    onDone();
  };

  return (
    <Input
      autoFocus
      aria-label="重命名对话"
      value={value}
      maxLength={50}
      className="h-7 min-w-0 flex-1 ps-2.5 pe-9 text-sm"
      onFocus={(event) => event.currentTarget.select()}
      onChange={(event) => setValue(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing) return;
        if (event.key === "Enter") {
          event.preventDefault();
          commit();
        } else if (event.key === "Escape") {
          event.preventDefault();
          settled.current = true;
          onDone();
        }
      }}
    />
  );
}

// 样式改写自 assistant-ui 官方 registry 的 thread-list：悬停或选中时显示"更多"菜单。
function ConversationItem() {
  const aui = useAui();
  const title = useAuiState((state) => state.threadListItem.title) ?? "新对话";
  const [renaming, setRenaming] = useState(false);

  return (
    <ThreadListItemPrimitive.Root className="group relative flex h-8 items-center rounded-md transition-colors hover:bg-muted data-active:bg-muted has-data-[state=open]:bg-muted">
      {renaming ? (
        <RenameInput onDone={() => setRenaming(false)} />
      ) : (
        <ThreadListItemPrimitive.Trigger className="flex h-full min-w-0 flex-1 items-center rounded-md px-2.5 text-start text-sm outline-none group-hover:pe-9 group-data-active:pe-9 group-has-data-[state=open]:pe-9 focus-visible:ring-1 focus-visible:ring-ring/50">
          <span className="min-w-0 flex-1 truncate">
            <ThreadListItemPrimitive.Title fallback="新对话" />
          </span>
        </ThreadListItemPrimitive.Trigger>
      )}
      <ThreadListItemMorePrimitive.Root>
        <ThreadListItemMorePrimitive.Trigger asChild>
          <Button
            variant="ghost"
            size="icon"
            aria-label={`更多操作 ${title}`}
            className="absolute end-1.5 top-1/2 size-6 -translate-y-1/2 p-0 opacity-0 group-hover:opacity-100 group-data-active:opacity-100 focus-visible:opacity-100 data-[state=open]:bg-accent data-[state=open]:opacity-100"
          >
            <MoreHorizontalIcon className="size-3.5" />
          </Button>
        </ThreadListItemMorePrimitive.Trigger>
        <ThreadListItemMorePrimitive.Content
          side="right"
          align="start"
          sideOffset={6}
          className="z-50 min-w-32 overflow-hidden rounded-xl border bg-popover p-1.5 text-popover-foreground data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
        >
          <ThreadListItemMorePrimitive.Item className={menuItem} onSelect={() => setRenaming(true)}>
            <PencilIcon className="size-4" />
            重命名
          </ThreadListItemMorePrimitive.Item>
          <ThreadListItemMorePrimitive.Item
            className={cn(menuItem, "text-destructive hover:bg-destructive/10 hover:text-destructive focus:bg-destructive/10 focus:text-destructive")}
            onSelect={() => {
              if (window.confirm(`删除对话"${title}"？删除后无法恢复。`)) aui.threadListItem().delete();
            }}
          >
            <TrashIcon className="size-4" />
            删除
          </ThreadListItemMorePrimitive.Item>
        </ThreadListItemMorePrimitive.Content>
      </ThreadListItemMorePrimitive.Root>
    </ThreadListItemPrimitive.Root>
  );
}

// 会话列表由 assistant-ui ThreadList 渲染：切换、新建、重命名、删除都经过运行时的
// threadList 适配器（见 runtime/RagRuntime.tsx），这里负责按更新时间分组和按标题搜索。
function ConversationList() {
  const { conversations } = useApp();
  const { listError } = useRagThread();
  const [search, setSearch] = useState("");
  const query = search.trim().toLowerCase();
  // 下标必须取自 assistant-ui 自己的 threadIds：会话列表刷新后，运行时的线程列表
  // 会晚一次渲染才同步，直接按 conversations 的下标渲染 ItemByIndex 会越界崩溃。
  const threadIds = useAuiState((state) => state.threads.threadIds);
  const byId = new Map(conversations.map((conversation) => [conversation.id, conversation]));

  const groups = new Map<string, number[]>();
  threadIds.forEach((id, index) => {
    const conversation = byId.get(id);
    const title = conversation?.title ?? "新对话";
    if (query && !title.toLowerCase().includes(query)) return;
    const label = conversation ? groupLabel(conversation.updatedAt) : "今天";
    groups.set(label, [...(groups.get(label) ?? []), index]);
  });

  return (
    <ThreadListPrimitive.Root className="flex min-h-0 flex-1 flex-col gap-0.5">
      <ThreadListPrimitive.New asChild>
        <Button variant="ghost" className="h-8 justify-start gap-2 rounded-md px-2.5 text-sm font-normal hover:bg-muted">
          <PlusIcon className="size-4" />
          新对话
        </Button>
      </ThreadListPrimitive.New>
      {threadIds.length > 0 && (
        <div className="relative px-0.5 py-1">
          <SearchIcon className="pointer-events-none absolute start-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            aria-label="搜索对话"
            placeholder="搜索对话"
            className="h-8 ps-8 text-sm"
          />
        </div>
      )}
      <div className="-mx-1 flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-1">
        {threadIds.length === 0 && (
          <div className="px-2.5 py-4 text-sm text-muted-foreground">还没有对话</div>
        )}
        {threadIds.length > 0 && query && groups.size === 0 && (
          <div className="px-2.5 py-4 text-sm text-muted-foreground">没有找到对话</div>
        )}
        {[...groups.entries()].map(([label, indexes]) => (
          <Fragment key={label}>
            <div className="px-2.5 pt-3 pb-1 text-xs font-medium text-muted-foreground">{label}</div>
            {indexes.map((index) => (
              <ThreadListPrimitive.ItemByIndex
                key={threadIds[index]}
                index={index}
                components={{ ThreadListItem: ConversationItem }}
              />
            ))}
          </Fragment>
        ))}
        {listError && (
          <div role="alert" className="mx-1 mt-2 rounded-md bg-destructive/10 p-2 text-xs text-destructive">{listError}</div>
        )}
      </div>
    </ThreadListPrimitive.Root>
  );
}

function NavItem({ to, icon, label }: { to: string; icon: ReactNode; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => cn(
        "flex h-8 items-center gap-2 rounded-md px-2.5 text-sm text-foreground transition-colors hover:bg-muted [&_svg]:size-4 [&_svg]:text-muted-foreground",
        isActive && "bg-muted",
      )}
    >
      {icon}
      <span className="truncate">{label}</span>
    </NavLink>
  );
}

function Sidebar({ open }: { open: boolean }) {
  const { user, logout } = useApp();

  return (
    <nav
      aria-label="主导航"
      className={cn(
        "fixed inset-y-0 left-0 z-40 flex w-64 shrink-0 flex-col gap-2 border-r bg-sidebar p-2 text-sidebar-foreground transition-transform md:static md:translate-x-0",
        open ? "translate-x-0" : "-translate-x-full",
      )}
    >
      <div className="flex h-10 items-center gap-2 px-2.5">
        <span className="flex size-6 items-center justify-center rounded-md bg-primary text-xs font-semibold text-primary-foreground">R</span>
        <span className="text-sm font-semibold">RAG 知识库</span>
      </div>

      <ConversationList />

      <div className="flex flex-col gap-0.5 border-t pt-2">
        <NavItem to="/kb" icon={<BookOpenIcon />} label="知识库" />
        {user?.role === "admin" && (
          <>
            <div className="px-2.5 pt-2 pb-1 text-xs font-medium text-muted-foreground">管理后台</div>
            <NavItem to="/admin/kb" icon={<SettingsIcon />} label="知识库管理" />
            <NavItem to="/admin/insights" icon={<BarChart3Icon />} label="问题洞察" />
            <NavItem to="/admin/settings" icon={<SlidersHorizontalIcon />} label="模型配置" />
          </>
        )}
        <div className="mt-1 flex h-10 items-center gap-2 px-2.5">
          <span className="flex size-6 items-center justify-center rounded-full bg-muted text-xs font-medium">
            {user?.name.slice(0, 1) ?? "?"}
          </span>
          <span className="min-w-0 flex-1 truncate text-sm">{user?.name ?? "加载中"}</span>
          {api.mode === "mock" && (
            <span className="rounded border px-1.5 text-[11px] text-muted-foreground">演示</span>
          )}
          {api.mode === "live" && (
            <TooltipIconButton tooltip="退出登录" aria-label="退出" className="size-7" onClick={() => void logout()}>
              <LogOutIcon />
            </TooltipIconButton>
          )}
        </div>
      </div>
    </nav>
  );
}

export function Topbar({ title, children }: { title: string; children?: ReactNode }) {
  const openMenu = useContext(MenuContext);
  return (
    <header className="flex h-12 shrink-0 items-center gap-2 px-4">
      <TooltipIconButton tooltip="打开导航" aria-label="打开导航" className="-ms-1 size-8 md:hidden" onClick={openMenu}>
        <MenuIcon />
      </TooltipIconButton>
      <h1 className="m-0 min-w-0 truncate text-sm font-medium">{title}</h1>
      {children}
    </header>
  );
}

export function Layout() {
  const { loadError } = useApp();
  const [open, setOpen] = useState(false);
  const location = useLocation();
  // 移动端抽屉在路由变化（切换会话、新建对话）后自动收起。
  useEffect(() => setOpen(false), [location.pathname]);
  // 首页与会话页是同一个聊天视图，切换时不重新挂载，避免首问跳转时闪烁。
  const isChat = location.pathname === "/" || location.pathname.startsWith("/c/");

  return (
    <RagRuntimeProvider>
      <TooltipProvider>
        <MenuContext.Provider value={() => setOpen(true)}>
          <div
            className="flex h-full min-h-0 bg-background text-foreground"
            onKeyDown={(event) => { if (event.key === "Escape") setOpen(false); }}
          >
            <Sidebar open={open} />
            {open && (
              <button
                type="button"
                aria-label="关闭导航"
                className="fixed inset-0 z-30 bg-black/20 md:hidden"
                onClick={() => setOpen(false)}
              />
            )}
            <main className="flex min-h-0 min-w-0 flex-1 flex-col" key={isChat ? "chat" : location.pathname}>
              {loadError && (
                <div role="alert" className="m-3 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                  加载失败：{loadError}
                </div>
              )}
              <Outlet />
            </main>
          </div>
        </MenuContext.Provider>
      </TooltipProvider>
    </RagRuntimeProvider>
  );
}

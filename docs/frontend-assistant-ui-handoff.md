# 前端 assistant-ui 重构：接口交接文档

> 日期：2026-11-15 · 适用分支：`codex/postgresql` · 前端依赖：`@assistant-ui/react@0.15.36`

本文写给后端同学，说明三件事：

1. 前端如何用 [assistant-ui](https://github.com/assistant-ui/assistant-ui) 重构，以及它怎样对接现有接口；
2. 后端已有的接口，前端用到了哪些、还有哪些可以接；
3. assistant-ui 能提供、但**后端还缺接口**的能力，以及建议的接口约定，需要和后端一起确认。

---

## 1. 重构概述

### 1.1 架构

assistant-ui 只管聊天界面和交互状态，数据仍然走现有的 `src/api`（live / mock 两套实现不变）。两者之间用 **ExternalStoreRuntime** 适配：

```
src/api (live.ts / mock.ts)          ← 现有接口层，未改动
        │
src/runtime/RagRuntime.tsx           ← useExternalStoreRuntime 适配层
  - 会话状态（turns + 进行中的一轮）→ assistant-ui 消息
  - onNew / onCancel / onReload      → createConversation / streamQuery / AbortController
  - threadList 适配器                → 会话列表的切换、新建、重命名、删除
        │
assistant-ui primitives
  - ThreadPrimitive       消息列表、自动滚动、空状态、推荐问题
  - MessagePrimitive      问题 / 回答消息，正文分片（Parts）
  - ComposerPrimitive     输入框（回车发送、输入法组字、自动增高）、发送 / 停止
  - ActionBarPrimitive    复制、重试
  - ThreadListPrimitive   侧栏会话列表
```

一轮问答在前端拆成两条消息：

| assistant-ui 消息 | 来源 | 说明 |
|---|---|---|
| user | `turn.user_message` | 问题 |
| assistant | `turn` 全部字段 | 正文作为 text part；引用、拒答、澄清、证据统计、反馈等放在 `metadata.custom.turn` |
| assistant（running） | SSE `status` / `token` | 流式过程中的阶段与正文；`complete` 后替换为正式轮次 |

### 1.2 页面变化

- 首页和会话页合并成一个线程视图：`/` 是空线程（欢迎页 + 推荐问题），首个问题发出时先 `POST /api/conversations` 再跳到 `/c/:id`，不再靠路由 state 传问题。
- 侧栏会话列表改用 `ThreadListPrimitive`，重命名 / 删除都经过运行时适配器。
- 回答卡片（过程摘要、引用、拒答换库重查、澄清选项、反馈）保留原有交互，渲染挂在 assistant-ui 的消息上下文里。
- 知识库页、管理后台、登录页没有改动。

---

## 2. 现有接口清单与前端使用情况

所有接口都在 `/api` 前缀下，认证用 HttpOnly 会话 Cookie，写请求带 `X-CSRF-Token`（取自 `rag_csrf` Cookie）。

### 2.1 前端已在用

| 接口 | 用途 | 对应 assistant-ui 能力 |
|---|---|---|
| `POST /auth/login` · `POST /auth/logout` · `GET /me` | 登录、身份、提问额度 | — |
| `GET /knowledge-bases` | 知识库列表（范围选择、名称映射） | Composer 扩展区 |
| `GET /knowledge-bases/{kb_id}/documents?limit=100` | 知识库页文档列表 | — |
| `GET /suggested-questions?limit=20` | 欢迎页推荐问题 | `ThreadPrimitive.Suggestion` |
| `GET /conversations` | 会话列表（全量模式） | ThreadList `threads` |
| `POST /conversations` | 新建会话（带默认 `kb_scope`） | 首问时调用 |
| `GET /conversations/{id}` | 会话历史（全量模式） | 线程 `messages` |
| `PATCH /conversations/{id}` | 重命名 | ThreadList `onRename` |
| `DELETE /conversations/{id}` | 删除 | ThreadList `onDelete` |
| `POST /conversations/{id}/query/stream`（SSE） | 流式问答：`status` / `token` / `complete` / `error` | `onNew`、`isRunning` |
| 断开 SSE 连接 | 停止回答（后端监听断连并取消） | `onCancel` |
| `GET /documents/{doc_id}/chunks/{chunk_id}?context=1` | 来源面板原文上下文 | 引用点击 |
| `PUT /turns/{turn_id}/feedback` | 赞 / 踩 / 取消（只传 `rating`） | 自定义反馈按钮 |
| `GET /documents` · `DELETE /documents/{doc_id}` | 管理后台文档 | — |
| `POST /admin/knowledge-bases/{kb_id}/documents` · `GET /admin/ingestion-jobs/{job_id}` | 上传入库与进度轮询 | — |
| `POST /admin/knowledge-bases` · `PATCH /admin/knowledge-bases/{kb_id}` | 新建 / 编辑知识库 | — |
| `GET /admin/insights/overview?days=7` · `GET /admin/insights/refusals?limit=50` | 问题洞察 | — |

### 2.2 后端已有、前端还没接（前端待办，后端不用改）

| 接口 | 可以用来做什么 | 备注 |
|---|---|---|
| `GET /conversations/{id}?turns_limit=&turns_cursor=` | 长会话分页，接 assistant-ui 的 `hasEarlier` / `onLoadEarlier`（向上滚动加载更早轮次） | **需确认**：页内轮次是不是按时间正序、`turns_next_cursor` 是不是指向更早一页 |
| `GET /conversations?limit=&cursor=` | 会话列表分页 | ExternalStore 的 threadList 适配器目前没有"加载更多"回调，要自己加按钮 |
| `GET /turns/{turn_id}/feedback` | 读取单轮反馈 | 历史轮次已经带 `feedback`，暂时不需要 |
| `PUT /turns/{turn_id}/feedback` 的 `reasons` / `comment` | 点踩后选原因（`incorrect` / `incomplete` / `irrelevant` / `outdated` / `citation_wrong`）、补充说明 | 前端目前只传 `rating` |
| `POST /turns/{turn_id}/shares` · `GET /shares/{token}` | 分享单轮回答，返回 `url: "/s/{token}"` | 前端还没有 `/s/:token` 页面；另见 3.7 |
| `DELETE /admin/knowledge-bases/{kb_id}?force=` | 删除知识库 | 管理后台还没有入口 |
| `PUT /admin/knowledge-bases/{kb_id}/suggested-questions` | 维护推荐问题 | 管理后台还没有入口 |
| `GET /admin/documents/{doc_id}/chunks` | 分块诊断 | — |
| `GET /admin/insights/feedback` · `PATCH /admin/insights/feedback/{id}` | 反馈列表与分拣 | — |
| `GET /admin/traces` · `GET /admin/traces/{trace_id}` | 链路追踪 | 回答卡片已显示 `trace_id`，可以加跳转 |
| `GET /admin/usage` | 用量统计 | — |
| `GET /admin/evaluations/latest` · `/schema` | 评测报告 | — |
| `POST /query` · `POST /query/stream`（管理员） | 单轮调试（`strategy` / `top_k` / `lane` / `debug`） | 可以做管理员调试台 |

---

## 3. 缺少的接口（需要后端配合）

优先级：**P1** = assistant-ui 的核心交互，现在只能降级处理；**P2** = 体验增强。

### 3.1 [P1] 重新生成回答 / 编辑已发送的问题

**assistant-ui 能力**：`onReload`（重新生成）、`onEdit`（编辑问题后重答），配合 `BranchPicker` 在多个版本之间切换。

**现状**：后端只能在会话末尾追加新的一轮。前端的"重试"只用于失败或被停止的那一轮，把同一个问题重新发一次，结果是一条新轮次；"编辑问题"和"重新生成已完成的回答"都没有开放。

**建议接口**（二选一，倾向方案 A）：

- 方案 A：查询接口加一个父节点参数
  ```http
  POST /api/conversations/{id}/query/stream
  { "question": "...", "kb_scope": {...}, "parent_turn_id": "<uuid>|null", "regenerate_of": "<turn_id>|null" }
  ```
  - `regenerate_of`：用同一个问题重答这一轮，生成同级的新版本；
  - `parent_turn_id`：从某一轮之后分叉（编辑问题），之后的轮次成为另一条分支；
  - 历史接口返回每轮的 `parent_turn_id`，以及同级版本信息（`branch_index` / `branch_count`，或者直接返回整棵树），前端据此显示 `1/2` 版本切换。
- 方案 B：不做分支，只支持"替换最后一轮"：`POST /api/conversations/{id}/turns/{turn_id}/regenerate`（SSE，事件格式和 query/stream 相同），旧轮次标记为被替换。

**需要后端决定**：Agent checkpoint 要能回退到指定轮次之前的状态，否则分支之后的上下文会混在一起。

### 3.2 [P1] 刷新或断线后恢复进行中的回答

**assistant-ui 能力**：`canResume` / `onResume`（恢复中断的运行）。

**现状**：流式回答进行中如果刷新页面或网络断开，前端什么都看不到；后端会把断连当成取消，这一轮直接丢失。

**建议接口**：

```http
GET /api/conversations/{id}/runs/active
→ 200 { "run_id": "...", "question": "...", "kb_scope": {...}, "started_at": "..." }
→ 204（没有进行中的运行）

GET /api/conversations/{id}/runs/{run_id}/stream
Last-Event-ID: <最后收到的事件 id>
→ text/event-stream，从断点之后补发事件
```

需要配套：

- SSE 每个事件带 `id:` 字段；
- 新增一个开始事件 `event: start`，`data: {"run_id": "...", "turn_id": "..."}`，让前端尽早拿到运行 ID；
- 断连不再立刻取消运行，而是保留一个可配置的宽限期，由 3.3 的显式取消接口来终止。

### 3.3 [P1] 显式取消运行

**现状**：停止回答靠断开 SSE 连接，后端轮询 `request.is_disconnected()` 后取消。中间有反向代理或缓冲时，断连可能晚很久才被发现，或者根本发现不了。做了 3.2 的断线恢复以后，"断连"和"取消"也必须分开处理。

**建议接口**：

```http
POST /api/conversations/{id}/runs/{run_id}/cancel
→ 202 { "cancelled": true }
```

取消之后，流里推送 `event: error`，`data: {"code": "cancelled", "message": "已停止回答"}` 作为终止事件。

### 3.4 [P1] `complete` 事件带上会话标题

**现状**：首轮回答后后端会自动给会话改名，但 `complete` 事件里没有标题，前端只能在每轮结束后重新拉一遍 `GET /conversations` 来同步侧栏和顶栏标题。

**建议**：`complete` 的 data 里加一个 `conversation`，内容是会话摘要（至少包含 `title`、`updated_at`、`turns_count`）。或者单独推一个 `event: title`，`data: {"title": "..."}`。

### 3.5 [P2] 回答后的推荐追问

**assistant-ui 能力**：`suggestions`（`ThreadPrimitive.Suggestions` 在最后一条回答下方显示可点击的追问）。

**建议**：`complete` 的 data 里加 `follow_up_questions: string[]`（0–3 条），历史轮次同样返回，或者只给最后一轮。

### 3.6 [P2] 会话归档

**assistant-ui 能力**：ThreadList 的 `archivedThreads`、`onArchive` / `onUnarchive`，以及 `ThreadListItemPrimitive.Archive` 按钮。

**现状**：`ConversationStatus` 只有 `active` / `deleting` 两种状态。

**建议**：
- `PATCH /api/conversations/{id}` 支持 `{ "archived": true | false }`；
- `GET /api/conversations?archived=true|false`，默认 `false`。

### 3.7 [P2] 分享页的访问权限

**现状**：`GET /api/shares/{token}` 挂在 `require_user` 下面，没登录的人打不开分享链接。

**需要确认**：分享链接是只给已登录的同事看，还是允许匿名访问？如果允许匿名，要把这个路由移出 `require_user`，并单独做限流。前端的 `/s/:token` 页面会按确认后的结论来做。

### 3.8 [P2] 会话搜索

**建议**：`GET /api/conversations?q=<关键词>&limit=&cursor=`，按标题搜索，也可以包含问题正文。侧栏顶部会加一个搜索框。

### 3.9 [P2] 流式错误码清单

**现状**：前端只单独处理了 HTTP 409（"上一个问题还在回答中"），其他 `event: error` 的 `code` 都直接显示 `message`。

**请后端提供**：SSE `error.code` 和 HTTP 错误 `error.code` 的完整枚举，以及每个码是否可以重试，例如 `conversation_busy`、额度用尽、`cancelled`、超时、模型不可用等。前端会按码决定显示"重试"按钮还是倒计时。

### 3.10 [暂不需要] 对话内附件

assistant-ui 支持 `AttachmentAdapter`（在输入框里拖入文件或图片）。目前产品上没有"带附件提问"的需求，先记在这里。

---

## 4. 需要后端确认的现有约定

| # | 问题 | 前端当前假设 |
|---|---|---|
| 1 | `token` 事件是不是只在正文校验通过之后才发？ | 是。所以流式阶段先只显示 `status`，正文晚一些才出现 |
| 2 | `complete` 的 payload 和历史轮次的字段是否完全一致（`outcome` / `grounding` / `refusal` / `clarification` / `evidence_stats` / `kb_scope`）？ | 前端 `normalizeTurn` 两种外壳都兼容，缺失字段会按保守值推导 |
| 3 | 历史分页每页内是不是按时间正序？cursor 是不是指向更早一页？ | 见 2.2，接入 `onLoadEarlier` 之前需要确认 |
| 4 | `status.phase` 后续会不会新增？ | 前端会忽略不认识的阶段；有 `label` 时优先显示后端给的文案 |
| 5 | 同一会话并发提问返回 409，提示文案能不能由后端给？ | 目前前端写死了文案 |

---

## 5. 前端改动文件

| 文件 | 说明 |
|---|---|
| `frontend/src/runtime/RagRuntime.tsx` | 新增：ExternalStoreRuntime 适配层（消息转换、流式问答、取消、重试、会话列表适配器） |
| `frontend/src/components/chat/Thread.tsx` | 新增：线程视图（欢迎页 / 消息列表 / 回到底部） |
| `frontend/src/components/chat/Messages.tsx` | 新增：问题消息、回答消息（正式轮次与流式中两种状态） |
| `frontend/src/components/chat/Composer.tsx` | 新增：基于 ComposerPrimitive 的输入框和查询范围选择 |
| `frontend/src/components/TurnView.tsx` | 重构为 `TurnBody`（回答正文部分，不依赖 assistant-ui，方便单测） |
| `frontend/src/components/Layout.tsx` | 侧栏改用 ThreadListPrimitive；布局层挂载运行时 |
| `frontend/src/pages/ChatPage.tsx` | 首页和会话页合并 |
| 删除 `pages/HomePage.tsx`、`components/Composer.tsx`、`components/PendingTurn.tsx` | 功能已迁移到上面的新组件 |

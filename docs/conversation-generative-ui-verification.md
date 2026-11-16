# 会话 Generative UI 一期验证记录

2026-11-15，接口字段按用户约定 v1 实施。二期表格输出暂不生成，`data_blocks=[]`。

## 实现结果

- Fast、扩查、升级后的实际 LangGraph 检索调用均接入请求级收集器；管理员无状态查询不启用收集器。
- 每次实际调用先发 running；终态在最终引用校验完成后、首个 token 前按调用顺序发送。每个 ID 一次终态。断连走协作取消，不补发终态，不提交未完成轮次。
- hits 只取实际调用范围内的检索块，至多 10 条；snippet 截断到 200 字，屏蔽控制字符和文件系统路径；路径可能含空格/引号，因此从检测起点隐藏至行尾。filename 去除父目录。错误固定为 retrieval_failed / 检索失败，不发送原始异常或模型消息。
- used 按最终引用的 kb_id/doc_id/chunk_id 精确匹配；relevant 计该次返回中最终通过证据校验的数量。可选 score 暂不返回，避免混合/重排分数伪归一化。
- JSONB 数组与正式轮次同事务保存，complete、非流式响应、完整及分页历史一致；旧轮次默认两个空数组。

## 与提示的差异

- 新增 **v5**，不是 v4：v4 已发布给全文检索；已在实施前汇报。未改写 v1–v4。
- data/table 按约定二期延期：定义了类型、行列及单元格约束，本期不发送 data，也不开放未完成引用溯源验证的表格生成。
- 可选 score 省略。其他字段名和枚举未改。

## 验证

- 定向会话/Fast/Agent：38 passed。
- 既有 SSE/会话接口、PG 仓储及迁移：21 passed。
- 新 UI 模块、会话及管理员调试接口（含路径安全修复与旧表结构升级）：34 passed。
- 独立审查发现带空格路径的后半段泄漏；3 项路径回归和 1 项不可见字符绕过回归均先失败后通过。其余无阻塞项。审查者单独验证了实际图节点 ContextVar 传播与管理员接口隔离。
- 全量 `uv run pytest -q --tb=short`：**1217 passed、16 skipped、5 deselected、2 warnings，507.30 秒**。警告为既有依赖/评估序列化提示。该轮收集初始 8 项 UI 用例；随后补充路径及不可见字符安全用例，最新整个 UI 专项 **12 passed**，不是将交叉运行的用例重复相加。
- 实际数据库：应用及评估 schema 均已登记 v5；原应用库 7 轮全部保留并返回空 UI 数组，评估库 0 轮。迁移重复执行已验证。
- 当前本地 8000 端口没有服务，未启动新的常驻服务。
- 凭据扫描通过；未 commit、push。本任务未修改 frontend/，该目录有其他任务并行修改。

## 本任务文件清单

新增：
- app/domain/generative_ui.py
- app/application/conversations/generative_ui.py
- app/infrastructure/persistence/schema_v5.sql
- tests/test_generative_ui.py
- docs/superpowers/plans/2026-11-15-conversation-generative-ui.md
- docs/conversation-generative-ui-verification.md

修改：
- app/domain/conversations.py
- app/application/conversations/events.py
- app/application/conversations/progress.py
- app/application/conversations/service.py
- app/application/query/fast_lane.py
- app/infrastructure/conversations/langgraph_agent.py
- app/infrastructure/conversations/postgres_repository.py
- app/infrastructure/persistence/migrations.py（仅新增 v5 分支，本任务未改 v1–v4 分支）
- app/interfaces/http/schemas.py
- docs/frontend-assistant-ui-handoff.md（第 6 节）

详细执行产物位于 Git 忽略目录 data/repair/generative-ui/。

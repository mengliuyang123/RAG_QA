# Agent 收尾修复验证记录

2026-11-15。问题：追踪 `f0f90784-bc87-48d0-b13e-746a4799fffb`，提问“langchain 怎么使用”。实际追踪有两次 `sufficient=true` 后生成正文均为 0，模型仍返回工具调用，最终受限额分支影响返回拒答。原历史记录保留，不作事后改写。

## 修复

- `langgraph_agent.py`：规划阶段可用工具；检索后的回答生成使用原始未绑定工具的模型及收尾提示。补查继续由证据审核的有效 `next_query` 控制，保持最多 3 次检索。
- 达到限额而没有新检索时，有批准证据就进入生成；没有批准证据才安全拒答。合法的第三次检索不会因同批第四次调用被拒绝而丢弃。
- 若无工具模型仍异常输出工具调用，后端不执行它，进入已有的一次无工具修复；持续无正文/工具响应归为 `LLMGenerationError`。
- 已判充分的证据如果仍不能生成或修复有效引用，返回生成错误；审核真正判为不足时仍允许正常拒答。证据不足但能回答部分内容时仍保留部分依据回答。
- 引用校验、答案审核及共享预算均保留，未批准正文不进入生成。
- `orchestrator.py`：防止“证据充分但通道返回拒答”被重新标为 `insufficient_evidence`。
- 会话 SSE 继续使用已存在的 `generation_error` 错误码，未完成生成的提示为“回答生成未能完成，请重试”，不写成正式拒答轮次。没有新增拒答枚举或前端字段。
- 生成追踪增加 `tools_enabled`、`unexpected_tool_calls`，便于区分模型请求配置与异常响应。

## 真实模型验证

用新代码的管理查询入口执行一次相同问题，不创建或改写正式会话：

- 新追踪：`55347bee-f3dc-4e5e-83a4-7174e9a0627c`。
- 实际检索 2 次，累计 10 个去重候选，4 个相关并被引用。
- 第二次证据审核判充分；生成请求 `tools_enabled=false`，但云模型仍异常返回一次工具调用，正文 0 字。
- 后端没有执行第三次检索；一次无工具修复生成 1175 字正文。
- `answer_judge` 实际执行，`supported=true`、`complete=true`，最终 `outcome=answered`、`grounding=grounded`。

本次实测证明的是有界收尾和审核链路，不能保证模型每次都在首个生成请求中正常输出。当前策略把异常工具响应阻断并交给唯一修复机会，避免再次消耗检索预算。

## 测试与运行状态

- 最初 4 项新回归复现旧行为；编排元数据与 SSE 提示、真正证据不足边界的新增回归也均先失败后通过。
- 初步定向 83 passed；更新 SDK 无工具生成断言、批量工具规划和错误语义后，相关 6 个测试模块 80 passed。
- 一次独立审查无阻塞问题，审查者独立运行 49 项通过。
- 首轮全量：1225 passed、4 failed；4 项均为旧收尾策略的测试断言/装配，已针对新行为更新并复测通过；最终全量复测 **1230 passed、16 skipped、5 deselected、2 个既有警告，479.47 秒**，无失败。
- 已重启本地原有 8000 后端加载修复，健康检查 200。没有修改模型服务或前端文件，未提交或推送。
- 日志及安全摘要位于 Git 忽略目录 `data/repair/agent-finalization/`；未记录凭据或整段文档正文。

## 本任务文件

生产代码：
- app/infrastructure/conversations/langgraph_agent.py
- app/infrastructure/conversations/prompts.py
- app/application/query/orchestrator.py
- app/application/conversations/service.py

测试：新增 tests/test_agent_finalization.py；更新 tests/test_agent_quality_graph.py、tests/test_langgraph_conversation_agent.py、tests/test_citation_versions.py、tests/test_conversation_agent_factory.py、tests/test_final_execution_boundaries.py。

文档：本文件及 docs/superpowers/plans/2026-11-15-agent-finalization.md。

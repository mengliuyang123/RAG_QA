# Conversation Generative UI Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement inline, with a final independent review.

**Goal:** 会话 SSE 增量 tool 事件及正式轮次回放，保持既有事件，data_blocks 暂为 []。
**Spec:** 用户接口约定 v1；docs/frontend-assistant-ui-handoff.md。
**Architecture:** domain 类型化 UI 数据；请求 ContextVar 收集 Fast/Agent 的真实检索调用，running 即时发送，终态在最终引用校验后且 token 前发送。仅 ConversationService 开启收集，管理员单轮不变。安全边界仅接收实际检索范围内的返回块，禁止内部路径，截断 200 字。新增 v5 JSONB 数组（v4 已为全文检索使用，已汇报）。

- [x] RED：工具顺序/终态唯一、错误、取消、范围和截断、持久化与旧数据兼容测试。
- [x] GREEN：domain UI 模型、typed tool/data events、请求收集器；Fast/Agent 调用点接入。
- [x] 服务 token 前终态提交，正式结果携带快照；v5 迁移、完整/分页/reverse 历史、响应模型。
- [x] 后端定向及全量测试；一次独立审查；实际 schema 迁移、服务恢复。
- [x] handoff 已实现章节与差异、验证记录。只改后端及文档，不 commit/push。

Rulings: v5 替代用户提示的 v4，避免覆写已发布版本。score 可选，v1 不返回原始混合/重排分数，避免伪归一化。relevant 是该次命中中最终通过证据校验的数量。data 仅定义类型，本次不生成表格，data_blocks=[]。

完成记录：docs/conversation-generative-ui-verification.md。当前无 8000 监听，保持服务未启动状态。

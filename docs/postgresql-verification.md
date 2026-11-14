# PostgreSQL 替换验证记录

2026-11-15 完成。分支 `codex/postgresql`，改动保留在工作区，未自动提交。

## 验证结果

- PostgreSQL 18.7、pgvector 0.8.7，使用用户指定的全新数据库。
- `uv run --group eval pytest -q --tb=short`：1199 passed、16 skipped、5 deselected，2 个既有依赖/序列化警告。
- 另补时间列与反馈表达式索引升级用例；`tests/test_postgres_database.py`：3 passed。
- 前端 `npm.cmd test`：39 passed；live 生产构建通过。前端同时有其他修改，本任务未编辑前端代码。
- 真实模型验证：上传入库、2560 维 pgvector、混合检索和重排序、带引用 SSE、非流式追问、两轮历史保存通过。
- 重启后读取两轮历史及向量维度成功；usage、insights、trace 管理接口均返回 200。
- 临时账号、会话、知识库及文档已清理。新应用库没有默认管理员，请按 quickstart 在交互终端创建账号。
- Python SQLite 审计钩子下执行新后端存储读取成功，未打开 SQLite；本次运行检查前后保留的 14 个旧数据库相关文件大小和修改时间一致。
- 详细运行产物保存在被 Git 忽略的 `data/repair/postgres/`，不含数据库连接凭据。

## 独立审查修复

1. 删除标记与正式轮次保存共用数据库写锁。已用两个连接的真实并发测试复现旧竞态，再验证删除等待保存事务完成。
2. 会话分页、分享授权/正文使用 REPEATABLE READ 只读事务。已复现并发删除导致正文消失，再验证一致快照。
3. 评估集合初始化增加数据库咨询锁，使多个容器/不同工作目录不能同时准备并清理同一集合。
4. 时间列升级前在同一事务内删除、重建反馈 COALESCE 表达式索引，避免 TEXT 与 TIMESTAMPTZ 混用时升级失败。

## 实施选择及边界

- 复用原工作区并创建功能分支，保留用户原有修改；未自动提交、合并或推送。
- JSON 保留领域原有字符串表示；时间字段最终使用 TIMESTAMPTZ，并在行映射中转 UTC ISO 字符串。最初保留 TEXT 的选择因日历边界测试暴露问题而修正。
- 状态变更使用 schema 级事务咨询锁，以简单方式保证队列容量、租约、发布和删除检查的正确性；代价是并发写入吞吐受限。任务领取同时使用 SKIP LOCKED。
- 向量保存完整 float32 精度并使用 cosine 精确检索；当前 2560 维不创建普通 vector HNSW 索引。大规模语料应另行评估索引方案及精度/召回取舍。
- 原文件、修复日志和评估报告仍是文件产物；所有原数据库存储均已替换，不导入旧 SQLite/Chroma 数据。
- SQLite 专用钩子和旧格式导入测试改为真实 PostgreSQL SQL 观察、事务、隔离和迁移检查，没有批量跳过持久化测试。
- Windows Python 3.11 的 monotonic 使用 GetTickCount64，追踪使用 QueryPerformanceCounter；准入计时改为与追踪共用 perf_counter，以保证排队耗时不大于总请求耗时。已有八项相关不变量先复现失败，再通过验证。
- 实施过程中发现的前端并行修改保持原样；未改变前端产品行为。

未遗留独立审查认定的阻塞问题。

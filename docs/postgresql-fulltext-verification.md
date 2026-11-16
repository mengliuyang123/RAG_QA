# PostgreSQL 中文全文检索实施与验证

2026-11-15，后端改动；前端现有修改保持原样。

## 实现

- 默认关键词适配器改为 `PostgresKeywordIndex`，不持有进程内语料或 BM25 索引。
- 固定 Jieba 0.42.1；文档使用搜索模式保留子词，查询使用精确模式，均禁用 HMM 并共用规范化及词典。例如“数据库”不会作为 OR 查询展开成“数据”，避免误召回“数据报表”。
- `vector_chunks.search_text` 与正文/向量同一事务写入；`search_vector` 是使用显式 `pg_catalog.simple` 配置的生成列，以 `vector_chunks_search` GIN 索引查询。
- 排序使用 `ts_rank_cd`，集合、知识库和未发布文档排除在 Top-K 前生效；混合召回继续 RRF 和原重排链路，接口字段不变。追踪中的 `bm25_ms` 字段保留原名以兼容现有消费者，现在记录关键词全文召回耗时。
- 默认入库、删除及检索不再全库加载文本以重建关键词索引。保留旧 BM25 实现用于原有单元测试和显式替换适配器，生产默认工厂不使用它。

## 旧数据与部署

应用和评估 schema 均已完成迁移 v4。应用库 1102 个块全部回填；评估库 0 个块，索引结构已创建。应用数据按稳定顺序对 collection/id/doc_id/kb_id/metadata/text/embedding 聚合摘要，迁移前后均为 `f702eac91fca212e610b10972b23524e`，未重算或修改向量、正文和元数据。

迁移以每批 500 条读取旧数据；DDL、回填、GIN 和版本登记同事务提交，失败整体回滚。实际 1102 条回填覆盖多个批次。部署时停止旧后端后执行初始化 CLI，再运行新代码；后端健康检查 200，真实已有语料的只读全文查询返回 5 条结果。模型服务和前端不受本次部署操作影响。

## 验证

- 定向后端回归：109 passed。
- 将发布状态集成测试切换到 PostgreSQL 全文适配器后，发布过滤和全文模块联合运行：18 passed。
- 独立审查单独运行全文模块：9 passed，无阻塞问题。
- 全文模块覆盖中文词语/误召回、英文及数字、操作符安全、知识库与集合隔离、Top-K 前排除、真实 GIN 查询计划、旧块回填及幂等、迁移异常原子回滚、未提交和回滚修改不可见、独立子进程新增/删除后立即可见、旧 rebuild 快照不可覆盖事实数据、默认流水线不扫描全库。
- 接口与调试回归：23 passed。原测试中没有数据库连接的内存向量替身改为真实 PostgreSQL 文本存储，并保持向量召回为空，确保仍验证关键词候选及重排异常响应。
- 全量后端回归：`uv run --group eval pytest -q --tb=short`，**1209 passed、16 skipped、5 deselected、2 warnings，378.82 秒**。警告来自既有依赖及评估报告序列化。
- 凭据扫描和 git diff 空白检查通过；既有示例配置的占位值已规范化，真实 `.env` 未改动。

运行日志和只包含统计及摘要的部署记录位于 Git 忽略目录 `data/repair/fulltext/`。

## 运维边界

所有后端/入库 worker 必须在迁移后使用新代码；旧代码不会写入新增的 NOT NULL 分词字段。直接 SQL 导入或修改正文必须同步用后端 `search_text(text)` 更新分词字段。调整分词版本或词典时，需要新迁移统一回填，不能让不同 worker 使用不同规则。关键词相关性改为 PostgreSQL 排名，不能沿用 BM25 分数阈值；现有 RRF 使用排名，不依赖原始分数尺度。

参考：[PostgreSQL 全文索引](https://www.postgresql.org/docs/current/textsearch-tables.html)、[查询与相关性排序](https://www.postgresql.org/docs/current/textsearch-controls.html)、[Jieba 搜索模式](https://github.com/fxsjy/jieba)。

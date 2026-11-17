# Design

## Context

动机见 `proposal.md` 的 Why 部分，行为契约见 `specs/document-chunking/spec.md` 和 `specs/retrieval-evidence-context/spec.md`。和方案相关的现状如下：

- **解析**：解析器产出 `DocumentSegment(text, page, filename)`。Markdown、Word 整篇只产出一个片段；PDF、PowerPoint 每页（每张幻灯片）一个片段。所有解析器都不保留标题信息。
- **分块**：`RecursiveDocumentChunker` 对每个片段调用 LangChain 的 `RecursiveCharacterTextSplitter`（500 字、重叠 50）。块 ID 为 `{doc_id}:{chunk_index}`，`chunk_index` 在整篇文档内连续。
- **存储**：`vector_chunks` 一行一块，`metadata` 为 JSONB，内容是 `DocumentChunk` 除 `id`、`text` 外的字段。`search_text` 在写入时由 jieba 分词生成，`search_vector` 是由它派生的生成列。Embedding 在 `job_pipeline._embed` 中直接用 `chunk.text`。
- **元数据读取**：`knowledge_reader._decode_header` / `_validate_group` 会严格校验块元数据。`read_chunks` 一次最多读 7 块，并校验读取期间文档没有变化。前端的引用前后文用的就是这套接口。
- **证据**：快速通道在 `fast_lane.py` 中由 `RetrievalResult` 构造 `EvidenceItem`；Agent 通道在 `langgraph_agent.py`（约 674-692 行）中由检索结果构造 evidence 字典。证据判断（`infrastructure/conversations/quality.py`）、快速生成（`PromptBuilder.build_numbered`）和 Agent 生成都直接读 `item.text`。
- **预算**：`PromptBudgetGuard.ensure` 超限时只会抛出 `PromptContextTooLargeError`，不会裁剪。
- **替换版本**：`IngestionJobs.submit(..., replace_doc_id=...)` 已经实现了“新版本发布成功后才替换旧版本”。源文件保存在 `data/` 下，由 `ingestion_sources` 端口管理。

## Goals / Non-Goals

**Goals:**
- 标题路径只依赖文档自带的结构标记，结果可以复现，相同输入一定得到相同的块。
- 不改表结构，不写 SQL 迁移；新旧块可以在同一个集合里共存。
- 父块扩展只是在检索服务内部多出的一步。快速通道和 Agent 通道拿到的检索结果已经带好上下文，两条通道不各自实现扩展。
- 引用、分享、历史会话的数据格式不变。

**Non-Goals:**
- 不对 PDF 做版面分析或 OCR，也不按字号推测 PDF 标题（留给后续的 PDF 解析改进）。
- 不做语义分块，也不在入库时调用 LLM 生成摘要或问题。
- 不改变 `top_k`、`top_k_rerank` 的默认值和检索策略选择逻辑。
- 前端不新增“查看父块”的界面，侧栏前后文保持现状。

## Decisions

### D1：解析器按标题拆分片段，标题路径放在片段上
`DocumentSegment` 新增 `heading_path: tuple[str, ...] = ()`。Markdown 和 Word 解析器在遇到标题时结束当前片段、开始新片段，每个片段只属于一条标题路径；PDF 和 TXT 不变，PowerPoint 把幻灯片标题占位符（`type="title"` / `"ctrTitle"`）的文字作为路径。

- 标题行本身不进入正文，只出现在路径里。
- Markdown 用逐行状态机识别 ATX 标题，跳过围栏代码块。暂不支持 Setext 标题（`===` 下划线写法）。
- Word 通过 `word/styles.xml` 把样式 ID 解析为样式名和大纲级别，同时兼容 `Heading N` 和中文的“标题 N”，以及直接设置在段落上的 `w:outlineLvl`。

**备选方案**：解析器保持一个大片段，由分块器重新扫描 Markdown 标题。没有采用，因为 Word 的标题信息在解析成纯文本时就已经丢失，只能在解析器里提取；把结构提取统一放在解析器，分块器就不需要知道文件格式。

### D2：父块不单独存储，由连续子块拼接
每个子块的 metadata 增加 `parent_index`（整篇文档内从 0 开始连续）和 `heading_path`。父块全文等于同一 `parent_index` 下的子块按 `chunk_index` 顺序拼接。为了让拼接结果与原文一致，子块之间不再重叠：`chunk_overlap` 只在父块之间起作用，父块内部为 0。

- 切分顺序：片段 →（按 `PARENT_CHUNK_SIZE` 递归切成）父块 →（按 `CHUNK_SIZE`、重叠 0 切成）子块。仍然使用 `RecursiveCharacterTextSplitter` 和现有的中文分隔符表。
- 片段天然在标题和页码处断开，所以父块不会跨章节，也不会跨页。

**备选方案 A**：新建 `parent_chunks` 表保存父块正文。查询时少一次拼接，但要写迁移，删除和替换文档时还要多维护一张表的一致性。现有的删除修复日志和发布快照逻辑都是围绕 `vector_chunks` 写的，代价太大。
**备选方案 B**：把父块全文复制进每个子块的 metadata。读取最简单，但存储会放大 2 到 3 倍，`metadata` 也会变成大字段，所有只读头部的查询都会变慢。

### D3：索引文本与展示文本分离
新增纯函数 `index_text(chunk) -> str`，返回 `"{filename} › {h1} › {h2}\n{text}"`；路径为空时只用文件名作前缀。以下三处改为使用它：

1. `job_pipeline._embed` 生成 Embedding 时；
2. `PostgresVectorStore.add_chunks` 计算 `search_text` 时；
3. 重排时发给重排服务的 `documents`。

`vector_chunks.text` 仍存子块原文，因此浏览、引用、高亮都不受影响。为了让第 3 点不必再读一次数据库，`RetrievalResult` 新增可选字段 `heading_path`，从 metadata 带出来。

**备选方案**：把前缀直接写进 `text`。实现最简单，但引用和高亮会显示前缀，与 `document-chunking` 规格冲突。

### D4：在 `RetrievalService` 内完成父块合并与扩展
在 `_retrieve_snapshot` 返回之前新增 `_expand_to_parents` 步骤：

1. 现有流程照常产出子块排名，但内部候选数从 `top_k` 放大到 `min(top_k * 3, top_k_rerank)`，保证合并后仍有足够的父块。
2. 按 `(doc_id, parent_index)` 合并，保留每组中排名最高的子块作为代表，截取前 `top_k` 组。
3. 批量读取这些父块的全部兄弟子块：新增向量存储方法 `read_parent_chunks(keys)`，用一条 SQL 按 `doc_id` 加 `metadata->>'parent_index'` 读取，并沿用相同的集合、范围和排除文档条件。
4. 拼出 `context = "文件名：…\n章节：h1 › h2\n页码：…\n内容：\n" + 父块全文`，写进 `RetrievalResult.context`。
5. 兄弟子块与命中子块的 `doc_updated_at` 不一致，或者兄弟子块数量不连续时，回退为 `context = None`，表示只用命中子块，满足规格中的“读取期间文档被替换”场景。旧块没有 `parent_index`，直接走回退。

`EvidenceItem` 新增 `context: str | None`。以下三处统一改为 `item.context or <现有格式化>`：证据判断的 payload、`PromptBuilder.build_numbered`、Agent 的生成提示词。快速通道和 Agent 通道构造证据时都把 `result.context` 复制过去。

**备选方案**：在快速通道和 Agent 通道各自扩展。没有采用，因为两条通道会出现两份逻辑，而且 Agent 的检索结果要经过 LangGraph 状态序列化，更容易漏掉。

### D5：超出预算时整条裁剪证据
新增 `fit_evidence_to_budget(evidence, build_messages, guard)`：先按完整证据组装消息，超出预算时从排名最低（编号最大）的证据开始整条移除后重试，至少保留 1 条；保留 1 条仍然超限时，照旧抛出 `PromptContextTooLargeError`。证据判断和生成都要调用它。被移除的编号不会出现在批准证据里，所以引用校验的逻辑不用改。

### D6：重新入库命令原地重建文本块
新增 `python -m app.interfaces.cli.reindex [--kb <id> ...] [--dry-run] [--all] [--limit N]`。

**为什么不复用替换版本流程**：最初设想用 `replace_doc_id=<旧 doc_id>` 重新提交同一份源文件，但实施时发现走不通。`register_upload` 会拒绝同库内容相同的上传；数据库还有唯一索引 `documents_live_kb_sha256`，要求同库未删除文档的 sha256 不能重复，替换期间新旧两行必然冲突。要放宽这两处，就得改查重规则、加迁移、调整发布顺序。因此改为原地重建，不触碰查重规则和数据库约束。

- **候选**：已发布（`status='ready'`）、未删除、没有替换任务在进行的文档，并且在 `ingestion_jobs` 中有内容摘要一致的成功任务（用它定位并校验受控源文件）。默认只处理仍有旧格式块（没有 `parent_index`）的文档，`--all` 连新格式的也重建，`--limit` 用于分批。没有任务记录的文档列为“找不到源文件”并跳过。
- **重建**：通过 `verified_snapshot` 取得大小和摘要都经过校验的源文件快照，在内存里完成解析、分块和分批向量化，这期间旧文本块照常可检索。
- **替换**：在一个事务里先取得与上传、删除、发布相同的业务写锁，再确认文档仍是同一已发布版本（内容摘要一致，没有被删除，也没有替换任务），然后删除该文档的全部旧块并写入新块。确认不通过时不做任何修改，结果记为“跳过”。`doc_id` 保持不变，`doc_updated_at` 更新为重建时间，历史引用打开前后文时会按“文档已更新”处理，和替换版本一致。
- **失败**：源文件缺失或校验不通过记为跳过，解析失败、向量化失败或写入失败记为失败，这些情况下旧文本块都不变。
- **标准答案（FAQ）文档**：它的源文件就是当前已发布的那篇 Markdown，和其他文档一样从源文件重建。它的格式本来就是 `## 问题` 加答案，切出来正好每个问题一个父块。
- 不经过入库任务队列，所以管理页上看不到进度；进度和结果由命令行逐篇输出。

### D7：配置
`app/config.py` 新增 `parent_chunk_size: int = Field(default=1500, gt=0)`，并校验 `parent_chunk_size >= chunk_size`。`chunk_overlap` 的含义改为“父块之间的重叠”，默认值仍为 50。`.env.example` 和 README 的“常用配置项”同步更新。

## Risks / Trade-offs

- [生成上下文变长，单轮 Token 费用最多约为原来的 3 倍] → `PARENT_CHUNK_SIZE` 可配，设成和 `CHUNK_SIZE` 相同就退化为现在的行为；D5 保证不会超出预算；上线前用评估脚本对比费用和质量。
- [子块不再重叠，句子在边界处可能被切开，影响向量召回] → 父块扩展后，生成模型能看到完整的句子；召回影响用评估集中的检索命中率来衡量，必要时允许子块内部有少量重叠，同时把拼接规则改成去重拼接（这只改实现，不改规格）。
- [Word 文档没有使用标题样式（全文手动加粗）时，标题路径为空] → 规格只承诺依据结构标记识别标题，这类文档的表现与现在相同，不会变差。
- [重新入库产生 Embedding 费用，大库耗时较长] → 支持 `--dry-run` 先看数量，支持按知识库分批执行，期间旧版本始终可查。
- [`read_parent_chunks` 用 `metadata->>'parent_index'` 过滤，没有索引] → 查询条件同时带 `doc_id`，可以先走已有的 `vector_chunks_doc_id` 索引，每篇文档的块数有限，代价可以接受。
- [旧文档长期不重新入库时，效果提升不明显] → 在 README 的升级说明中提示执行 reindex；`list_document_headers` 可以统计旧格式块的数量，供后续在管理页展示（本次不做）。

## Migration Plan

1. 部署新版本：新上传的文档自动使用新规则，旧块按回退逻辑正常工作。不需要停机，不需要数据库迁移。
2. 运行 `python -m app.interfaces.cli.reindex --dry-run` 确认范围，再按知识库分批执行。
3. 回滚：回滚代码即可。旧代码读取带新增 metadata 字段的块时，`_decode_header` 只取已知字段，可以正常工作；新块在旧代码下表现为不带重叠的普通块，只是没有父块扩展。实施时要用测试确认旧版本的读取路径确实会忽略未知字段。

## Open Questions

- Markdown 的 Setext 标题和 Word 的自定义样式（名字不含 Heading 或“标题”、但设置了大纲级别）是否需要支持，可以等真实文档反馈后再补。只改解析器，不影响规格。
- `top_k * 3` 的放大倍数是经验值，可以在评估后再调整。

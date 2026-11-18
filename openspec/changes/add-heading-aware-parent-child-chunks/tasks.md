# Tasks

## 1. 领域模型与配置

- [x] 1.1 `DocumentSegment` 新增 `heading_path: tuple[str, ...] = ()`；`DocumentChunk` 新增 `parent_index: int | None = None` 和 `heading_path: tuple[str, ...] = ()`。验证：现有领域模型测试全部通过，并新增测试确认默认值能构造出旧格式对象。
- [x] 1.2 `app/config.py` 新增 `parent_chunk_size`（默认 1500），并校验 `parent_chunk_size >= chunk_size`；同步更新 `.env.example` 和 README 的“常用配置项”。验证：新增配置测试，覆盖默认值、非法组合报错、环境变量覆盖。
- [x] 1.3 新增纯函数 `index_text(chunk)`，返回“文件名 › 标题路径”前缀加正文，路径为空时只用文件名。验证：单元测试覆盖空路径、多级路径和标题中含特殊字符的情况。

## 2. 解析器提取标题路径

- [x] 2.1 Markdown 解析器改为逐行状态机：识别 ATX 标题 `#` 到 `######`，跳过围栏代码块，每条标题路径产出一个片段，标题行不进入正文。验证：测试覆盖多级标题、同级切换、代码块内的 `#`、文档开头没有标题的正文、没有任何标题的文档。
- [x] 2.2 Word 解析器读取 `word/styles.xml`，识别 `Heading N`、“标题 N” 和 `w:outlineLvl`，按标题拆分片段；表格行仍然归属所在章节。验证：用 `tests/` 中构造的 `.docx` 样例，覆盖中英文样式名、表格在章节内、没有标题样式这三种情况。
- [x] 2.3 PowerPoint 解析器把标题占位符（`title` / `ctrTitle`）的文字作为该页片段的 `heading_path`。验证：测试覆盖有标题的幻灯片、没有标题的幻灯片，以及页码保持不变。
- [x] 2.4 确认 PDF 和 TXT 解析器产出的 `heading_path` 为空，行为与现在相同。验证：现有解析器测试通过，并补一条断言。

## 3. 父子两级分块

- [x] 3.1 改写 `RecursiveDocumentChunker`：先按 `parent_chunk_size` 切出父块（父块之间重叠 `chunk_overlap`），再在父块内部按 `chunk_size` 切出子块（重叠 0）；写入连续的 `parent_index` 和 `chunk_index`，`heading_path` 从片段继承。验证：测试覆盖规格中的“短章节”“长章节”“章节边界不合并”“子块拼回父块”“父块小于子块上限”。
- [x] 3.2 组合根（`app/bootstrap/factories.py`、`container.py`）把 `parent_chunk_size` 传给分块器。验证：现有组合根和架构测试通过。

## 4. 索引文本与存储

- [x] 4.1 `job_pipeline._embed` 改为对 `index_text(chunk)` 做向量化。验证：入库流水线测试中，用假 Embedding 记录收到的文本，断言带有前缀。
- [x] 4.2 `PostgresVectorStore.add_chunks` 用 `index_text(chunk)` 计算 `search_text`，`text` 列仍存原文；`metadata` 中写入 `parent_index` 和 `heading_path`。验证：PostgreSQL 集成测试中，只出现在标题里的词能被 `PostgresKeywordIndex.search` 命中，而文本块浏览接口返回的正文不含前缀。
- [x] 4.3 `knowledge_reader._decode_header` 和 `PostgresVectorStore._chunk` 兼容缺少新字段的旧 metadata，`RetrievalResult` 新增 `heading_path` 和 `parent_index` 并从 metadata 带出。验证：测试分别用旧格式和新格式的 metadata 读取、浏览文本块、打开引用前后文。
- [x] 4.4 新增 `read_parent_chunks(keys)`：按 `(doc_id, parent_index)` 批量读取兄弟子块，沿用集合、知识库范围和排除文档条件。验证：集成测试覆盖范围外的知识库读不到、未发布文档读不到、多个父块一次读出。

## 5. 检索合并与上下文扩展

- [x] 5.1 重排时把 `index_text` 作为发给重排服务的文本。验证：用假重排器断言收到的是子块加前缀，不是父块全文。
- [x] 5.2 `RetrievalService` 新增 `_expand_to_parents`：内部候选数放大到 `min(top_k * 3, top_k_rerank)`，按父块合并，截取前 `top_k` 个，拼出 `context`；版本不一致、兄弟块不连续或旧块时回退为 `context=None`。验证：单元测试覆盖规格中的“两个子块命中同一父块”“候选不足 top_k 个父块”“读取期间文档被替换”“旧文本块”。
- [x] 5.3 `EvidenceItem` 新增 `context`；快速通道（`fast_lane.py`）和 Agent 通道（`langgraph_agent.py` 中由检索结果构造 evidence 的位置）都复制 `result.context`。验证：两条通道的测试都断言证据带有上下文，且 LangGraph 检查点序列化后能还原。
- [x] 5.4 证据判断 payload、`PromptBuilder.build_numbered` 和 Agent 生成提示词统一使用 `item.context or <现有格式化>`。验证：提示词快照测试，覆盖有上下文和没有上下文两种情况。
- [x] 5.5 新增 `fit_evidence_to_budget`，证据判断和生成都先经过它，按编号从大到小整条移除证据，至少保留 1 条。验证：测试覆盖不超限时原样保留、超限时移除最低排名、只剩 1 条仍超限时抛出 `PromptContextTooLargeError`。
- [x] 5.6 确认引用仍指向命中的子块：`validate_grounded_citations` 产出的 `chunk_id` 和 `chunk_text` 是代表子块的。验证：端到端问答测试断言引用的 `chunk_text` 是子块原文，并且前端引用侧栏测试（`frontend/src/components/SourcePanel` 相关测试）不需要修改就能通过。

## 6. 重新入库命令

- [x] 6.1 新增 `app/interfaces/cli/reindex.py`，支持 `--kb`、`--dry-run`、`--all`、`--limit`；按 design D6 原地重建（校验源文件快照 → 内存中解析、分块、向量化 → 业务写锁下确认版本后整体替换），源文件缺失时跳过并汇总，FAQ 文档同样从源文件重建。验证：测试覆盖试运行不调用模型也不写入、源文件缺失被跳过且旧块不变、向量化失败旧块不变、重建期间文档被删除时不复活、默认只处理旧格式文档、`--limit` 分批。
- [x] 6.2 README 增加“升级到父子分块”说明（先试运行再按库执行，会产生 Embedding 费用）。验证：按文档里的命令在本地测试库执行一遍，输出与文档描述一致。

## 7. 集成验证

- [x] 7.1 运行后端全部测试（`pytest`）和前端测试（`npm test`），全部通过；架构测试确认没有新增跨层依赖。
- [x] 7.2 用 `evaluation/run_eval.py` 在固定语料上分别跑改动前后的版本，对比检索命中率、回答质量和平均 Token 费用，把结果记录在本变更目录的 `evaluation.md` 中；命中率或质量下降时，回到 design 的风险项调整参数。（范围调整，2026-11-19 经用户决定：检索命中率做了前后对比；回答质量只跑了改动后的完整评估，不再花钱跑改动前的基线；Token 费用按上下文字数估算，未实测。语料缩为 10 篇 / 29 题。）
- [x] 7.3 回滚演练：用改动前的代码读取新格式入库的测试库，确认浏览、检索、引用都正常（对应 design 迁移计划第 3 步）。

## Workflow follow-up

- 评审通过并全部实现后，执行 `/openspec-archive-change` 归档，把两份规格合并进 `openspec/specs/`。

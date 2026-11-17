# Proposal

## Why

现在的分块器（`app/infrastructure/chunking/recursive.py`）只按字符数把文档切成约 500 字的块，块里不带标题和章节信息。生成模型也只能看到命中的那一小块。结果有两个：一是“第三条”“注意事项”这类脱离上下文的小块很难被检索到；二是即使检索到了，模型也不知道这段话属于哪一章、前后讲了什么，回答容易缺信息或误解。代码审查发现，这是当前问答效果最主要的短板，主流 RAG 项目（例如 RAGFlow）普遍用“标题感知 + 父子块”来解决。

## What Changes

- 解析器为每个片段提供**标题路径**（heading path）：
  - Markdown：识别 ATX 标题（`#` 到 `######`）。
  - Word（`.docx`）：识别标题样式（Heading 1-9 / 标题 1-9）。
  - PowerPoint（`.pptx`）：幻灯片标题作为一级标题。
  - PDF / TXT：没有可靠的结构信息，标题路径只含文件名。
- 分块器改为**两级结构**：
  - **父块**：按章节边界和父块字数上限切出，作为生成时的上下文单位，默认最多 1500 字。
  - **子块**：在父块内部切出，作为检索单位，默认 500 字，子块之间不重叠。
  - 父块不单独存储，由同一父块下的连续子块拼接而成。
- 每个子块做向量化和建关键词索引时，文本前面加上“文件名 › 标题路径”前缀。块正文和引用高亮仍使用原文，不带前缀。
- 检索仍然以子块为单位召回和重排，之后按父块合并：多个子块命中同一父块时，只占一个证据位。交给证据判断和生成模型的上下文改为“标题路径 + 父块全文”。引用编号仍然指向命中的子块，前端高亮位置不变。
- 新增配置项 `PARENT_CHUNK_SIZE`，默认 1500；`CHUNK_SIZE` 的含义变为子块字数上限。
- 新增命令行工具，用已保存的源文件把现有文档按新规则原地重建（见 design D6），新文本块写入前旧版本照常可查。
- 兼容旧数据：没有父块和标题元数据的旧块，各自视为一个只含自身的父块，混合语料可以正常检索，不需要停机迁移。

## Capabilities

### New Capabilities
- `document-chunking`：文档解析出的结构信息（标题路径）以及父子两级分块规则，包括块元数据、索引文本前缀和旧数据兼容。
- `retrieval-evidence-context`：检索命中后如何按父块合并、扩展成交给证据判断和生成模型的上下文，以及引用如何对应回子块。

### Modified Capabilities
<!-- openspec/specs/ 下目前没有任何已有规格，本次没有需要修改的能力。 -->

## Impact

- **代码**：
  - `app/domain/documents.py`：`DocumentSegment` 增加标题路径；`DocumentChunk` 增加父块序号和标题路径。
  - `app/infrastructure/parsing/`：`markdown.py`、`office.py`、`pdf.py`、`text.py` 按标题拆分片段并附带标题路径。
  - `app/infrastructure/chunking/recursive.py`：父子两级切分。
  - `app/application/ingestion/job_pipeline.py`：向量化时使用带前缀的索引文本。
  - `app/infrastructure/vectorstores/postgres.py`：写入关键词索引时使用带前缀的索引文本，读取父块兄弟块。
  - `app/application/retrieval/service.py`：召回后按父块合并和扩展。
  - `app/domain/retrieval.py`、`app/domain/agent_quality.py`：结果和证据增加上下文字段。
  - `app/application/query/prompting.py`、`app/infrastructure/conversations/`：快速通道和 Agent 通道的提示词改用父块上下文。
  - `app/config.py`：新增 `PARENT_CHUNK_SIZE`。
  - `app/interfaces/cli/`：新增重新入库命令。
- **数据**：`vector_chunks.metadata` 增加可选字段，不改表结构，不需要 SQL 迁移。新文档的子块数量和现在差不多，重新入库会产生一次 Embedding 费用。
- **API**：对外接口不变；引用里的 `chunk_text` 仍是子块原文。
- **成本**：每次生成的上下文变长，最多约为现在的 3 倍，Token 费用相应上升；`top_k` 默认值保持 5。
- **评估**：需要用 `evaluation/run_eval.py` 对比改动前后的检索命中率和回答质量。

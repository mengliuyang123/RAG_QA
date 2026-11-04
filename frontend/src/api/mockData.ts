// 演示用的固定知识库与文档。只在 mock 模式和 live 模式下后端尚未
// 实现的接口中使用，所有内容均为虚构示例。

export interface MockChunk {
  text: string;
  page: number | null;
}

export interface MockDocument {
  docId: string;
  kbId: string;
  filename: string;
  updatedAt: string;
  fileStatus: "available" | "missing";
  chunks: MockChunk[];
}

export const MOCK_KNOWLEDGE_BASES = [
  {
    id: "dev",
    name: "研发文档",
    description: "框架用法、代码规范、API 说明、LangGraph 持久化与检索实现。",
    icon: "code",
  },
  {
    id: "ops",
    name: "运维手册",
    description: "部署流程、服务启停、监控告警、日志位置、常见故障排查与恢复步骤。",
    icon: "server",
  },
  {
    id: "product",
    name: "产品资料",
    description: "功能说明、版本差异、定价方案与更新记录。",
    icon: "package",
  },
];

export const MOCK_DOCUMENTS: MockDocument[] = [
  {
    docId: "d1f0c2a4-0000-4000-8000-000000000001",
    kbId: "dev",
    filename: "langgraph_persistence.md",
    updatedAt: "2026-10-18T09:00:00Z",
    fileStatus: "available",
    chunks: [
      { text: "Checkpointer 负责在图的每个超步保存状态，使对话可以中断后恢复，也支持时间回溯调试。", page: null },
      { text: "SQLite checkpointer 适合本地开发和单进程部署。建议开启 WAL 模式，并设置 busy_timeout，以减少并发写入时的数据库锁定错误。", page: null },
      { text: "生产环境推荐使用 PostgresSaver，它支持多个进程并发读写同一个 thread，需要安装 langgraph-checkpoint-postgres。", page: null },
      { text: "删除会话时应同时调用 delete_thread 清理对应的全部 checkpoint，避免数据库持续增长。", page: null },
    ],
  },
  {
    docId: "d1f0c2a4-0000-4000-8000-000000000002",
    kbId: "dev",
    filename: "retrieval-strategies.md",
    updatedAt: "2026-10-12T02:00:00Z",
    fileStatus: "available",
    chunks: [
      { text: "hybrid 策略同时执行向量检索和 BM25 检索，再用 RRF 融合两路排名，兼顾语义相似和精确术语匹配。", page: null },
      { text: "hybrid_rerank 在 hybrid 的基础上，把前 20 个候选交给重排模型精排，准确率更高，但会增加一次远程调用的延迟。", page: null },
      { text: "在本项目的评估集上，hybrid_rerank 的 Hit Rate 高于 hybrid，平均延迟约多出 0.6 秒。", page: null },
    ],
  },
  {
    docId: "d1f0c2a4-0000-4000-8000-000000000003",
    kbId: "ops",
    filename: "deploy-guide.md",
    updatedAt: "2026-10-20T03:00:00Z",
    fileStatus: "available",
    chunks: [
      { text: "部署前确认 .env 中已配置 LLM、Embedding 与 Reranker 服务地址，并执行 uv sync 安装依赖。", page: null },
      { text: "使用 SQLite 存储会话时，生产环境只允许启动一个 worker；多个进程同时写同一个数据库文件会导致锁定错误。", page: null },
      { text: "服务启动后访问 /api/health 确认返回 status=ok，再开放给用户访问。", page: null },
    ],
  },
  {
    docId: "d1f0c2a4-0000-4000-8000-000000000004",
    kbId: "ops",
    filename: "monitoring.pdf",
    updatedAt: "2026-09-02T03:00:00Z",
    fileStatus: "missing",
    chunks: [
      { text: "应用日志默认输出到标准输出，由进程管理器统一收集；检索阶段耗时会记录在 retrieval_ms 字段中。", page: 2 },
    ],
  },
  {
    docId: "d1f0c2a4-0000-4000-8000-000000000005",
    kbId: "product",
    filename: "release-notes.md",
    updatedAt: "2026-10-21T08:00:00Z",
    fileStatus: "available",
    chunks: [
      { text: "0.2 版本新增多轮会话，支持基于最近历史改写查询，并在回答中标注可点击的引用来源。", page: null },
      { text: "0.2 版本新增文档管理页面，支持查看入库状态和删除文档。", page: null },
    ],
  },
];

export const MOCK_SUGGESTED_QUESTIONS = [
  { id: "sq1", kbId: "dev", text: "checkpoint 怎么配置" },
  { id: "sq2", kbId: "dev", text: "hybrid 和 hybrid_rerank 有什么区别" },
  { id: "sq3", kbId: "ops", text: "服务启动报数据库锁定怎么办" },
  { id: "sq4", kbId: "ops", text: "日志在哪里看" },
  { id: "sq5", kbId: "product", text: "最新版本更新了什么" },
  { id: "sq6", kbId: "product", text: "Redis 缓存怎么配置" },
];

export const MOCK_INSIGHTS = [
  { kbId: "dev", questions: 612, refusalRate: 0.06, thumbsDownRate: 0.03, expandedRate: 0.08 },
  { kbId: "ops", questions: 388, refusalRate: 0.17, thumbsDownRate: 0.05, expandedRate: 0.22 },
  { kbId: "product", questions: 240, refusalRate: 0.04, thumbsDownRate: 0.02, expandedRate: 0.05 },
];

export const MOCK_REFUSALS = [
  { id: "r1", question: "Redis 缓存怎么配置", kbIds: ["dev"], missing: ["Redis 作为缓存的配置方法"], askedAt: "2026-10-23T06:10:00Z" },
  { id: "r2", question: "灰度发布的流程是什么", kbIds: ["ops"], missing: ["灰度发布步骤与回滚条件"], askedAt: "2026-10-22T10:30:00Z" },
  { id: "r3", question: "企业版支持私有化部署吗", kbIds: ["product"], missing: ["企业版部署方式说明"], askedAt: "2026-10-21T02:15:00Z" },
];

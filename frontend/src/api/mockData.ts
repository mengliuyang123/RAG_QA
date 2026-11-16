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
  /** 上传的原文，用于模拟后端的内容查重。 */
  content?: string;
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

// 管理员账号管理的演示账号；u_demo 是 mock 模式下的当前管理员。
export const MOCK_USERS = [
  { id: "u_demo", username: "xiaoliu", name: "小刘", role: "admin" as const, disabled: false, createdAt: "2026-09-01T02:00:00Z" },
  { id: "u_ops", username: "ops.wang", name: "王工", role: "member" as const, disabled: false, createdAt: "2026-09-12T03:30:00Z" },
  { id: "u_old", username: "intern", name: "实习生", role: "member" as const, disabled: true, createdAt: "2026-10-02T08:00:00Z" },
];

// 知识缺口候选：结构与拒答聚类、点踩反馈接口转换后一致。
export const MOCK_GAP_CANDIDATES = [
  {
    key: "cluster:cl_redis0000000000000000000",
    source: "refusal_cluster" as const,
    title: "Redis 缓存怎么配置",
    questions: ["Redis 缓存怎么配置", "Redis 的过期时间在哪里设置", "缓存用 Redis 还是本地内存"],
    kbId: "dev",
    count: 6,
    lastAskedAt: "2026-10-23T06:10:00Z",
  },
  {
    key: "cluster:cl_canary000000000000000000",
    source: "refusal_cluster" as const,
    title: "灰度发布的流程是什么",
    questions: ["灰度发布的流程是什么", "灰度失败怎么回滚"],
    kbId: "ops",
    count: 3,
    lastAskedAt: "2026-10-22T10:30:00Z",
  },
  {
    key: "feedback:7d7c3f1e-1111-4222-8333-944455556666",
    source: "feedback" as const,
    title: "企业版支持私有化部署吗",
    questions: ["企业版支持私有化部署吗"],
    kbId: "product",
    count: 1,
    lastAskedAt: "2026-10-21T02:15:00Z",
  },
];

// 点踩反馈示例（问答内容为虚构）。
export const MOCK_FEEDBACK = [
  {
    id: "fb1",
    turnId: "t_fb1",
    question: "生产环境数据库连接池上限是多少",
    answer: "根据运维手册，连接池上限为 50。",
    citations: [{
      position: 1, kbId: "ops", docId: "d1f0c2a4-0000-4000-8000-000000000003", chunkId: "d1f0c2a4-0000-4000-8000-000000000003:0", filename: "deploy-guide.md",
      page: null, chunkText: "连接池上限 50。", docUpdatedAt: "2026-06-01T00:00:00Z",
    }],
    reasons: ["outdated" as const],
    comment: "已经调整为 100 了",
    revision: 1,
    triage: null,
    createdAt: "2026-11-10T03:00:00Z",
  },
  {
    id: "fb2",
    turnId: "t_fb2",
    question: "专业版和企业版的区别",
    answer: "资料中没有找到两者的完整对比。",
    citations: [],
    reasons: ["incomplete" as const],
    comment: null,
    revision: 1,
    triage: "missing_content" as const,
    createdAt: "2026-11-09T08:30:00Z",
  },
];

// 用量示例：费用为按配置费率的估算。
const usageRow = (key: string, label: string, questions: number, calls: number, input: number, output: number, amount: number) => ({
  key, label, questions, modelCalls: calls, inputTokens: input, outputTokens: output,
  costs: [{ currency: "CNY", amount }], costStatus: "estimated" as const,
});

export const MOCK_USAGE = {
  day: [
    usageRow("2026-11-15", "2026-11-15", 42, 160, 210_000, 31_000, 3.12),
    usageRow("2026-11-16", "2026-11-16", 57, 221, 289_000, 40_500, 4.36),
    usageRow("2026-11-17", "2026-11-17", 18, 70, 92_000, 13_000, 1.37),
  ],
  user: [
    usageRow("u1", "用户 A", 61, 240, 320_000, 45_000, 4.81),
    usageRow("u2", "用户 B", 56, 211, 271_000, 39_500, 4.04),
  ],
  kb: [
    usageRow("dev", "dev", 70, 270, 352_000, 50_000, 5.31),
    usageRow("ops", "ops", 47, 181, 239_000, 34_500, 3.54),
  ],
};

export const MOCK_TRACES = [
  { id: "tr1", question: "LangGraph 检查点怎么持久化", createdAt: "2026-11-17T02:10:00Z", outcome: "answered" as const, lane: "agent" as const, latencyMs: 8200, modelCalls: 7, inputTokens: 18_400, outputTokens: 1_900, costs: [{ currency: "CNY", amount: 0.28 }] },
  { id: "tr2", question: "服务启动报数据库锁定", createdAt: "2026-11-17T01:55:00Z", outcome: "answered" as const, lane: "fast" as const, latencyMs: 2100, modelCalls: 3, inputTokens: 5_200, outputTokens: 600, costs: [{ currency: "CNY", amount: 0.08 }] },
  { id: "tr3", question: "Redis 缓存怎么配置", createdAt: "2026-11-17T01:20:00Z", outcome: "refused" as const, lane: "agent" as const, latencyMs: 6400, modelCalls: 6, inputTokens: 14_100, outputTokens: 900, costs: [{ currency: "CNY", amount: 0.2 }] },
];

# RAG 评估报告

- Run ID: `20261118T090647Z-39acb214`
- Git commit: `39acb214c5a4d73c3e0451ce8d29b5a0f0651679`
- Dirty worktree: `true`
- 本地记录: `87`
- RGB 记录: `150`

## 模型

- Generator: `qwen3.8-flash`
- Embedding: `qwen3-embedding:4b`
- Reranker: `bge-reranker-v2-m3`
- Independent judge: `gemma4:12b`

## 本地指标

### hybrid

- answer_correctness: `0.510329`
- answer_relevancy: `0.719617`
- citation_precision: `0.740230`
- citation_recall: `0.718391`
- faithfulness: `0.807675`
- hit_rate: `0.965517`
- judge_semantic_correctness: `0.793103`
- latency_mean_ms: `3700.481414`
- latency_p50_ms: `3692.216000`
- latency_p95_ms: `7440.407400`
- mrr: `0.942529`

### hybrid_rerank

- answer_correctness: `0.580524`
- answer_relevancy: `0.876442`
- citation_precision: `0.910920`
- citation_recall: `0.936782`
- faithfulness: `0.970058`
- hit_rate: `1.000000`
- judge_semantic_correctness: `0.960345`
- latency_mean_ms: `4267.959655`
- latency_p50_ms: `3974.345000`
- latency_p95_ms: `6717.646000`
- mrr: `0.982759`

### vector

- answer_correctness: `0.528667`
- answer_relevancy: `0.689617`
- citation_precision: `0.747126`
- citation_recall: `0.758621`
- faithfulness: `0.748851`
- hit_rate: `1.000000`
- judge_semantic_correctness: `0.758621`
- latency_mean_ms: `3433.370207`
- latency_p50_ms: `3190.870000`
- latency_p95_ms: `7257.881000`
- mrr: `1.000000`

## RGB 中文子集生成评估

许可证：CC BY-NC-SA 4.0；仅限非商业用途。

### multi_document_integration

- failed_count: `0.000000`
- judge_semantic_correctness: `0.854444`
- latency_mean_ms: `4436.813423`
- latency_p50_ms: `2384.152450`
- latency_p95_ms: `13470.124070`
- official_accuracy: `0.933333`

### noise_robustness

- failed_count: `2.000000`
- judge_semantic_correctness: `0.950000`
- latency_mean_ms: `2445.467736`
- latency_p50_ms: `2011.526050`
- latency_p95_ms: `5431.377135`
- official_accuracy: `0.988636`

### noise_robustness:noise_0

- failed_count: `0.000000`
- judge_semantic_correctness: `0.950000`
- latency_mean_ms: `2286.410113`
- latency_p50_ms: `2042.481300`
- latency_p95_ms: `4228.815350`
- official_accuracy: `0.966667`

### noise_robustness:noise_0.4

- failed_count: `1.000000`
- judge_semantic_correctness: `0.958621`
- latency_mean_ms: `2473.509748`
- latency_p50_ms: `1593.467500`
- latency_p95_ms: `5730.323800`
- official_accuracy: `1.000000`

### noise_robustness:noise_0.8

- failed_count: `1.000000`
- judge_semantic_correctness: `0.941379`
- latency_mean_ms: `2581.968093`
- latency_p50_ms: `2104.902000`
- latency_p95_ms: `6001.045580`
- official_accuracy: `1.000000`

### rejection

- failed_count: `1.000000`
- judge_semantic_correctness: `0.551724`
- latency_mean_ms: `4546.642638`
- latency_p50_ms: `3180.778100`
- latency_p95_ms: `11997.643020`
- official_accuracy: `0.517241`

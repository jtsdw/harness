# 目标二实现设计：三层 profiling 成本归因

这是 `inspect_ai_roadmap.md` 里标注为"需要单独立项讨论"的设计文档。写作时间点：跑完目标一四条需求的真实验证之后，趁着本地 vLLM 环境已经跑通、`inspect_trace` 基础设施已经稳定，直接推进。

## 这次新确认的关键事实（比 roadmap 写作时更精确）

`inspect_ai_roadmap.md` 当时的判断是"model invocation 层的服务端指标拿不到，除非接入 vLLM `/metrics`，且关联精度存在不确定性"。这次做了真实 spike（起本地 vLLM，打一个真实请求，同时轮询 `/metrics`），结果比预期好：

1. **我们锁定的 vLLM 版本（`0.6.3.post1`）比 roadmap 写作时假设的更丰富**——`/metrics` 里其实有 `vllm:time_to_first_token_seconds`、`vllm:time_per_output_token_seconds`、`vllm:e2e_request_latency_seconds`、`vllm:request_prompt_tokens`、`vllm:request_generation_tokens` 这些 histogram，不是只有 gauge。真实校准请求（44 输入 token，300 输出 token）测出 TTFT=55.6ms，平均 ITL=32.2ms/token——跟同一份数据早前用"帐面 working_time + 真实 token 数"做线性回归估出的 decode 吞吐（~29 token/s ≈ 34.5ms/token）几乎吻合，交叉验证通过。
2. **关联方法本身是可行的**——用 `MAX_CONNECTIONS=1`（我们已有的真实 benchmark 惯例，为了避开本地 vLLM 并发崩溃）跑，每次模型调用之间不会有其他请求插进来，所以在调用前后各拉一次 `/metrics`，对 histogram 的 `_sum`/`_count` 取差值，只要 `_count` 差值恰好是 1，这次差值就精确对应"这一次调用"——不需要额外的时间窗口启发式匹配。
3. **仍然拿不到的**：`queueing time`（单独拆分的排队耗时）、`prefill time`/`decode time` 的单独拆分（现在只有合并的 TTFT 和 ITL，没有"排队"和"prefill"分开的字段——这两个是更新版本 vLLM 才加的）、`batch size`（每次调用时 GPU 上同批次请求数——本地这套单请求串行跑，这个值恒为 1，没有观测意义）、`peak GPU memory`（`/metrics` 只有 KV cache 使用率的百分比，不是显存的绝对/峰值字节数）。

## 三层设计

### Token 层（新模块：`inspect_trace/analysis/token_layer.py`，纯离线分析）

不需要新的实时采集——`efficient-harness.md` 要的 `input/output/reasoning/tool-call/observation/repeated-context token` 全部已经由 `token_attribution`/`prefill_diff`/`segment_tokens` 三个既有记录覆盖，这一层的工作是**把已经写盘的 JSONL 聚合成一份逐 episode 的汇总表**：

- 每个 episode：`total_input_tokens`/`total_output_tokens`（对 `token_attribution` 逐条求和）、`repeated_context_tokens`（`prefill_diff.reused_tokens_estimate` 之和，含消息和工具 schema 两部分分开列）、`reasoning_tokens`/`tool_call_tokens`/`final_response_tokens`（`token_attribution` 三个输出类别之和）。
- `discarded/retry/rollback token`：`retry` token 浪费 = 每个 `attempt_group` 里 `outcome=="error"` 的失败尝试，如果它确实产生过 `model_event_uuid`（有些失败尝试在真正拿到响应前就被打断，没有 `model_event_uuid`），去查对应 `segment_tokens.estimated_output_tokens_total` 累加，代表"这次白白生成、后来被丢弃的 token"。`discarded`（因为压缩/裁剪从 context 中被移除的历史消息）和 `rollback`（因为语义回滚被撤销的内容）——目标一验证阶段已经确认 inspect_ai 没有真正的回滚机制，这两类目前都如实记为"未观测到"，不编造字段，等真的观测到 compaction/rollback 场景再补。

### Episode 层（新模块：`inspect_trace/analysis/episode_layer.py`，纯离线分析）

同样不需要新采集，全部基于目标一已有的 `execution_topology`（时间线数据）、`attempt_group`（重试）、`token_attribution`（token 归因）、真实 `.eval` 日志的 `EvalSample.scores`（成功率）：

- `end-to-end latency` = episode 时间线首尾时间差（`execution_topology` 时间线数据已有）。
- `critical-path latency` vs `total model compute time`/`total tool execution time`：**这是这一层唯一需要真正写分析代码的地方**。目标一的执行时间线本质就是关键路径本身（因为我们从未观测到真并行——见 `goal1_r3_r4_real_benchmark_findings.md`，857 次真实 tool call、98 次多重调用机会，0 次真并行），所以在当前真实数据下 `critical-path latency == end-to-end latency`，`exclusive time` 和 `inclusive time` 在没有嵌套/并行 span 的情况下也重合。分析代码要按标准定义实现（区分三种时间口径，支持未来出现并行 span 时的正确关键路径计算），但如实注明：**在目前的真实数据上，三者数值相同，这是被测系统没有并行的必然结果，不是分析代码算错**。
- `model waiting for tool` / `tool waiting for model`：`execution_topology` 已经算过，这一层直接复用，不重复实现。
- `number of LLM calls`/`number of tool calls`/`retries`：直接数 `token_attribution`/`action_parsing`/`attempt_group` 记录条数。
- `success rate`：真实 `bfcl_scorer` 分数（已经在目标一验证里用了 200+15 条真实数据算过：`multi_turn_base` 4.5%、`live_parallel` 66.7%）。
- `cost per successful episode`：需要引入定价表（`inspect_trace/analysis/pricing.py`，模型名到单价的映射），乘以每个 episode 的真实计费 token 数，除以成功的 episode 数（本地 vLLM 场景下单价是 0，因为不计费；这一项主要是为 hosted 模型场景准备的，本地场景下如实报告"边际成本 0，只有摊销的硬件/电费成本，不在这次范围内估算"）。

### Model invocation 层（新模块：`inspect_trace/vllm_metrics.py`，实时 Hook 采集）

这是唯一需要新增实时采集逻辑的一层，设计上跟 `attempt_groups.py` 同构（复用同一个已验证的"用 anyio 任务身份做稳定关联"手法，而不是 `BeforeModelGenerate.sample_id`，原因见 `goal1_real_benchmark_findings.md` 里那个 `active.id` 不稳定的 bug）：

- `on_before_model_generate`：异步拉一次 `http://localhost:8000/metrics`（地址可配置，环境变量 `INSPECT_TRACE_VLLM_METRICS_URL`，默认这个值；如果拉取失败——比如根本没在用 vLLM——直接跳过，不报错、不阻塞正常流程），解析出这几个 histogram 的 `_sum`/`_count`（TTFT、ITL、e2e latency、prompt tokens、generation tokens）和当前 gauge 值（`num_requests_running`/`num_requests_waiting`/`gpu_cache_usage_perc`）以及累计 `num_preemptions_total`，存一份"调用前快照"，用任务身份做 key。
- `on_sample_event` 的 `ModelEvent` 分支：再拉一次 `/metrics`，跟"调用前快照"算差值：
  - 如果 TTFT/ITL 的 `_count` 差值恰好是 1，这次差值的 `_sum` 就是这次调用真实的 TTFT/ITL——精确写入。
  - 如果差值不是 1（比如没有用 `MAX_CONNECTIONS=1`，导致并发混进来），标记 `attribution_confidence: "ambiguous"`，把差值原样记下但注明不能确定完全对应这一次调用，不静默假装精确。
  - `num_requests_running`/`num_requests_waiting` 用"调用前快照"的值（代表这次调用发起时观测到的排队深度）。
  - `gpu_cache_usage_perc` 用"调用后快照"的值（代表这次调用结束时的 KV cache 占用）。
  - `preemptions_delta` = 调用前后 `num_preemptions_total` 的差值（这次调用期间是否被抢占过）。
- 新增 schema：`VLLMMetricsRecord`，字段：`ttft_seconds`/`itl_seconds_avg`/`e2e_latency_seconds`/`attribution_confidence`（`"exact"` 或 `"ambiguous"`）/`queue_depth_running`/`queue_depth_waiting`/`gpu_cache_usage_perc`/`preemptions_delta`/`vllm_prompt_tokens`/`vllm_generation_tokens`（这两个是 vLLM 自己统计的，可以和我们自己的 `billed_input_tokens`/`billed_output_tokens` 交叉核对）。
- **这个模块只在目标模型确实是本地 vLLM 时才有意义**——如果跑的是 DeepSeek 这类 hosted 模型，`/metrics` 请求会直接连接失败，模块要能优雅地识别这种情况并跳过（不产出 `VLLMMetricsRecord`，不是产出一堆 null 字段），文档里明确写清楚这一层**目前只覆盖本地 vLLM 路径**，hosted 路径这一层永远不可得（跟目标一"原始调用语句"那个发现是同一类结构性限制）。

## 不做的部分（如实标注，不假装完成）

- `tokenization/template time` 单独拆分：`/metrics` 没有这个字段，vLLM 内部把这部分并入了 prefill，暂不单独拆分。
- `prefill time`/`decode time` 单独拆分：现有字段只有 TTFT（近似 prefill+首 token）和 ITL（近似 decode 单 token耗时），没有更细的拆分，暂用这两个做代理，如实标注"不是严格意义上的 prefill/decode 时间，是它们的近似代理"。
- `batch size`/`peak GPU memory`（绝对值）：`/metrics` 没有对应字段，本地单请求串行场景下 `batch size` 恒为 1 也没有观测意义，暂不做。
- `queueing time`（数值化的排队耗时，而不是排队深度）：`/metrics` 没有直接暴露，`num_requests_running`/`num_requests_waiting` 只是深度指标不是耗时指标，暂用深度指标代理。

## 实现顺序

1. Token 层 + Episode 层（纯离线分析，不需要新的实时采集，风险最低、价值高）。
2. `vllm_metrics.py` 实时采集模块 + 接入 `hooks.py`。
3. 用真实 benchmark（复用已有的 `runs/goal1_bfcl_multi_turn_base_full`/需要新跑一次以获得 `VLLMMetricsRecord` 数据，因为这是全新字段，旧的 run 里没有）验证三层输出。
4. 写 findings 文档，更新 `inspect_ai_roadmap.md` 里目标二这一节的状态。

## 相关文档

- 可行性判断的原始版本：[`inspect_ai_roadmap.md`](./inspect_ai_roadmap.md)"目标二"一节
- 目标一的方法论延续（真实数据、如实报告限制）：[`goal1_r3_r4_real_benchmark_findings.md`](./goal1_r3_r4_real_benchmark_findings.md)

# 从 inspect_ai 到 efficient-harness 四个目标：现状、可行性与工作量分析

本文档回答一个问题：以 [inspect_ai](https://inspect.aisi.org.uk/) 为底座，实现 `efficient-harness.md` 定义的四个目标，各自还需要做什么、有多可行、量级多大。

结论写在最前面：

| 目标 | 状态 | 可行性 | 量级工作量（相对值，目标一=1） |
|---|---|---|---|
| 一：完整执行轨迹 | **已实现**（`inspect_trace` 包） | — | 1 |
| 二：三层 profiling 成本归因 | **已实现**（`inspect_trace/analysis/` + `vllm_metrics.py`），真实数据验证通过 | 三层全部确认可行，比最初预估更好 | 已完成 |
| 三：标准化加速干预接口 | 未开始 | 高（大部分干预点已存在，是封装问题不是造轮子问题） | 1.5 |
| 四：online/offline 对照实验 | 未开始 | 中，有可参考的数据模型但无可复用实现 | 2，且有调研不确定性 |
| 五：加速方法 insight/method 分析 | **刚启动**（2026-08-05 新增目标，已完成 SPORK 一篇） | 高，方法论已验证可行（详见分析过程），瓶颈是调研广度不是技术难度 | 持续性工作，非一次性交付，见 [`acceleration_methods_survey.md`](./acceleration_methods_survey.md) |

以下逐项展开。方法论：每一项先说 inspect_ai *现在* 提供什么（有代码证据的才算，不猜测），再说要补什么，最后给可行性判断和工作量估计。工作量用相对量级而非人天数——目标一已经做完，用它的实际工作量作为量级 1 的参照系。

本文的部分判断已根据 [`inspect_ai_docs_examples_audit.md`](./inspect_ai_docs_examples_audit.md)（对 inspect_ai 自带 `docs/`/`examples/` 的复用价值审计）做了更新/修正，具体改动点在对应小节里标注。

---

## 目标一：完整执行轨迹 — 已实现

结论：inspect_ai 自带的 event/transcript 体系覆盖了六个问题里的四个（上下文快照、invalid action 留痕、并行/等待/回滚、HTTP 重试留痕），真正缺的是四类衍生事实，已通过 `src/inspect_trace/`（一个基于官方 Hooks entry_points 机制的独立包，不改动 inspect_ai 源码）补齐：

| 问题 | inspect_ai 原生覆盖 | inspect_trace 新增 |
|---|---|---|
| Q1 上下文膨胀 | ✅ `ModelEvent.input` 每步完整快照 | — |
| Q2 重复 prefill | 仅调用级聚合 cache token 数 | ✅ 消息级 new/reused 分类，全历史比较 |
| Q3 reasoning/tool/response 分 token | 仅 `reasoning_tokens` 真实值，无 tool-call/text 拆分 | ✅ tiktoken 估算拆分，标注 `estimated_*`，不覆盖真实计费值 |
| Q4 哪类 observation 反复复用 | 无 | ✅ 按 `ChatMessageTool.function` 聚合复用统计 |
| Q5 retry/invalid action 成本 | ✅ invalid action 天然留痕；HTTP retry 每次尝试独立成事件 | ✅ 把同一逻辑请求的多次尝试串成 attempt group，统计浪费的 wait time |
| Q6 并行/等待/回滚 | ✅ 并行 tool call、`collect()` 并行 subagent、working_time 分离、`InterruptEvent` | — |

已通过端到端测试验证（`mockllm/model` 驱动的确定性轨迹 + 交叉核对 `.eval` 日志里的真实 `ModelEvent` 数量）。细节见 `src/inspect_ai/src/inspect_trace/README.md`。

---

## 目标二：三层 profiling 成本归因 — 已实现，真实数据验证通过

**2026-08-03 更新**：三层全部已实现并用真实数据验证，结论比下面原始判断更好——完整过程见 [`goal2_design.md`](./goal2_design.md)（设计，含真实 spike 结果）和 [`goal2_real_validation_findings.md`](./goal2_real_validation_findings.md)（真实数据验证结果）。以下原始判断保留作记录，关键修正点：

- Token 层 / episode 层：判断成立，已实现为 `inspect_trace/analysis/{token_layer,episode_layer}.py`，200 样本真实数据验证通过（跟 benchmark 自己报告的 token 数/accuracy 完全吻合）。
- Model invocation 层：**比原判断更可行**——真实 spike 发现我们锁定的 `vllm==0.6.3.post1` 其实已经暴露 `time_to_first_token_seconds`/`time_per_output_token_seconds`/`e2e_request_latency_seconds` 这几个 histogram（原判断以为完全没有/需要等新版本），且 `MAX_CONNECTIONS=1`（已有的真实 benchmark 惯例）恰好让"按时间窗口关联到具体 request"这件事变得精确可行（真实验证：49/49 次调用 100% 精确归因）。真正拿不到的是 `queueing time` 数值、`prefill`/`decode` 时间的严格拆分、`batch size`、`peak GPU memory` 绝对值——这部分原判断成立。

### Token 层 — 可行性高，量级小

`input_tokens`/`output_tokens`/`reasoning_tokens`/`input_tokens_cache_write`/`input_tokens_cache_read` 已由 `ModelUsage` 提供（真实计费值，非估算）；`inspect_trace` 已补上 segment 级估算和 dedup/repeated-prefill 计数。这一层基本是"整理现有数据 + 目标一产出"的聚合工作，不需要新的采集点。

### Episode 层 — 可行性高，量级小到中

`end-to-end latency`、`number of LLM calls`、`number of tool calls`、`retries`、`success rate` 均可从现有 `EvalLog`/`EvalStats`/事件流直接统计。`critical-path latency` 需要额外计算（把 span 树的时间区间做并行感知的关键路径分析，而不是简单相加），`model waiting for tool` / `tool waiting for model` 需要基于 `working_time` 和 span 时间戳做区间运算——这些是纯离线分析代码，不需要新的运行时插桩，可以直接消费目标一产出的数据。`cost per successful episode` 需要引入定价表（模型单价 × token 数），是配置问题不是技术问题。`stats.connection_limit_history`（并发限额变化历史，含变化原因：`slow_start`/`steady_state_up`/`rate_limit`/`manual`）已经在 `.eval` log 里现成记录，可直接作为排队延迟分析的协变量，不用额外埋点。

**新发现两项可直接复用的资源**（详见 [`inspect_ai_docs_examples_audit.md`](./inspect_ai_docs_examples_audit.md)），能进一步降低这一层和下一层的工作量：
- `inspect_ai.analysis` 的 `events_df()`——官方提供的 event 级 DataFrame，`ModelEventColumns` 直接暴露 `input_tokens_cache_read`/`working_time`/`retries`/原始 request-response，支持并行读取大批量 log。建议下游分析脚本直接基于它写，替代手写 JSONL parser。
- `on_model_usage`/`on_model_cache_usage` Hook（`ModelUsageData.call_duration`）——inspect_ai 框架层面算好的"排除 retry backoff 等待时间的纯调用耗时"，且天然区分本地缓存命中 vs 真实调用。`inspect_trace` 目前没用这两个 Hook，是比自己从 `on_sample_event` 过滤再手算 `working_time` 更干净的起点，也顺带排查出一个待核实的潜在问题：如果实验开了 inspect_ai 本地 `--cache`，现有耗时统计方式可能把缓存命中也计入真实调用耗时。

### Model invocation 层 — 可行性中低，是目标二里工作量的主要来源

这一层要的指标是 `queueing time / tokenization time / prefill time / decode time / TTFT / inter-token latency / peak GPU memory / KV cache size / batch size / cache hit-miss / tokens per second`。已核实的现状：

- **完全没有**：对 `src/inspect_ai/model/_providers/*.py` 做过 grep，没有任何 provider 记录 TTFT、decode time、KV cache size、batch size、queueing time。标准 OpenAI-compatible `/v1/chat/completions` 响应体本身也不包含这些字段——这不是 inspect_ai 没做，是这一层信息在协议层面就不存在，除非切到流式响应或直连 vLLM 的扩展接口。
- **有一半基础设施，没有采集逻辑**：inspect_ai 的 vLLM provider（`src/inspect_ai/model/_providers/vllm.py`）已经支持流式请求（`self.client.chat.completions.stream(**request)`），但目前只用流式来拿 `get_final_completion()`，逐 chunk 到达的时间戳被直接丢弃。也就是说，**客户端可测的 TTFT / inter-token latency，管道已经打通了一半**——把这段流式消费循环包一层时间戳记录，是一个具体、可评估工作量的任务，不需要 vLLM 服务端配合。
- **服务端内部指标（KV cache size / batch size / peak GPU memory / queueing time）拿不到，除非额外接入 vLLM 的 `/metrics` Prometheus 端点**。这是一个独立的側信道：需要另起一个采集进程定期 scrape vLLM `/metrics`，再按时间窗口/request 时间戳把这些聚合指标（Prometheus 计数器通常是全局的，不天然按 request 切分）和具体的某次 model call 关联起来。这是目标二里唯一一块"需要新起基础设施、且关联精度存在不确定性"的部分，建议作为独立子任务单独立项评估，不要和 token/episode 层混在一起估计。

工作量结论：token 层 + episode 层量级 1（跟目标一体量相当，主要是分析代码）；model invocation 层里客户端可测部分（TTFT/ITL）量级 1；vLLM 服务端指标側信道量级 1.5–2 且有较大不确定性（先用一个最小验证：跑一次 vLLM + `/metrics`，确认能否按时间窗口把 GPU 内存/KV cache 占用和某次具体请求对上,再决定要不要做）。三者相加，目标二总量级给 3–4。

---

## 目标三：标准化加速干预接口

可行性判断：**高，工作量相对最小**，因为 inspect_ai 本身已经把"标准化干预点"这件事解决了大半——不需要造拦截层，只需要设计一套公平比较的实验协议封装在这些干预点之上：

- **Context 构造干预** = inspect_ai 的 `Solver`。任何 context 压缩/裁剪策略都可以写成一个 `solver()`，天然可插拔、可组合、可在同一 `Task` 定义下切换对比。
- **Generation 干预** = `on_before_model_generate` hook（已在目标一里验证可用，`data.input`/`data.tools`/`data.config` 在这里是可读的；文档里提到它在 cache 查找之前触发，"hook mutations to inputs/tools/config are reflected in cache keys and in the actual API call"——即这个 hook 本来就设计成可以修改请求，不是只读观测点）+ 自定义 `ModelAPI` provider（同样走 entry_points 注册，跟 `inspect_trace` 用的是同一套机制）。
- **Runtime 干预**（比如换后端、换 batching 策略）= 自定义 `ModelAPI` provider 本身就是这层的标准接口，vLLM/OpenAI-compatible/自定义后端只是不同的 provider 实现。
- **Tool execution 干预** = `Tool` 包装/`ToolEnvironment` 的等价物已经是 inspect_ai 的一等公民。

需要新建的东西：一套"实验协议"封装——固定模型、任务、trace schema，只切换某一层的实现，跑多组对比，产出统一格式的对比报表（EAL 项目里 `eal compare` 的思路可以直接借鉴）。这是应用层封装，不是新的拦截机制。工作量给量级 1.5。

**待跟进线索**：`docs/compaction.qmd`（Automatic/Native/Summary/Edit/Trim Compaction、Token Counting）粗看目录跟"context 层干预接口"直接相关，inspect_ai 内置了多种 context compaction 策略和一个 `compact()` provider 扩展点——这次审计只扫了目录没细读，下次应优先补上，可能会降低这一层的实际工作量（如果现成的 compaction 策略已经覆盖了我们想对比的干预方法）。

---

## 目标四：online/offline 对照实验（含 replay）

可行性判断：**中，且有明确的调研缺口**。核实结果如下：

- Online execution：inspect_ai 本身就是在线执行的评测框架，这部分不需要额外工作。
- Offline replay：已确认存在 `BranchEvent`（`src/inspect_ai/event/_branch.py`）和 `AnchorEvent`（`src/inspect_ai/event/_anchor.py`），数据模型描述的正是"标记轨迹重放的分界点""标记一个可回滚的锚点"这类语义，看起来和目标四要的东西高度吻合。**判断已修正（详见 [`inspect_ai_docs_examples_audit.md`](./inspect_ai_docs_examples_audit.md)）**：之前认为这是"预留但没有任何地方发出的死代码"，实际上它们被 `src/inspect_ai/agent/_deepagent/agent_tool.py` 使用，服务于 **deepagent 子代理 fork 场景的 log viewer 时间线可视化标注**——但只存锚点 ID 用于画图，不存重放所需的完整状态（工具响应、随机种子等）。结论不变：跟目标四"固定 trajectory 只重放 model call 或 context construction"这种受控对照实验依然无关，只是"死代码"这个定性不准确，应改为"用途方向不同"。
- 同一轮审计也确认了 `docs/checkpointing.qmd`（sandbox checkpoint/resume 子系统，`src/inspect_ai/util/_checkpoint/`，含 restic 快照）**不能替代目标四**：它解决的是"崩溃后原样恢复继续跑同一次评测"（抗基础设施故障），不追求确定性重放，跟"给定固定轨迹/工具响应做受控对照实验"是方向完全不同的问题，不能直接复用，判断维持不变。

工作量结论：无法照抄现成实现，需要基于目标一已经产出的完整轨迹数据（`ModelEvent.input`/`output` 逐步快照）自己实现 replay runner——用记录下来的历史响应喂给一个 mock/replay provider，只重跑 context construction 或某个优化环节，其余部分保持轨迹不变。`BranchEvent`/`AnchorEvent` 的字段设计可以作为命名/概念上的参考，但不能当作复用的代码。给量级 2，且标注"有调研不确定性"——具体是否需要更贴近 `BranchEvent` 的语义（例如未来想跟 inspect_ai 官方的 checkpoint/resume 生态对齐）需要在真正立项前再确认一次官方是否有相关路线图。

---

## 目标一/二现状批判性复盘（迁移前自查，2026-08-04）

在决定迁移到新服务器之前，对照最初的目的——"快速理解 agent 执行流程、观察性能瓶颈"——重新审视了一遍目标一/二至今积累的全部真实数据（`goal1_r3_r4_real_benchmark_findings.md` + `goal2_real_validation_findings.md`），不是走形式确认"已实现"，而是问一个更具体的问题："现在到底能看到什么、看不到什么"。

### 目标一（流程重建）：已经达到目的，而且质量超出预期

统一时间线 + token 归因 + action parsing 追溯，三者叠加确实能让人几分钟内看清一条 episode 完整发生了什么。比预期更有价值的是两类顺带挖出来的发现：

- **具体、可执行的瓶颈**：`multi_turn_base` 200 样本上，工具 schema 重复消耗的 token（862 万）是对话历史重复量（23 万）的 **37 倍**——这是目前这套 harness 产出的最扎实的一条效率发现，直接指向"该优化工具 schema 的复用，而不是对话历史压缩"。
- **此前任何检测器都看不到的真实故障**：深挖 `ModelEvent.call.response` 原始文本才发现的两类 `<tool_call>` 标签解析故障——标签未闭合导致的**静默丢失**（模型意图凭空消失，无报错无 `ToolEvent`）、多标签互相干扰导致的**张冠李戴**（退化成 `function="unknown"`）。这两类此前完全不在任何指标里，只有手工对照原始 provider 响应才挖得出来。

也如实暴露了两处读数陷阱，不算 bug 但用这套工具的人必须知道：`emulate_tools=true` 路径下系统指令被塞进 user 消息，导致 `content_category` 的 system_template 分类失效；`final_response` token 归因是内容块类型驱动、不是语义驱动的，会把解析失败后残留的 `<tool_call>` 文本误计成"模型的最终自然语言回答"。详见 `goal1_r3_r4_real_benchmark_findings.md` 的"附加发现"两节。

### 目标二（成本归因）：三层基础设施对了，但"瓶颈"至今没有被真正观察到

原始需求文档里最想回答的问题——"tool execution、network I/O、GPU inference 之间的重叠，naive sum 会不会超过真实端到端时间"——在我们目前积累的全部真实数据上，答案永远是"不会，因为从来没有重叠过"：

- `concurrency_savings_seconds` 在全部 200 个 episode 上是 0（算法本身在专门的 mock 场景里验证过能正确算出正数，问题是被测对象从没触发过重叠）。
- `queue_depth_running`/`queue_depth_waiting`/`preemptions_delta` 全部是 0——`MAX_CONNECTIONS=1` 串行执行下不可能有排队，这些字段"在真排队场景下是否正确"至今**没有被验证过**，只验证了"无排队时正确显示 0"。
- `batch_size`、`peak GPU memory`（绝对值）、`queueing time`（数值化）：vLLM 这个版本的 `/metrics` 拿不到，单请求串行场景下 `batch size` 恒为 1 也没有观测意义。
- `gpu_cache_usage_perc_at_end` 采样时机不对，49 条记录全部是 0，已知限制未修。
- `cost_usd` 恒为 $0（本地模型不计费，定价表只有一条记录），跨模型成本比较目前没有任何真实数据支撑；`multi_turn_base` 只有 9/200 个成功样本，"cost per successful episode"这类指标在这个样本量下统计意义也很薄弱。

换句话说：目标二能回答"这次调用花了多少 token/时间/钱"，但还回答不了"GPU 层面的瓶颈（并发、排队、显存压力）到底在哪"。

### 一个共同的根因：至今所有真实数据都来自同一种结构性不会暴露瓶颈的测试条件

三条叠加：本地 3B 小模型（`multi_turn_base` 准确率只有 4.5%）、`MAX_CONNECTIONS=1`（为绕开旧 vLLM 并发崩溃刻意加的限制）、BFCL 注册的工具默认不开 `parallel=True`。这不是代码缺陷——是我们至今为止一直在用一种"保证跑得稳但也保证看不到并发/排队现象"的方式测试。

### 对迁移的直接影响

新服务器（H100 80GB）是第一次有条件打破这三条限制的机会——真并发大概率不再让 vLLM 崩溃，显存也足够跑更大的模型。这件事不应该只是"多人共用 GPU"这个协作需求的副产品，而应该单独当成一次**验证实验**来对待：迁移后除了常规复现确认环境正确，应该专门跑一次刻意制造负载的实验（部分工具标 `parallel=True` + `MAX_CONNECTIONS>1`），episode 层和 model invocation 层的并发/排队字段才会第一次产生非零数据，目标二才算真正被完整验证过。已经把这一条写进 [`deployment_migration_guide.md`](./deployment_migration_guide.md)。

## 优先级建议

目标二已完成（token/episode 层 + model invocation 层的 vLLM `/metrics` 侧信道，spike 验证通过后直接投入实现，见上）。目标三是下一个"确定可行、主要是工程整理工作"的部分，建议排在前面；目标四仍有明确的不确定性，建议先做一个小规模验证 spike（"replay runner 的最小可行版本能不能正确重放一条目标一已经记录下来的轨迹"），确认可行后再投入完整实现。

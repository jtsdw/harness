# inspect_ai 自带 docs/examples 复用价值审计

审计对象：`/home/liuyingen/code/inspect_ai/docs/` 和 `examples/`（上游 inspect_ai 的只读参考克隆，不是我们自己的项目——我们自己的代码/文档在 `/home/liuyingen/code/efficient-harness/`），目的是找出对我们四个目标（`efficient-harness.md`）有直接复用价值的官方文档/示例，避免重复造轮子，也避免误用跟我们目标撞名但实际无关的机制。

## 可直接采纳，建议后续落地

### `inspect_ai.analysis`（`events_df()`）—— 目标二最实用的一条

inspect_ai 自带 event 级 DataFrame 分析模块，四层粒度：`evals_df()`（每个 log 一行）→ `samples_df()`（每个 sample 一行）→ `messages_df()`（每条消息一行）→ `events_df()`（**每个 event 一行，我们能拿到的最细粒度**）。`ModelEventColumns` 列组直接暴露 `input_tokens_cache_read`、`working_time`、`retries`、`cache`、原始 request/response，支持 `parallel=True` 并行读取大量 log、`duckdb` 联表分析。

建议：目标二的下游分析脚本迁移到用 `events_df()` + 自定义 `filter`/`columns`，替代手写 JSONL parser + pandas 拼装。它不能解决"chunk 级 TTFT/ITL 缺失"这个已知缺口（因为 `ModelEvent` 本身就没存这些），但能大幅减少"把已有字段聚合成报表"这部分的工作量。

参考：`docs/dataframe.qmd`（"Events" 一节）、`src/inspect_ai/analysis/_dataframe/events/columns.py`。

### `on_model_usage`/`on_model_cache_usage` Hook —— 我们目前完全没用

官方 MLflow/W&B tracking 示例（`examples/hooks/mlflow_tracking.py`）用到的两个 Hook，`inspect_trace` 目前没用。关键字段 `ModelUsageData.call_duration`：**已经排除了 retry backoff 等待时间的纯调用耗时**，且本地缓存命中时没有这个字段（天然区分"真实调用"和"缓存命中"）。

对目标二的价值：比我们自己从 `on_sample_event` 过滤 `ModelEvent` 再手算 `working_time` 更干净，是"queueing/prefill/decode 时间拆分"任务的更好起点。

**顺带发现一个需要核查的潜在问题**：如果我们跑实验时开了 inspect_ai 的本地 `--cache`，`inspect_trace` 现在从 `on_sample_event` 过滤 `ModelEvent` 算耗时的方式，可能会把缓存命中的调用也当成真实调用计入统计——`on_model_cache_usage` 天然规避了这个问题。**待办**：检查我们目前的实验有没有开本地 cache（据实验记录应该没开，但要确认），以及是否需要把 `inspect_trace` 的耗时统计迁移到基于这两个 Hook。

参考：`src/inspect_ai/hooks/_hooks.py:227-261`（`ModelUsageData`/`ModelCacheUsageData` 定义）；`examples/hooks/mlflow_tracking.py`（用法示例）。

### `enabled()` 门控 —— 低成本改进

官方 Hooks 示例（`mlflow_tracing.py`/`mlflow_tracking.py`/`wandb_weave.py`）都实现了 `enabled()` 方法，用环境变量控制这个 Hook 是否整体激活。我们的 `TraceHooks`（`src/inspect_trace/src/inspect_trace/hooks.py`）没有实现，意味着只要包被安装就永远激活。**待办**：加一个 `INSPECT_TRACE_ENABLED`（默认 true）之类的环境变量门控，方便跑纯 baseline 对照时临时关掉记录开销。

### `on_before_model_generate` 支持原地改写请求 —— 目标三的现成起点

官方文档明确写了这个 Hook"可以修改 `data.input`/`data.tools`/`data.config`，会实际影响发出的 API 调用和缓存 key，每次 retry 都会触发一次"。这是我们已经在用的机制（`inspect_trace` 的 attempt-grouping 就靠它），但目标三设计"标准化 generation 层干预接口"时应该先评估能不能在这个 Hook 基础上扩展出"短路/替换返回结果"的能力，而不是从零参照 pydantic-ai 的 `AbstractCapability` 重新设计一套。细节见 `framework-selection.md` 更新。

参考：`docs/extensions-hooks.qmd` 第 79-95 行；`src/inspect_ai/hooks/_hooks.py:264-278`。

### `stats.connection_limit_history` —— 目标二 episode 层协变量，不需要额外埋点

`.eval` log 的 `EvalStats` 里已经记录了并发限额变化历史（timestamp、model、old/new limit、变化原因：`slow_start`/`steady_state_up`/`rate_limit`/`manual`）。做 GPU 显存/batch size 关联分析、或者排查"某个 episode 为什么慢"时，这是现成的协变量数据源，不用自己额外埋点。

参考：`docs/models-concurrency.qmd`（"Limit History" 一节）。

## 纠正的判断

### `BranchEvent`/`AnchorEvent` 不是死代码（但依然跟目标四无关）

之前（`inspect_ai_roadmap.md` 目标四小节）判断这两个事件"预留但没有任何地方真正构造/发出"——这个判断**不准确**，是之前 grep 没找全导致的。实际上它们被 `src/inspect_ai/agent/_deepagent/agent_tool.py` 使用，用于 deepagent **子代理 fork 场景的 log viewer 时间线可视化标注**（标记"这段是从主线哪个锚点分叉出来的子代理轨迹"）。

修正后的结论：这套 schema 确实在用，但只存锚点 ID 用于 UI 画图，不存重放所需的完整状态（工具响应、随机种子等），跟目标四"固定 trajectory 只重放 model call 或 context construction"这种受控对照实验诉求依然无关。**结论不变，只是"死代码"这个定性错了，改成"用途跟目标四方向不同"更准确。**

参考：`src/inspect_ai/event/_timeline.py:530`（`timeline_branch`）；`src/inspect_ai/agent/_deepagent/agent_tool.py`（调用点）。

## 解决了一个悬而未决的问题

### `examples/bridge/pydantic-ai/` —— pydantic-ai 和 inspect_ai 不是二选一

`framework-selection.md` 第一轮分析讨论"如果只能选一个起点，选 Pydantic AI"、第二轮分析讨论"继续用 inspect_ai"，隐含前提都是"两者互斥，选一个"。这次调研确认这个前提不完整：`agent_bridge()`（`docs/agent-bridge.qmd`）能让一个**原生 pydantic-ai agent**（model 参数写成特殊值 `"inspect"`）正常运行的同时，它发出的每一次模型调用都被路由进 inspect_ai 自己的 model provider，从而正常触发 `ModelEvent`、被我们的 Hooks 捕获、记进 `.eval` log——**目标一已经建好的整套观测基础设施不受影响**。

限制（不是万能药）：`agent_bridge()` 只解决"事后记录不丢失"，不会让 pydantic-ai 的 `AbstractCapability`（`wrap_model_request`/`wrap_tool_execute`）拦截点和 inspect_ai 自己的 Hooks 机制自动统一成一套——真要同时用两边的干预能力，仍需分别维护，只是不再是"选 pydantic-ai 就必须放弃 inspect_ai 的观测能力"这种非此即彼的困境。

参考：`examples/bridge/pydantic-ai/agent.py`；`docs/agent-bridge.qmd`（"Agent Bridge" 一节）；`src/inspect_ai/agent/_bridge/bridge.py`。

## 确认跟我们目标无关，不用再花时间看

| 文档/示例 | 实际是什么 | 为什么不相关 |
|---|---|---|
| `docs/tracing.qmd` + `docs/reference/inspect_trace.qmd`（`inspect trace` CLI） | 运行时诊断日志（HTTP/subprocess/docker 调用的健康状况追踪），用于排查"评测卡住/报错" | 面向"进程健康状况"，不是"agent 轨迹效率分析"；纯粹跟我们的 `inspect_trace` 包撞名 |
| `docs/checkpointing.qmd` + `examples/checkpoint_ctf.py` | 崩溃后原样恢复继续跑同一次评测（sandbox restic 快照 + host 端状态文件） | 目的是抗基础设施故障，不追求确定性重放；跟目标四"给定固定轨迹/工具响应做受控对照实验"是不同问题 |
| `docs/intervention.qmd` + `examples/intervention/` | 人在环运行时干预（ACP 协议，人类可以连接到跑着的 eval 打断/发消息） | 面向 human-in-the-loop，不是"程序化拦截/替换 model 调用或 tool 执行做效率对比" |
| `examples/prefill.py` | 预先写入一段 assistant 消息前缀引导模型输出（prompt engineering 技巧） | 跟"repeated prefill 计算成本"纯属术语撞车，语义完全不同 |
| `docs/models-batch.qmd` | 托管 provider（OpenAI/Anthropic/Google）的批处理 API | 跟我们用的本地 vLLM OpenAI-compatible endpoint 无关 |
| `docs/control-channel.qmd`（`inspect ctl`） | 长跑评测的远程运维/调参工具（暂停/恢复/动态改并发） | 不改变 agent 逻辑本身，是外部调度旋钮，跟目标三/四无关；但跑大规模 benchmark 实验时值得用（工程便利，非目标本身） |
| `docs/extensions-model-api.qmd` | 自定义 `ModelAPI` provider 的扩展指南 | 是"整体替换 provider 实现"式扩展点，不是 pydantic-ai `AbstractCapability` 那种"拦截+短路+替换"的装饰器式接口；`ModelCall.create(filter=...)` 的过滤回调模式可以借鉴到我们写 raw request/response 时做脱敏 |

## 待跟进线索（这次没细看，下次优先看）

**`docs/compaction.qmd`**——粗看目录（Automatic/Native/Summary/Edit/Trim Compaction、Token Counting），明显跟目标一"上下文膨胀"和目标三"context 层干预接口"直接相关。inspect_ai 内置了多种 context compaction 策略，还有一个 `compact()` provider 扩展点。这次只看了目录没细读，优先级不低于本次审计里已经细看的项目，下次单独调研。

**`examples/http_proxy/`**——如果以后目标四想做"录制真实 HTTP 请求/响应之后重放"，这套 mitmproxy + addon 的架构（`request()`/`response()` hook 函数）是一个可以直接抄骨架的通用模式。现阶段我们的 BFCL/GSM8K 实验走的是 inspect_ai 原生 solver（模型调用天然经过 `ModelAPI` 层，不需要这套代理），只有未来架构变成"跑不受控制的第三方 CLI agent"时才用得上，优先级不高，先记录在案。

## 相关文档

- [`framework-selection.md`](./framework-selection.md) —— pydantic-ai bridge 发现已同步进该文档
- [`inspect_ai_roadmap.md`](./inspect_ai_roadmap.md) —— 目标二/四的现状判断已根据本次审计更新

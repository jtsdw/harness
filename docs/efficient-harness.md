
我们对于一个面向效率研究的agent harness，有如下三个主要目标。涉及到观察，定量分析和改进方案。

# 目标一：重建完整执行轨迹
建立统一的 Agent execution trace，完整记录从 prompt 构造、模型生成、action parsing、tool execution 到 observation 回填的全过程，并支持 token-level 和 event-level inspection。

这是一份需求文档，下面四条是 harness 必须具备的记录能力，不是"我们想研究什么"（后者见文末"潜在科研价值举例"一节）。

## 需求一：完整的 token 级记录

对每一次模型调用，都要记录：

- 真实（计费）的 prefill/decode-等价 token 数——即实际送进 model 的输入 token 和模型产生的输出 token，不是估算值；
- 按 pipeline 阶段的 token 归因：system template（系统提示词）/ tool schema（调用 tool 时的 schema 定义）/ reasoning / tool call / final response。

这个归因需要覆盖两个层面：model 层面（原始的输入 vs 输出拆分）和 agent 层面（按 ReAct 框架定义，reasoning/tool-calling/final-response 各阶段涉及的 token）。现在的 log 和可视化在这个方面仍然不够直观。

## 需求二：每一步的新增 vs 复用上下文

每一步的 context，必须能判定哪些内容是这一步新增的，哪些是历史内容原样重发的——需要能快速定位到哪些上下文被重复使用，哪些又是新增的上下文内容，包括 system template 这类静态内容要能与普通新增内容区分开来单独统计。

## 需求三：执行拓扑

trace 必须能重建一次 episode 的执行结构：是线性执行，还是存在并行 tool call、等待（model waiting for tool / tool waiting for model）、重试。

## 需求四：action parsing 与 observation 回填追踪

模型 tool-call 的解析/校验失败要留痕，工具结果写回 context（observation 回填）的过程要能追溯，不能只在成功路径上有记录。

---

我们目标一的核心需求就是为了让我们能够快速了解agent的执行过程。
我们设计这个项目的核心是为了agent efficiency。这一点和普通的agent 框架不同，他们追求的是agent功能性的最大化，需要agent去完成实际问题，他们追求结果。
而我们是为了寻找agent过程中的效率瓶颈，追求的是agent过程的清晰和可解释，需要追求过程。



# 目标二：完成成本归因

构建 token、model invocation 和 episode 三层 profiling 体系，对 Agent 成本进行阶段级和关键路径级归因，并联合评估 latency、memory、token consumption、task success 和 monetary cost。


需要建立三层指标体系


### Token 层

包括：

- input token；
- newly appended token；
- repeated context token；
- output token；
- reasoning token；
- tool-call token；
- observation token；
- discarded、retry、rollback token。

### Model invocation 层

每次模型调用记录：

- queueing time；
- tokenization/template time；
- prefill time；
- decode time；
- time to first token；
- inter-token latency；
- peak GPU memory；
- KV cache size；
- batch size；
- cache hit/miss；
- generated tokens per second。


### Episode 层

整个任务记录：

- end-to-end latency；
- critical-path latency；
- total model compute time；
- total tool execution time；
- model waiting for tool；
- tool waiting for model；
- number of LLM calls；
- number of tool calls；
- retry次数；
- success rate；
- cost per successful episode。


此外，Agent 中存在 tool execution、network I/O、GPU inference 之间的重叠。简单地把各阶段 latency 相加，可能超过真实端到端时间。因此要区分：

- exclusive time：一个阶段自身占用的时间；
- inclusive time：包含子事件的总时间；
- critical-path time：真正决定 episode 完成时间的路径。






# 目标三：提供标准化加速接口



提供标准化的 context、generation、runtime 和 tool execution 干预接口，支持效率方法的可插拔实现，并在相同轨迹、模型、任务和质量约束下进行公平比较。


# 目标四：对照实验分析


Agent 任务具有较强随机性。不同方法如果运行在不同的模型输出和不同的 tool trajectory 上，很难判断性能差异到底来自优化方法，还是来自轨迹变化。

因此 harness 应当支持两种模式。

### Online execution

正常运行 Agent，产生新的 action、observation 和 trajectory，用于测量真实端到端效果。

### Offline replay

固定已有 trajectory 或固定 tool response，只重放模型调用、context construction 或优化环节，用于控制变量。

例如，你想比较 prefix caching 对 prefill latency 的影响，可以在完全相同的 prompt 序列上重放。你想比较 observation compression，则可以固定工具输出，观察压缩前后 action 和 success 的变化。

没有 replay，很多实验只能得到相关性，无法形成可靠的因果结论。


# 推荐的框架组合与改进方向

本项目不应重新实现一个通用 Agent 框架，而应在成熟框架之上增加面向效率研究的观测、回放和分析能力。当前调查了 Pydantic AI、Inspect AI 和 smolagents，建议采用分层组合：

```text
Inspect AI
任务、数据集、批量实验、评分与结果归档
                    │
                    ▼
Pydantic AI
Agent、模型、工具、消息历史与运行时事件
                    │
                    ▼
efficient-harness core
统一 trace、profiling、replay、critical-path 分析
                    │
                    ▼
smolagents adapter
轻量 Agent 基线与兼容性验证
```

## Pydantic AI：核心 Agent runtime

Pydantic AI 最适合承载被测 Agent 的真实运行。它已经提供模型、消息、工具、事件流、重试、历史处理和 OpenTelemetry instrumentation，能够较低成本补齐目标一和目标二所需的运行时数据。

需要在其上增加：

- 独立的 append-only trace recorder，不能只依赖外部 OTel 后端；
- 统一记录 prompt、模型请求、模型响应、action parsing、tool execution、observation、retry 和 rollback；
- 将每次模型调用的 Usage、cache token、cost、TTFT 和流式 chunk 时间戳写入本地实验产物；
- 将历史处理前后的消息序列进行规范化，支持 new、reused、discarded 和 repeated-prefill 分类；
- 增加 replay model 和 replay tool，使固定响应可以重放，而不是再次访问真实模型或工具；
- 将 Pydantic AI 的历史处理器、模型封装和工具集封装为 context、generation、runtime、tool 四类标准干预接口。

需要注意：Pydantic AI 的 OpenTelemetry 能力主要解决可观测性，不自动提供面向因果实验的完整 offline replay。因此，trace 文件和 replay 协议必须由本项目拥有。

## Inspect AI：评测与实验管理层

Inspect AI 最适合负责任务、数据集、批量运行、评分和实验结果比较。它的 Task、Solver、Model、Tool、Scorer、Metric、EvalLog 和 event/transcript 结构可以作为本项目的评测外壳。

需要在其上增加或适配：

- 将 efficient-harness 的 trace 与 Inspect AI 的 EvalLog 建立稳定映射；
- 用 Solver 表达 context compression、history pruning 等上下文干预；
- 用 Model hook 或自定义 provider 表达 generation 和 runtime 干预；
- 为流式模型响应记录 TTFT、inter-token latency 和 chunk 时间戳；
- 将 token/episode profiling 和 critical-path 分析作为独立离线分析器，不把复杂指标全部塞进评测执行路径；
- 增加 replay provider，使同一个任务可以固定模型输出、工具响应或完整 trajectory；
- 明确区分 Inspect AI 原生日志、efficient-harness 扩展 trace 和估算指标，避免把估算 token 当成真实计费数据。

Inspect AI 的 `AnchorEvent`、`BranchEvent` 和 sandbox checkpoint 只能作为概念参考，不能直接假设它们已经提供完整的 trajectory replay。离线回放仍需要基于实际记录的模型事件和工具响应实现。

## smolagents：轻量基线与适配器

smolagents 代码量小、Agent 主循环清晰，适合作为第一批兼容性基线，也适合快速验证 trace schema 和 adapter API。

需要补充：

- 把 `RunResult`、`AgentMemory`、`MemoryStep` 和 `step_callbacks` 转换为统一 trace event；
- 将简单的累计 token 和 step timing 扩展为 invocation、tool、retry 和 episode 级数据；
- 对 CodeAgent 的代码生成、代码执行、工具调用和执行结果分别建模；
- 为 ToolCallingAgent 和 CodeAgent 提供统一的 replay tool/model 接口；
- 明确 LocalPythonExecutor 的安全边界，不能把本地代码执行器当成安全沙箱；
- 将 smolagents 作为 baseline adapter，而不是整个研究平台的唯一基础。

smolagents 当前的监控能力较轻，缺少成熟的事件、回放和细粒度性能归因模型，因此不建议从它开始建设核心 harness。

## 实现优先级

建议按以下顺序实施：

1. 先在 Pydantic AI 外部建立统一 trace schema 和 recorder；
2. 捕获 Agent、model、tool、observation、retry、等待和并发事件；
3. 实现本地 trace artifact、replay model 和 replay tool；
4. 完成 token 层、invocation 层和 episode 层 profiling；
5. 实现 critical path、exclusive/inclusive time 和并发重叠分析；
6. 接入 Inspect AI 的 Task、Dataset、Scorer 和批量实验；
7. 最后加入 smolagents adapter，验证 harness 对不同 Agent runtime 的兼容性。

详细选型依据见 [`framework-selection.md`](./framework-selection.md)。




# 潜在科研价值举例


必须明确：**harness 本身通常不是最重要的科研贡献。**

它首先是一个研究仪器。真正可以进一步发展成论文贡献的内容可能是：

1. 首个细粒度 Agent generation/runtime profiling study；
2. 发现 Agent execution 中此前被忽略的主要效率瓶颈；
3. 证明传统 LLM acceleration 在 Agent 场景下存在不同收益规律；
4. 建立 Agent acceleration 的统一评估方法；
5. 基于观察提出新的 Agent-specific acceleration mechanism。

例如，最终可能得到这样的结论：

- 普通 LLM 中 decode 占主要开销，但多步 Agent 随轨迹增长逐渐转为 repeated prefill 主导；
- 某个模型级方法单次生成加速明显，但由于 tool latency 和 retry，端到端收益很低；
- observation compression 比 reasoning token reduction 对长轨迹 Agent 更有效；
- structured tool-call span 的 speculative acceptance 与自然语言 span 存在系统性差异；
- prefix cache 的理论收益因为 prompt template 重构而没有真正实现；
- 不同 Agent benchmark 的瓶颈结构完全不同，不存在统一最优的加速方法。

这些才是 harness 带来的研究产出。

# 需要提前防止的三个风险

### 风险一：变成重复造 Agent 框架

你不需要重新实现 LangChain、AutoGen 或完整 benchmark ecosystem。你的差异应该是：

> 现有框架关注功能编排，你的系统关注细粒度可观测性、成本归因和受控效率实验。

Agent 功能尽量简单，profiling 能力尽量深入。

### 风险二：只测单次模型调用

如果最终图表都是 TTFT、tokens/s 和 GPU memory，那么它仍然只是普通 LLM serving benchmark。

必须保留 Agent 特有指标：

- episode success；
- trajectory length；
- number of actions；
- retries；
- observation growth；
- tool waiting；
- repeated prefill；
- cost per successful episode。

### 风险三：先集成方法，后寻找问题

不要一开始就把 speculative decoding、quantization、context compression 全部接进去。正确顺序仍然是：

1. 跑通 Agent；
2. 建立 trace；
3. profiling；
4. 找到实际瓶颈；
5. 再选择匹配的优化方法。

否则很容易成为一个“各种加速技术在 Agent 上跑一遍”的 benchmark 项目，缺少核心研究问题。


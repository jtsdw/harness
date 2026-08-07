# efficient-harness 下一阶段需求：面向 Agent 推理效率的可复现实验平台

文档日期：2026-08-07  
文档状态：需求设计，尚未开始实现  
前置文档：[`efficient-harness.md`](./efficient-harness.md)、[`inspect_ai_roadmap.md`](./inspect_ai_roadmap.md)

本文档定义项目完成目标一、目标二并启动目标五之后的下一阶段需求。它不取代原有五个目标，而是把尚未闭环的目标三、目标四，与现代 vLLM Serving、受控负载验证和面试交付要求收敛成一条可执行主线。

本文是**需求文档**：描述必须解决的问题、边界和验收标准。模块类名、具体 API 签名和文件拆分只有在约束需求时才出现，不把尚未评审的实现草案提前写成既定架构。

## 结论写在最前面

项目下一阶段的定位是：

> **基于 Inspect AI 与 vLLM 的 Agent 推理效率分析与优化实验平台：重建执行过程，在 token、model invocation 和 episode 三层归因成本，通过标准化干预与受控 replay，比较推理加速方法能否在任务质量约束下转化为 Agent 端到端收益。**

项目不应转成通用 Agent 应用框架，也不应缩成单一的投机解码研究代码。投机解码是第一批重点案例，但必须和 prefix/context 复用、Tool Schema 成本、受控负载验证及 Agent episode 指标放进同一评测体系。并发不是独立研究方向，只用于检查单请求结论在 continuous batching 和资源竞争下是否仍然成立。

下一阶段按以下顺序推进：

| 优先级 | 目标 | 为什么现在做 | 完成标志 |
|---|---|---|---|
| P0 | vLLM 逐请求观测与受控负载验证 | 先解决“是否测得准”，再用小规模并发检查结论是否稳健 | 单请求归因可信；受控并发下仍能关联 request、Agent step 与 Serving 指标 |
| P1 | 标准化加速干预接口 | 再解决“如何统一实施优化”，让项目从观察工具变成实验平台 | 至少两类真实干预使用同一协议运行 |
| P2 | Offline replay MVP | 原目标四尚未实现，当前对比仍受 trajectory 随机性影响 | 固定一条真实轨迹并只替换指定干预层完成重放 |
| P3 | Agent 加速案例研究 | 把测量和干预能力转化成可答辩结论 | 至少两类方法、三个 workload 完成质量约束下对比 |
| P4 | 可复现交付与可靠性收口 | 技术主线稳定后再统一入口、测试和面试材料，避免过早包装反复返工 | 新环境可在明确时间内跑通测试、最小 demo 和报告 |
| 可选 | Agent-phase-aware speculation | 研究创新有不确定性，不能阻塞工程闭环 | 先证实阶段差异，再决定是否实现新策略 |

这里的章节仍按 A-E 编号，方便和需求项稳定对应；**章节出现顺序不代表实施顺序**。实际执行顺序是 B → C → D → E → A。B/C 阶段不要求先完成统一 CLI 和面试 demo，但必须从第一天保存原始 artifact、实验配置和版本信息，不能等到 A 阶段再凭记忆补实验记录。

---

# 一、项目边界与目标用户

## 1.1 目标用户

第一目标用户是进行 Agent/LLM 推理效率实验的研究者或工程师。他需要回答：

- 一个 Agent episode 的时间和 token 花在哪里；
- 单次模型调用的加速能否转化成 episode 端到端加速；
- 优化是否改变了 Tool Call、轨迹或任务成功率；
- 不同方法是否在相同模型、请求、轨迹和质量约束下公平比较；
- 某篇论文声称的 insight 是否在目标 workload 中真实存在。

第二目标用户是项目协作者和面试评审者。他需要在不了解历史上下文的情况下：

- 十分钟内理解项目定位、架构和核心发现；
- 在无 GPU 环境跑通最小 trace/replay demo；
- 在有 GPU 环境按文档复现实验；
- 从原始 artifact 追溯 Dashboard 中的每个结论。

## 1.2 非目标

下一阶段明确不做：

- 不重新实现 LangGraph、Pydantic AI、AutoGen 一类通用 Agent 编排框架；
- 不建设生产级多租户 Agent 平台、完整 Memory 服务或通用 Tool Marketplace；
- 不为了“显得底层”重写 vLLM Scheduler、PagedAttention 或 CUDA Kernel；
- 不在缺少公平 baseline 和质量验证时宣称提出了新的投机解码算法；
- 不追求一次支持所有 Agent 框架，优先保证 Inspect AI 主路径正确；
- 不把 Prometheus 聚合值、估算 token、客户端计时和逐请求服务端指标混为同一证据等级。

## 1.3 项目成功标准

项目完成下一阶段后，必须能用一套 artifact 回答以下主问题：

> 在给定 Agent workload、模型、硬件、并发和质量约束下，瓶颈位于 repeated prefill、decode、queue、tool execution 还是 retry；应用某项优化后，单调用收益有多少、episode 收益有多少、收益为何被放大或稀释。

如果只能报告 tokens/s，而不能报告 Agent task success、端到端延迟和实验条件，则不算完成。

---

# 二、现状基线：必须继承，不能重新实现

以下能力已存在，下一阶段必须复用：

- `inspect_trace` 基于公开 Hooks 记录 prefill diff、segment token、token attribution、attempt group、execution topology 和 action parsing；
- token、model invocation、episode 三层 profiling 已有实现；
- BFCL、tau2-bench、ToolSpec 已有适配器和真实运行记录；
- 本地 vLLM baseline 与 n-gram speculative decoding 已跑通；
- ToolSpec 原生和 Harness 路径已做结果对齐；
- Dashboard 已能从原始 run artifact 重建；
- NSCC H100 + EAGLE-3 只有设计和脚本，尚未现场验证。

必须保留的真实限制：

1. `vllm_metrics` 在一组 49 次调用验证中能够精确归因，但在 ToolSpec/vLLM 场景中完全不产出记录，可靠性结论不能泛化；
2. 已有主要实验以 `MAX_CONNECTIONS=1` 运行，queue、preemption 和 batching 没有得到真实验证；
3. 目标三的标准化干预接口、目标四的 offline replay 尚未实现；
4. `gpu_cache_usage_perc_at_end` 的调用后采样错过峰值，不能代表真实 KV Cache 峰值；
5. EAGLE-3 的模型、版本、显存和指标检查尚未在 NSCC 真实硬件验证；
6. 尚无统一的 `verify`/`demo`/`benchmark` 入口（当前测试/格式化/类型检查要分别调用 `inspect_trace/scripts/verify.sh` 等脚本），这是 A1 要解决的问题；2026-08-07 复核确认 `verify.sh` 本身在这台机器上能在数十秒内跑完并全部通过，不存在"测试挂起/长时间不完成"的问题，此前的说法不准确，已更正。

这些限制不是附注，而是下一阶段验收测试必须覆盖的输入。

---

# 三、需求 A：可复现交付与可靠性收口（P4）

## A1：统一入口

仓库根目录必须提供三个稳定入口，具体使用 `make`、shell script 或 Python CLI 可在实现设计阶段决定：

- **verify**：只运行不需要 GPU/API Key 的格式检查和测试；
- **demo**：使用确定性 mock/replay 数据产生一份 trace、三层报告和最小可视化；
- **benchmark**：在显式提供模型服务后运行一个小规模真实 benchmark。

入口不得隐式下载大模型，不得依赖开发者个人目录中的未声明文件。

## A2：运行清单

每次真实实验必须写出机器可读 manifest，至少包括：

- git commit、dirty worktree 标记；
- 时间、主机、GPU 型号和数量；
- Python、PyTorch、CUDA、vLLM、Inspect AI 与 adapter 版本；
- target/draft 模型及 revision；
- dataset、split、limit、seed；
- sampling 参数；
- concurrency/QPS 和服务端关键参数；
- 开启的干预方法及配置；
- 原始 `.eval`、trace、服务日志和报告位置。

不能只把命令写在 Markdown 中；Markdown 可解释，manifest 负责复现。

## A3：测试分层

测试至少分为：

- unit：纯函数、schema、分析逻辑；
- integration-no-gpu：Inspect AI mock model + Hooks + artifact；
- integration-vllm：需要服务端但可限制为单请求；
- benchmark-smoke：少量真实样本，只验证全链路；
- full-benchmark：不进入默认测试。

默认 `verify` 必须明确打印正在执行的阶段，不允许长时间无输出。GPU/网络测试必须 opt-in。

## A4：面试交付物

必须提供：

- 一张当前真实架构图，而不是理想终态图；
- 一页“已实现/未实现/已知限制”；
- 一份十分钟 demo 路径；
- 一份统一结果表；
- 三个可以从原始数据复核的核心发现；
- 一份设计取舍：为什么基于 Inspect AI、为什么不是通用 Agent 框架、为什么需要 replay。

## A 类验收标准

- 新 checkout 按文档完成 `verify` 和 `demo`；
- 默认测试在预设时间预算内结束，超时会返回非零状态并指出阶段；
- demo 产物可从原始记录重新生成，不提交手工改写的结果；
- README 不把规划中的功能写成已实现；
- 任意结果表中的数字均可追溯到 manifest 和 artifact。

---

# 四、需求 B：vLLM 逐请求观测与受控负载验证（P0）

本需求的核心是建立可信的逐请求测量与 Agent invocation 归因，不是研究工业级流量调度。并发只承担稳健性验证：检查单请求下观察到的加速收益，在小规模 continuous batching、排队和资源竞争出现后是否仍然成立。项目不拥有生产流量，因此只声明“受控合成负载”或“Agent trace replay 负载”，不声明模拟了真实工业流量。

## B1：逐请求指标优先

NSCC 路径升级 vLLM 后，应优先使用服务端逐请求 timing 数据关联：

- queue time；
- time to first token；
- generation/decode time；
- mean inter-token latency；
- output tokens per second。

新版 vLLM 已提供可选的 per-request metrics；是否启用、版本下具体响应结构和性能开销必须现场验证，不能只根据文档假设。官方参考：<https://docs.vllm.ai/en/latest/features/per_request_metrics/>。

Prometheus `/metrics` 保留用于服务级观测：running/waiting requests、prefill/decode/queue histogram、preemption、KV Cache、prefix cache 命中率、speculative acceptance 和 MFU 等。它不再承担“通过前后 histogram 差值猜测某个并发请求”的主要关联职责。官方参考：<https://docs.vllm.ai/en/latest/usage/metrics/>。

## B2：统一请求关联

每次模型调用必须建立以下稳定关联：

```text
run_id
  └── eval_id
      └── sample_uuid / episode_id
          └── model_event_uuid / invocation_id
              └── serving_request_id
```

不允许只靠时间窗口匹配。缺少 `serving_request_id` 时必须标记 `unattributed`，不得静默归因到最近请求。

## B3：指标证据等级

每个指标必须记录 `source` 和 `confidence`：

| source | 示例 | 证据等级 |
|---|---|---|
| server_per_request | queue/TTFT/ITL | 逐请求直接值 |
| server_prometheus | running requests/KV usage | 服务级聚合值 |
| client_stream | 首 chunk/相邻 chunk 时间 | 客户端观测值 |
| inspect_event | working_time/usage | Agent/Eval 层值 |
| estimated | 分段 token、近似显存 | 估算值 |

同名指标来自不同来源时不能覆盖；应并列保存并允许交叉验证。

## B4：必须采集的指标

### Invocation 层

- queue time；
- TTFT；
- prefill time（服务端版本可用时）；
- decode/generation time；
- mean ITL/TPOT；
- prompt/generated/cached token；
- request tokens/s；
- finish/cancel/error reason；
- speculative draft/accepted token 和 acceptance rate（启用时）；
- guided/structured decoding（工具调用 JSON schema 等语法约束解码）的开销（启用时）。这类约束会在每步解码时过滤非法 token，可能与投机解码的接受率相互影响（约束越严格，草稿被拒绝的概率可能越高）；这个交互目前只是推测，尚未实测，先在此记录，具体验证放到用到该功能时再做，不阻塞 B 阶段。

### 服务层

- running/waiting requests；
- preemption；
- KV Cache 使用率；
- prefix cache 命中率（衡量 agent 场景下重复系统提示/工具 schema/多轮历史的复用效果；具体 Prometheus 字段名以实际部署的 vLLM 版本现场核实为准）；
- batch/iteration token；
- GPU utilization、显存和功耗（可通过独立采样器补充）；
- 指标采集本身的 CPU/latency overhead。

### Episode 层

- P50/P95 end-to-end latency；样本量充分时再报告 P99；
- critical path；
- model/tool/waiting 时间；
- success rate；
- cost per successful episode；
- 单调用 speedup 到 episode speedup 的转化率。

## B5：最小稳健性实验矩阵

第一阶段只要求一个小而可解释的矩阵，不建设通用压测平台：

| 维度 | 最小取值 |
|---|---|
| concurrency | 1 / 4 / 8；只有 8 未产生资源竞争时才增加 16 |
| workload | 一组真实 Agent trace 中抽取的输入，保留其 prompt/output 长度分布 |
| method | baseline；可选使用一个已有方法验证指标兼容性，不在 B 阶段研究方法收益 |
| arrival | 第一阶段使用固定并发；Poisson、burst 和 open-loop QPS 为扩展实验 |
| repetition | 每个配置至少 3 次，并记录样本数和离散程度 |

固定并发用于回答“收益是否随资源竞争变化”，不能包装成生产流量。后续若增加请求到达率实验，必须区分 closed-loop concurrency 与 open-loop QPS，二者不能混用同一个“并发数”字段。Agent trace replay 在需求 D 建立稳定 replay artifact 后再加入，不作为 B 的首轮阻塞项。

## B 类验收标准

- concurrency=1 下逐请求指标与现有客户端/Prometheus方法交叉验证；
- concurrency=4/8 下每个 invocation 能通过 request ID 关联，不能依赖 `_count delta == 1`；
- baseline 在相同的 1/4/8 受控负载下均能完成逐请求关联；若使用已有方法，只验证指标兼容性，不提前承担 C/E 的方法研究任务；
- queue、batch、KV Cache 和 preemption 如实报告；不为了制造“工业场景”强制把系统压到过载；
- 报告包含 P50/P95、样本量和离散程度，不只报告平均值；样本不足时不报告 P99；
- 指标采集开销单独测量并报告；
- vLLM 指标不可用时，运行继续但产物显式记录缺失原因。

---

# 五、需求 C：标准化加速干预接口（P1，对应原目标三）

## C1：干预类型

统一协议至少覆盖四类：

```text
ContextPolicy
GenerationPolicy
RuntimePolicy
ToolExecutionPolicy
```

- `ContextPolicy`：Tool Schema 选择/压缩、历史裁剪、observation 压缩、prompt canonicalization；
- `GenerationPolicy`：采样配置、speculative proposer、模型级联；
- `RuntimePolicy`：prefix caching、batch/scheduler 配置、KV Cache 策略；
- `ToolExecutionPolicy`：并行度、timeout、cache、mock latency、retry。

这四类是概念边界，不要求第一次实现就设计一个复杂继承体系。

## C2：共同契约

每个干预必须声明：

- `name` 和版本；
- 作用层与配置；
- 是否可能改变模型输出/trajectory；
- 是否声称 lossless，以及 lossless 的准确定义；
- 所需服务端能力；
- 质量守卫和失败回退；
- 对应 trace 中的 before/after 记录；
- 能否用于 online execution、offline replay，或两者都可以。

## C3：第一批实现

只要求两个端到端插件：

1. 一个 context 类干预：优先选择 Tool Schema selection/compression 或稳定 prompt canonicalization；
2. 一个 generation/runtime 类干预：封装 vLLM n-gram/EAGLE-3/ToolSpec 中至少一种，而不是重新实现解码算法。

第一批不要求同时实现四类接口。接口设计必须由真实插件反推，不能先写一个没有用户的抽象层。

## C4：配置与结果隔离

- baseline 与 intervention 必须产生独立 run；
- 每个 run 的 manifest 必须完整记录干预配置；
- 干预不得原地修改原始 dataset 或 replay artifact；
- Dashboard 必须能按方法、workload、并发和 seed 分组；
- 失败回退到 baseline 时必须记录，不能把回退结果计为干预成功。

## C 类验收标准

- 两个真实插件通过共同发现/配置/manifest 机制运行；
- 至少一个插件来自外部方法适配，证明协议不是只服务自研代码；
- before/after context 或 runtime 配置可从 trace 追溯；
- 干预关闭时，行为与现有 baseline 一致；
- 接入第三个同类方法不需要修改 benchmark adapter 核心逻辑。

---

# 六、需求 D：Offline Replay MVP（P2，对应原目标四）

## D1：Replay 的目的

Replay 用于控制变量，不是为了恢复中断作业，也不是重新播放 Dashboard 动画。MVP 必须支持：

- 固定原始输入；
- 固定 Tool Response；
- 固定或校验模型输出；
- 只重放 context construction、model invocation 或某个干预层；
- 将 replay 结果与来源 trajectory 建立双向链接。

## D2：三种模式

按实现优先级定义：

1. **Tool-fixed replay**：工具返回固定，模型重新运行；用于比较 context/generation 方法；
2. **Model-fixed replay**：模型输出固定，只重放 context、tool 和分析；用于验证分析器和 Tool 策略；
3. **Full trajectory replay**：模型与工具均固定；用于确定性回归和 Dashboard 重建。

MVP 只强制完成 Tool-fixed 和 Full trajectory；Model-fixed 可在接口自然时一并实现。

## D3：Replay artifact

必须包含：

- source run/eval/sample；
- 每步规范化模型输入与输出；
- Tool Call ID、参数、返回或异常；
- 采样配置、seed 和停止原因；
- 内容 hash；
- 可选择脱敏或不保存原始敏感内容的策略；
- schema version。

## D4：分叉处理

当重新运行模型产生不同 Tool Call 或轨迹分叉时：

- 不得偷偷继续消费不匹配的旧 Tool Response；
- 默认中止并报告第一个 divergence；
- 可选的 tolerant 模式必须明确 Tool 匹配规则；
- divergence 本身应成为结果指标。

## D 类验收标准

- 从现有真实 BFCL 或 tau2 样本生成 replay artifact；
- Full trajectory replay 两次产生相同 canonical trace；
- Tool-fixed replay 能在更换一个干预方法后运行；
- 参数不匹配时在正确步骤失败，并输出可读 diff；
- replay 结果不会被计入真实 online latency 结论；
- artifact schema 有版本并通过兼容性测试。

---

# 七、需求 E：Agent 加速案例研究（P3）

## E1：研究问题

第一轮案例研究不问“哪种投机解码普遍最好”，而问：

1. 不同 workload 的 repeated prefill、decode、tool wait 和 queue 占比是否不同；
2. 单调用 speedup 有多少能转化为 episode speedup；
3. Tool Schema/Prefix 优化和 speculative decoding 分别作用在哪个瓶颈；
4. 并发升高后 speculative decoding 是否产生负收益；
5. 优化是否改变 Tool Call 合法率、参数正确率、trajectory 和 success。

## E2：最小方法集合

- vLLM baseline；
- prefix cache off/on；
- 一个 context/Tool Schema 方法；
- vLLM n-gram speculative decoding；
- EAGLE-3（现场验证成功后）；
- ToolSpec（作为 Tool Calling 专用方法）。

若某方法无法在同一 serving 栈运行，必须按“各自相对 baseline speedup”比较，不能直接横比原始 tokens/s。

## E3：最小 workload 集合

- API-Bank 或 BFCL：结构化 Tool Calling；
- tau2-bench：多轮、带 Tool 等待的 Agent；
- 一个长自然语言输出任务：Decode 主导对照。

三个 workload 必须使用同一套结果 schema，但不要求使用完全相同的 scorer。

## E4：质量评估

必须区分：

- greedy/确定性设置下的逐 token 或结构化输出一致性；
- 随机采样下的分布保持，不把“文本不完全相同”直接判为算法错误；
- Tool Call schema validity；
- Tool name/argument correctness；
- episode task success；
- replay divergence rate。

任何 `lossless` 声明都必须写明是“相同 token”“相同采样分布”还是“任务质量无显著下降”。

## E5：报告格式

每个方法至少报告：

- TTFT、TPOT/ITL、E2E P50/P95；样本量充分时再报告 P99；
- request/output-token throughput；
- draft acceptance 指标（适用时）；
- repeated/cached prefill；
- task success 与 Tool Call 指标；
- episode speedup / invocation speedup；
- GPU、并发、模型、采样和服务配置；
- 置信区间或多 seed 方差；
- 失败案例和适用边界。

## E 类验收标准

- 至少两个加速方法完成三个 workload 的统一对照；
- 所有结论都能从 manifest + artifact 重新生成；
- 报告至少包含一个“单调用快但 episode 不快”或相反的机制解释；
- 不把不同模型/serving 栈的原始 tokens/s 当作公平横向结论；
- 结论明确限定模型、硬件、并发与数据集，不外推成普遍规律。

---

# 八、可选研究：Agent-phase-aware speculative inference

这个方向只有在 A-E 完成或至少形成稳定实验闭环后才进入实现阶段。

## 8.1 先验证的假设

Agent 输出应按阶段拆分：

- Tool Schema/JSON 固定结构；
- Tool name；
- Tool argument；
- reasoning/natural language；
- final response。

先测各阶段的 token 占比、proposal acceptance rate/accepted length、verification cost、schema validity，以及对 episode 关键路径的贡献。

如果阶段之间没有稳定差异，就停止该研究方向，不为了完成预设故事继续造策略。

## 8.2 可能的方法形态

只有假设成立后，再考虑：

- Tool 结构阶段使用 schema/retrieval proposer；
- 自然语言阶段使用 EAGLE/MTP/ngram 或关闭投机；
- 根据并发和阶段动态选择 proposer/K；
- 优化目标从 tokens/s 改为质量约束下的 episode latency。

当前 vLLM 已有基于 batch size 调整 speculative length 的 Dynamic Speculative Decoding，因此“只根据并发动态调整 K”不能单独视为 Agent-specific 创新。项目创新必须来自 Agent 阶段、结构约束或 episode-level objective。

## 8.3 研究继续/终止门槛

继续条件：

- 至少两个 workload 上观察到可重复的阶段接受率差异；
- 潜在节省落在 episode 关键路径；
- 现有单一方法不能同时覆盖结构化与自然语言阶段；
- 切换策略的开销小于预期收益。

任一关键条件不成立时，把结果写成 profiling finding，停止新方法开发。负结果仍然是 harness 的有效产出。

---

# 九、统一实验协议

## 9.1 实验单元

一个可比较 run 由以下元组唯一描述：

```text
(code revision, environment, hardware, target model, draft model,
 workload, dataset revision, sampling config, load config,
 intervention config, seed)
```

任一维度不同都视为不同实验条件。

## 9.2 Baseline 规则

- 每种 serving 栈有自己的无加速 baseline；
- 同一对比内 target model、量化、sampling、dataset 和硬件必须一致；
- baseline 与 intervention 交错或随机化运行顺序，避免温度/后台负载随时间漂移；
- 统一 warm-up，warm-up 不计入正式样本；
- OOM、超时、回退和失败请求计入结果，不静默删除。

## 9.3 Online 与 Replay 的结论边界

- Online 用于真实 E2E、trajectory 和 success；
- Replay 用于控制变量和定位因果机制；
- Replay latency 不能直接替代 Online E2E；
- 两者结论必须分栏展示，不能混合平均。

## 9.4 Overhead

必须测量 Hooks/trace 写盘、per-request metrics、Prometheus/GPU 高频采样、intervention wrapper 的开销，并明确 replay 自身不计入被测推理时间的边界。

---

# 十、非功能需求

## 10.1 正确性

- 真实值与估算值字段分离；
- 所有 join key 稳定且可验证；
- 并发测试中不得使用最近时间戳启发式做“精确”归因；
- schema、manifest 和 replay artifact 有版本；
- 缺失指标显式记录 `unavailable`/`unattributed`，不默认填零。

## 10.2 可观察失败

- Hook 超时、指标抓取失败、服务端版本不支持、回退 baseline 都必须写入 artifact；
- 失败不能阻塞被测 Agent 主路径，除非该指标是当前实验的硬性前置条件；
- 报告必须列出未完成样本和失败原因。

## 10.3 可移植性

- 本地旧 GPU 路径和 NSCC H100 路径保持独立依赖；
- 不为迁就本地旧驱动继续限制 NSCC 的 vLLM 版本；
- adapter 与 `inspect_trace` 通过版本化包依赖连接，不复制源码；
- 所有路径可配置，不把个人绝对路径写入运行逻辑。

## 10.4 数据与安全

- Tool 参数、Observation 和模型输出可能包含敏感信息；
- replay artifact 必须支持字段级脱敏或 hash-only 模式；
- Dashboard 默认不将 Secret、Token 或完整环境变量嵌入 HTML；
- 公开示例只使用可公开数据集和模型输出。

---

# 十一、里程碑与交付顺序

## M0：Serving 观测闭环

范围：B1-B5。  
交付：H100 vLLM 现场验证、逐请求指标、请求关联，以及 concurrency=1/4/8 下测量链路本身的稳健性报告。  
退出条件：先证明单请求归因可信，再证明受控负载下仍能正确关联和解释指标；不要求在本阶段研究加速收益，也不构建工业压测平台。

## M1：干预接口 MVP

范围：C1-C4。  
交付：一个 context 插件、一个 generation/runtime 插件。  
退出条件：两种方法共享配置和 artifact 协议，adapter 不感知具体方法。

## M2：Replay MVP

范围：D1-D4。  
交付：版本化 replay artifact、Tool-fixed 和 Full trajectory replay。  
退出条件：固定轨迹可重复，分叉能检测，online/replay 结果不混淆。

## M3：统一案例研究

范围：E1-E5。  
交付：三 workload、多方法、受控负载/质量联合报告和 Dashboard，其中并发只作为方法收益的稳健性维度。  
退出条件：回答“单调用收益怎样转化为 episode 收益”。

## M4：可靠性与面试交付收口

范围：A1-A4。  
交付：稳定 `verify`、mock demo、run manifest、测试分层，以及基于 M0-M3 真实产物整理的架构图、统一结果表和十分钟演示。  
退出条件：新 checkout 能重复运行；测试无长时间静默；所有对外结果都能追溯到原始 artifact。此阶段是收口，不得为了包装重新发明实验接口。

## M5：可选科研验证

范围：第八节。  
交付：阶段级 acceptance/critical-path 分析；仅在继续门槛通过后实现策略。  
退出条件：形成有证据的新方法，或形成同样有价值的负结果。

顺序约束：M0 → M1 → M2 → M3 → M4 → M5。M0 与 M1 是当前最优先主线；M2 可以提前设计，但 M3 不得绕过 Serving 观测与 Replay；M4 在技术能力和案例结果稳定后统一收口；M5 不得阻塞 M0-M4。

---

# 十二、面试声明门槛

为了避免简历描述超过实际完成度，声明必须满足：

| 可以声明 | 前置证据 |
|---|---|
| “构建 Agent 效率分析 Harness” | 目标一/二代码、测试、真实 artifact |
| “支持三层成本归因” | token/invocation/episode 报告及已知限制 |
| “支持受控负载下的 Serving 性能分析” | M0 完成，逐请求归因可信，并完成 concurrency=1/4/8 下的关联与指标解释验证 |
| “提供可插拔加速接口” | M1 至少两个真实插件 |
| “支持受控 Replay” | M2 固定真实 trajectory 可重复 |
| “比较 Agent 投机推理方法” | M3 公平 baseline、质量和 E2E 结果 |
| “提出 Agent-specific 投机策略” | M5 假设验证、实现、强 baseline 与消融 |

尚未达到门槛的内容只能写“设计/进行中”，不能用完成时态。

---

# 十三、主要风险与应对

## 风险一：继续扩张文档而缺少可运行入口

应对：B/C 优先不等于只写研究代码；M0-M3 每个里程碑都必须留下可执行命令、原始 artifact 和最小测试，M4 再统一整理入口、测试分层与展示材料。

## 风险二：新版 vLLM 迁移吞噬全部时间

应对：本地旧版本保持不动；NSCC 独立环境先完成最小 baseline 和逐请求指标 spike，再迁移完整 benchmark。

## 风险三：投机解码只有 tokens/s 收益

应对：以 episode E2E、P50/P95 和 success 为主结果；只有样本量充分时报告 P99。若 Tool 等待主导导致收益小，如实作为结论。

## 风险四：Replay 设计过度通用

应对：先固定一条现有 BFCL/tau2 轨迹，支持 Tool-fixed 和 Full trajectory；不先解决跨框架通用 replay。

## 风险五：抽象接口没有真实用户

应对：C 类接口由 Tool Schema 方法和一种 speculative 方法反推；两个插件跑通前不冻结 API。

## 风险六：把随机输出差异误判成投机错误

应对：确定性 token 对齐、随机分布保持、Tool Call 正确性和 episode success 分开评估。

---

# 十四、相关文档

- [项目原始五目标](./efficient-harness.md)
- [Inspect AI 路线与未完成目标](./inspect_ai_roadmap.md)
- [目标二设计](./goal2_design.md)
- [目标二真实验证与指标采集缺口](./goal2_real_validation_findings.md)
- [ToolSpec 原生复现与 Harness 接入](./toolspec_integration_findings.md)
- [ToolSpec 与 vLLM n-gram 对比](./toolspec_vllm_speculative_comparison.md)
- [NSCC H100 / EAGLE-3 设计阶段计划](./nscc_h100_speculative_decoding_plan.md)
- [加速方法调研](./acceleration_methods_survey.md)
- [Benchmark 接入手册](./benchmark_integration_playbook.md)
- [远程计算工作流](./remote_compute_workflow.md)

下一份应新增的文档不是另一份总路线，而应是 M0（需求 B）的 Serving 指标接入与验收设计；随后为 M1（需求 C）的干预接口建立设计与真实插件记录，再为 M2 的 Replay schema 建独立设计文档。

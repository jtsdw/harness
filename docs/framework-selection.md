# Efficient Harness 框架选型与改进计划

本文合并了两轮独立的框架选型分析：一轮是概念性的组合架构设计（未落地实现前的分析），另一轮是基于**实际读代码 + 已经在 inspect_ai 上跑通目标一（trace）实现**之后的实测验证。两轮结论不完全一致，本文如实保留分歧并说明分歧的来源，不强行统一成一个结论。

## 两轮结论对比

| | 第一轮（概念分析） | 第二轮（实测验证，本次追加） |
|---|---|---|
| 出发点 | 从零假设三个框架都未使用，纯按能力清单打分 | 已经用 inspect_ai 的 Hooks 机制实现并跑通目标一（重复 prefill 检测、segment token 估算、retry attempt 分组），并有可运行的测试为证 |
| 结论 | 推荐**组合架构**：Pydantic AI 做核心 Agent runtime，Inspect AI 做外层任务/评测编排，smolagents 做轻量 baseline；若只能选一个起点，选 **Pydantic AI** | 推荐**继续用 inspect_ai 单一框架**，把 pydantic-ai 的 `AbstractCapability`（`wrap_*` 系列）设计思路吸收进目标三的干预接口设计；smolagents 不作为底座，留作后续研究对象 |
| 分歧原因 | 未考虑"已经验证过的实现"这一变量，纯粹按框架能力清单比较 | 发现 inspect_ai 已解决的 HTTP-retry-attempt 可观测性问题，恰好是 pydantic-ai 明确承认的缺口（retry 被 tenacity transport 吃在 span 边界之下，默认不可见），而这正是 `efficient-harness.md` 目标一 Q5 的核心诉求；切换框架意味着把这个已解决的问题重新做一遍 |

两轮分析在"框架能力清单"层面高度一致（见下方两节的具体对比内容），分歧主要出在**是否把"已验证实现"的沉没成本和风险对冲价值计入决策**。如果你们现在处于"完全没有开始实现"的阶段，第一轮的组合架构思路仍然值得认真考虑，尤其是 Pydantic AI 的 `wrap_*` 干预机制和纯 OTel 可观测性；但既然目标一已经在 inspect_ai 上跑通并有测试验证，第二轮的"不切换，吸收优点"是风险更低的路径。

## 第三轮补充：pydantic-ai 和 inspect_ai 不是二选一

前两轮分析都隐含一个前提——"pydantic-ai 和 inspect_ai 只能选一个当底座"。在审计 inspect_ai 自带的 `docs/`/`examples/` 时（完整过程见 [`inspect_ai_docs_examples_audit.md`](./inspect_ai_docs_examples_audit.md)）发现这个前提不完整：inspect_ai 官方提供 `agent_bridge()`（`docs/agent-bridge.qmd`，示例 `examples/bridge/pydantic-ai/agent.py`），能让一个**原生 pydantic-ai agent**（model 参数写成特殊值 `"inspect"`）正常运行的同时，它发出的每一次模型调用都被路由进 inspect_ai 自己的 model provider——正常触发 `ModelEvent`，被我们已经建好的 Hooks（`inspect_trace`）捕获，记进 `.eval` log。也就是说：**可以用 pydantic-ai 写 agent 逻辑，同时不丢失目标一已经建好的整套 inspect_ai 观测基础设施**。

限制（不是万能药，别高估）：`agent_bridge()` 只解决"事后记录不丢失"，**不会**让 pydantic-ai 的 `AbstractCapability`（`wrap_model_request`/`wrap_tool_execute`）拦截点和 inspect_ai 自己的 Hooks 机制自动统一成一套。真要同时用两边的干预能力（比如目标三想用 pydantic-ai 的 `wrap_*` 做效率方法的插拔），仍然需要在两层各自维护相应逻辑，只是不再是"选 pydantic-ai 就必须放弃 inspect_ai 观测能力"这种非此即彼的困境，多了一个"组合使用"的可行选项。

同一轮审计还发现一个降低目标三设计成本的信息：inspect_ai 自己的 `on_before_model_generate` Hook 已经**明确支持原地修改即将发出的请求**（`data.input`/`data.tools`/`data.config`，会实际影响发出的 API 调用和缓存 key），且每次 retry 都会触发一次——这是我们已经在用的机制（`inspect_trace` 的 attempt-grouping 就靠它）。目标三设计"标准化 generation 层干预接口"时，第一步应该评估能不能在这个已有 Hook 基础上扩展出"短路/替换返回结果"的能力（目前它只支持"观测+改写"，还不支持"不发起真实调用直接返回结果"），而不是照抄 pydantic-ai 的 `AbstractCapability` 从零设计一套平行体系——这样能少一层"两套干预机制并存"的复杂度。

## 选型标准

理想 harness 需要同时满足：

1. 能重建完整 Agent execution trace；
2. 能完成 token、model invocation 和 episode 三层成本归因；
3. 能提供 context、generation、runtime 和 tool execution 的标准干预点；
4. 能支持 online execution 与 offline replay；
5. 代码质量、测试、维护状态足以支撑长期研究。

## 第一轮：概念性对比

### Pydantic AI

优势：

- 生产级 Agent runtime，模型和工具抽象完整；
- 有消息历史、事件流、重试、流式输出和历史处理器；
- `InstrumentedModel` 已覆盖 OpenTelemetry span、输入输出消息、token usage、cost 和 time-to-first-chunk 等观测；
- Usage 模型区分输入、输出、缓存读写等 token 类型；
- 适合研究上下文增长、重复 prefill、模型调用和工具等待。

缺口：

- OTel 观测不等于本地、可审计、可重放的实验轨迹；
- 需要补充统一 trace artifact 和 replay provider；
- GPU memory、KV cache、batch、prefill/decode 等底层指标需要对接推理后端；
- 需要增加 critical-path 和并发区间分析。

### Inspect AI

优势：

- 评测架构成熟，Task、Dataset、Solver、Scorer、Metric 和 EvalLog 分工清晰；
- 有 event、transcript、viewer 和 analysis 体系；
- 适合固定任务、批量实验、结果归档和统一评分；
- 工程测试和 CI 规模较大，适合作为长期评测外壳。

缺口：

- 细粒度 model invocation profiling 不是其主要目标；
- 需要补充流式 chunk 时间、TTFT 和 ITL 记录；
- trajectory replay 和固定模型/工具响应需要自定义 provider；
- `AnchorEvent`、`BranchEvent` 或 sandbox checkpoint 不能直接视为完整 replay 实现；
- 应将性能分析放在独立 analyzer 中，避免污染评测执行核心。

### smolagents

优势：

- 代码量小，Agent 主循环易读易改；
- 提供 AgentMemory、MemoryStep、step callback、RunResult 和基础 token/timing 信息；
- 适合快速实现 baseline 和验证 adapter 接口。

缺口：

- 缺少统一、持久化的细粒度事件模型；
- 缺少完整 offline replay 和阶段级 profiling；
- CodeAgent 的代码生成与执行需要额外的动作事件建模；
- LocalPythonExecutor 不应被当作安全沙箱；
- 测试和基础设施规模小于另外两个项目。

## 第二轮：实测验证补充

在真正读代码（而非仅凭文档/印象）之后，对上面三个框架做了逐项核实，重点针对 `efficient-harness.md` 目标一里明确列出的六个问题（上下文膨胀、重复 prefill、token 分类、observation 复用、retry/invalid action 成本、并行/等待/回滚）。

| | inspect_ai（已验证并已实现目标一） | pydantic-ai | smolagents |
|---|---|---|---|
| 控制边界（自己调模型、自己实现 loop） | ✅ | ✅ | ✅ |
| 消息级完整轨迹快照 | ✅ `ModelEvent.input` | ✅ OTel span 里的 `gen_ai.input.messages` | ✅ `ActionStep.model_input_messages` |
| Token 细粒度（cache/reasoning） | ✅ 真实值 | ✅ 真实值（`RequestUsage` + provider `details`） | ⚠️ 只有粗粒度 input/output/total，reasoning 需从 `raw` 二次解析 |
| TTFT / decode 等 model-invocation 指标 | ❌（已在 `inspect_ai_roadmap.md` 里确认缺失） | ✅ 有 TTFT histogram，但仍无 queueing/decode 拆分 | ❌ 完全没有 |
| **HTTP retry 可观测** | ✅ 每次尝试独立事件（已基于此做出 attempt 分组，见 `inspect_trace` 实现） | ❌ retry 被 tenacity transport 吃在 span 边界之下，默认不可见 | ❌ 完全不可见，`token_usage` 只反映最终成功那次 |
| 干预/hook 机制 | Hooks + entry_points 自动发现（观测为主，`before_model_generate` 可改） | `AbstractCapability`：before/after/**wrap** 覆盖 run/node/model_request/tool_validate/tool_execute/output_validate 几乎每个阶段，wrap 可短路/重试/替换——**四类里最强** | 仅 `step_callbacks`（事后回调），无生成前拦截点，需子类化 `Model`/`Agent` |
| 并行工具调用 | ✅ | ✅ 更精细（`end_strategy`: early/graceful/exhaustive + barrier） | ⚠️ 仅 `ToolCallingAgent` 支持，`CodeAgent`（其招牌模式）不支持 |
| Mock/Replay | ✅ 官方 `mockllm/model`（已用于测试） | ✅ `FunctionModel`（好用但要自己接线，无官方 cassette 公共 API） | ❌ 无官方 mock，测试里靠手写 if/else 的 Fake 类 |
| 成熟度 | 6952 commits，UK AISI 官方 | 2602 commits，**CI 强制 100% 覆盖率**，Production/Stable | 1052 commits，HuggingFace 官方，API 仍在演化（`dev0`） |

### 关键发现

**pydantic-ai 的干预机制其实比 inspect_ai 更强**：`AbstractCapability` 不只是观测点，`wrap_model_request`/`wrap_tool_execute`/`wrap_node_run` 这类"wrap"钩子可以真正拦截、短路、重试、替换执行——这正是目标三"标准化加速干预接口"想要的东西，比 inspect_ai 的 Hooks（主要是观测+改写单点）覆盖面更全。而且它的可观测性是纯 OTel（不绑定商业 Logfire），意味着目标二的下游分析可以直接复用整个 OTel 生态（collector/exporter/存储），比自定义 event schema 更"标准"。这一点跟第一轮分析的判断一致（"`InstrumentedModel` 已覆盖 OpenTelemetry span"），第二轮补充了具体的代码证据（`_instrumentation.py::open_model_request_span`、`TIME_TO_FIRST_CHUNK_HISTOGRAM_BOUNDARIES` 等）。

**但 pydantic-ai 有一个和目标一 Q5 直接对应的真实缺口**：HTTP 级 retry 被吃在 tenacity transport 里，span 边界之下默认不可见——这正是已经在 inspect_ai 上花功夫解决（并踩坑修复了 `anyio.get_current_task()` 返回非稳定 wrapper 对象的 bug）的 attempt-grouping 能力。要在 pydantic-ai 上拿到同等能力，得自己接 tenacity 的 `before`/`after` 回调重新实现一遍。第一轮分析没有覆盖到这个具体缺口。

**smolagents 不建议作为通用底座**：三类干预/观测能力（hook、retry 可见性、并发）都是三者中最弱的，且它的招牌 `CodeAgent` 模式把 reasoning 和 tool-call 混在一段自由文本里，做 token 级归因还要自己按代码块边界二次分词——这对"通用效率研究基础设施"是减分，但反而让它更适合作为**研究对象**（而不是底座）：代码优先 agent 范式下的效率归因难题本身就是一个有意思的科研问题，可以在底座建好之后作为一个 benchmark target 接进来。这个"研究对象而非底座"的定位比第一轮分析的"轻量 baseline"定位更明确。

## 推荐架构

第一轮设计的组合架构图仍然是一个合理的**理想终态**参考，尤其是如果未来要支持多种 agent 框架接入同一套 profiling/replay 基础设施：

```text
Inspect AI Task / Dataset / Scorer
              │
              ▼
Pydantic AI Agent runtime
              │
              ├── model instrumentation
              ├── tool instrumentation
              ├── event stream capture
              └── history/context capture
              │
              ▼
efficient-harness core
              ├── canonical trace
              ├── replay model/tool
              ├── token profiler
              ├── latency profiler
              ├── critical-path analyzer
              └── intervention API
              │
              ▼
smolagents adapter / baseline
```

**当前实际路径**（基于已完成的目标一实现）与此图的差异：`inspect_trace`（`/home/liuyingen/code/efficient-harness/inspect_trace/`）目前直接扮演了图中 "efficient-harness core" 的角色，但挂在 inspect_ai 的 Hooks 之下，而不是挂在 Pydantic AI runtime 之下。这不等于放弃了组合架构的方向——inspect_ai 本身也天然承担着图中 "Task / Dataset / Scorer" 的外层编排角色，相当于把图中的两层合并成了一层。如果未来确实需要 Pydantic AI 更强的 `wrap_*` 干预能力（尤其是做目标三时），可以考虑把 "model instrumentation / tool instrumentation" 这两层换成 Pydantic AI 的 `AbstractCapability`，同时保留 inspect_ai 作为外层任务编排——即向图中的组合架构靠拢，而不是推倒重来。

## 第一阶段改进清单

第一轮给出的清单在"如果从零开始"的场景下依然成立，列在这里供后续目标（二/三/四）参考；目标一部分已经用 inspect_ai + `inspect_trace` 完成，不需要重做：

- ~~定义稳定的事件类型和 trace schema~~ ——已用 `inspect_trace/schema.py` 完成（`PrefillDiffRecord`/`SegmentTokenRecord`/`AttemptGroupRecord`）；
- ~~为每个事件记录 `trace_id`、`episode_id`、`parent_id`、开始结束时间和来源框架~~ ——已用 `TraceEnvelope`（`run_id`/`eval_id`/`sample_uuid`/`recorded_at`）完成，粒度对应到 inspect_ai 的 event/sample/task 层级；
- ~~记录模型请求前的规范化 context 与请求后的 response~~ ——已通过 `on_before_model_generate`/`on_sample_event` 完成；
- ~~记录真实 Usage 与估算 segment token，并明确标记二者~~ ——已完成（`billed_*` vs `estimated_*` 字段分离）；
- 包装工具执行，记录输入、输出、等待、异常和重试——**部分完成**（inspect_ai 原生已覆盖工具执行本身的记录，`inspect_trace` 目前只针对模型调用做了 retry 分组，工具执行侧的等待/异常细分留给目标二）；
- 实现 replay model、replay tool 和固定 trajectory runner——**未开始**，对应目标四，详见 `inspect_ai_roadmap.md`；
- 对所有干预方法保持相同任务、模型、初始输入和 replay 数据——**未开始**，对应目标三/四；
- 用 Inspect AI 负责任务评分，用 efficient-harness analyzer 负责效率报告——当前架构下这一条自然成立（同一个框架内分工），不需要额外的框架间桥接。

## 不能混淆的边界

- Agent framework 负责产生行为；harness 负责记录、控制和比较行为；
- OTel 负责观测传输；trace artifact 负责实验重现；
- EvalLog 负责评测结果；profiling trace 负责性能归因；
- 真实计费 token、服务端性能指标和本地估算值必须分开保存；
- online 结果用于测量真实效果，offline replay 结果用于控制变量和因果比较。

## 相关文档

- [Efficient Harness 总体目标与设计](./efficient-harness.md)
- [Inspect AI 路线与现状分析](./inspect_ai_roadmap.md)
- [本地与远程服务器同步指南](/home/liuyingen/code/doc/sync-guide.md)（通用指南，不专属于这个项目，所以没有跟着这次重构搬过来）
- 目标一实现：`/home/liuyingen/code/efficient-harness/inspect_trace/README.md`

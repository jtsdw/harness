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

## 第四轮补充：tau2-bench 能不能当底座（不是当被测对象）

前三轮比较的都是"通用 agent 框架能不能当底座"这个问题的候选项（Pydantic AI/Inspect AI/smolagents）。后来我们深度接入了 tau2-bench（Sierra 的 τ³-bench，一个双控客服 agent 评测框架）——但那是把它当**被测 benchmark** 接进我们已经选定的 inspect_ai 底座（完整过程见 [`tau2_bench_integration_findings.md`](./tau2_bench_integration_findings.md)，可复用方法论见 [`benchmark_integration_playbook.md`](./benchmark_integration_playbook.md)）。这次接入过程中读了 tau2-bench 相当一部分核心代码（`Orchestrator`/`Environment`/`Evaluator`/`registry`），足够反过来认真回答一个前几轮都没问过的问题：**tau2-bench 自己的 harness 机制，能不能反过来当底座、取代 inspect_ai？**

答案是不能。逐项对照本文"选型标准"一节的五条标准，附真实代码证据（不是读文档/印象，全部是读源码得出的结论）。

### 逐项核实

**① 能不能重建完整 Agent execution trace——部分具备，但缺执行拓扑**

`SimulationRun`/`Message` 数据模型（`src/tau2/data_model/simulation.py`、`src/tau2/data_model/message.py`）本身相当完整，含 cost、usage、`raw_data`（litellm 原始响应全量保留）、review 等字段，这是 tau2-bench 现有能力里离"可用"最近的一块。但没有 per-step 结构化耗时/并行标记：`Orchestrator._execute_tool_calls()`（`orchestrator.py:313-329`）即使一次 assistant 消息里有多个 tool_call，也是 `for` 循环顺序执行，没有"这批调用是否并行发起"的元数据；重试信息完全不落盘——`run_with_retry()`（`runner/progress.py:19-52`）对整次 simulation 做 try/except 重试，重试次数/原因只打印到 console/日志，`SimulationRun` 类里没有对应字段，"这个任务重试了几次才成功"这件事在最终产物里完全看不出来。

**② 能不能做 token / model-invocation / episode 三层成本归因——episode 层有，另外两层几乎空白**

episode 层（每次 simulation 的总 cost/reward，`metrics/agent_metrics.py`）可以直接复用。但 model-invocation 层（TTFT/ITL/queueing）完全没有，而且不是"没顾上加字段"，是架构上不支持采集：唯一的模型调用入口 `llm_utils.generate()`（`llm_utils.py:406-418`）用的是 litellm **同步非流式** `completion()`，`generation_time_seconds` 是"发请求到拿到完整响应"的单一标量，没有逐 chunk 时间戳可拆。语音模块里唯一貌似细粒度的 `response_latency_mean`/`yield_latency_mean`（`metrics/voice_interaction_metrics.py`）衡量的是"对话轮转"延迟（谁等谁说话），数据源是离散仿真 tick，不是真实模型 API 的流式 token 输出——`config.py:82-86` 的 `DEFAULT_TEXT_STREAMING_CHUNK_BY = "words"` 证实所谓"流式语音"其实是把已经完整生成好的文本按词切块模拟语速，跟真实 TTFT 无关。token 层同样被截断：`get_response_usage()`（`llm_utils.py:134-141`）只从 litellm 的 `Usage` 对象里取 `completion_tokens`/`prompt_tokens` 两个数字，reasoning/cache 相关的细分字段（很多 provider 的 `usage.completion_tokens_details.reasoning_tokens` 等）被显式丢弃——虽然完整原始响应保留在 `raw_data` 里理论上可以二次解析，但 tau2 自己的统计链路（`get_token_usage()`、`agent_metrics.py`）完全不读这部分，等于要绕开现有抽象自己重做一层。

**③ 能不能提供标准化的 context/generation/runtime/tool 干预点——没有等价的 Hooks 机制**

`registry.py` 的 `register_domain`/`register_agent_factory`/`register_user` 是"接入新领域/新实现"的组件注册表（本质是几个 `Dict[str, Callable]`，`registry.py:70-275`），跟 inspect_ai 那种"第三方包通过 `entry_points` 自动挂进任意一次运行、观测每一次模型调用"的插件机制不是一回事——全仓库搜索 `entry_points` 零命中，`pyproject.toml` 只声明了一个 CLI 入口。全仓库范围搜 `hook`/`observer`/`listener`/`callback`，命中的要么是语音全双工模式的打断/背景应答逻辑（`agent/base/streaming.py`，领域特定，跟通用 instrumentation 无关），要么是借用 litellm 自带的 `success_callback` 接 Langfuse（`llm_utils.py:66-69`，第三方可观测性 SaaS 的挂钩，不是 tau2 自己暴露的接口）。想做统一的跨领域 instrumentation，唯一现实路径是直接 monkeypatch `llm_utils.generate`——侵入式改造，不是插件化接入。

**④ 能不能支持 online execution 与 offline replay——online 没问题，replay 名不副实**

`checkpoint.py` 的 `--save`/断点续跑机制解决的是"避免重复计算"（跳过已经跑完的 `(trial, task_id, seed)` 组合），不是"固定住某一层、只重放另一层"；`evaluate_trajectories.py` 能对已存的静态消息序列重新跑 evaluator 算分（评估标准变了之后重新打分），但完全不涉及重新调用 LLM/agent/user，帮不上"固定 user simulator 回复、只让 agent 侧变量做对照实验"这个目标四的核心诉求。全仓库搜 `replay`/`cassette`/`fixture` 零命中。litellm 的响应缓存（`LLM_CACHE_ENABLED`）在 prompt 完全一致时能顺带实现某种"确定性重放"，但这是缓存的副作用不是设计意图，只要对话历史分叉一步就失效，也没法只固定某一层放开另一层。

**⑤ 代码质量、测试、维护状态——活跃但量级和保障远小于 inspect_ai**

`git log` 显示 185 commits（对比 inspect_ai 6952 commits，约 1/37）、13 个贡献者、项目历史约 14 个月，近 90 天 57 次提交——**活跃度不低，不是废弃项目**，这一点要如实肯定。但核心 Python 包 `src/tau2/` 完全没有对应的 GitHub Actions CI（`.github/workflows/` 下 3 个 workflow 全部只覆盖 leaderboard 前端），60 个测试文件、约 2.1 万行测试代码是真实存在的,但执行依赖本地 `pre-commit` hook（`.pre-commit-config.yaml` 里 `entry: make check-all`），不是服务端强制 gate，`git commit --no-verify` 就能绕过。

### 一个额外的架构性限制：同步执行核心

`BaseOrchestrator.run()`/`step()`（`orchestrator.py:260-291`）是普通 `def`，核心循环里没有一处 `await`；模型调用走的是 litellm 同步 `completion()` 而非 `acompletion()`。并发的唯一来源是跨 simulation 的线程池（`runner/batch.py:810`），单次 simulation 内部完全串行。连 FastAPI 服务层的 `async def` 端点（`api_service/simulation_service.py:42-52`）内部也是直接同步阻塞调用，没有 `await`/线程池 offload——这就是为什么这次接入 tau2-bench 时，必须用 `anyio.to_thread.run_sync` + `anyio.from_thread.run` 做同步/异步桥接才能让 inspect_ai 的异步 Hooks 生效（见 `tau2_bench_integration_findings.md`）。反过来想："如果拿 tau2-bench 的 Orchestrator 当异步 instrumentation 的载体"，需要把 `step()`/`Environment.get_response()`/`llm_utils.generate()` 整条链路重写成 `async def`，这基本等于重写整个执行引擎。

### 一个跟通用性直接相关的限制：Domain 是重量级、领域绑定的概念

加一个新 domain 需要 `data_model.py`（DB 子类）+ `tools.py`（工具集）+ `environment.py` + `policy.md`（自然语言策略文本）+ `tasks.json` 全套（`domains/README.md`），即便是专门做到最小的 `mock` domain 也有约 338 行。这套抽象天然假设"DB 状态 + 工具 API + 自然语言政策 + 多轮客服对话"——想拿它跑我们已经跑过的 GSM8K（单轮数学问答，没有 DB/工具/user simulator）或 BFCL（纯函数调用准确率，通常不需要多轮 user 交互），都得硬造一堆不符合它原生语义的空壳组件，工作量接近"用一个为客服场景设计的框架削足适履"，而且 tau2-bench 没有 inspect_evals 那样的第三方 benchmark 生态——它自己内置的 4-5 个领域（`mock`/`airline`/`retail`/`telecom`/`banking_knowledge`）就是全部，`CONTRIBUTING.md` 鼓励的贡献方式是把新 domain 直接合并进这个仓库,而不是独立发布成可即插即用的第三方包。

### 结论

tau2-bench 是一个为"双控客服 agent 评测"精心设计、执行成熟的**同步、领域绑定型 benchmark 框架**——它在"跑通一个客服模拟任务并按 DB/沟通/工具调用维度打分"这件事上做得很扎实（`Orchestrator`/`Evaluator`/checkpoint-resume 都是生产级质量），这也是这次能够顺利把它接进来当被测对象、复用它的 `Environment`/`Evaluator` 不用重新发明评分逻辑的原因。但它从设计初衷上就不是"通用 agent 执行观测底座"：没有 Hooks/插件系统，model-invocation 级指标因为同步非流式调用架构而拿不到，token 归因被现有抽象截断，replay 能力名不副实，Domain 概念绑定了具体业务场景、没有第三方生态。要把它改造成能替代 inspect_ai 的底座，意味着要**从零建 Hooks/插件系统 + 从零建流式 TTFT/ITL 采集 + 从零建细粒度 token 归因 + 从零把执行核心异步化 + 从零建轻量任务接入层**——这五项改造叠加在一个 185 commits、核心包无 CI 强制门禁的中等规模项目上,风险和工作量都明显高于继续用 inspect_ai（已有 Hooks、已有异步核心、已有 inspect_evals 生态）、把 tau2-bench 仅作为被测 benchmark 接入的现有路线,跟第二轮对 smolagents 的结论（"不建议作为通用底座,但适合当研究对象"）是同一类判断,只是这次是事后用真实接入经验反向验证,而不是选型阶段的预判。

tau2-bench 值得保留使用的部分，仅限于它作为**被测 domain**时提供的高质量客服场景任务集和四维评测标准（ENV/ACTION/COMMUNICATE/NL_ASSERTIONS），以及它的 `SimulationRun`/`Message` 数据模型可以作为"如何记录一次完整多轮 agent 交互轨迹"的设计参考。

## 相关文档

- [Efficient Harness 总体目标与设计](./efficient-harness.md)
- [Inspect AI 路线与现状分析](./inspect_ai_roadmap.md)
- [本地与远程服务器同步指南](/home/liuyingen/code/doc/sync-guide.md)（通用指南，不专属于这个项目，所以没有跟着这次重构搬过来）
- 目标一实现：`/home/liuyingen/code/efficient-harness/inspect_trace/README.md`
- [tau2-bench 接入的完整过程与真实结果](./tau2_bench_integration_findings.md)（第四轮补充的调研基础，把 tau2-bench 当被测 benchmark 接入的详细记录）
- [接入新 benchmark 操作手册](./benchmark_integration_playbook.md)（"当被测对象接入" vs "当底座"这两件事的方法论都在这篇里，本文第四轮是后者的具体案例）

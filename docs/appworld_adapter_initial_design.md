# AppWorld Adapter 设计与验证

> 状态（2026-08-11）：0→1 执行接入、P0 测量闭环和完整 57 题 dev baseline 均已完成。
> 当前 adapter 已达到 baseline 采集与 success-conditioned 效率分析可用；尚未进入受控干预阶段。
> 下一项代码工作仍是 P1-1 的上游 tool-menu 参数开放。

## 压缩后的核心目标

我们不是要“复现 AppWorld 排行榜”，而是要把 AppWorld 接成 harness 的第三种真实 workload：

> 在不改变 AppWorld 任务、状态转移和评分语义的前提下，让目标模型的所有推理与真实工具执行进入 Inspect 事件流，从而度量“完成任务需要付出的模型与工具成本”。

研究对象始终是 harness：

\[
\text{Efficiency} \mid \text{Task Success}
\]

也就是只有在任务正确完成或达到相同质量时，延迟、token、prefill、上下文增长、工具并发等指标才有比较意义。

## 第一性原理：我们真正要得到什么

主仓库的目标不是积累 adapter，也不是把 benchmark 跑到某个样本数。`inspect_trace`、benchmark
和本地模型共同组成一件研究仪器，最终必须完成下面的因果链：

```text
真实 closed-loop workload
  → 重建完整 action / observation / state trajectory
  → 在 token、model invocation、episode 三层定位瓶颈
  → 提出只改变一个因素的干预
  → 在相同质量约束下比较 online 结果
  → 必要时固定 trajectory 做 offline replay，排除轨迹随机性
  → 得到可跨 workload 检验的效率结论
```

因此 AppWorld 的职责不是“第三个能跑的 benchmark”，而是补齐仓库此前缺失的实验条件：

- 比 BFCL 更长、更状态化，足以让 observation 增长、重复 prefill 和失败恢复成为主效应；
- 比 tau2 更可控，不依赖用户模拟模型，并能把目标模型和真实工具执行放进同一事件链；
- 比 ToolSpec 更接近完整 agent episode，能够检验单个 generation 加速最终能否转化为
  end-to-end、success-conditioned 收益；
- 初始状态、工具结果和最终评分均由本地程序决定，适合后续在线干预和固定轨迹重放。

AppWorld 是否接入成功，应由“能否支持上述归因和对照实验”判断，而不是由目录是否齐全、
5 题是否无异常或 57 题是否跑完判断。

## Harness 核心实践落地状态

这里的“嵌入到位”指评测控制面、事件链和官方语义已经进入真实执行路径，而不是运行结束后再拼接日志。当前结论是：**执行面和 baseline 研究语义已经嵌入，受控干预与配对比较尚未完成。**

| 核心实践 | 状态 | 当前证据与边界 |
|---|---|---|
| Inspect 持有评测控制面 | 已落实 | `Task`、dataset、solver、scorer 和 `.eval` 都由 Inspect 统一承载 |
| 上游 benchmark 保持语义权威 | 已落实 | AppWorld 继续负责 prompt/agent 逻辑、状态转移、参数校验和 `evaluate_task()`；adapter 不复制应用或 grader |
| 模型阶段进入原生事件流 | 已落实 | predictor/main 均产生真实 `ModelEvent`，通过原生 `role` 区分 |
| 工具执行进入原生事件流 | 已落实 | 被上游接受的调用通过公开 `execute_tools()` 产生 `ToolEvent`，call ID 保持不变 |
| 决策与执行可关联 | 已落实 | `ModelEvent.tool_call.id == ToolEvent.id`；AppWorld environment I/O 没有 call ID，只能按 interaction 顺序和内容交叉核对，不能声称四类产物都可按 call ID 直连 |
| predictor/main 分阶段归因 | 已落实 | 标准 token/episode 汇总按 `model_event_uuid → role` 输出 predictor/main/unknown，并用 `EvalSample.role_usage` 校验 billed usage |
| 官方成功条件优先于效率指标 | 已落实 | scorer 透传完整官方 `TestTracker`，不重算任务正确性 |
| 关键语义有确定性回归检查 | 已落实 | 除原生 `batch_execute()` 边界外，测试覆盖 provider error、invalid-name、max-call truncation、environment failure、observation 写回和终止后未消费 |
| action/observation 生命周期 | 已落实 | scorer metadata 透传逐轮 call ledger，区分 proposed/accepted/dropped/executed、Inspect transport outcome、AppWorld environment outcome 与 observation 状态 |
| 并发边界显式且不改变任务 | 已落实 | 模型可提出多调用 batch，但 AppWorld 工具始终串行；进程内锁和 `--max-samples 1` 双重保证 sample 串行 |
| 跨任务隔离 | 已落实（串行边界） | 完整 57 题无 sample/transport error 并完成状态复位；sample 并发仍不开放，也不是当前需求 |

AppWorld 的 freezegun 仍负责 benchmark 内部任务时间，但 ignore 配置同时排除 `inspect_ai` 和
`inspect_trace`。新固定 5 题中 Model/ToolEvent 与 trace `recorded_at` 均处于 2026 实际运行时间，
AppWorld 的 `DateTime.now()` 仍保持各任务设定的 2023 时间。129 个 `ModelEvent` 和 193 个
`ToolEvent` 均满足 `completed > timestamp` 且 `working_time > 0`。其他工程边界是：setup 对 OpenAI 版本做受控覆盖；
本地模型配置采用零单价，因此 token limit 有效而 cost limit 不触发；AppWorld 把环境执行失败作为
普通结果字符串返回，因此 `ToolEvent.error` 只表示 Inspect 执行层异常，不能直接代表 AppWorld API 成功。

## 当前仓库已经覆盖什么

- BFCL：结构化函数调用，偏工具选择和参数正确性。
- tau-bench：多轮对话、用户模拟器、环境状态变化，偏交互闭环。
- `inspect_trace`：采集 `ModelEvent`、`ToolEvent`，并依赖 tool-call ID 建立模型决策到工具执行的拓扑关系。
- tau2 adapter 已验证了一个可复用模式：Inspect 持有样本和模型调用，上游同步 orchestrator 在 worker thread 中运行。

三者的接入边界并不相同：

| workload | 编排循环 | 模型调用 | 工具执行 | 评分 |
|---|---|---|---|---|
| BFCL | Inspect 原生 solver | Inspect 原生 | Inspect `execute_tools()` | BFCL scorer |
| tau2 | 上游 orchestrator | agent bridge 回到 Inspect | tau2 环境直接执行，因此没有 `ToolEvent` | 上游 evaluator |
| AppWorld | adapter 持有的薄循环 | predictor/main bridge 回到 Inspect | Inspect `execute_tools()` 包装 AppWorld 执行 | 上游 `evaluate_task()` |

AppWorld 不能完全照搬 tau2：如果继续让上游 `Agent.solve_task()` 直接调用
`world.batch_execute()`，模型事件能被采集，但工具事件仍会像 tau2 一样结构性缺失。它也不应
照搬 BFCL 的完整 Inspect agent loop，因为 AppWorld 已经定义了 step、interaction、usage limit
和结束条件。正确位置是在两者之间：adapter 只持有薄循环，逐步调用上游 agent 的单步逻辑。

此前缺少的是：

> 无外部网络、无用户模拟器，但具有本地持久状态、长轨迹和确定性状态评分的工具调用 workload。

AppWorld 正好补这一格。

### “与已有 benchmark 对齐”的准确含义

对齐的是研究契约，不是实现形状。四类 workload 在仓库中的真实角色如下：

| workload | 能可靠回答什么 | 结构性不能回答什么 | 在实验矩阵中的角色 |
|---|---|---|---|
| BFCL multi-turn | 原生 tool-call/schema、局部拓扑和受控多轮成本 | 长状态闭环和跨应用副作用较弱 | micro/control |
| tau2 | 带 user simulator 和业务 policy 的交互有效性、被测 agent 的模型成本 | user、环境工具和部分评分不在统一 Inspect 事件链中 | interactive external validation |
| ToolSpec API-Bank | generation-side 方法的解码速度和输出偏离 | 单轮、无真实工具执行和状态评分，不能代表 episode 级收益 | method-specific microbenchmark |
| AppWorld Base | predictor、main、真实 API、状态变化和程序化 final-state score 的本地长闭环 | 第一版工具必须串行，不能验证 tool parallelism | primary closed-loop workload |

所以 AppWorld 当前在 **Inspect 控制面和 baseline 研究读数** 上都已对齐 BFCL/tau2：Inspect
持有 sample，模型调用进入 `ModelEvent`，上游 environment/evaluator 保持权威；标准分析器也能
按 predictor/main、success/termination 和两类失败直接汇总。尚未对齐的是干预层：当前入口只
能运行固定的 predicted/20 tool-menu policy，还不能做 baseline 与 menu-policy 的受控配对比较。

## 已实现架构

```text
Inspect Task
    │
    ▼
AppWorld adapter solver
    │
    ├─ API predictor ──► Inspect model ──► ModelEvent(role=predictor)
    │
    ├─ Main agent ─────► Inspect model ──► ModelEvent(role=main, tool_calls)
    │                                         │ call_id
    │                                         ▼
    ├─ Inspect execute_tools ─────────────► ToolEvent
    │                                         │
    │                                         ▼
    └────────────────────────────────────► AppWorld world.execute
                                              │
                                              ▼
                                  upstream AppWorld evaluator
```

### 1. 上游语义保持不动

直接复用：

- AppWorld 数据集和 split；
- `SimplifiedFunctionCallingAgent` 的 prompt、API predictor、tool-call 解析、截断和无工具回合处理；
- `AppWorld` 本地环境及状态转移；
- 官方 `evaluate_task()` 评分。

不重写 benchmark，不复制应用逻辑，不自建 grader。

不能直接复用上游 `Agent.solve_task()`，因为它在内部调用 `world.batch_execute()`，绕过 Inspect
工具执行。adapter 复用 `initialize()` 和 `next_execution_inputs_usage_and_status()`，仅接管
它们之间的执行边界，并保持以下原生循环语义：

- predictor 是第 1 个 step，main agent 从第 2 个 step 开始；
- tool calls 先经过上游的数量截断和函数名检查。需要注意：Inspect 的 `ToolCall.arguments`
  到 bridge 时已经是字典，bridge 再序列化成合法 JSON，因此 AppWorld 原生 provider 路径中的
  “非法 JSON → `{}`”fallback 在当前 adapter 路径上结构性不可触发；更早的 provider 解析失败
  必须从 `ModelEvent.error`/原始 response 观察，不能声称由 AppWorld fallback 覆盖；
- 没有 tool call 时，上游仍用 `world.execute("")` 计一次 interaction；
- 每轮 usage 继续进入 AppWorld `UsageTracker`，保留 token limit；本地模型使用零单价，cost limit 不触发；
- 每轮结束后仍用 `world.task_completed()` 和上游 max-steps 条件判断终止。

### 2. 只替换两个模型调用入口

AppWorld 实际有两种模型调用：

- API predictor：从大量 API 中筛选候选工具；
- main agent：根据候选工具进行多轮调用。

将两者的 `LanguageModel.generate()` 替换为同一个轻量 Inspect bridge，使它们都产生真实 `ModelEvent`。桥接层只负责：

- AppWorld message/tool schema 转成 Inspect 类型；
- 调用当前 Inspect model；
- 将 `ModelOutput` 转回 AppWorld 预期格式；
- 将 Inspect usage 转回 AppWorld `Usage`，避免改变 usage limit；
- 显式映射已支持的 temperature、seed、max tokens、top-p、stop、tool choice 和 parallel-tool-call 参数，并拒绝未知参数，不做盲目透传。

阶段归属不放在 sample metadata。Inspect 已原生支持 `ModelEvent.role`，bridge 分别通过
`get_model(role="predictor", default=active_model)` 和
`get_model(role="main", default=active_model)` 复用同一个被测模型并标记阶段。原始 `.eval`
日志已经包含 role；`inspect_trace` 派生记录可先用 `model_event_uuid` 回连，不修改事件模型。

上游单步方法是同步的，而 Inspect model 是异步的。只把上游单步方法放入 worker thread，
bridge 用 `anyio.from_thread.run()` 回到当前 Inspect sample 的事件循环。不能把整个
`solve_task()` 放入 worker，因为每轮工具调用还必须回到 Inspect `execute_tools()`。

### 3. 工具执行是唯一关键边界

不能只在运行结束后读取 AppWorld 日志，因为当前拓扑分析需要真实 `ToolEvent`，而且：

```text
ToolEvent.id == ModelEvent 中对应 tool_call.id
```

正式实现链路是：

1. main model 经 Inspect 返回 tool calls；
2. 将 `ModelOutput` 交还上游 agent，得到上游实际接受的 `ExecutionIO` 列表；
3. 依据 `ExecutionIO.metadata.id` 过滤和重建待执行的 assistant tool calls，保留原始 call ID；
4. 优先使用 predictor 提供的 schema；若缺失则回退到当前 task 的完整权威 API docs；合法形状但不存在的函数使用只负责执行的最小 schema，继续交给 AppWorld 产生原生环境失败；
5. 动态构造仅用于执行的 `ToolDef`，wrapper 接受原始 kwargs，由 AppWorld 作为唯一参数 validator，并显式设置 `parallel=False`；
6. 调用 Inspect 的公开 `execute_tools()`；
7. 每个工具 wrapper 调用 `world.execute()` 执行对应的原始 `ExecutionIO.content`；
8. 将工具结果转回 `ExecutionIO`，交还上游 agent 继续下一轮。

call ID 不再是未知风险：Inspect `execute_tools()` 会直接构造
`ToolEvent(id=call.id, ...)` 和 `ChatMessageTool(tool_call_id=call.id, ...)`。第一版不扩展
`inspect_trace` 接受 AppWorld 事件。

### 4. 串行执行与 interaction 语义

这里有两种不同的“并行”，必须分开：

- 模型可以像上游配置一样，在一个响应中提议多个 tool calls；
- AppWorld 原生 `batch_execute()` 实际逐个调用 `execute()`，不并发修改状态。

因此所有 AppWorld `ToolDef` 都必须 `parallel=False`。不能为了得到并发指标而并发执行这些
调用，否则数据库状态转移、返回结果顺序和 benchmark 语义都会改变。

同时，`batch_execute()` 中的多个调用虽然逐个执行，却共享一个 interaction number，并以
sub-interaction 编号。逐个直接调用 `world.execute()` 会错误地把它们计成多个 interaction。
solver 使用一个最薄的 per-turn batch 上下文，复用上游的计数规则：记录本轮开始时的
`num_interactions`，执行每个 wrapper 前恢复该值并设置对应 `num_sub_interactions`，本轮结束
后清零 sub-interaction。确定性对照已经验证 interaction number、环境日志和终止状态与原生
路径一致。

### 5. Sample 隔离

AppWorld 初始化会调用全局 `AppWorld.close_all()`，并使用全局 DB cache、TestClient 和冻结
时间状态。因此 solver 使用进程内全局锁，正式运行仍显式使用：

```text
--max-samples 1
```

`--max-connections 1` 只限制模型请求并发，不能替代 sample 串行。进程内锁防止误配
`--max-samples` 时破坏状态，但不能保护多个 eval 进程。固定 5 题只证明单进程串行状态复位稳定；
若未来需要 sample 并发，应使用独立进程和独立 AppWorld 输出命名空间。

### 6. 评分边界

solver 在 AppWorld world 关闭并完成状态落盘后调用官方 `evaluate_task()`，将完整
`TestTracker.to_dict()` 保存到 sample store；Inspect scorer 只透传其中的 task success，
不重算判分。

`evaluate_task()` 在 sample event-loop 线程同步执行；不能送入 worker thread，因为 AppWorld
的 SQLite 连接与模型 cache 对象具有创建线程亲和性。模型生成所需的上游同步单步逻辑仍按
第 2 节所述放入 worker。

`scenario_goal_completion` 不是单题指标。`dev --limit 1` 只验 task-level `TestTracker`；多题
完成后再用 AppWorld `Metric.compute_metrics()` 对所有官方 TestTracker 聚合 task goal
completion 和 scenario goal completion。

### 7. 最小目录形态

```text
appworld_adapter/
├── README.md
├── pyproject.toml
├── scripts/
│   ├── setup.sh
│   └── run_adapter.sh  # 唯一正式入口；唯一命名并拒绝复用输出目录
├── src/appworld_adapter/
│   ├── task.py       # dataset、Task、官方 scorer
│   ├── model.py      # predictor/main 的 Inspect bridge
│   └── solver.py     # 最薄的 AppWorld 闭环
└── tests/
    ├── test_execution.py  # 原生/adapter 执行边界等价性
    ├── test_model.py      # 最小模型格式桥接检查
    └── test_run_adapter.sh # full 默认值和目录冲突保护
```

不建立通用 benchmark adapter 基类，也不预先抽象事件协议。只有确认 BFCL、tau2、AppWorld 存在稳定共性后再抽象。

## 第一轮验收结果

先用固定 task ID 跑一遍 AppWorld 原生 function-calling 路径作为参照，再做正式 adapter 的
`dev --limit 1 --max-samples 1`，不另建 spike。以下条件均已满足：

- [x] predictor 和 main agent 都产生带正确 role 的 `ModelEvent`；
- [x] 每个经上游过滤后实际进入 AppWorld 执行的模型工具调用都产生一个 `ToolEvent`；
- [x] tool-call ID 全部正确关联，没有孤立工具事件；
- [x] 工具执行顺序、interaction/sub-interaction 编号和结果回填顺序与原生路径一致；
- [x] AppWorld 官方 evaluator 完成评分，并保留完整 `TestTracker`；
- [x] `inspect_trace` 产生非空 token、延迟和执行拓扑记录；
- [x] 每个真实模型调用都有 model-invocation/vLLM metrics；
- [x] 运行时清除代理变量，除本地被测模型 endpoint 外不访问网络；
- [x] AppWorld 原生 API log、environment I/O 与 ToolEvent 可逐项对账。

不把“API log 条数等于 ToolEvent 条数”作为无条件断言。合法形状但不存在的函数会产生一次
环境执行和 ToolEvent，却不会产生真实 API request；无工具回合还会产生没有 ToolEvent 的空
interaction。数量差异必须能归类并解释，正常已执行 API 调用则必须与对应 ToolEvent 对上。

固定多题检查已经通过。它只关闭“单进程串行跑多个 sample 会不会污染状态”这一项工程风险，
不自动授权完整 57 题运行，也不代表研究读数已经完备。

## 已验证的单样例

本地 Qwen3.6-27B AutoRound 对 `dev` 任务 `50e1ac9_1` 的串行真实运行已通过：

- Inspect eval 状态 success，官方 score 为 1.0；记录的完整 tracker 与重新调用官方 evaluator
  所得字典完全相同，2/2 tests passed；
- 11 个 `ModelEvent`（predictor 1、main 10），89 个唯一且顺序匹配的 `ToolEvent`，无 Inspect
  tool error 或孤立 call ID；
- 10 个 AppWorld batch interaction 对应 89 个有序 environment sub-interaction 和 89 条 API log；
- `inspect_trace` 写出非空真实记录，11 个 model invocation 均有 attribution=exact 的 vLLM metrics；
- `.eval` 与 trace JSONL 均非空。成功退出后的 uvicorn lifespan `CancelledError` 仅发生在
  AppWorld 内嵌服务清理阶段，不改变 eval 的 success 状态或官方评分。

修复 AppWorld 冻结时间对 Inspect 时钟的影响后，同一任务再次真实运行仍为 score 1.0，事件数量
保持 11/89；11 个模型事件的 `working_time` 为 1.28–13.90 秒，89 个工具事件为
0.004–0.128 秒，全部具有严格递增的开始/结束时间。

## 固定 5 题 scale gate

本地 Qwen3.6-27B AutoRound 对 `dev --limit 5 --max-samples 1` 的串行运行已完成：

- Inspect eval 状态 success，5/5 样本无基础设施错误；
- 官方 task accuracy 为 2/5；其中 3 题以 `task_completed` 终止，2 题达到 50-step 上限；
- 共记录 129 个 `ModelEvent` 和 193 个 `ToolEvent`，全部计时有效且 tool-call ID 唯一；
- 达到 max-steps 的样本仍完成官方评分和清理，后续样本可正常重新初始化；
- 该结果验证的是 adapter 的规模稳定性，不把当前模型的 40% task accuracy 解释为 adapter 成败。

P0-1 至 P0-4 完成后的新固定 5 题运行复现了相同的 2/5 accuracy、5 predictor/124 main 调用和
39,241/7,584、1,636,842/202,790 两阶段 billed token 总量，并补齐了正式验收读数：

- 124 个 main round 共提出、接受和执行 193 个调用，0 dropped；与 193 个同 ID `ToolEvent`
  一一对应，另有 88 个空 proposed round；
- 179 次环境成功、14 次 AppWorld environment failure、0 次 Inspect transport failure；190 个
  observation 在后续 main input 中写回，3 个因任务随该轮结束而标记为
  `not_consumed_after_termination`；
- 5 个 sample 均满足 `sample_end_to_end_latency_seconds >= model_tool_window_seconds >=
  total_busy_seconds`，并保持 interval-union concurrency identity；
- 129 条 token attribution 全部且只归入 predictor/main。vLLM metrics 为 127/129，缺失的是两个
  sample 的最终 main 调用；129 条 `ModelEvent` completion/working latency 均完整，因此该 endpoint
  末调用采集缺口不阻塞 P0-2 的阶段 model-latency 验收，缺失值也未写成 0。

更重要的是，这 5 题已经说明 AppWorld 为什么值得接入。把 `.eval` 中的 `ModelEvent.role` 与
现有 trace 按 `model_event_uuid` 事后关联后得到：

| 阶段 | 调用数 | billed input tokens | billed output tokens | repeated message tokens（估算） | repeated tool-schema tokens（估算） |
|---|---:|---:|---:|---:|---:|
| predictor | 5 | 39,241 | 7,584 | 0 | 0 |
| main | 124 | 1,636,842 | 202,790 | 781,782 | 396,702 |

这不是正式总体统计，但已经给出一个必须继续验证的真实假设：BFCL 上重复 tool schema 是重复
message 的 37 倍；AppWorld 这 5 题上方向反转，main 阶段的重复 message 约为重复 schema 的
1.97 倍。也就是说，不同 workload 的主要瓶颈可能确实不同，AppWorld 已经开始完成它的研究补位。

同一批数据还暴露了当前读数缺口：193 个 `ToolEvent` 中有 14 个 AppWorld 环境执行结果以
`Execution failed...` 开头，但 193 条 `action_parsing` 记录的 `error_present` 全为 false。
原因不是环境没有失败，而是 AppWorld 把应用/参数失败作为 observation 字符串返回，Inspect
执行 wrapper 本身成功。若直接使用现有错误字段，会得出错误的 action reliability 结论。

## 完整 57 题 dev baseline

2026-08-11，本地 Qwen3.6-27B AutoRound 通过正式 `scripts/run_adapter.sh` 完成完整 `dev`：

- Inspect `.eval` 为 `success`，57/57 completed、57/57 scored、0 sample error；总耗时 2:17:14；
- 官方 task accuracy 为 35/57 = 0.6140（stderr 0.0651）；47 题以 `task_completed` 结束，
  其中 35 题通过、12 题未通过；另 10 题达到 `max_steps`，均未通过；
- 57 次 predictor 和 1,023 次 main 调用共消耗 15,889,119 input、1,202,125 output tokens；
- 2,005 个 proposed call 全部 accepted/executed，0 dropped、0 provider-error round、0 Inspect
  transport failure；260 次 AppWorld environment failure 是进入 observation 的 agent/application
  结果，不是 harness 基础设施错误；
- 57 个 trace sample 全部被标准 token/episode 汇总关联，`unknown` role 为 0；成功 episode 平均
  173,587 tokens、70.1 秒，失败 episode 平均 500,713 tokens、262.7 秒；
- main 的 repeated-message 估算为 6,629,047 tokens，repeated-schema 为 2,556,829，继续支持
  AppWorld 更偏 history/observation-dominated 的工作假设；
- vLLM metrics 覆盖 1,075/1,080 次模型调用；缺失的 5 个 main 调用保留为 missing，不写成 0。
  Inspect event latency、token、score 和 episode join 均完整，因此这是小范围 invocation-metric
  缺口，不阻塞 baseline 使用。

正式产物位于 `runs/appworld-dev-full-20260811T013633Z`。tmux 下曾因 Inspect 自动选择 Textual
display，与 AppWorld 进程全局 safety guard 的文件写保护发生冲突；正式入口现固定
`--display plain`。失败尝试没有复用，后续全量使用全新的 run/experiment 目录并正常退出。

## 0→1 已经证明什么，尚未证明什么

已经证明：

- Inspect 可以持有 AppWorld sample，并在同一执行上下文中采集 predictor/main 模型调用；
- 经上游接受的 tool call 可以保持 call ID，经 `execute_tools()` 进入 AppWorld 状态机；
- 单轮 batch 的顺序、sub-interaction、输出和状态语义与原生 `batch_execute()` 一致；
- 官方 final-state evaluator 可以在 adapter 路径上完成评分；
- 完整串行 dev 可以复位，且真实轨迹足以产生 BFCL 没有暴露的 context-growth 信号；
- token/latency 已形成 57 题 success-conditioned 分布，可区分快速失败和真实效率收益。

尚未证明：

- 任一加速方法在相同质量约束下改善了 AppWorld episode；
- online 结果中的差异来自干预本身，而不是模型生成了不同 trajectory。

## 1→100：从可运行 adapter 到研究闭环

“1→100”不是把 `--limit 5` 改成 `--limit 57`，而是依次关闭下面四个证据缺口。

### A. 先让读数符合 AppWorld 的真实结构

1. **阶段归因**：标准 token/model-invocation/episode 分析必须按 `ModelEvent.role` 分出
   predictor 与 main，同时保留总 episode 数。AppWorld 的 API retrieval 成本和主循环成本不能
   混成一个“LLM 调用均值”。优先复用 `.eval` 中已有 role，以 `model_event_uuid` join；没有理由
   在 adapter 再造一套 stage event。
2. **action 生命周期**：对每个 main `ModelEvent` 区分 proposed、accepted、executed、
   environment-failed、written-back；记录被函数名过滤、max-call 截断和空调用的去向。Inspect
   provider 先于 AppWorld 发生的 tool-argument 解析失败另列，不能误写成 AppWorld JSON fallback。
   Inspect transport error 与 AppWorld application error 必须分列。
3. **时间语义**：保留 AppWorld 冻结的任务时间，同时让 Inspect 和 `inspect_trace` 的观测时间都
   使用真实时钟。该冻结问题现已修复，并由确定性测试和新固定 5 题共同验证。
4. **统计边界**：明确 episode latency 是否只覆盖首个 model event 到最后一个 model/tool event，
   还是包含 prompt construction、world initialization 和 evaluator。当前 `episode_layer.py` 采用前者，
   文档和报表必须使用准确名称，不能把未测的初始化/评分时间算进“端到端”。

这四项已在 P0 完成，并由完整 57 题 baseline 再次验证；后续扩大到其他 split 或干预 run 时
继续沿用同一口径。

### A.1 已落地的 P0 代码工作（完整 dev 之前）

下面四项是 full baseline 的前置条件，不是可选优化。实现时修改已有边界，不新增 controller、
adapter 基类或 AppWorld 专用 trace 框架。

#### P0-1：修正观测时钟（已完成）

- **修改位置**：`appworld_adapter/src/appworld_adapter/model.py`；回归检查放进现有
  `appworld_adapter/tests/`。
- **原问题**：`freezegun.configure()` 只忽略 `inspect_ai`，所以 `.eval` 中 Model/ToolEvent
  使用真实时钟，但 `inspect_trace.hooks._now_iso()` 仍被固定为 AppWorld 的 2023 任务时间。
- **已实现**：在同一个 freezegun ignore 配置中同时排除 `inspect_ai` 和 `inspect_trace`；
  AppWorld 自己的任务时间仍保持冻结，不修改 benchmark 时间语义。
- **验收**：真实运行中 ModelEvent、ToolEvent 和 trace `recorded_at` 都处于实际运行时间；
  AppWorld 任务内依赖 `DateTime.now()` 的结果仍等于任务设定时间。

#### P0-2：把 predictor/main 纳入标准三层归因（已完成）

- **修改位置**：`inspect_trace/src/inspect_trace/analysis/_loader.py`、`token_layer.py`、
  `episode_layer.py`，以及现有 `inspect_trace/tests/test_analysis_layers.py`。
- **原问题**：raw `.eval` 已有每个 `ModelEvent.role`，`EvalSample.role_usage` 甚至已经提供按
  role 聚合的真实 usage；但 trace 汇总丢掉了这层维度，当前 5 题数字来自一次性 join 脚本。
- **已实现**：复用 manifest 已有的 `.eval` 路径，在离线分析时建立
  `model_event_uuid → role` 索引。真实 billed usage 优先使用/校验 `role_usage`；schema、history、
  reasoning、tool-call、vLLM latency 等细项按 UUID join 到 predictor/main。保留现有 episode 总计，
  不向 adapter 增加重复的 stage event。
- **验收**：标准分析入口直接复现当前 5 题的 predictor 5 次/main 124 次及对应 token 总量；
  所有带 UUID 的 model 派生记录都能归入且只归入一个 role，未知 role 显式报告而不是并入 main。

#### P0-3：记录完整 action 生命周期和两类失败（已完成）

- **修改位置**：`appworld_adapter/src/appworld_adapter/model.py`、`solver.py`、`task.py`；
  扩充现有 `tests/test_execution.py`/`test_model.py`。`inspect_trace/action_parsing.py` 保持其
  “Inspect transport/parsing error”原义，不把 AppWorld 字符串错误硬塞进去。
- **原问题**：`ModelEvent` 保存 provider 提出的调用，`ToolEvent` 只保存最终执行的调用，二者
  之间的上游截断/函数名过滤没有结构化原因；同时 AppWorld 的执行失败以
  `Execution failed...` observation 返回，`ToolEvent.error` 仍为 null。
- **已实现**：bridge 保留 main model 本轮的原始 tool-call 快照；solver 在上游单步返回后，
  按 call ID 记录 proposed、accepted、dropped（至少区分 invalid-name、max-call truncation）、
  executed 和 AppWorld environment outcome，并在下一次 model input 中核对 written-back。
  这份逐轮 ledger 写入 `state.store`，scorer metadata 只做透传。应用失败与 Inspect 执行失败分列。
- **验收**：每个 accepted call 恰有一个同 ID ToolEvent；`proposed = accepted + dropped`；每个
  executed call 恰有一个 observation；当前 5 题的 14 个 `Execution failed...` 被计为 AppWorld
  environment failure，同时 Inspect transport error 仍为 0。最终一步因任务已经结束而没有再次
  送入模型的 observation 标为 `not_consumed_after_termination`，不能误报为丢失。

#### P0-4：修正 episode latency 的名称和边界（已完成）

- **修改位置**：`inspect_trace/src/inspect_trace/analysis/episode_layer.py` 及其现有测试。
- **原问题**：`end_to_end_latency_seconds` 是首个 model/tool event 到最后一个
  model/tool event 的窗口，不包含首个事件前的 world/prompt 初始化，也不包含末事件后的 evaluator；
  raw `EvalSample` 已有 `started_at`、`completed_at`、`total_time` 和 `working_time`，但分析器未使用。
- **已实现**：把该口径明确命名为 `model_tool_window_seconds`；另用 `EvalSample.total_time`
  报告 sample end-to-end latency，并保留 `working_time`。现有所谓 model/tool waiting 字段实际是
  相邻阶段间空档，应报告为 `model_to_tool_gap_seconds`/`tool_to_model_gap_seconds`，不能解释成
  某一方真实占用资源等待。
- **验收**：`sample_end_to_end >= model_tool_window >= total_busy`；并发合成用例仍满足
  `naive busy sum - interval union = concurrency savings`；现有 BFCL 分析不会因没有 role 而失败。

### A.2 代码层完成门槛

P0 完成不是“文件写完”，而是同一条 5 题运行能由仓库正式入口直接产出：

- predictor/main 分阶段 token 与 model latency；
- proposed/accepted/executed/environment-failed/written-back 计数；
- Inspect failure 与 AppWorld failure 分列；
- sample end-to-end、model/tool window、busy time 和 transition gap 四个不混淆的时间口径；
- 全部真实观测时间未被 AppWorld freezegun 污染。

只扩充现有测试中的关键断言，再复跑固定 5 题；不为这些条件建设新的 gate runner。上述门槛
先由固定 5 题关闭，随后完整 57 题 baseline 已通过正式入口完成。

### B. 建立 AppWorld baseline，而不是只报一个总 accuracy（已完成）

完整 dev 已提供逐 episode token、latency、termination、action lifecycle 和官方 score，可直接按
success/failure 分层；上节记录的是第一份 baseline 事实。后续 report 仍应保留下列维度，不能退化
成单一 accuracy：

在固定 dev task 集合上报告分布，并始终按 success/failure、termination reason 分层：

- predictor 与 main 各自的 calls、tokens、TTFT、decode/e2e model latency；
- 每步工具菜单大小、schema token、history/observation token 和随 step 的增长曲线；
- proposed/accepted/executed/failed action 比率及失败恢复成本；
- model/tool busy time、两阶段间 gap、episode latency；
- task goal completion；有完整 scenario 集合时再报告 scenario goal completion；
- accuracy–cost Pareto 或 success-conditioned cost，绝不把快速失败当成更高效率。

完整 57 题 dev 已作为这一阶段的 baseline 数据集；它关闭了规模和分布证据，但不替代下一阶段
的配对干预实验。

### C. 用 AppWorld 做第一个受控干预，而不是继续新增 adapter

首选干预应利用 AppWorld 上游已经存在的机制：API predictor 的 `predicted`、`ground_truth`、
`all` 模式和 `max_predicted_apis`，而不是新建通用插件框架。围绕同一批 dev 任务比较：

```text
tool-menu policy / budget
  → predictor 开销
  → main 每步 schema 与 repeated-prefill 开销
  → action 可用性、轨迹长度和最终 success
```

这个实验直接检验 AppWorld 独有、也是仓库当前最重要的系统问题：花一次 retrieval/prediction
成本换取更小工具菜单，是否真的降低整条 episode 成本，以及收益会不会被漏掉关键 API 导致的
成功率下降抵消。`ground_truth` 只作为 dev 上的 oracle 下界，不作为可部署方法；`all` 需要明确
是否仍受 `max_predicted_apis` 截断。

第二个优先实验才是 prefix caching：固定已记录的 prompt 序列，对同一序列比较 cache on/off，
把 serving 收益与 trajectory 变化分开。这是目标四 offline replay 的具体用例，不应先建设一个
没有实验消费者的通用 replay 系统。只有 baseline 继续确认 observation/history 是主导项后，
才进入 context compaction；只有 tool-call generation 的占比足够大时，才把 ToolSpec 类方法搬到
AppWorld episode 上评估 Amdahl 上限。

### C.1 受控实验前必须落地的代码工作

#### P1-1：开放上游已有的 tool-menu 干预参数

- **修改位置**：`appworld_adapter/src/appworld_adapter/solver.py`、`task.py`、README 和现有测试。
- **当前问题**：`_build_agent()` 把 predictor mode 固定为 `predicted`、候选上限固定为 20，
  因而当前入口只能重复 baseline，无法运行设计中最重要的 menu-policy 对照。
- **必要修改**：把 `api_predictor_mode` 和 `max_predicted_apis` 作为 Inspect task 参数传到上游
  `APIPredictor`；只接受上游已有的 `predicted|ground_truth|all`，不再发明策略接口。将实际 mode、
  budget、最终暴露工具数写入 score metadata，使实验产物自描述。
- **验收**：同一 task 可分别运行 predicted、ground-truth oracle 和 all；predicted 有一个
  predictor ModelEvent，非 predicted 模式不伪造 predictor 调用；main ModelEvent.tools 与上游最终
  菜单完全一致。`all` 的含义连同 `max_predicted_apis` 截断必须在产物中显式记录。

#### P1-2：形成一个正式的 AppWorld 对比汇总入口

- **修改位置**：优先扩充 `inspect_trace/analysis/` 的现有汇总；只有现有入口无法表达
  accuracy–cost Pareto 时，才增加一个薄的 AppWorld report 脚本。
- **当前问题**：单 run 的 57 题 token/episode 和 success/termination 分层已经可由标准分析入口
  复现，但还没有把两个 run 按 task ID 配对并报告 success flip/Pareto 的正式入口。
- **必要修改**：消费 P0 的标准输出和 scorer metadata，以 task ID 对齐 baseline/intervention，
  报告 success flip、阶段成本变化、菜单大小、轨迹长度和 success-conditioned cost。不要生成新的
  dashboard 或实验数据库；JSON/CSV 加终端表格足够。
- **验收**：给定两个 run 目录，一条命令产生逐 task 配对结果和聚合 Pareto；任务集合、模型、
  seed、mode/budget 不一致时拒绝或明确标注，不能悄悄比较不同样本集合。

#### P1-3：清理可复现安装边界，但不为 provider 新造框架

- **修改位置**：`appworld_adapter/pyproject.toml`、`scripts/setup.sh`、README。
- **当前问题**：AppWorld 运行实际依赖 `inspect_trace`，本机命令又借用了名为
  `tau2-agent-vllm` 的 provider；二者都由 setup 脚本 `--no-deps` 注入而未出现在项目依赖中，
  `uv sync` 本身不能复现可运行环境。OpenAI 版本覆盖也只存在于 shell 命令中。
- **必要修改**：把真实依赖和 OpenAI 版本覆盖理由变成单一、可检查的安装路径。短期可以显式复用
  现有 no-strict provider，但必须把该耦合写进依赖和命令；只有 BFCL/tau2/AppWorld 确认共同需要
  同一个 provider 后，才把它移动并改成 benchmark-neutral 名称，不能现在先建 provider 框架。
- **验收**：从干净 venv 执行文档唯一的 setup 命令后，pytest 和 1 题运行无需额外手工安装；
  `inspect model` 能列出 README 指定的 provider。

### C.2 明确暂不实现的代码

- 不实现通用 replay framework。prefix-cache 实验立项时，只实现一个消费者：读取已记录的
  `ModelEvent.input/tools/config`，按原顺序向同一 provider 重放，并验证请求 hash 一致。
- 不实现 AppWorld tool 并发。上游 `batch_execute()` 是有序状态转移，强行并发会改变 benchmark。
- 不实现 full-run controller、恢复状态机或 dashboard；Inspect 自己的 eval/log 已经覆盖运行管理。
- 不把 AppWorld application error 塞进 `ToolEvent.error`，否则会破坏 Inspect transport error 的
  跨 benchmark 可比语义。

### D. 形成跨 workload 结论

最终交付不应是“AppWorld accuracy/latency 表”，而应回答：

- BFCL 的 schema-dominated repeated prefill 是否会在 AppWorld 变成 history/observation-dominated；
- menu retrieval、prefix cache、context compaction 分别在哪类 workload 有收益；
- 单次 generation 加速在完整 episode 上被 model/tool/trajectory 哪一部分稀释；
- 上述收益在 tau2 的交互式、带用户模拟条件下是否仍成立。

至少完成一个受控干预并得到 success-conditioned、可复核的结论，AppWorld 才从“0→1 adapter”
进入了“1→100 research workload”。

## 当前判断

当前已经打通下面这一条纵向链路：

```text
main model tool call → Inspect ToolEvent → AppWorld 状态变化
```

执行链路、P0 测量闭环和完整 dev baseline 均已成立，但完整研究链路尚未成立。下一项代码工作
是 P1-1：仅开放上游已有的 `api_predictor_mode` 与 `max_predicted_apis`，随后用同一 57 题集合做
baseline/tool-menu 配对干预；在此之前不新增 adapter、dashboard 或通用 replay 框架。

这里的“成立”不仅是状态最终发生变化，还包括：调用经过上游过滤、Inspect 事件 ID 可关联、
AppWorld batch interaction 语义未改变、官方 evaluator 对同一落盘状态完成评分。

# 接入新 benchmark 操作手册

这份手册回答一个会反复出现的问题：**遇到一个新的 benchmark，想用我们的 harness 做效率画像（目标一/二），该怎么接？** 不是每次都要重新想一遍架构——tau2-bench 已经把这套方法论真实验证过一次（[`tau2_bench_integration_findings.md`](./tau2_bench_integration_findings.md) 是完整过程记录），这篇手册把可复用的部分提炼出来，下次直接照着走前置条件检查，不用从头推导。

**这不是"造好一个通用适配器工具，以后接新 benchmark 零成本"**——每次还是要写真的代码、踩真的坑（这篇手册也如实列出了会反复遇到的那几类坑）。它省的是"该怎么设计"这一步的思考成本，不是实现成本。

## 第一步：分类，不是所有 benchmark 都一样难

### 情况一：benchmark 本来就建在 inspect_ai / inspect_evals 之上

比如 BFCL、GSM8K。**零适配器工作**——`inspect_trace` 的 Hooks 自动生效，因为它本来就在 inspect_ai 自己的 eval 循环里跑，直接用 `run_bfcl_benchmark.sh` 那套脚本模式复制一份即可。大多数标准 benchmark（`inspect_evals` 目录里几十个）都落在这一类。如果目标 benchmark 属于这一类，这篇手册剩下的内容都不需要看。

### 情况二：完全独立的框架，但满足三个前提

这是 tau2-bench 的情况，也是这篇手册主要讲的内容。往下看三个前提检查。

### 情况三：不满足情况二的前提

见文末"情况三：退化方案"一节——能做，但代价和局限都更大，如实说明,不要假装能做到情况二的效果。

## 第二步（情况二）：三个前提检查

三个全部满足，才值得走完整的适配器流程；有一个不满足，先看会带来什么代价，再决定要不要做。

### 前提 A：目标 benchmark 有没有干净的库入口，不是只有 CLI

检查方法：找一个"跑一次模拟 + 拿到真实评分"的 Python 函数，不经过 argparse/CLI 解析。tau2-bench 有 `tau2.runner.simulation.run_simulation(orchestrator)`——直接 import 调用，跑完就有真实 `reward_info`。

**如果没有**：只能退化成"包一层子进程 + 解析日志"，Hook 级别的细粒度追踪会丢失（子进程里发生的模型调用不在 inspect_ai 自己的 Sample 执行上下文里，`inspect_trace` 拿不到任何东西）。

### 前提 B：agent 那一侧是不是可插拔的

检查方法：有没有一个 base class/protocol，只要求实现"给定输入生成输出"这一个方法，其余编排逻辑（跟环境交互、跟其他角色交互）完全跟具体 agent 实现解耦。tau2-bench 的 `tau2.agent.base_agent.HalfDuplexAgent` 就是这样——只要实现 `get_init_state()`/`generate_next_message()` 两个方法。

**如果没有**（agent 逻辑硬编码死在 orchestrator 内部换不出来）：只能自己重新实现一遍编排循环，风险明显变大——容易做成"看起来接上了，但已经不是同一个 benchmark 的语义"，这正是我们从一开始就想避免的（复用对方的 evaluator，不重新发明）。

### 前提 C：消息/工具的 wire format 跟 OpenAI-style chat/tool-calling 差多远

检查方法：对方的模型调用是不是也走 OpenAI-compatible 接口或者 LiteLLM。tau2-bench 走 LiteLLM，message/tool 格式跟 inspect_ai 高度同构，转换层（`convert.py`）只有几十行,是字段对字段的直接映射，不需要语义转译。

**如果差异很大**（比如对方是纯 RL gym 那种动作/观测空间，不是对话式的）：转换层要重新设计，工作量明显增加，且更容易在转换过程中悄悄丢失或曲解语义。

## 第三步（情况二）：核心架构原则

1. **必须由 inspect_ai 自己的 Task/Solver 持有 Sample 执行上下文，不能反过来**。原因（已经用真实源码验证过）：`inspect_trace` 几乎全部数据都挂在 `on_sample_event` 上，这个回调只在 inspect_ai 自己的 `sample_active()` 有效时才触发——`inspect_ai/hooks/_hooks.py` 的 `emit_sample_event()` 有明确的 `if active is None: return` 守卫。"在对方的 Agent 里直接调 `inspect_ai.model.generate()`，其余用对方自己的编排循环"这条路子行不通，必须反过来。
2. **复用对方的 Environment/Evaluator，不重新实现评分逻辑**。这是保证"这仍然是同一个 benchmark"的关键——tau2-bench 这次验证过，两条路径（原生 CLI vs 适配器）用的是完全同一份 `evaluate_simulation()` 代码，分歧从来不出在评分这一层。
3. **只追踪被测 agent 一侧**，其他角色（user simulator、judge 等）留在对方自己的模型调用路径上，不接入 inspect_ai。这是明确的设计选择——避免过度扩大追踪范围导致复杂度失控，代价是"整个 episode 的完整开销画像"不包含这些角色那一半的真实成本，需要在文档里如实说明,不要含糊带过。
4. **同步/异步桥接**（如果对方的编排循环是同步的）：整个同步循环调用扔进 `anyio.to_thread.run_sync`，agent 适配器内部真正要发起模型调用时，用 `anyio.from_thread.run()` 跳回原来的事件循环——这样调用仍然发生在 inspect_ai 自己 Sample 执行所在的那个异步任务里，`sample_active()` 才认得。tau2-bench 的 `Orchestrator.run()`/`step()` 全是同步方法，这个桥接是必需的。
5. **消息/工具类型转换要"薄"**：双向转换函数只做字段对字段的映射，不引入任何语义解读或猜测。转换错了比转换不完整更危险——会悄悄改变被测 agent 实际看到的内容。
6. **撞上 wire-format 不兼容时，写自定义 `ModelAPI` 子类，不要长期活在 workaround 里**。这次先用 `emulate_tools=true` 绕开了 `strict` 字段不兼容问题，能跑但引入了一个真实的机制不对称（agent 侧文本解析 vs 其余角色原生 tool-calling），后来发现这个不对称确实在制造额外的、可归因的分歧（reward_diff 从 2/10 降到 1/10），于是补了一个继承官方 provider、只重写冲突方法的薄子类,通过 entry_points 注册。**判断要不要修的标准**：这个不对称是不是在制造额外的、可归因的分歧，不是所有不对称都值得花时间修。

## 第四步（情况二）：具体实现步骤

照抄 `tau2_adapter/` 的文件划分，这是已经验证过的组织方式：

| 文件 | 职责 |
|---|---|
| `convert.py` | 消息/工具双向转换（对方类型 ↔ inspect_ai 类型） |
| `agent.py` | 被测 agent 的适配器子类（实现对方的 agent 接口）+ 同步/异步桥接 |
| `dataset.py` | 从对方的任务数据构造 inspect_ai `Dataset`——尽量复用对方自己的任务加载函数，不要重新解析原始 JSON |
| `solver.py` | 驱动整个模拟：构造对方的 `Environment`/其他角色，实例化我们的 agent 适配器，跑对方的编排循环，拿到对方的真实评分结果，写进 `state.store` |
| `task.py` | 组装成 inspect_ai `Task`：dataset + solver + 一个"直接透传对方评分"的 `Scorer`（不重新计算） |
| `_registry.py`（按需） | 自定义 `ModelAPI` provider，走跟 Hooks 一样的 `entry_points` 机制注册 |

新项目应该是独立的 uv 项目（跟 `inspect_trace`/`local-model-server` 平级，不共享 venv），依赖 `inspect_ai`（PyPI）、`inspect_trace`（path 依赖）、目标 benchmark 自己的包（path 依赖，指向对方的本地克隆）。

## 第五步（情况二）：验证顺序，不要跳步骤

1. **先用对方自己的原生方式跑通至少 1 个任务**（CLI 或最小 Python 脚本），拿到一个真实基线结果。这是后续所有对比的参照系，也顺便验证环境本身装对了——tau2-bench 这一步就先踩了 Python 3.13 的 `audioop` 坑和 `TAU2_DATA_DIR` 路径层级的坑。
2. **写适配器，先跑单任务**，检查两件事：(a) 能不能拿到跟原生路径同量级的真实评分结果；(b) `inspect_trace` 是不是真的产出了非空数据——这是核心技术假设（同步/异步桥接能不能让 Hooks 正确触发）的验证点，**不能跳过**，跳过等于没验证过整套方案真的成立。
3. **全量跑一遍，跟原生基线逐条对比，不要只看聚合数字**。聚合数字可能因为方向相反的误差抵消而"看起来一致"——tau2-bench 第一次跑聚合 accuracy 两边都是 0.30，但逐条看只有 6/10 一致，2/10 reward 直接翻转，一升一降刚好抵消。**这个陷阱很容易踩，必须逐条核对，不能只看汇总指标就下结论。**
4. **判断发现的机制不对称值不值得修**：不对称是不是在制造额外的、可归因的分歧？修复前后重新跑一遍全量对比，用真实数字（不是手算，写代码算）验证修复是否真的有效果——这次手算错了一次（口算成"6/10 提升到 7/10"，实际脚本算出来是"6/10 没变，reward_diff 从 2 降到 1"），这也是个真实教训：**逐条对比这类结果必须用代码算，不能靠人工数**。

## 会反复遇到的坑（分类记录，来自 tau2-bench 这次真实踩坑）

### 环境/依赖类

- **目标 benchmark 用了新特性或依赖了已被移除的标准库**：tau2-bench 的 voice 模块无条件 import Python 3.13 已经移除的 `audioop`，即使完全不需要 voice 功能也会被主 `__init__.py` 的导入链拖进去。解法：换一个还没移除该模块的 Python 版本（3.12），而不是装 shim 包硬凑。
- **目标 benchmark 自己代码里可能有真实 bug**：tau2-bench 的 `to_litellm_messages()` 给每个 tool_call 多写了一个不属于 OpenAI schema 的顶层字段，新版 openai SDK 严格校验直接拒绝——这种 bug 往往只在"多轮工具调用 + 我们这条不常见的调用路径"组合下才会触发，遇到报错先怀疑自己，但排查到底之后如果确认是对方代码的问题，直接本地打补丁（前提是我们完全拥有这份本地克隆，不打算走上游 PR）。
- **`uv pip install <name>` 不认 `[tool.uv.sources]` path 覆盖**：这次手滑用 `uv pip install --reinstall-package tau2 --no-cache tau2` 装出来的是 PyPI 上一个同名但完全无关的包。改环境要用 `uv sync --reinstall-package <name>`，会正确读 `pyproject.toml` 里的 source 覆盖。
- **环境变量的路径层级容易搞错**：`TAU2_DATA_DIR` 要指到 `data/` 这一级，不是 `data/tau2/` 那一级——每个新 benchmark 大概率有自己的一套环境变量约定，遇到 `FileNotFoundError` 先去读对方自己解析这个变量的代码，不要照抄上一个 benchmark 的假设。

### Provider 兼容性类

- **inspect_ai 的 provider 默认行为可能跟目标推理服务端的旧版本 schema 冲突**：这次是 inspect_ai 的 `openai-api` provider 无条件给工具加 `"strict"` 字段，旧版 vLLM 的请求 schema 不认识直接拒绝。遇到这类"字段层面的 400 错误"，先怀疑是不是 provider 默认加了什么目标服务端不认识的字段,再决定是绕开（比如切 `emulate_tools=true`）还是正式修（写薄的 provider 子类去掉冲突字段）——第四步已经给过判断标准。

### Non-determinism 类

- **多角色模拟（user simulator/judge 等）本身自带随机性，`temperature=0` 不保证确定性**：tau2-bench 原生 CLI 自己重复跑同一个任务两次都不一致。做对比实验之前，先测一下"同一条路径重复跑"的基线噪声有多大，不要一次性跑完就直接下结论说"harness 导致了这个差异"——差异可能压根不是 harness 造成的。要严格拆分"harness 引入的差异"和"benchmark 本身固有的随机性"，需要真正的固定轨迹重放（对应我们项目自己的目标四）或者多次重复实验取分布，不是跑一次对比一次就能定论。

## 情况三：退化方案（前提不满足时）

如果前提 A（没有干净的库入口）不满足：只能包一层子进程去跑对方的 CLI，事后解析对方自己产出的日志/结果文件（tau2-bench 的 `results.json` 就是这样的格式）。能拿到最终评分结果，**拿不到任何 Hook 级别的细粒度数据**（token 归因、执行拓扑等目标一二的核心产出全部没有）——如实把这个降级说清楚，不要包装成"接上了"。

如果前提 B（agent 不可插拔）不满足：只能重新实现一遍编排循环逻辑，照抄对方的语义（读对方源码，逐步复刻状态机）。工作量和风险都明显更大，做完之后要格外仔细地做"逐条结果对比"这一步（第五步），因为重新实现的循环逻辑跟对方原始实现产生细微偏差的可能性比"复用对方编排循环"这种情况高得多。

如果前提 C（消息格式差异很大）不满足：转换层要重新设计,可能没法做到"薄",需要投入更多精力验证转换的正确性(比如构造边界用例手工核对转换前后是否语义等价)，不要假设"看起来转过去了"就是转对了。

## 相关文档

- 完整案例（这份手册里每一条原则的来源）：[`tau2_bench_integration_findings.md`](./tau2_bench_integration_findings.md)，配套可视化面板 [`tau2_dashboard.html`](./tau2_dashboard.html)
- 目标一/二实现基础（`inspect_trace` 的 Hooks 机制、为什么必须在 inspect_ai 自己的 Sample 执行上下文里）：[`inspect_ai_roadmap.md`](./inspect_ai_roadmap.md)
- 适配器代码：`/home/liuyingen/code/efficient-harness/tau2_adapter/`

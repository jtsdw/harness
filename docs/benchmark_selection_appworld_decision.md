# Harness 下一 benchmark 选型：优先 AppWorld Base

状态：选型已执行；AppWorld adapter、P0 测量闭环和 57 题 dev baseline 已验证
日期：2026-08-09

> 2026-08-11 更新：本文件的选型判断已经落地。当前实现和全量验证结果见
> [`appworld_adapter_initial_design.md`](./appworld_adapter_initial_design.md)；正式运行入口见
> [`appworld_adapter/README.md`](../appworld_adapter/README.md)。

## 结论

如果当前只能新增一个 benchmark，应暂停把 ToolBench 作为下一主目标，优先接入
**AppWorld Base 的 function-calling 路径**。

这里的 Base 指不带额外用户模拟器的原始 AppWorld，而不是 AppWorld-UL。第一阶段应复用
上游 `simplified_function_calling_agent` 的评测语义，而不是先接 code-agent、MCP 远程服务
或自行设计新的 agent scaffold。

这个判断并不是认为 AppWorld 在所有维度都优于其他 benchmark，而是认为它最适合成为
当前 harness 的下一块主 workload：它能在相对可控的条件下提供比 BFCL 更深的状态化
工具闭环，又避免 tau-bench、ToolSandbox、ToolBench 和真实网页/桌面 benchmark 引入的
额外模型、judge、网络或环境噪声。

## 从第一性原理定义选型目标

本项目研究对象首先是 harness，而不是某个模型的 leaderboard 排名。因此，benchmark 的
价值不应主要由题目数量或流行度决定，而应由它能否支持可靠的效率归因决定。

下一 benchmark 应尽量同时满足以下条件：

1. **闭环深度**：包含多轮 `model -> tool -> state -> model`，能够产生上下文增长、重复
   prefill、工具结果膨胀、失败恢复和潜在并行调用。
2. **完整可观测**：目标模型调用、工具请求、工具结果、环境状态变化、终止原因和评分依据
   都能用稳定 ID 关联。
3. **可控与可重放**：初始状态固定且可复位；相同动作得到相同结果；不依赖会变化的网页、
   搜索结果或第三方 API。
4. **评测不引入额外模型**：核心成功判定不依赖 LLM judge；基础版本不要求另一个用户模拟
   模型参与闭环。
5. **结果有效**：按最终状态和约束评分，允许多条正确轨迹，并能识别未请求的副作用；不能
   只比较参考 action sequence。
6. **系统研究价值**：既能做 closed-loop 的能力—效率联合评测，也能导出真实轨迹用于
   replay、并发、cache 和 serving 实验。
7. **接入成本可控**：上游语义可以复用，不需要维护数千个外部服务、真实账号或另一套大型
   模型后端。

效率比较还必须以成功为条件，或报告 accuracy–cost Pareto。一个快速失败的 agent 不是高效
agent，因此不能把所有成功和失败 episode 的平均 token/latency 直接解释为效率。

## 当前 harness 已有的两个锚点

### BFCL multi-turn：受控的 function-calling control

BFCL multi-turn 的本地 Python 工具在 Inspect 内执行，因此 `ModelEvent`、`ToolEvent`、工具
错误和执行拓扑可以完整观测。现有 200 样本运行平均每题 7.03 次模型调用、4.285 次工具
调用，且已经暴露出工具 schema 重复 prefill 是主要成本。

它的局限是环境和任务范围较窄，现有运行没有观测到真实并行，不能单独代表更长的跨应用
agent workload。

证据：[`goal2_real_validation_findings.md`](./goal2_real_validation_findings.md)。

### tau-bench：交互式业务任务和目标模型推理 profile

tau-bench 提供多轮用户交互、业务规则和最终任务 reward，是重要的外部有效性验证。但当前
adapter 只把被测 agent 的模型调用重新接入 Inspect；用户模拟器、环境工具执行和部分评分
过程仍在 tau-bench 自己的运行边界中。因此它不是一份统一的全 agent trace。

证据：[`tau2_adapter/src/tau2_adapter/solver.py`](../tau2_adapter/src/tau2_adapter/solver.py)
和 [`tau2_bench_integration_findings.md`](./tau2_bench_integration_findings.md)。

下一 benchmark 最有价值的补位，是在不引入第二个模型和真实网络的前提下增加闭环深度、
应用/API 多样性和完整状态观测。

## 为什么 AppWorld Base 最匹配

### 1. 深度和多样性足以产生 agent 系统效应

AppWorld 包含 750 个任务、9 个日常应用和 457 个 API。论文报告每个任务平均涉及 1.8 个
应用、9.5 个 API，最多涉及 6 个应用、26 个 API。它不只是根据一句话生成一次函数调用，
而是需要根据中间结果继续决策和修改状态。

这比当前 BFCL 提供更长、更丰富的闭环，同时仍保持为结构化 API 工具任务，而不是退化成
通用网页问答或桌面操作。

### 2. 环境可以完全本地、同进程和确定性运行

AppWorld 使用任务专属 SQLite 状态和固定时间。默认通过 FastAPI `TestClient` 在同一进程
模拟 HTTP 请求，不要求启动真实 API server，也不要求访问外部网页。环境支持 reset、状态
checkpoint 和恢复。

因此模型端以外的环境延迟可以被测量并控制，而不会把公网抖动混入 TTFT、tool gap 或
episode latency。

### 3. scorer 是程序化的最终状态检查

AppWorld 使用数据库状态和执行单测检查任务完成情况，同时检查 collateral damage。评分不
依赖复现参考轨迹，也不需要 LLM judge。这使不同但正确的调用计划可以通过，而不必要的
删除、付款或修改仍会失败。

### 4. 上游日志足以验证 harness trace 的完整性

每个任务原生保存：

- `logs/environment_io.md`：每次环境执行的输入输出；
- `logs/api_calls.jsonl`：每次 API/HTTP 调用；
- 最终数据库差分；
- 逐项 evaluator pass/fail 和 assertion trace；
- 代码与数据版本。

这些产物可以与 Inspect 的 `ModelEvent`、`ToolEvent` 和 scorer metadata 做逐任务交叉核对，
从而区分“指标真实为零”和“事件没有被采集”。这正是当前 tau-bench 接入中最欠缺的验证面。

### 5. 官方 function-calling 路径本身包含值得研究的系统阶段

当前 `simplified_function_calling_agent` 不是把 457 个 schema 全部塞进每一轮。它先调用
API predictor，最多选择 20 个 API，再把这组 function schemas 交给主 agent 连续调用，并
允许一次返回多个 tool calls。

这个过程同时包含：

- 工具检索/菜单裁剪成本；
- 动态 schema 暴露；
- schema 在后续轮次中的重复 prefill；
- 多工具串行或并行执行；
- 工具错误与恢复；
- 随上下文增长变化的 TTFT 和 decode 成本。

API predictor 是 agent scaffold 的真实组成部分，必须作为单独 stage 被 trace 和计费，不能
隐藏为 benchmark 预处理，也不能使用 ground-truth API 列表替代。

## 候选比较

| 候选 | 主要增量 | 对 harness 的主要干扰 | 当前判断 |
|---|---|---|---|
| **AppWorld Base** | 750 题、457 API、状态化跨应用、程序化 scorer、完整原生日志 | 需要编写 Inspect adapter；官方 function-calling 含 API predictor | **下一主目标** |
| Toolathlon-Verified | 604 工具、约 20 轮、真实软件和严格 evaluator，长轨迹最强 | 完整本地运行要求逐任务容器、外网、真实账号/token 和多套应用；仅 108 题 | 后续 frontier validation 或轨迹 replay |
| ToolSandbox | 完整状态快照、局部/最终 milestone、可组合 Python 工具 | 默认引入 LLM user simulator；搜索场景需要 RapidAPI；工具世界较小，与 tau 重叠 | 非首选 |
| STATE-Bench | 450 个企业任务、task-local DB、确定性状态断言 | 中心闭环仍有 LLM user simulator；发布时间较新，与 tau 的结构相近 | 观察候选 |
| MCP-Atlas | 1,000 题、220 工具、MCP、多服务器编排 | 通常仅 3–6 次工具调用；主要按最终答案 claims 打分；容器/真实 server | 工具发现研究候选，不是主效率 workload |
| ToolBench / StableToolBench | 最大规模的开放 API 空间，工具检索研究价值高 | simulator/MirrorAPI、retriever、judge 和版本语义增加变量；弱状态环境；轨迹相对短 | 保留为工具规模专项，不作为下一主目标 |
| GAIA / WebArena / OSWorld | 真实长轨迹和通用 agent 能力 | 公网、浏览器、GUI、多模态和容器噪声大，难以做因果效率归因 | 外部有效性验证 |
| SWE/Terminal 类 | 很长的 coding/terminal 轨迹，可用于 serving 压测 | 仓库构建、测试、容器和文件系统成本主导；不专注结构化工具调用 | 更适合 trace replay，不作主 closed-loop benchmark |

## 建议的最小实验矩阵

完成 AppWorld 后，项目不应把所有 benchmark 合并成一个无差别总分，而应维持三个角色：

1. **BFCL multi-turn：micro/control。** 验证原生 function call、schema 成本、工具解析和局部
   执行拓扑。
2. **AppWorld Base：primary closed-loop workload。** 研究深度工具闭环中的成功率、上下文
   增长、cache、延迟、无效动作和状态副作用。
3. **tau-bench：interactive external validation。** 验证结论在带用户模拟和业务 policy 的
   agent 交互中是否仍成立。

Toolathlon-Verified 的公开轨迹可以作为后续第四层：不立即复现其完整在线环境，而是在
harness 增加 trace-replay/SLO 模式后，用长轨迹测试并发 agent、prefix/cache 策略和 serving
容量。Replay 不能替代 closed-loop 成功率，但可以把模型服务实验与环境噪声分开。

## 接入边界与验收条件

第一阶段只应实现 AppWorld Base 的 function-calling 路径：

- 复用上游任务加载、API schema、API 执行和 evaluator；
- API predictor 和主 agent 的所有模型调用都进入 Inspect，并用 stage 区分；
- 每个实际 AppWorld API 调用形成 Inspect `ToolEvent` 或语义等价的结构化事件；
- 保留并关联 AppWorld 原生 API log、环境 I/O、DB diff 和 evaluator 结果；
- 使用同进程本地环境，不启动外部 AppWorld HTTP/MCP server，不访问真实网络；
- 先在固定 dev task IDs 上逐样本比较原生路径和 adapter，再扩到官方 test split；
- 分别报告 task goal completion、scenario goal completion 和成功条件下的效率指标；
- 失败 episode 单独分析，不能用快速失败降低出来的均值宣称优化；
- 第一阶段不做 AppWorld-UL、code-agent、多模态、训练、排行榜提交或 Toolathlon 环境部署。

在宣称 model-invocation 指标可靠前，还必须重新验证当前 vLLM metrics collector。仓库已经
记录了一个真实场景中 `sample_event` 的第二次 metrics snapshot 不返回、最终完全不产出记录
的问题；AppWorld 的长轨迹不能建立在未经重新验证的采集器上。

证据：[`goal2_real_validation_findings.md`](./goal2_real_validation_findings.md#2026-08-05-补充发现一个真实场景下-vllm_metrics-完全不产出记录的-bug跟本文档4949-100-精确归因的结论存在未解决的冲突)。

## 哪些证据会改变这个判断

出现以下任一情况，应重新评估 AppWorld 的优先级：

1. 固定版本的 AppWorld Base 无法在断开外网后完成任务执行和官方评分。
2. function-calling 路径无法在不改变上游 agent 语义的情况下把 predictor、主模型和单个 API
   调用映射为可关联事件。
3. 本项目的科学问题被明确收窄为“超大工具库检索”，而不再研究完整 agent inference 和
   closed-loop 执行；此时 ToolBench/MCP 类 benchmark 更合适。
4. 本项目的科学问题被明确改为“多用户并发下的模型 serving 容量”，此时真实轨迹 replay
   应优先于新增 closed-loop benchmark。
5. Toolathlon 或其他新 benchmark 提供完全离线、无需账号、固定镜像且可程序化评分的
   verified 子集，并且安装和复位成本达到 AppWorld 同一量级。

## 主要外部资料

- AppWorld 论文：[ACL 2024](https://aclanthology.org/2024.acl-long.850/)
- AppWorld 实现与运行说明：[官方仓库](https://github.com/StonyBrookNLP/appworld)
- AppWorld function-calling agent：[上游实现](https://github.com/StonyBrookNLP/appworld/blob/main/experiments/code/simplified/function_calling_agent.py)
- Toolathlon：[论文](https://arxiv.org/abs/2510.25726)；[运行要求](https://toolathlon.xyz/docs/test)
- ToolSandbox：[官方仓库](https://github.com/apple/ToolSandbox)
- STATE-Bench：[官方仓库](https://github.com/microsoft/STATE-Bench)
- MCP-Atlas：[论文](https://arxiv.org/abs/2602.00933)
- ToolBench：[官方仓库](https://github.com/OpenBMB/ToolBench)
- OSWorld-Human：[MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html)
- Tool-integrated reasoning 的硬件成本度量：[PTE](https://arxiv.org/abs/2604.05404)
- 真实 agent 轨迹 replay 的 serving 实践：[AA-AgentPerf methodology](https://artificialanalysis.ai/methodology/agentperf)

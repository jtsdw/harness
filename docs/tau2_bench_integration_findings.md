# tau2-bench 接入：环境部署、框架对接、真实运行、结果分析

延续项目一贯的方法论：先设计（见上一轮的架构调研，已并入下文"背景"一节），再真实跑通，如实报告结果——包括踩过的坑、遇到的真实 bug、以及"两条路径结果不完全一致"这类不方便但真实的发现。

## 背景：为什么要接 tau2-bench，以及一开始的技术判断

用户想知道 tau2-bench（Sierra 的 τ³-bench，一个面向客服场景的双控 agent 评测框架）能不能接入我们的 harness，以及"跑同一个 bench 是否能跑出同样效果"。tau2-bench 是完全独立的定制框架——自己的 `Orchestrator`/`Environment`/`Evaluator`/`UserSimulator`，模型层用 LiteLLM，不基于 inspect_ai。核心特点是"双控"：`User` 不是数据集里的固定脚本，是另一个真实跑的 LLM（`UserSimulator`），跟 agent 动态对话；评分基于 DB 状态哈希比对 + 字符串匹配 + 可选 LLM 判断的自然语言断言，不是 AST 匹配。

调研阶段读 `/home/liuyingen/code/inspect_ai/src/inspect_ai/hooks/_hooks.py` 确认了一个关键约束：`emit_sample_event()`（第 710 行）有守卫 `if active is None: return`，`sample_active()` 是 inspect_ai 自己 Task/Sample 执行时才会设置的 contextvar——`inspect_trace` 几乎全部实际写数据的逻辑都挂在 `on_sample_event` 上，脱离 inspect_ai 自己的 eval 循环就一条记录都写不出来。这意味着"在 tau2 自己的 Agent 里直接调 `inspect_ai.model.generate()`，其余用 tau2 自己的 orchestrator"这条路子行不通，必须反过来：**由 inspect_ai 的 Task/Solver 驱动整个模拟循环**，在 Solver 内部把 tau2-bench 自己的 `Environment`/`UserSimulator`/`Evaluator` 当库函数调用，只把"被测 agent 侧的模型调用"这一步换成 `inspect_ai.model.generate()`。

另一个已知障碍：`tau2.orchestrator.orchestrator.Orchestrator.run()`/`step()` 全部是同步方法，但 `inspect_ai.model.generate()` 是异步的，且必须在 inspect_ai 自己 Sample 执行所在的那个异步任务里调用，`sample_active()` 才认得。

## 环境部署

tau2-bench 数据集**已经内置**在仓库里（`data/tau2/domains/{mock,airline,retail,telecom,banking_knowledge}/tasks.json`），不需要另外下载——这跟最初"数据集下载"这一步的预期不同，如实更正。

真实踩到的环境坑：

1. **Python 3.13 装不上**：`tau2.voice` 模块在 import 时无条件依赖 Python 3.13 已经移除的标准库 `audioop`（`tau2/voice/utils/audio_preprocessing.py`），即使完全不需要 voice 功能也会被 `tau2/__init__.py` 的导入链拖进去。tau2 自己的 `requires-python = ">=3.12,<3.14"` 声称支持 3.13，实际在纯净环境下装不上。改用 Python 3.12（`uv sync --python 3.12`）绕开，不需要额外的 `audioop-lts` shim 包。
2. **`TAU2_DATA_DIR` 要指到 `data/` 这一级，不是 `data/tau2/`**——tau2 自己的代码（`utils/utils.py`）会在这个环境变量后面再拼一段 `tau2/domains/...`，指错一级会得到 `FileNotFoundError`。
3. **`uv pip install <name>` 不认 `pyproject.toml` 里的 `[tool.uv.sources]` path 覆盖**——排查 litellm 版本问题时手滑跑了 `uv pip install --reinstall-package tau2 --no-cache tau2`，装出来的是 PyPI 上一个同名但完全无关的包（`tau2==2.3.3`），不是我们本地的 tau2-bench。用 `uv sync --reinstall-package tau2`（走 `uv sync`，会读 pyproject.toml 的 source 覆盖）才对。当场发现当场改回，如实记录这个操作失误。

## 框架对接：三个真实 bug，不是同一类问题

对接过程中，`tau2 run` 原生 CLI 本身（不涉及我们任何代码）就连续踩到两个环境相关的真实 bug，加上我们适配器自己踩到一个——分开记录，因为它们的根因、影响范围、修复方式都不一样。

### Bug 1（更正此前文档的错误判断）：这版 vLLM 其实支持原生 tool-calling

`local_model_deployment.md`/`serve.sh` 此前一直断言 `vllm==0.6.3.post1` 没有 `--enable-auto-tool-choice`/`--tool-call-parser` 机制，所以全程靠 `emulate_tools=true` 走客户端模拟。这次实测重新验证：**这个判断是错的**——用 `--enable-auto-tool-choice --tool-call-parser hermes` 重启服务，`curl` 直接打一个真实带 `tools` 的请求，返回了结构良好的原生 `tool_calls`，完全正常。之前的判断大概率是没有真正试过，凭旧印象写的。这个更正已经在 `deployment_migration_guide.md` 的"原生 tool-calling"一节留了痕迹，这里是首次真正证实。

### Bug 2：tau2-bench 自己 `to_litellm_messages()` 里一个真实的多余字段

跑原生 CLI 时（agent 和 user 都指本地 vLLM，`tool_choice=auto`），第二轮开始必现：
```
litellm.BadRequestError: OpenAIException - 5 validation errors for ValidatorIterator
0.ChatCompletionMessageFunctionToolCallParam.name
  Extra inputs are not permitted ...
```
排查过程：先怀疑是 `litellm`/`openai` 包版本不兼容（升级 litellm 到 1.95.0、把 openai 降到 1.109.1，都没用），再怀疑是 vLLM 服务端自己的 `openai` 版本问题（同样无效）。最后用最小复现脚本直接调 `tau2.utils.llm_utils.generate()` 走两轮对话，拿到完整 traceback，定位到 `tau2/utils/llm_utils.py:180`——`to_litellm_messages()` 把 assistant 消息的每个 tool_call 转成 dict 时，除了正确的 `{"id", "type", "function": {"name", "arguments"}}` 之外，**多写了一个顶层 `"name"` 字段**，这个字段不属于 OpenAI 的 `ChatCompletionMessageFunctionToolCallParam` schema（`total=False` 但 extra-forbidden），新版 `openai` SDK 的严格 pydantic 校验直接拒绝。这是 tau2-bench 自己代码里的一个真实 bug，只在"多轮工具调用 + 严格校验的新版 openai SDK"这个组合下才会触发，大概率之前没人踩到是因为大多数用户对着真正的 hosted API 跑，走的是另一条内部转换路径。

修复：本地直接删掉这个多余字段（我们完全拥有这份本地克隆，不打算走上游 PR，这不是"绕过问题"，是"改对了"）。改动位置：`/home/liuyingen/code/tau2-bench/src/tau2/utils/llm_utils.py`，`to_litellm_messages()` 函数。

### Bug 3（我们适配器自己的问题）：inspect_ai 默认给工具加的 `strict` 字段，这版 vLLM 不认

改用 Bug 2 的修复之后，tau2 原生 CLI 已经能正常跑通原生 tool-calling。但换成我们的适配器（走 `inspect_ai.model.generate()`）时，遇到了第三个、性质完全不同的问题：

```
{'type': 'extra_forbidden', 'loc': ('body', 'tools', 0, 'function', 'strict'), ...}
```

`inspect_ai` 的 `openai-api` provider（`model/_providers/openai_compatible.py:398`）**无条件**给每个工具的 `function` 加一个 `"strict"` 字段（`tool["function"]["strict"] = self.strict_tools`，没有 `if` 判断，`-M strict_tools=false` 也只是把值设成 `false`，字段本身还在），这是给"结构化输出"场景用的新字段，vLLM 0.6.3.post1 的请求 schema 完全不认识，直接 400。tau2 自己的工具构造代码没有这个字段，所以原生路径不受影响，只有 inspect_ai 这条路径会撞上。

修复：适配器这边把 agent 侧模型也切回 `emulate_tools=true`（走我们全项目一直在用的客户端文本解析路径），彻底绕开这个字段问题——这也是我们最熟悉、验证最多的路径。**代价**：agent 侧现在是 `emulate_tools`（文本解析 tool_calls），user simulator + 原生基线是真正的原生 tool-calling——两条路径底层生成 tool_calls 的具体机制不同，这是一个真实存在、需要如实说明的不对称，不是掩盖不谈。好在两者最终都会被规整成结构化的 `ToolCall` 对象（`model_output_to_tau2_assistant_message()` 转换函数不关心底层是怎么解析出来的），所以对 tau2 环境/评估器来说是透明的,不影响评分逻辑。

## Hooks 触发验证：核心技术假设成立

单任务验证（`create_task_1`）后，`inspect_trace` 的输出目录里真实产出了：

| kind | 条数 |
|---|---|
| prefill_diff | 6 |
| segment_tokens | 6 |
| token_attribution | 6 |
| attempt_group | 6 |
| vllm_metrics | 6 |
| execution_topology | 1 |

不是空文件，`sample_id` 正确关联到 `create_task_1`（tau2 自己的任务 ID）。这证实了"背景"一节的核心技术判断：`anyio.to_thread.run_sync`（把整个同步的 `Orchestrator.run()` 扔进 worker 线程）+ `anyio.from_thread.run`（agent 内部真正要发起模型调用时跳回原来的事件循环）这套桥接，确实能让 `sample_active()` 在真正调用 `model.generate()` 的那一刻正确解析，Hooks 按预期触发。

**一个真实的、结构性的局限，如实记录**：`execution_topology`/`action_parsing` 两类记录里 `total_tool_calls` 恒为 0——不是 bug，是因为 tau2 自己的 `Environment` 执行工具调用（`InspectAIAgent` 只负责"提议" tool_calls，不负责执行），完全不经过 inspect_ai 自己的 `ToolEvent` 机制。也就是说，**这种集成方式下，需求三（执行拓扑）/需求四（action parsing）这两类数据结构性地拿不到**——跟目标一在 `live_parallel` category 上"结构性看不到 ToolEvent"是同一类限制，不是这次新引入的缺陷。真正能拿到的是需求一（token 归因，含 `vllm_metrics` 的 model invocation 层）,不是全部四条需求。

## 全量运行结果对比：mock domain 10 个任务

两条路径都用同一个本地模型（`Qwen/Qwen2.5-3B-Instruct`，本地 vLLM，`temperature=0.0`），agent 和 user simulator 都指向同一个服务。

| task_id | 原生 CLI reward | 原生 termination | 适配器 reward | 适配器 termination | 一致？ |
|---|---|---|---|---|---|
| create_task_1 | 0.0 | max_steps | 0.0 | max_steps | ✅ |
| create_task_1_nl_eval | 0.0 | max_steps | 0.0 | max_steps | ✅ |
| create_task_1_with_env_assertions | 0.0 | max_steps | 0.0 | max_steps | ✅ |
| impossible_task_1 | 0.0 | max_steps | 0.0 | max_steps | ✅ |
| update_task_1 | **1.0** | user_stop | **0.0** | user_stop | ❌ reward 不同 |
| update_task_with_history_and_env_assertions | 1.0 | user_stop | 1.0 | user_stop | ✅ |
| update_task_with_initialization_actions | **0.0** | max_steps | **1.0** | user_stop | ❌ reward 不同 |
| update_task_with_initialization_data | 0.0 | max_steps | 0.0 | user_stop | ⚠️ reward 同，termination 不同 |
| update_task_with_message_history | 1.0 | user_stop | 1.0 | user_stop | ✅ |
| update_task_with_user_tools | 0.0 | user_stop | 0.0 | max_steps | ⚠️ reward 同，termination 不同 |

聚合结果：**两条路径的 accuracy 都是 0.30（3/10）**——但逐条看，只有 6/10 完全一致（reward 和 termination 都相同），2/10 的 reward 直接翻转（一个从 1→0，一个从 0→1，恰好互相抵消，聚合数字才"看起来一样"），另外 2/10 reward 相同但对话轮数/终止方式不同。

**一个更进一步的诚实发现**：原生 CLI 本身，同一个任务重复跑两次也不完全一致——单独跑 `create_task_1`（阶段 3 那次单任务验证）得到 reward=1.0，但在这次 10 任务批跑里，同一个 `create_task_1` 却是 reward=0.0（`max_steps`）。也就是说，**"原生 vs 适配器"观察到的差异，不能全部归因于"harness 不一样"**——哪怕两次都走一模一样的原生 CLI 代码路径，`temperature=0.0` 也没能让 dual-control 模拟完全确定：user simulator 的具体措辞、vLLM 批处理调度的时序，都会让对话轨迹产生真实的、非 harness 造成的随机性。

## "同一个 bench，能不能跑出同样效果"——实证结论

回答上一轮那个问题，现在有真数据了，不是推测：

**不完全能，但也不是"完全跑不出"**。10 个任务里 6 个逐条一致，聚合准确率巧合地相同（0.30），但个体层面 2/10 direction 相反的翻转说明这不是"稳定可复现"的一致,是运气抵消。根因分层：
- 一部分差异（至少 2/10 那种 reward 相同但 termination 不同的情况）大概率来自 dual-control 模拟本身固有的非确定性（user simulator 措辞、批处理时序），不是我们适配器的问题——这条从"原生 CLI 自己都不能精确复现自己"这个事实反推出来。
- 另一部分差异可能来自适配器和原生路径两侧机制上的真实不对称（agent 侧 `emulate_tools=true` 文本解析 vs user simulator 原生 tool-calling，见 Bug 3），但这次的数据量（10 个任务、每边各跑一次）不足以把这部分和上面的固有随机性彻底分开——需要多次重复跑（比如每条路径跑 5 次取分布）才能定量拆解,这次没有做到那个规模,如实说明这是当前证据的边界,不是回避。

## 已知限制（如实列出）

- 只接了 `mock` domain（10 个任务），`airline`/`retail`/`telecom`/`banking_knowledge` 没有验证过，工具集更大更复杂，可能暴露新的转换边界问题（比如更复杂的 JSON schema、`additionalProperties`、嵌套对象参数）。
- `execution_topology`/`action_parsing`（需求三/四）在这种集成方式下结构性拿不到数据，见上文。
- user simulator 的模型调用完全不经过 `inspect_trace`，只追踪被测 agent 一侧——这是设计选择，不是遗漏，但意味着"整个 episode 的完整 token/延迟画像"不包含 user simulator 那一半的真实开销。
- 本地 3B 模型演"客服 agent"和"用户"两个角色，能力都偏弱（10 个任务里 7 个没通过，多数因为超出 `max_steps`）——这是模型能力限制，不是集成本身的问题，但让"两条路径到底有没有真正对齐"这个问题在这批小样本上更难看清（弱模型的行为方差本来就大）。
- 只跑了 `temperature=0.0`、单次重复——没有做多次重复实验来定量拆分"harness 差异"和"固有随机性"两类误差来源。

## 复现命令

环境搭建（一次性）：
```bash
cd /home/liuyingen/code/tau2-bench && uv sync --python 3.12
# 应用本地 bug 修复：删除 src/tau2/utils/llm_utils.py 的 to_litellm_messages() 里多余的
# 顶层 "name" 字段（第 182 行附近），见上文 Bug 2

cd /home/liuyingen/code/efficient-harness/tau2_adapter && uv sync --extra dev --python 3.12
```

启动本地 vLLM（原生 tool-calling，供 tau2 原生 CLI + user simulator 使用）：
```bash
cd /home/liuyingen/code/efficient-harness/local-model-server
uv run vllm serve Qwen/Qwen2.5-3B-Instruct \
  --port 8000 --gpu-memory-utilization 0.85 --max-model-len 16384 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

原生 CLI 基线：
```bash
cd /home/liuyingen/code/tau2-bench
TAU2_DATA_DIR=/home/liuyingen/code/tau2-bench/data uv run tau2 run \
  --domain mock \
  --agent-llm "openai/Qwen/Qwen2.5-3B-Instruct" \
  --agent-llm-args '{"temperature": 0.0, "api_base": "http://localhost:8000/v1", "api_key": "not-needed"}' \
  --user-llm "openai/Qwen/Qwen2.5-3B-Instruct" \
  --user-llm-args '{"temperature": 0.0, "api_base": "http://localhost:8000/v1", "api_key": "not-needed"}' \
  --num-trials 1 --num-tasks 10 --max-steps 20 --save results_mock_baseline
```

适配器路径（agent 侧走 `emulate_tools=true`，见 Bug 3）：
```bash
cd /home/liuyingen/code/efficient-harness/tau2_adapter/src/tau2_adapter
TAU2_DATA_DIR=/home/liuyingen/code/tau2-bench/data \
TAU2_USER_MODEL="openai/Qwen/Qwen2.5-3B-Instruct" \
TAU2_USER_API_BASE="http://localhost:8000/v1" TAU2_USER_API_KEY="not-needed" \
VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
INSPECT_TRACE_DIR="/home/liuyingen/code/efficient-harness/runs/tau2_adapter_full/.inspect_trace" \
uv run --project /home/liuyingen/code/efficient-harness/tau2_adapter inspect eval task.py \
  --model "openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" -M emulate_tools=true \
  --max-connections 1 \
  --log-dir /home/liuyingen/code/efficient-harness/runs/tau2_adapter_full/logs
```

原始数据留档：`runs/tau2_native_baseline/`（原生 CLI 的 `results_1task`/`results_mock_baseline`）、`runs/tau2_adapter_full/`（适配器路径的 `.eval` 日志 + `inspect_trace` JSONL）。

## 相关文件

- 适配器代码：`/home/liuyingen/code/efficient-harness/tau2_adapter/`（`agent.py` 同步/异步桥接、`solver.py` 驱动 tau2 模拟、`dataset.py`/`task.py` 组装、`convert.py` 消息/工具类型转换）
- tau2-bench 本地 bug 修复：`/home/liuyingen/code/tau2-bench/src/tau2/utils/llm_utils.py`（本地补丁，不打算走上游 PR）
- 目标一实现（`inspect_trace`）：`/home/liuyingen/code/efficient-harness/inspect_trace/`

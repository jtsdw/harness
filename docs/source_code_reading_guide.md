# inspect_ai 源码阅读指南：从 prompt 输入到模型输出的完整链路

这份指南不按目录结构介绍代码（`model/`、`solver/`、`agent/`...），而是按**一条真实请求实际流经的顺序**带你读——从数据集的一条 `Sample` 变成第一条 prompt，到 harness 怎么组织上下文、调用模型、解析工具调用、把结果拼回去，直到这条样本跑完。每一段都标了具体文件+函数+行号（已逐条验证，不是凭印象写的），并且对照我们已经观察到的真实数据（`.eval` 日志字段、`runs/compare/` 里的实验结果）——目的是让你读代码的同时能马上在真实数据里找到对应的东西，而不是抽象地理解。

## 阅读方法论：为什么按执行顺序读，不按目录读

inspect_ai 是个成熟框架，目录很多（`model/`/`solver/`/`agent/`/`tool/`/`event/`/`hooks/`/`log/`...），如果按目录挨个看，很容易在还没建立全局图景之前就陷入某个子系统的细节。更有效的方式是：**先跟着一条请求实际怎么流转走一遍主干道**，把每个目录在这条路上"负责哪一段"先定位清楚，再回头深入某个你关心的目录。这也是我们做 `inspect_trace` 时实际用的方法——先搞清楚 `ModelEvent`/Hooks 在整条链路里卡在哪个位置，再决定要在哪个点插桩。

## 全局地图

```
Sample.input (数据集原始输入)
        │
        ▼
sample_messages()                          ──► TaskState.messages 初始化
        │
        ▼
plan(state, generate)                      ──► solver 链条依次跑（system_message/prompt_template/...）
        │                                       每个 solver 可以直接改 state.messages（prompt 构造）
        ▼
generate(state, tool_calls="loop")  ────┐
        │                               │  这是 harness"组织 token"的核心循环
        ▼                               │  （task_generate，见下）
   ┌─────────────────────────────┐      │
   │ model.generate(state.messages,│    │
   │   state.tools, ...)          │◄────┘
   │      │                       │
   │      ▼                       │
   │ Model.generate() 公开方法      │
   │      │ (retry 包裹 + Hooks)    │
   │      ▼                        │
   │ ModelAPI.generate()（provider）│  ──► 真实 HTTP 请求（OpenAI/vLLM/...)
   │      │                        │
   │      ▼                        │
   │ ModelOutput / ModelEvent 落盘  │
   └──────────┬─────────────────────┘
              │ 有 tool_calls？
              ▼
        execute_tools()            ──► 解析、并行执行工具，产出 ChatMessageTool
              │
              ▼
        messages 追加进 state       ──► 回到 model.generate()，循环
              │
        （没有更多 tool_calls）
              ▼
        state 返回给 plan 的下一个 solver / sample 结束
              │
              ▼
        scoring → EvalSample 组装 → 写入 .eval 日志
```

## `TaskState` 详解：贯穿全程的核心对象

在读逐段拆解之前，先把 `state`（也就是 `TaskState`）这个反复出现的参数搞清楚，后面每一段读起来会顺很多。

**源码位置**：`src/inspect_ai/solver/_task_state.py`，类定义在第 140 行，构造函数 150-190 行，属性一路到 467 行。类的 docstring（141-148 行）原话：

> `TaskState` represents the internal state of the `Task` being run for a single `Sample`. It is passed to and returned from each solver during a sample's evaluation.

**它是什么**：一个常见的误解是把它类比成"token 之于 model"那种最小执行单元——这个类比方向反了。`TaskState` 不是最小单元，恰恰相反，是**装着一条样本执行期间几乎所有东西的容器**，从头到尾只有这一个（除非显式 `fork()` 出分支）。更贴切的类比是 web 框架里的 **request context**：一个请求进来生成一个 context 对象，依次穿过若干 middleware，每个 middleware 都能读/改这个 context，最后拿它生成 response——`Sample` 对应请求，`TaskState` 对应 context 对象，每个 `Solver`（`system_message`/`prompt_template`/`basic_agent()`/...)对应一个 middleware，函数签名就是 `(state, generate) -> state`，`plan(state, generate)` 就是 middleware 链条依次执行（第 1 段）。如果一定要找"最小单元"，那是 `ChatMessage`（`TaskState.messages` 列表里的一条）,不是 `TaskState` 本身。

### 完整字段清单（按用途分组）

**对话本身**（harness 组织上下文的核心，第 2 段的循环改的就是这几个）：

| 字段 | 行号 | 说明 |
|---|---|---|
| `messages: list[ChatMessage]` | 261-273 | 对话历史，docstring 原话"generally get appended to every time a `generate` call is made" |
| `tools: list[Tool]` | 294-303 | 模型可用的工具 |
| `tool_choice: ToolChoice \| None` | 305 | 注意这个不是 `@property`，是普通类属性，写法跟其他字段不太一样 |
| `output: ModelOutput` | 276-287 | "最终"输出；docstring 提醒"对简单 eval 来说通常就是对话里最后一条消息，但复杂 solver 可能直接手动设置它" |

**这条样本的身份信息**（构造时传入，多数只读）：

| 字段 | 行号 | 说明 |
|---|---|---|
| `model`/`sample_id`/`epoch`/`uuid` | 192-205, 438-440 | `uuid` 就是 `inspect_trace`/`.eval` 日志里那个 `sample_uuid` |
| `input: str \| list[ChatMessage]` | 208-210 | docstring 明确写"should be considered immutable"——`Sample.input` 的原始值，不随对话推进而变，对应第 0 段"`sample.messages` 最终态 vs `TaskState.input` 原始态"的区别 |
| `input_text` | 213-236 | `input` 的字符串便捷访问器 |
| `target: Target` | 424-426 | 数据集给的标准答案，打分用 |
| `choices` | 61-137（`Choices` 类单独定义） | 多选题专用 |

**任务生命周期控制**（限流熔断相关，setter 里有额外副作用，不是被动存值）：

| 字段 | 说明 |
|---|---|
| `completed: bool`（402-421 行） | **getter 不是简单返回值**，还会调用 `set_active_sample_total_messages()` 顺手同步控制面板状态，读的时候留意这一点 |
| `message_limit`/`token_limit`/`token_limit_type`/`token_usage`/`cost_limit`/`cost_usage` | 每个 limit 的 setter 都会立刻做一次越界检查（`check_message_limit`/`check_token_limit`/`check_cost_limit`） |

**附加数据**：

| 字段 | 行号 | 说明 |
|---|---|---|
| `metadata: dict[str, Any]` | 252-258 | 来自 `Sample.metadata` |
| `store: Store` | 290-292 | 跨 solver 共享的自由格式数据仓库，`store_as()`（456-467 行）提供 Pydantic 模型化访问 |
| `scores: dict[str, Score] \| None` | 429-435 | |

### 两个容易忽略的细节

1. **它不保证全程是"同一个对象"**。`set_sample_state()`（474-498 行）的注释写得很直白："a solver can return a deepcopy or a state it got from `fork()` rather than the state it was passed"——多数内置 solver（`system_message`/`prompt_template`）原地改 `state.messages` 再把同一个对象 `return state`，但框架允许某个 solver 返回全新/deepcopy 的 `TaskState`（比如 `solver/_fork.py` 的 `fork()`，可以把一条样本的执行分叉成多条并行 `TaskState`，用于多路探索）。这也是为什么 `plan()` 每跑完一个 solver 都要重新 `state = await solver(state, generate)` 接收返回值，不能假设 solver 一定原地改完直接往下走。

2. **`.eval` 日志里的 `state` 事件，diff 的就是它**。文件末尾的 `state_jsonable()` 函数（504-525 行）把 `TaskState` 序列化成字典：`messages`/`tools`（转成 `ToolInfo`）/`tool_choice`/`store`/`output`/`completed`/`metadata`——`eval_log_format.md` 第 7 节讲过的那些 JSON Patch 格式的 `state` 事件，本质就是"这个 solver 把这份字典改成了什么样"的增量快照，字段跟上面这份清单基本一一对应。

## 逐段拆解

### 第 0 段：Sample 变成第一条消息

- **入口**：`src/inspect_ai/_eval/task/run.py:1000-1023`（`create_sample_state()`）
- 数据集的一条 `Sample.input`（字符串或消息列表）经过 `sample_messages(sample)` 转成 `TaskState.messages` 的初始值，同时 `TaskState` 还带上了这条样本的 `id`/`epoch`/`target`/`message_limit`/`token_limit` 等。
- **对照真实数据**：GSM8K 那条样本的 `sample_init` 事件（见 `eval_log_format.md` 第 7 节）里 `state.messages` 就是这一步的产物——此时还只有原始数据集问题，system prompt 和 few-shot 还没加进去。

### 第 1 段：solver 链条——这是"prompt 怎么被组织出来"的地方

- **入口**：`src/inspect_ai/_eval/task/run.py:1959`，`state = await plan(state, generate)`
- `plan` 是 `EvalPlan`（`src/inspect_ai/solver/_plan.py`），本质是一串 `Solver`——`Solver` 协议定义在 `src/inspect_ai/solver/_solver.py:79-97`：`async def __call__(self, state: TaskState, generate: Generate) -> TaskState`，可以直接改 `state.messages`（塞 system prompt、few-shot 示例、套 prompt 模板），也可以调用 `generate` 触发模型调用。
- **对照真实数据**：GSM8K 那条 `.eval` 日志的 `plan.steps` 字段（`eval_log_format.md` 第 3 节）——`system_message`（塞进 10 个 few-shot 示例）→ `prompt_template`（套"Solve the following math problem..."模板）——这两步就是这一段代码实际跑出来的产物，日志里 `state` 事件的 JSON Patch（`{"op": "replace", "path": "/messages/0/content", ...}`）逐条记录了这个变换过程。
- **想看更复杂的 agent 循环，读这两个具体 solver**：
  - `src/inspect_ai/solver/_basic_agent.py`——我们测试里一直用的 `basic_agent()`，一个"允许多轮工具调用直到调用 `submit()`"的 ReAct 循环 solver。
  - `src/inspect_ai/agent/_react.py`——更完整、生产级的 ReAct agent 实现（`react()`），带了 refusal 重试、`AgentSubmit` 等更多机制。

### 第 2 段：`generate()`——harness 组织上下文的核心循环，只有 66 行

这是整个链路里最值得精读的一段代码，直接决定了"每一步模型实际看到多大的上下文"。

- **位置**：`src/inspect_ai/_eval/task/generate.py`，函数 `task_generate()`（全文件只有 66 行，建议直接整个读完）
- 核心逻辑：
  ```python
  while True:
      state.output = await model.generate(
          input=state.messages, tools=state.tools, tool_choice=tool_choice, config=config
      )
      state.messages.append(state.output.message)        # 把模型的回复也塞进历史
      if state.completed: return state
      if message.tool_calls:
          messages, output = await execute_tools(state.messages, state.tools, ...)
          state.messages.extend(messages)                # 工具结果也塞进历史
          # ... 循环
      else:
          return state
  ```
- 每一轮循环，`state.messages` 只增不减（除非某个 solver 主动做了压缩/裁剪）——**这就是目标一 Q1"上下文如何逐步膨胀"字面意义上对应的那一行代码**。`inspect_trace` 的 `prefill_diff.py` 分析的正是这个循环每一轮传给 `model.generate()` 的 `input` 快照。
- **对照真实数据**：`runs/compare/deepseek_bfcl/` 里那条 41 步的轨迹，`prefill_diff` 记录显示每步稳定新增 2 条消息（一次 tool call + 一次 tool result）——对应的就是这个 while 循环每转一圈，`state.messages` 被 `append` 一次（assistant 消息）+ `extend` 一次（tool 结果消息）。

### 第 3 段：`model.generate()` —— 从"发起调用"到落盘成 `ModelEvent`

- **公开入口**：`src/inspect_ai/model/_model.py:764`，`Model.generate()`——`task_generate()` 调的就是这个方法。
- 内部被 `tenacity` 的 `@retry` 包裹（第 1179 行装饰器，第 1191 行 `async def generate()` 内层闭包），每次尝试（含每次重试）：
  1. `await emit_before_model_generate(...)`（第 1197 行）——**这是 inspect_ai 官方 Hooks 机制的介入点**，`inspect_trace` 用它来追踪 retry attempt。
  2. 缓存查找（如果开了 `--cache`）
  3. `self.api.generate(...)`——调到具体 provider（`ModelAPI` 抽象方法，定义在第 314 行；具体实现在 `src/inspect_ai/model/_providers/openai.py`、`vllm.py`、`openai_compatible.py` 等文件，这里才是真正拼 HTTP 请求、发出去的地方）
  4. `_record_model_interaction()`（第 1631 行）——创建/更新 `ModelEvent`，先以 `pending=True` 状态写入 transcript
  5. `complete()`（第 1669 行）——调用成功或失败后调用，把 `pending` 设为完成态，触发 `on_sample_event` Hook（`inspect_trace` 的三个检测器全靠这个 Hook 拿数据）
- **对照真实数据**：`eval_log_format.md` 第 7 节那条完整 `ModelEvent` JSON——`input`/`tools`/`output`/`call.request`/`call.response` 这几个字段，就是上面第 3、4 步分别往里塞的东西。

### 第 4 段：工具调用解析与执行

- **位置**：`src/inspect_ai/model/_call_tools.py:103`，`execute_tools()`（真正干活的是 `_execute_tools_impl`，第 134 行起）
- 从最后一条 assistant 消息的 `tool_calls` 字段出发，解析每个调用、按 `ToolDef.parallel` 标记分组成"stage"，同一 stage 内用 `anyio.create_task_group()` 并发执行（这是目标一 Q6"是否支持并行"字面意义上对应的代码），串行工具单独成一个 stage 起 barrier 作用。
- **对照真实数据**：`model_dataset_comparison_findings.md` 发现一里"工具 schema 占了重复 prefill 的 90%+"——这些工具 schema 就是从这里（`state.tools`）传给 `model.generate()` 的 `tools` 参数，每一轮循环都原样重发一遍，正是我们发现的那个 bug 的源头。

### 第 5 段：sample 收尾

- 循环退出（没有更多 tool_calls，或触发了 `message_limit`/`token_limit`）后，`state` 返回给 `plan` 里的下一个 solver（如果链条还没走完）或者直接进入打分阶段。
- 打分：`src/inspect_ai/scorer/`（比如我们用过的 `match`/`includes`/`bfcl_scorer`），产出记进 `sample.scores`。
- 组装：`EvalSample.summary()`（`src/inspect_ai/log/_log.py:542-579`）把完整 sample 收敛成一份摘要，连同完整版一起写进 `.eval` 文件。

## `inspect_trace` 在这条链路里的介入点

对照上面几段，我们自己 `src/inspect_trace/` 的检测器分别挂在哪：

| 检测器 | 挂在哪个 Hook | 对应源码里的哪一段 |
|---|---|---|
| `prefill_diff.py`（重复 prefill，含 `content_category`/`tool_call_id`） | `on_sample_event`，`ModelEvent` 分支 | 读第 3 段 `ModelEvent.input`（也就是第 2 段循环每轮传给 `model.generate()` 的 `state.messages` 快照） |
| `segment_tokens.py`（分段 token 估算） | 同上 | 读第 3 段 `ModelEvent.output` |
| `token_attribution.py`（需求一的归因视图） | 同上，接在 `prefill_diff`/`segment_tokens` 之后 | 无状态组合，不读新的源码位置——纯粹重组前两者算出的字段 |
| `attempt_groups.py`（retry 分组） | `on_before_model_generate` + `on_model_retry` + `on_sample_event` | 挂在第 3 段的 `@retry` 装饰器内层，`emit_before_model_generate` 每次尝试都触发 |
| `execution_topology.py`（需求三：执行拓扑） | `on_sample_event`，新增 `ToolEvent` 分支；`on_sample_end` 时 `finalize()` | 读第 4 段 `execute_tools()`/`_execute_tools_impl()` 产生的 `ToolEvent`（`timestamp`/`completed`/`working_time`） |
| `action_parsing.py`（需求四：解析失败 + 回填追踪） | `on_sample_event`，`ToolEvent` 分支 | 读第 4 段 `parse_tool_call()`/`validate_tool_input()` 失败时落在 `ToolEvent.error` 上的 `ToolCallError` |

这也是为什么之前调研 `docs/extensions-hooks.qmd` 会发现 `on_before_model_generate` 明确支持"修改 `data.input`/`data.tools`/`data.config`"——它就是在第 3 段第 1 步触发的，改了这里，第 3 步真正发出去的 HTTP 请求就会变。目标三如果要做"generation 层标准化干预接口"，这就是要扩展的确切位置。

### 目标一需求一至四的实现

重新梳理 `efficient-harness.md` 目标一之后（"研究问题"→"需求"），新增的三个模块分别对应需求一/三/四，都是在原有三个检测器的基础上做的：

- **需求一（`token_attribution.py`）**：不新建有状态的 tracker——`prefill_diff`（输入侧）和 `segment_tokens`（输出侧）已经把需要的数字都算出来了，`token_attribution.compose()` 只是把两者拼成一行，同时清楚标注哪些字段是真实计费值（`billed_*`，直接来自 `event.output.usage`）、哪些是 tiktoken 估算值（`*_tokens_estimate`）。这也是为什么 R1 没有对应的新 Hook 挂载点：它接在 `prefill_diff`/`segment_tokens` 写盘之后，纯粹是组合逻辑。
- **需求三（`execution_topology.py`）**：并行度靠时间戳推断，不靠 inspect_ai 内部的 `ToolDef.parallel` 声明（那个字段根本没暴露在 `ToolEvent` 上）——同一 stage 内任意两个 `ToolEvent` 的 `[timestamp, completed]` 区间如果真的重叠，才算 `observed_parallel`。这个设计选择是刻意的：对效率研究来说，"实际跑的时候是不是真的并发"比"这个工具声明自己可以并发"更有意义。`tool_call_id -> parent_model_event_uuid` 这份索引在 `observe_model_event()` 里建立（从 `event.output.message.tool_calls` 提取），`observe_tool_event()` 到达的 `ToolEvent` 靠它归属到正确的 stage——这个顺序能保证是因为第 4 段的 `execute_tools()` 一定在第 3 段这次 `ModelEvent` 落盘之后才被调用。**没有回滚字段**是刻意的：读了 `solver/_fork.py`（`fork()` 让每个分支都跑完，不丢弃任何分支）、`event/_branch.py`/`event/_anchor.py`（只是 UI 时间线标记）、`event/_checkpoint.py`（崩溃恢复用的磁盘持久化）之后确认 inspect_ai 没有真正的语义回滚机制，能拿到的最接近的类比就是 `attempt_group` 里已经记录的重试。
- **需求四（`action_parsing.py`）**：`tool_call_id`（即 `ToolCall.id` == `ChatMessageTool.tool_call_id`）是贯穿"这次调用→这次解析结果→下一步 `prefill_diff` 里的新消息"三者的稳定 join key。触发一次真实的解析错误很简单：给 `ToolCall.arguments` 漏填一个必填参数，`model/_call_tools.py` 里 `validate_tool_input()` 就会真的报错，走到 `ToolCallError("parsing", ...)`，不需要在测试里手工模拟异常。

## 建议的阅读顺序（照着勾）

1. `src/inspect_ai/solver/_task_state.py`——先搞清楚 `TaskState` 这个"贯穿全程被传递、被修改的对象"到底长什么样，见本文上面"`TaskState` 详解"一节。
2. `src/inspect_ai/_eval/task/generate.py`——整个文件只有 66 行，是全篇最该精读的一段，读完这个再看别的都会更顺。
3. `src/inspect_ai/model/_model.py` 的 `Model.generate()`（764 行起）——只看到 `self.api.generate()` 那一行为止，不用深入某个具体 provider。
4. 挑一个 provider 实现细看（比如 `src/inspect_ai/model/_providers/openai_compatible.py`，因为我们本地 vLLM 部署用的就是这个），对照真实发出去的 `call.request` JSON。
5. `src/inspect_ai/model/_call_tools.py` 的 `execute_tools`/`_execute_tools_impl`。
6. `src/inspect_ai/hooks/_hooks.py`——回过头看 Hooks 系统怎么织入上面 3、5 两步。
7. 对照着读我们自己的 `src/inspect_trace/src/inspect_trace/hooks.py`——这时候应该能一眼看出每个回调对应源码的哪一行。

## 动手练习（用我们已有的真实数据，不用重新跑实验）

1. 打开 `runs/compare/deepseek_bfcl/logs/*.eval`（`inspect view` 或 `read_eval_log`），挑一条样本，数一下 `events` 里 `event=="model"` 的条数，跟 `.inspect_trace/*/*/sample-*.jsonl` 里对应 `sample_uuid` 的 `prefill_diff` 记录条数应该完全一致——这是在验证第 2 段"每转一圈循环产生一个 `ModelEvent`"这句话是不是真的。
2. 挑 `runs/compare/local_bfcl/`（本地 vLLM 那次跑的，`emulate_tools=true`）对比同一条 BFCL 样本在 `runs/compare/deepseek_bfcl/`（原生 tool call）里的样子——因为本地这边是 client 端模拟工具调用（不走 `message.tool_calls` 这个原生字段），去读一下 `inspect_ai/model/_providers/openai_compatible.py` 里 `emulate_tools` 相关的解析逻辑，看它是怎么把纯文本输出"翻译"回结构化 `tool_calls` 塞进第 2 段循环期待的位置的。
3. 在 `src/inspect_ai/_eval/task/generate.py` 里临时加一行 `print(len(state.messages))`（本地实验用，不要提交），重新跑一次 `run_bfcl_benchmark.sh --limit 1`，肉眼看着这个数字一步步涨上去——比看日志更直观地感受第 2 段那个 while 循环。

## 相关文档

- 概念入门（不涉及源码）：[`inspect_ai_quickstart.md`](./inspect_ai_quickstart.md)
- `.eval` 文件每个字段详解：[`eval_log_format.md`](./eval_log_format.md)
- 这次没细读、下次该补的官方文档清单：[`inspect_ai_docs_examples_audit.md`](./inspect_ai_docs_examples_audit.md)

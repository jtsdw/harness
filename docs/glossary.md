# 术语表

按类别分组，不是字母序——查的时候先想"这个词是哪个层面的"，比纯字母序好定位。每条给最简说明 + 定义/详细讨论所在的文档或源码位置，不重复展开细节。

## inspect_ai 核心概念

| 术语 | 说明 | 详见 |
|---|---|---|
| **Task** | 一次评测的完整定义：数据集 + solver 链 + scorer。`@task` 装饰的函数返回一个 `Task` 对象 | `inspect_ai_essential_docs.md` 第一层 |
| **Sample** | 数据集里的一条数据（输入 + 目标答案）。`Sample.input` 是这条数据最原始的形态，还没经过任何 solver 加工 | `source_code_reading_guide.md` 第 0 段 |
| **Solver** | `(TaskState, Generate) -> TaskState` 的协议。可以直接改 `state.messages`（prompt 构造），也可以调用 `generate` 触发模型调用。`system_message`/`prompt_template`/`basic_agent()`/`react()` 都是具体的 Solver | `source_code_reading_guide.md` 第 1 段 |
| **Generate** | 传给每个 Solver 的一个可调用对象，真正调模型、可选地循环解析 tool call。`tool_calls="loop"` 时就是 ReAct 循环本身 | `source_code_reading_guide.md` 第 2 段（`task_generate()`） |
| **TaskState** | 贯穿一条 sample 执行始终、被各 Solver 传递/修改的对象，装着 `messages`/`tools`/`output`/`completed` 等字段。不是"最小执行单元"，更像 web 框架里贯穿请求生命周期的 request context | `source_code_reading_guide.md`"`TaskState` 详解"一节 |
| **Model** / **ModelAPI** | `Model` 是面向调用方的公开包装（含 retry、Hooks、缓存）；`ModelAPI` 是每个 provider（OpenAI/vLLM/...)要实现的抽象基类，真正拼 HTTP 请求 | `source_code_reading_guide.md` 第 3 段 |
| **Tool** / **ToolInfo** | `Tool` 是一个可被模型调用的 Python 函数；`ToolInfo` 是它的 JSON Schema 描述（名字/描述/参数），每次调用都会被发给模型 | `goal1_real_benchmark_findings.md`（工具 schema 重复占比发现） |
| **Scorer** | 给一条 sample 的输出打分的函数（比如 `match`/`includes`/`bfcl_scorer`），产出记进 `sample.scores` | `eval_log_format.md` 第 6 节 |
| **EvalLog** | `.eval` 文件反序列化后的顶层对象：`eval`(身份配置)/`plan`(solver 链)/`results`(打分结果)/`stats`(耗时用量)/`samples`(逐条数据) | `eval_log_format.md` 全文 |
| **ModelEvent** | 一次模型调用的完整记录：`input`(发送的消息列表)、`tools`、`output`、`call.request`/`call.response`(原始收发)。`inspect_trace` 三个检测器全靠它 | `eval_log_format.md` 第 7 节 |
| **ToolEvent** | 一次工具调用的记录 | `eval_log_format.md` 第 7 节 |
| **Span** (`span_begin`/`span_end`) | 标记"一段有起止的执行"的事件对，可以嵌套成树（比如 `solvers` 大 span 底下套 `system_message`/`prompt_template` 小 span）。目标一 Q6 分析并行/等待结构时会用到 | `eval_log_format.md` 第 7 节 |
| **Transcript** | 一条 sample 执行期间产生的完整 event 流的运行时载体 | `inspect_ai_essential_docs.md` |
| **Hooks** | 官方插件机制：继承 `Hooks` 基类实现回调（`on_before_model_generate`/`on_model_retry`/`on_sample_event`/`on_model_usage`/...），通过 entry_points 自动发现、无需改 inspect_ai 源码即可挂进去观测/干预。`inspect_trace` 整个包就是一个 Hooks 插件 | `inspect_ai_quickstart.md` 第 5 节 |
| **`emulate_tools`** | `openai-api` provider 的一个参数：当 provider 本身不支持原生结构化 tool call 时，让 inspect_ai 自己用 prompt 引导模型输出、再从文本里解析出结构化 tool call。我们本地 vLLM（`vllm==0.6.3.post1`，没有 `--tool-call-parser`）就是靠这个跑通的 | `local_model_deployment.md` |
| **`attachment://...`** | `.eval` 文件里的内容寻址去重机制：同一段长文本（few-shot 示例、系统提示词）只存一份在 `attachments` 字典里，各处用哈希引用它 | `eval_log_format.md` 第 8 节 |

## `inspect_trace`（我们自己的目标一实现）专属术语

| 术语 | 说明 | 详见 |
|---|---|---|
| **repeated prefill（重复 prefill）** | 同一段 context（消息或工具 schema）在连续多轮模型调用里被原样重新处理——这是整个项目最核心的研究对象。区分"消息级去重"（我们做的，语义近似）和"provider 真实 prefix-cache 命中"（字节级，两者不完全等价） | `goal1_real_benchmark_findings.md` 发现一/二 |
| **`prefill_diff`** | `inspect_trace` 三个检测器之一：对每步 `ModelEvent.input`（消息）和 `event.tools`（工具 schema）做 new/reused/dup_in_step 分类，全历史比较（不是滑动窗口） | `src/inspect_trace/src/inspect_trace/prefill_diff.py` |
| **`segment_tokens`** | 检测器之二：把模型输出按 reasoning/tool-call/server-tool-use/text 拆分做 token 估算（tiktoken，标 `estimated_*`），跟真实计费值（`billed_*`）并列、不覆盖 | `src/inspect_trace/src/inspect_trace/segment_tokens.py` |
| **`attempt_group`** | 检测器之三：把同一逻辑请求的多次 HTTP 重试尝试串成一组，统计浪费的 wait time | `src/inspect_trace/src/inspect_trace/attempt_groups.py` |
| **`tools_new`/`tools_reused`** | `prefill_diff` 记录里，工具 schema 这一维度的新增/复用计数（跟消息维度的 `new_messages`/`reused_messages` 是平行、独立的两组字段，不混在一起统计） | `goal1_real_benchmark_findings.md` |
| **`INSPECT_TRACE_DIR`** | 控制 `inspect_trace` 衍生数据落盘位置的环境变量，默认 `./.inspect_trace` | `inspect_ai_quickstart.md` 第 6 节 |
| **`content_category`** | `prefill_diff` 里每条消息的分类标签：`system_template`（系统提示词）还是 `conversation`（普通对话/工具结果）。目标一需求一/需求二共用的字段，用来把"静态重发的系统提示词"和"随轮次增长的对话历史"分开统计 | `source_code_reading_guide.md`"目标一需求一至四的实现"一节 |
| **`execution_topology`** | 目标一需求三的产出记录：把一个 sample 的 `ToolEvent` 按其所属的 `ModelEvent` 分组成 stage，标注每个 stage 是否观测到真并行（`observed_parallel`，靠时间窗口重叠推断，不是 inspect_ai 声明的 `ToolDef.parallel`）、`model_waiting_for_tool_seconds`/`tool_waiting_for_model_seconds`（推断值）与 `tool_semaphore_wait_seconds`（inspect_ai 自己的真实值）三种等待时间的区分。刻意没有"回滚"字段——inspect_ai 没有真正的回滚机制 | `goal1_r3_r4_real_benchmark_findings.md` |
| **`action_parsing`** | 目标一需求四的产出记录：每个 `ToolEvent` 的解析/校验错误（`error_type`/`error_message`），以及 `tool_call_id` 这个 join key——用它可以从这次工具调用一路追到下一步 `prefill_diff` 里对应的新消息，串起"调用→解析结果→回填" | `source_code_reading_guide.md`"目标一需求一至四的实现"一节 |
| **`token_attribution`** | 目标一需求一的产出记录：把 `prefill_diff`（输入侧）和 `segment_tokens`（输出侧）拼成一行，给出 system template / tool schema / conversation / reasoning / tool-calling / final-response 六类的 token 归因，同时标清哪些是真实计费值、哪些是估算值 | `source_code_reading_guide.md`"目标一需求一至四的实现"一节 |

## 效率研究一般术语（来自 `efficient-harness.md`）

| 术语 | 说明 |
|---|---|
| **token 层 / model invocation 层 / episode 层** | 目标二三层 profiling 体系：token 层管输入输出/重复/分类计数；model invocation 层管单次调用内部的 queueing/prefill/decode/TTFT 等；episode 层管整条样本的端到端 latency/success/cost |
| **exclusive time / inclusive time / critical-path time** | 三种时间统计口径：exclusive 是某阶段自己占用的时间，inclusive 是含子事件的总时间，critical-path 是真正决定端到端耗时的路径（并行阶段不能简单相加） |
| **online execution / offline replay** | 目标四的两种模式：online 是正常在线执行产生新轨迹；offline replay 是固定已有轨迹/工具响应，只重跑某一环节，用于控制变量的对照实验 |
| **cache hit/miss** | provider 真实的 prompt cache 命中情况，对应 `ModelUsage.input_tokens_cache_read`/`input_tokens_cache_write`（真实计费口径，不是我们的估算） |

## 基础设施 / 外部工具

| 术语 | 说明 | 详见 |
|---|---|---|
| **uv** | Python 项目/环境管理工具，本项目的标准环境管理方式（不用 conda）。"一个项目一个环境"是自动的，不需要手动 create/activate；用全局 cache + 硬链接去重，不会因为多个项目重复占用磁盘 | `inspect_ai_quickstart.md` 第 2.1 节 |
| **vLLM** | 本地模型推理服务框架，我们用它在 16GB 显存的 RTX 2000 Ada 上跑 `Qwen2.5-3B-Instruct` | `local_model_deployment.md` |
| **BFCL** | Berkeley Function-Calling Leaderboard，我们目标一真实验证用的主 benchmark（多轮、工具调用密集） | `datasets.md` |
| **GSM8K** | 数学应用题数据集，单轮、无工具调用，作为 BFCL 复杂轨迹之外的简单场景对照组 | `datasets.md` |
| **`multi_turn_base`** | BFCL 里的一个具体 category：基础多步文件系统操作任务 | `goal1_real_benchmark_findings.md` |

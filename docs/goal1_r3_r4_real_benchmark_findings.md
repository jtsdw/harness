# 目标一需求三/需求四：真实 benchmark 验证结果

目标一重新梳理为四条需求（见 [`efficient-harness.md`](./efficient-harness.md)）之后，`inspect_trace` 新增了 `execution_topology`（需求三）、`action_parsing`（需求四）、`token_attribution`（需求一）三个模块。本文记录在真实 BFCL 数据上跑这些新代码得到的结果——延续 [`goal1_real_benchmark_findings.md`](./goal1_real_benchmark_findings.md) 的风格：如实报告观察到的现象，包括"没观察到预期现象"本身。

配套可视化面板：[`goal1_r3_r4_dashboard.html`](./goal1_r3_r4_dashboard.html)（本地打开即可，不需要联网/起服务），按需求一至四逐条对照本文的真实数据。

## 背景

- 模型：本地 vLLM（`Qwen/Qwen2.5-3B-Instruct`，`vllm==0.6.3.post1`，`emulate_tools=true`）。
- 数据集：BFCL 两个 category——`live_parallel`（本来想直接验证"并行"）、`multi_turn_base`（已知会产生真实 tool 执行）。
- 复现命令：
  ```bash
  cd /home/liuyingen/code/efficient-harness/inspect_trace

  MODEL="openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" MODEL_ARGS="emulate_tools=true" \
  VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
  CATEGORIES="live_parallel" LIMIT=6 MAX_CONNECTIONS=1 \
  ./scripts/run_bfcl_benchmark.sh

  MODEL="openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" MODEL_ARGS="emulate_tools=true" \
  VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
  CATEGORIES="multi_turn_base" LIMIT=5 MAX_CONNECTIONS=1 \
  OUTPUT_DIR="/home/liuyingen/code/efficient-harness/runs/goal1_bfcl_multi_turn_base_r3r4" \
  ./scripts/run_bfcl_benchmark.sh
  ```
- 产出：`runs/goal1_bfcl_live_parallel/`、`runs/goal1_bfcl_multi_turn_base_r3r4/`（`.eval` 日志 + `inspect_trace` JSONL）。

`MAX_CONNECTIONS` 是这次新加到 `run_bfcl_benchmark.sh` 的参数——过程中第一次跑 `live_parallel`（`LIMIT=8`，默认并发）时，两个 `/v1/chat/completions` 请求同时打到本地 vLLM，直接把 `MQLLMEngine` 打死（`CRITICAL launcher.py:99] MQLLMEngine is already dead`），整个服务崩溃退出。这不是 `inspect_trace` 的 bug，是本地这套锁定版本（`vllm==0.6.3.post1`，为了兼容旧驱动）在并发请求下不够健壮——单卡跑本地小模型时用 `MAX_CONNECTIONS=1` 序列化请求就稳定了，之后两次跑都没再复现崩溃。

## 需求三：执行拓扑

### 发现一：`live_parallel` 这类 category 结构上就不会产生 `ToolEvent`

`live_parallel`（以及所有 `live_*`/`parallel`/`multiple` 这类 BFCL v1/v2 category）在 `inspect_evals/bfcl` 里走的是 `single_turn_solver`：只调用一次 `generate()`，直接对模型输出的 `tool_calls` 做 AST 匹配打分，**从不调用 `execute_tools()`**。真实验证：6 个 `live_parallel` 样本里，`execution_topology` 记录全部是 `total_stages=0`、`total_tool_calls=0`，`linear=True`（因为没有任何 stage，`all(...)` 在空列表上恒真）——即便某个样本的模型输出里确实包含了两个 `tool_calls`（真实看到过 `get_current_weather` 被同时提议调用两次，参数分别是"北京"和"上海"），因为这条路径根本不经过 `execute_tools()`，所以连不上任何 `ToolEvent`，我们的检测器对这类 category **结构性地看不到任何东西**——这不是 bug，是 `execution_topology`/`action_parsing` 两个检测器的设计前提（"读 `ToolEvent`"）跟这类 category 的实现方式不匹配。只有 `multi_turn_*`（v3，走 `multi_turn_solver`，真实调用一个模拟后端）这类 category 才会产生 `ToolEvent`。

这也纠正了当初选 `live_parallel` 这个名字来验证"并行"的直觉——category 名字里的"parallel"指的是"这条数据期望模型一次提议多个函数调用"（AST 匹配维度的"parallel"），跟我们目标三想测的"tool 执行时是否真的并发"完全是两个不同的"parallel"。

### 发现二：`multi_turn_base` 上真实观察到"一轮提议多个 tool_calls"，但从未观察到真并发

`multi_turn_base`（5 个样本）的 `execution_topology` 记录：

| sample | model_calls | stages | tool_calls | linear |
|---|---|---|---|---|
| L9kLDBwH | 4 | 2 | 2 | True |
| gTbuDm28 | 9 | 5 | 5 | True |
| 9fSBXU2J | 7 | 3 | 5 | **False** |
| PztbbKvR | 7 | 2 | 2 | True |
| CfNaZH3J | 7 | 4 | 6 | **False** |

`9fSBXU2J`/`CfNaZH3J` 两条里确实出现了同一个 model turn 提议 2 个 tool_calls 的 stage（比如 `['mkdir', 'mv']`、`['message_login', 'sendMessage']`），`linear` 因此正确判为 `False`。但这几个 stage 的 `observed_parallel` 全部是 `False`，`max_observed_concurrency` 全部是 `1`——两个 `ToolEvent` 的时间窗口完全不重叠，是严格先后执行的。原因跟设计阶段读 `inspect_ai/tool/_tool.py` 时确认的一致：`@tool` 装饰器的 `parallel` 参数**默认 `False`**（需要显式opt-in，且要求"审计过并发安全性"），BFCL 注册的这些工具没有设置 `parallel=True`，所以即便模型一次提议多个调用，inspect_ai 的 `_execute_tools_impl` 也会把它们当成独立的顺序 barrier 逐个执行，绝不会真的并发。

`unmatched_tool_event_uuids` 五个样本全部是 `[]`——`tool_call_id -> parent_model_event_uuid` 的索引在真实数据上没有出现任何漏配对，验证了这份索引设计（依赖"`ToolEvent` 一定晚于其 parent `ModelEvent` 落盘"这个顺序保证）在真实多轮轨迹上是可靠的。

没有观察到、也不可能观察到"回滚"——跟设计阶段的结论一致（inspect_ai 没有真正的回滚机制），这不是这次验证的新发现，只是再次确认。

**结论**：需求三的检测器本身工作正常（拓扑重建、并行推断、waiting time、join key 全部在真实数据上验证通过），但"观察到真并行"这件事在当前的 BFCL + 本地小模型 + 未开 `parallel=True` 的工具集组合下没有发生——这跟检测器无关，是被测对象（BFCL 的工具注册方式）决定的。想真的观察到 `observed_parallel=True`，需要换一个把工具标了 `parallel=True` 的 benchmark/agent，而不是换 BFCL category。

## 需求四：action parsing 与观察回填追踪

`multi_turn_base` 的 5 个样本一共产生 20 条 `action_parsing` 记录，其中 **6 条是真实的解析/校验错误**（不是构造的，是本地小模型自己犯的错）：

| 触发原因 | 出现次数 | 示例 |
|---|---|---|
| JSON 语法错误（漏逗号/多引号） | 2 | `{"name": "mv,""arguments": {...}}`（`name` 后面少了个逗号，字符串引号也错位了） |
| emulate_tools 解析器嵌套 `<tool_call>` 标签失败 | 1 | 模型在一个 `<tool_call>` 标签内又吐出了一个 `<tool_call>{"name": "message_login", ...}` |
| 幻觉出不存在的工具名 | 2 | `Tool sendMessage not found`（`sendMessage` 根本不在这条 sample 暴露的工具列表里） |
| schema 校验失败（多传了参数） | 1 | `view_messages_sent` 被传了一个 schema 里没定义的 `user_id` 参数，`validate_tool_input()` 报 `Additional properties are not allowed` |

全部 6 条的 `error_type` 都是 `"parsing"`，`is_parse_or_validation_error` 全部是 `True`——`ToolCallError("parsing", ...)` 这条真实路径（`model/_call_tools.py` 里 `parse_tool_call()` 的 JSON 解析失败 + `validate_tool_input()` 的 schema 校验失败）在小模型的真实输出上被完整触发了三种不同子类型，不需要像单测那样人工构造缺参数的 `ToolCall`——这验证了当初设计时的判断是对的：真实场景（尤其是弱一些的模型）确实会自然产生这类错误，`action_parsing` 记录的价值不是纸面上的。

`tool_call_id` join key 在全部 6 条错误记录上都正确关联到了下一步 `prefill_diff` 里对应的新消息（人工核对过其中 3 条，`tool_call_id` 完全匹配），"调用→解析结果→回填"这条链路在真实数据上是通的。

## 需求一/需求二：一个意料之外但值得记录的限制

按计划要交叉核对 `token_attribution` 里 `system_template_tokens_estimate` 是否符合预期（step 1 新增、step 2 起复用）。真实结果：**`multi_turn_base` 全部 5 个样本，所有步骤的 `system_template_messages`/`system_template_tokens_estimate` 都是 0**，不是"步骤 2 起变成 reused"，是从头到尾都不存在。

查了一下原始 `.eval` 日志：这条 BFCL 任务传给模型的第一条消息 `role` 是 `"user"`，不是 `"system"`——emulate_tools 路径下，系统指令（"You are a knowledgable assistant..." + 工具 XML schema 说明）被拼进了第一条 **user** 消息的文本里，而不是作为独立的 `ChatMessageSystem`。`content_category` 目前的判定逻辑就是简单的 `message.role == "system"`（见 `prefill_diff.py`），这个真实场景下完全命中不到。

这是一个**已确认的真实限制**，不是 bug：`content_category` 的检测依赖"系统指令用 `role="system"` 表达"这个假设，在原生 tool-call 支持的 provider（比如 DeepSeek 的 OpenAI-compatible 接口）上这个假设通常成立，但在 `emulate_tools=true` 这条 client-side 模拟路径上不成立——系统指令被合并进了 user 消息。工具 schema 的重复检测不受影响（`tools_new`/`tools_reused` 走的是完全独立的 `event.tools` 字段，跟消息 role 无关，这次验证里 20 个工具 schema 在 step 1 全新、step 2 起全部复用，逐步核对过完全正确）。

## 附加发现：原始调用语句、与 ToolSpec 一类加速研究的关联

这一节不对应四条需求里的某一条，是在回答"模型到底怎么调用 tool"这个问题时，顺着挖 `ModelEvent.call.response`（provider 原始 HTTP 响应，前几篇文档一直没细看过这个字段）挖出来的。

### 原始"调用语句"长什么样、藏在哪

我们展示过的所有内容（`prefill_diff`/`action_parsing`/仪表盘）全部建立在 inspect_ai 已经解析好的 `ChatMessageAssistant.tool_calls`（结构化 `ToolCall` 对象）这层抽象之上，从来没有展示过模型真正逐 token 生成的原始文本。真实调出来看（同一条样本 `multi_turn_base_0` 第 3 步），`ModelEvent.call.response`（provider 原始响应）里是这样的：

```json
"message": {
  "content": "<tool_call>{\"name\": \"mkdir\", \"arguments\": {\"dir_name\": \"temp\"}}</tool_call>\n\n<tool_call>{\"name\": \"mv\", \"arguments\": {\"source\": \"final_report.pdf\", \"destination\": \"temp/final_report.pdf\"}}"
}
```

这段 `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` 文本，就是 ToolSpec（`/home/liuyingen/code/ToolSpec`，一篇研究用 schema-aware + retrieval-augmented speculative decoding 加速 tool call 生成的论文）里研究的对象——两者格式几乎同构（`arguments` vs 论文里的 `parameters`，命名差异而已）。ToolSpec 能加速的原理是：这段文本里 `<tool_call>` 标签、JSON 标点、工具名、参数名全部能从 system prompt 里的 schema 提前推出来，真正需要模型"决定"的只有参数值，所以大部分 token 可以投机解码、批量验证。

**这个问题是否在我们的项目范式下存在，分路径回答**：

- **本地 vLLM + `emulate_tools=true`**：问题**完全存在**，跟 ToolSpec 的实验设置结构同构——模型确实在逐 token 生成这整段 JSON-ish 文本，且大部分是 schema 决定的模板。这也是我们系统里唯一一条能拿到"原始生成 token"、因而唯一有条件用 ToolSpec 这类方法去加速/介入的路径。
- **DeepSeek 原生 tool-calling（hosted）**：问题在客户端层面**不可见**——DeepSeek 的 API 直接返回解析好的 `tool_calls`，服务端内部是怎么生成这段调用语句的，我们完全看不到也控制不了。这跟目标二发现的"DeepSeek 拿不到 TTFT/decode time"是同一类结构性限制的另一种表现：凡是 provider 内部发生的事，这套 harness 一律看不见，也就无从测量或加速。

### 顺带发现的两类真实 bug：`<tool_call>` 标签解析故障

深挖了好几条真实样本的原始 `call.response` 之后确认：模型输出里"有没有显性 `<tool_call>` 文本"不是样本级别的属性，是**逐 step**的现象，取决于这一步的标签有没有真正闭合——而且实际观察到两种不同的故障机制，可见效果完全不同。

**机制一：标签没写完就停了（静默丢失）**

`multi_turn_base_0` 第 3 步的真实响应里，模型其实想调用两个工具（`mkdir` 和 `mv`），但第二个 `<tool_call>` 标签因为 `finish_reason: "stop"` 提前触发，**没有闭合**就结束了。`inspect_ai` 的 `emulate_tools` 解析器（`model/_providers/util/hf_handler.py`，正则 `<tool_call>((?:.|\n)*?)</tool_call>`）要求标签必须闭合才能匹配——用这段真实文本重放这个正则验证过，确实只能匹配到 `mkdir`，`mv` 那段完整地落进了"未匹配剩余文本"（对应 `xml_extract` 返回的 `other_content`）。

对照真实的 `execution_topology` 记录，这一步（`stage_index=3`，`parent_model_event_uuid=b4nHrRcCdySitruHfU5vPx`）确实只有 `['mkdir']`，`tool_count=1`——`mv` 这次调用**没有产生任何 `ToolEvent`、没有触发任何 `ToolCallError`，就这么静默消失了**。这跟需求四表格里那 6 条"有报错留痕"的解析失败不是一回事：那 6 条至少被 `action_parsing` 捕获、留了痕；这一条是**连痕迹都没有的静默丢失**，模型的真实意图凭空消失，我们现有的任何检测器都看不到它发生过——只能靠这次手工深挖原始 response 才挖出来。下一步（第 4 步）能看到模型自己重新发起了一次 `mv` 调用，是它"发现"第一次没生效后的自我修正，间接印证了这次丢失确实发生过。

这不是孤例：`multi_turn_base_0` 第 5-7 步、`multi_turn_base_10` 第 5-7 步都是同一种模式——`finish_reason: "stop"`，JSON 内容写完整了，但从没写 `</tool_call>`。`multi_turn_base_10` 第 5 步甚至更奇怪：`'<tool_call> {"name": "touch", "arguments": {"file_name": "notes.md"}} ⟶ {"error": "", "response": ""}'`——模型在没闭合标签的情况下，还自己"脑补"了一段 `⟶ {"error": "", "response": ""}`，像是在模仿训练数据里"调用 → 结果"的记录格式，而不是老老实实停下来等真实工具结果。

**机制二：多个标签互相"打架"，非贪婪正则抓错闭合位置**

`multi_turn_base_101` 第 4 步的真实响应是这样的：

```
...
<tool_call>
<tool_call>{"name": "message_login", "arguments": {"user_id": "USR001"} }</tool_call>

<tool_call>{"name": "sendMessage", "arguments": {...}} </tool_call>
```

模型多写了一个**孤立的、没内容的 `<tool_call>` 开标签**，后面才是真正想写的调用。因为正则是非贪婪匹配（找"最近的"闭合标签），第一次匹配从这个孤立开标签开始，一路匹配到**本该属于第二个标签**的那个 `</tool_call>`——被"吃"进去的内容变成 `\n<tool_call>{"name": "message_login"...`，开头多了一段 `<tool_call>` 前缀，不是合法 JSON，`json.loads` 直接失败，这次调用退化成 `function="unknown"`。正则从这里继续往后扫，后面那个 `sendMessage` 标签因为前面没有干扰了，反而解析成功——这就是为什么这一步真实的 `tool_calls` 是 `['unknown', 'sendMessage']`。

跟机制一不一样：这次标签其实"闭合"了，只是闭合到了错误的位置，所以看不到显性的 `<tool_call>` 原始文本残留（正则确实吃掉了），只会看到一个看起来莫名其妙的 `unknown` 调用——是需求四表格里"JSON 语法错误"那一类里，此前没细究过具体成因的一条。

**这跟 ToolSpec 一类方法的关联不只是"能加速"**：grammar-constrained 解码（强制闭合标签/合法 JSON、禁止在一个调用没结束前开始下一个）这类做法，除了能像 ToolSpec 一样加速生成，还能顺带**消除**机制一（格式没写完整丢调用）和机制二（标签互相污染导致张冠李戴）这两类正确性问题——这是加速研究之外一个意外但有价值的关联，值得在评估任何 schema-aware 解码方法时一并纳入"正确性收益"，不只是"速度收益"。

## 附加发现二：`final_response` 归因的一类错误分类

继续深挖同一条样本（`multi_turn_base_0`，7 步）时发现的另一个问题，跟上面"静默丢失"同根同源，但表现在需求一的 token 归因上。

真实数据：这条样本从第 3 步起，模型的结构化 `tool_calls` 变成空数组（不是模型不想调用工具，是它写的 JSON/标签格式又错了，属于需求四已经在追踪的那类失败），但 `token_attribution` 记录里的 `final_response_tokens_estimate` 从这一步起反而从 0 涨到 30～50：

| step | 结构化 `tool_calls` | `final_response_tokens_estimate` | 真实原始文本 |
|---|---|---|---|
| 1-2 | 有（部分解析失败） | 0 | — |
| 3-7 | **空数组** | 30～50 | `<tool_call>{"name": "mv", ...}` 这类**解析失败后原样留在 `message.content` 里的残留文本** |

看起来像是"模型开始给出自然语言最终答案了"，但对照原始文本，这几步模型其实**仍然在尝试调用工具，只是又失败了**——`segment_tokens.py` 按内容块类型分类（`ContentReasoning` → reasoning，`ContentToolUse` → server tool use，普通 `ContentText` → 归进 `final_response`），它没有办法区分"这个 `ContentText` 块是真的自然语言回答"还是"这个 `ContentText` 块只是解析失败后原样剩下的 `<tool_call>...` 语法残骸"——两者在数据类型上完全一样，都是一段字符串。

**结论**：`token_attribution`/`segment_tokens` 的 `final_response` 这一类目前是**内容块类型驱动，不是语义驱动**的，在 emulate_tools 场景下会把"解析失败的工具调用残留文本"误计入"模型给出的最终自然语言回答"。这跟需求一/需求二那条"system template 检测在 emulate_tools 下失效"是同一类根因（分类逻辑基于结构特征，而 emulate_tools 这条 client-side 模拟路径的结构跟原生 tool-calling API 不一样，会打破这些结构假设）——暂不改动分类逻辑（会引入"猜测哪段文本其实是失败的工具调用"这类脆弱启发式），先如实记录：**看到 `final_response` 有数值，不能默认这是模型真的给了最终答案，要对照当前步 `tool_calls` 是否为空、以及需求四同一步的 `action_parsing` 记录有没有报错**，三者一起看才是准确的。

## 结论与后续

- 需求三、需求四的检测器代码本身在真实数据上验证通过（拓扑重建、join key、real error capture 都对），意料之外的是被测对象（BFCL 工具注册、emulate_tools 的 prompt 构造方式）本身的两个特性限制了能观察到的现象：工具没开 `parallel=True` 导致永远看不到真并发；系统指令被塞进 user 消息导致 `content_category` 的 system_template 分类失效。
- 如果想真的看到 `observed_parallel=True`，需要一个把 `@tool(parallel=True)` 用起来的真实 agent/benchmark，不是继续在 BFCL 上换 category。
- 如果想让 `content_category` 在 emulate_tools 场景下也生效，需要把判定逻辑从"看 `role`"换成"看这条消息是不是第一条、且内容里包含固定的系统指令模板"这类更脆弱的启发式——目前判断不值得为了这一个 provider 特例牺牲现有逻辑的简单性，先如实记录限制。
- 本次顺带给 `run_bfcl_benchmark.sh` 加了 `MAX_CONNECTIONS` 参数（默认不设，行为不变），是从这次真实踩到的 vLLM 并发崩溃里直接得出的、必要的复现稳定性修复。
- 挖 `ModelEvent.call.response` 找到了模型真实生成的原始"调用语句"，确认本地 vLLM 路径下这个问题跟 ToolSpec 论文研究的场景结构同构（hosted DeepSeek 路径则完全不可见）；顺带在多条真实样本里确认了两类需求四现有检测器覆盖不到的标签解析故障——机制一是未闭合标签导致调用**静默丢失**（无报错、无 `ToolEvent`），机制二是多个标签互相干扰、非贪婪正则抓错闭合位置导致**张冠李戴**（退化成 `function="unknown"`，但至少还留了痕）——详见上面"附加发现"一节。
- 同一条样本还暴露出 `final_response` 归因是内容块类型驱动、不是语义驱动的，emulate_tools 场景下会把解析失败的工具调用残留文本误计成"模型的最终自然语言回答"——详见"附加发现二"一节，暂定为已知限制，不改动分类逻辑。

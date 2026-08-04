# 目标一真实 benchmark 验证：观察结果、发现的 bug 与结论

此前对 `inspect_trace`（目标一实现）的验证都是在手写的 3 步玩具轨迹上做的。本文记录第一次在真实 benchmark + 真实模型上跑通之后的观察、发现的一个真实 bug（已修复）、一个更深的科研发现，以及"目标一在真实场景下到底行不行"的结论。

## 实验设置

- Benchmark：`inspect_evals/bfcl`，`multi_turn_base` 类别（GorillaFileSystem 风格的多轮文件系统操作任务，官方数据集，200 条中取样本跑）。
- 模型：DeepSeek-chat，通过 `openai/deepseek-chat`（OpenAI-compatible 接口，`OPENAI_BASE_URL` 指向 `https://api.deepseek.com/`）真实调用，非 mock。
- 跑了三轮：第一轮（bug 修复前）2 条样本，分别产生 41 步和 24 步的真实多轮工具调用轨迹，用于发现下面的 bug；第二轮（修复后，验证性质，未持久化）；第三轮（修复后，持久化保存）2 条样本，分别产生 27 步和 22 步的轨迹。三轮步数不完全一致是模型输出随机性导致的正常现象，不影响下面的结论——尤其是工具 schema 部分的数值（每条样本的工具总数、每个工具的 token 估算）在三轮里完全一致，因为这部分只取决于数据集里固定的工具定义，不取决于模型怎么跑。
- `INSPECT_TRACE_DIR` 全程开启，`inspect_trace` 的三个检测器（`prefill_diff`/`segment_tokens`/`attempt_groups`）全程记录。
- **原始产出已持久化保存**在 `/home/liuyingen/code/efficient-harness/runs/goal1_bfcl_multi_turn_base/`（`.eval` 原始日志 + `.inspect_trace/` 衍生 JSONL），已加入 `.gitignore`（`/runs/` 和 `.inspect_trace/`），不会被提交，但不会被清理脚本误删，可以随时用 `inspect view --log-dir runs/goal1_bfcl_multi_turn_base/logs` 或直接读 JSONL 复核下面的每一条结论。

## 逐问题观察结果

**Q1 上下文膨胀**：完全符合预期。41 步的真实轨迹里，每步稳定新增 2 条消息（一次 tool call + 一次 tool result），线性增长到 83 条消息，没有任何异常跳变或计数错乱——说明检测器在真实、有一定深度的轨迹上是稳的，不是只在玩具例子的 3 步规模下才不出错。

**Q2 重复 prefill（消息级）**：消息级 diff 在 41 步深的轨迹上正确累积（`reused_messages` 随步数单调增长，`cumulative_distinct_messages` 与"全历史比较"设计预期一致），没有因为轨迹变长出现状态错乱。但把我们的估算值和 DeepSeek 真实返回的 `input_tokens_cache_read`/`input_tokens`（provider 真实计费的 cache 命中情况）逐步对比后发现：**两者走势并不紧密吻合**——有的步骤很接近，有的步骤差好几倍（比如某一步我们估算新增 165 token，但 provider 实际按全价计费的只有 38 token；另一步我们估算新增 153 token，provider 实际计费 70 token）。这不是 bug，是两种度量方式本质上衡量的不是同一件事：我们做的是**消息级**语义去重（"这条消息内容之前出现过吗"），provider 做的是**字节级 prefix cache**（"从请求开头连续匹配到第几个字节还和上次一样"）。只要序列化顺序、字段格式、乃至一个 tool_call_id 的差异出现在比较靠前的位置，就可能让 provider 的 cache 在消息级"看起来没变"的地方失效，反之亦然。这是一个真实的科研发现，直接对应 `efficient-harness.md` 里"潜在科研价值举例"一节提到的"prefix cache 的理论收益因为 prompt template 重构而没有真正实现"——现在有了具体的实测证据，而不只是假设。

**发现的 bug（已修复）**：第一轮跑完后发现，41 步轨迹里模型可用的工具定义有 31 个、序列化后约 23,000+ 字符，**每一步都通过 `event.tools` 字段原样重发一遍**，但 `prefill_diff.py` 当时只扫描 `event.input`（消息列表），完全没有扫描 `event.tools`——对"工具 schema 本身的重复"是完全失明的。这是目前已知在真实 tool-calling agent 里最大的一块结构性重复 payload（因为它在整条轨迹里通常完全不变，每一步都 100% 命中"重复"），却是最容易被"只看消息历史"的实现漏掉的部分。**已修复**：`prefill_diff.py` 现在对 `event.tools` 做独立的 new/reused/dup_in_step 分类（跟消息用同一套算法，但用单独的 seen-set，因为工具 schema 和会话历史是结构不同的两类重复内容，混在一起会让"到底是历史膨胀导致的重复、还是工具定义导致的重复"这个问题没法回答）。修复后在真实 benchmark 上回归验证：

| 样本 | 工具总数 | 每步工具 token 估算 | step 1（工具首次出现） | 后续每步 |
|---|---|---|---|---|
| 样本 A（持久化保存版，27 步轨迹） | 17 | 4,670 | `tools_new=17, tools_reused=0` | `tools_new=0, tools_reused=17`，全程稳定 |
| 样本 B（持久化保存版，22 步轨迹） | 31 | 7,899 | `tools_new=31, tools_reused=0` | `tools_new=0, tools_reused=31`，全程稳定 |

工具总数和每步 token 估算这两列在验证性质的第二轮（16/31 步）和持久化保存的第三轮（27/22 步）里完全一致——符合预期，因为这两个数只取决于数据集里固定的工具定义，不取决于模型具体怎么跑、跑了几步。步数不同纯粹是模型行为的随机性。修复前这部分数据完全不存在（字段不存在），修复后每一步都能准确识别"工具 schema 部分没有变化"，且数值在整条轨迹里保持稳定，符合预期。原始文件在 `runs/goal1_bfcl_multi_turn_base/.inspect_trace/*/*/sample-*.jsonl`，可以直接核对。

**Q3 分段 token 估算**：`reasoning_estimated_tokens` 全程正确为 0（DeepSeek-chat 不是推理模型，检测器没有乱报，是一个有意义的负向验证）；tool-call/text token 拆分数值合理。

**Q4 哪类 observation 被反复复用**：这是最直观、最有说服力的一组结果。BFCL `multi_turn_base` 是文件系统操作任务，轨迹后期的 `tool_reuse_breakdown` 显示复用最多的是 `ls`(13次)/`cd`(12次)/`pwd`(5次) 这类目录导航命令的历史返回值——跟任务性质完全吻合（agent 反复 `ls`/`cd` 探索目录结构是这类任务的典型行为模式），产出的数据是真正"可读、可解释"的，不是抽象数字。

**Q5 retry 分组**：两轮实验里 HTTP 调用全部一次成功，没有真实触发 retry 路径，所以这次没有产生新证据（重试路径本身已经在此前的合成测试里验证过，且这次真实跑的过程中机制本身没有报错或产生异常记录）。**这是一个待补的验证缺口**，需要专门构造会触发限流/超时的场景才能拿到真实场景下的 retry 证据。

**Q6 并行/等待/回滚**：这两轮实验都没有专门检查（BFCL multi_turn_base 是纯顺序单工具调用任务，本身不涉及并行）。同样是待补的验证缺口，需要挑一个官方标了"parallel"的 BFCL 类别（如 `live_parallel`/`parallel_multiple`）才能在真实场景下验证这部分。

## 结论

目标一在真实、有深度（十几到四十步）的多轮 agent 轨迹上，六个问题里能给出有意义答案的这几项（上下文膨胀、消息级重复 prefill、token 分段估算、observation 复用归因）都跑得稳，产出的数据可读、可解释，不是只在玩具例子里成立。真实场景暴露出一个此前玩具测试完全测不出来的真实 bug（工具 schema 未被追踪），说明"用真实 benchmark 验证"这一步是必要的，已经在这次实验里直接体现了价值，bug 已修复并有真实数据回归验证。

同时也如实暴露了两个还不完整的地方：(1) 消息级去重和 provider 真实 prefix-cache 命中之间存在系统性差异，这本身可能是值得深挖的科研问题而不只是实现缺陷；(2) retry 和并行这两类场景这次实验没有触发，仍然只有合成测试的覆盖，需要专门构造场景补上真实验证。

## 后续建议

1. 挑一个会触发限流/超时的场景（比如故意用较小的 `max_connections` 制造并发压力，或者故意选一个更容易被限流的模型/站点）专门验证 Q5 在真实场景下的 retry 分组。
2. 挑 BFCL 里标了 parallel 的类别（`live_parallel`/`parallel_multiple` 等）跑一遍，验证 Q6 的并行部分在真实场景下的记录是否正确。
3. 如果要深入研究"消息级 dedup vs 真实 prefix cache 边界"这个差异，需要拿到 provider 实际发送的原始请求字节（`ModelCall.request`，inspect_ai 已经在记录）做字节级/token 序列级的 diff，而不是我们现在做的消息对象级 diff——这可能是一个独立的、更精确的重复 prefill 检测方案，值得作为目标二 token 层 profiling 的一部分单独设计。

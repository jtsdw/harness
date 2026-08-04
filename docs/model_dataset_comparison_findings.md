# 模型 × 数据集对照实验：结果、分析与目标一/目标二契合度

用两个模型（DeepSeek-chat 真实 API、本地 vLLM 部署的 Qwen2.5-3B-Instruct）分别跑两个数据集（BFCL `multi_turn_base`、GSM8K），四组真实数据，验证 `inspect_trace` 在不同模型/不同任务类型下是否都能正确工作，并顺带挖出几个有实际意义的效率发现。原始数据全部持久化在 `runs/compare/{deepseek,local}_{bfcl,gsm8k}/`。

复现命令见 `inspect_trace/scripts/run_bfcl_benchmark.sh` 和 `run_gsm8k_benchmark.sh`（本地模型跑之前先 `local-model-server/scripts/serve.sh` 起服务）。

## 实验矩阵

| | BFCL multi_turn_base（3 样本） | GSM8K（5 样本） |
|---|---|---|
| **DeepSeek-chat**（hosted，原生 tool call） | accuracy 0.667，86 次模型调用，平均 28.7 步/样本 | accuracy 1.000，5 次调用，1 步/样本 |
| **Qwen2.5-3B-Instruct**（本地 vLLM，`emulate_tools` 模拟 tool call） | accuracy 0.000，28 次模型调用，平均 9.3 步/样本 | accuracy 0.800，5 次调用，1 步/样本 |

准确率数字不是这次实验的重点（样本量太小，3B 模型 + prompt 模拟工具调用天然弱于原生 tool call 支持的更大模型），重点是下面从 `inspect_trace` 产出里挖出来的效率现象。

## 发现一：工具 schema 重复占了"重复 prefill"的 90%+，不是消息历史

这是对我们之前修的那个 bug（`goal1_real_benchmark_findings.md`）的定量延伸。把 `prefill_diff` 记录里的两类复用来源加总：

| | 消息历史复用 token（估算） | 工具 schema 复用 token（估算） | 工具 schema 占比 |
|---|---|---|---|
| DeepSeek × BFCL | 51,403 | 497,396 | **90.6%** |
| 本地 Qwen2.5-3B × BFCL | 3,323 | 142,582 | **97.7%** |

两个模型上都是同一个结论：**工具 schema 的重复才是"重复 prefill"里的大头，会话历史增长反而是小头**。这跟直觉不完全一致——大家一般讨论"repeated prefill"时想到的都是"历史消息越堆越长"，但至少在 BFCL 这类工具丰富（本例 17-31 个工具）、单条 observation 不算长的场景下，真正该优化的目标其实是"别让 30 个工具定义在每一步都原样重发一遍"，而不是"压缩对话历史"。这是目标一意外发现的、值得在目标三"标准化干预接口"里优先做的一个具体优化方向（比如 provider 侧的 tool-schema 级 prefix caching，或者 client 侧只在工具集变化时才重发完整 schema）。

## 发现二：本地 vLLM 没开 prefix caching，实际重复计算的 token 量比 hosted 模型多一个数量级

对比两边真实的 provider 计费数据（`ModelUsage`，不是我们的估算）：

| | 总 model calls | 真实 billed input tokens | provider 报告的 cache-read tokens |
|---|---|---|---|
| DeepSeek × BFCL | 86 | 8,110 | **420,736**（命中缓存） |
| 本地 vLLM × BFCL | 28 | **120,562** | 无（该字段为 `None`） |

DeepSeek 86 次调用总共只被计费 8,110 个"真正新算的" input token，剩下 42 万+ token 全部命中了 provider 自己的 prompt cache；本地这次部署的 vLLM（`vllm==0.6.3.post1`，启动参数里 `enable_prefix_caching=False`，这个版本默认不开）在只有 28 次调用、消息总量明显更少的情况下，反而被计费了 12 万+ token——因为**每一次调用都把完整上下文当"全新"重新跑了一遍 prefill**。这不是模型能力问题，是一个纯粹的部署配置问题，而且直接可验证：只要重新起服务时加 `--enable-prefix-caching`，理论上这个数字应该大幅下降——这是一个现成的、可以直接用现有 harness 测的目标三对照实验（"prefix caching 开 vs 关，对同一批真实请求的效率影响"），细节见下面对照表里目标三那一行。

## 发现三：`inspect_trace` 在"没有工具调用的单轮任务"上表现是正确的健全性基线

GSM8K 两个模型上都是 1 步/样本、`tools_total=0`、`reused_tokens_estimate=0`、`reused_tool_tokens_estimate=0`——这是完全符合预期的"平凡情况"：单轮任务没有历史可复用，检测器没有报错也没有编造出奇怪的非零值。这跟 BFCL 那种几十步的复杂轨迹形成了两端对照，说明 `inspect_trace` 不是只在"复杂多轮工具调用"这一种场景下才凑巧能跑，边界情况（无工具、单步）也处理得干净。

## 发现四：wall-clock 时间上，本地部署反而比 hosted API 更快，尽管 prefill 效率更差

DeepSeek × BFCL 端到端跑了 50 秒（86 次调用，含网络往返），本地 vLLM × BFCL 只用了 37 秒（28 次调用，无网络延迟，3B 小模型 decode 快）。这提醒一件事：**目标二要衡量的"效率"不能只看 token 效率，episode 层的端到端 latency 和 token 层的重复计算量是两个独立、有时甚至反向的维度**——本地部署 token 层明显更浪费（发现二），但 episode 层反而更快，纯粹因为省掉了网络往返、模型更小。这正是 `efficient-harness.md` 目标二要求"联合评估 latency、memory、token consumption、task success 和 monetary cost"而不是只看单一指标的原因，现在有真实数据支撑这个设计判断。

---

## 目标一 / 目标二契合情况对照表

目标一已实现的 `inspect_trace` 产出，到底能覆盖目标二三层 profiling 体系里的哪些指标——用这四组真实数据实测验证，不是纸面推演。

### Token 层

| `efficient-harness.md` 要求的指标 | 目标一产出能否直接算出 | 证据 |
|---|---|---|
| input token | ✅ 已有（真实计费值） | `segment_tokens.billed_input_tokens`，如 DeepSeek×BFCL 累计 8,110 |
| output token | ✅ 已有（真实计费值） | `segment_tokens.billed_output_tokens` |
| newly appended token | ✅ 已有（估算值） | `prefill_diff.new_tokens_estimate` |
| repeated context token | ✅ 已有（估算值，且已拆分消息 vs 工具两个来源） | `prefill_diff.reused_tokens_estimate` + `reused_tool_tokens_estimate`，本文发现一 |
| reasoning token | ✅ 已有（真实值 + segment 估算） | `segment_tokens.billed_reasoning_tokens`（本次两个模型都是 0，符合预期，均非推理模型） |
| tool-call token | ✅ 已有（segment 估算） | `segment_tokens.tool_call_estimated_tokens` |
| observation token | ⚠️ 部分——工具返回内容本身的 token 数没单独统计，只统计了它作为消息被复用时的 token | 需要在 `prefill_diff.py` 里对 `role=="tool"` 的消息单独出一列 |
| discarded/retry/rollback token | ✅ 已有（本次实验零 retry，机制层面已在之前的合成测试验证过） | `attempt_group.total_wasted_wait_time` + 各 attempt 的 token（需要关联对应 `segment_tokens` 记录，目前无直接字段，可离线 join） |

### Model invocation 层

| 指标 | 能否直接算出 | 证据/缺口 |
|---|---|---|
| queueing time | ❌ | 未采集，需要 provider 侧支持 |
| tokenization/template time | ❌ | 未采集 |
| prefill time / decode time | ❌ | 未采集，`ModelCall.time` 只有总耗时，不拆分阶段 |
| time to first token / inter-token latency | ❌（管道已具备，未启用） | `inspect_ai_roadmap.md` 已指出 vLLM provider 支持流式但目前丢弃逐 chunk 时间戳 |
| peak GPU memory | ❌ | 未采集，需要接 vLLM `/metrics` 或 `nvidia-smi` 侧信道 |
| **cache hit/miss** | ✅ **本次首次拿到真实数据** | 发现二：DeepSeek `input_tokens_cache_read=420,736` vs 本地 vLLM `None`——这原本以为要接 vLLM 特有指标才能拿到，但至少 provider 自己上报的 cache 命中率（`ModelUsage.input_tokens_cache_read`）目标一已经在记录了，不需要额外工作 |
| batch size | ❌ | 未采集 |
| generated tokens per second | ⚠️ 可粗略算 | `output_tokens / (episode wall time)` 是近似值，不是逐次调用的真实 decode 速度 |

### Episode 层

| 指标 | 能否直接算出 | 证据 |
|---|---|---|
| end-to-end latency | ✅ | 本文发现四，直接读 `EvalStats.started_at`/`completed_at` |
| number of LLM calls | ✅ | `prefill_diff` 记录数即为 model call 数（本次 86/5/28/5） |
| number of tool calls | ✅ | 原始 `.eval` 日志里的 `ToolEvent` 数，inspect_ai 原生已覆盖 |
| retries | ✅ | `attempt_group.total_attempts > 1` 的记录数 |
| success rate | ✅ | scorer 结果直接给出（本次 0.667/1.000/0.000/0.800） |
| cost per successful episode | ⚠️ 需要外部定价表 | token 数已有，乘上模型单价即可，目标一没做（也不该做，定价表是配置问题） |
| critical-path latency / exclusive-inclusive 时间拆分 | ❌ | 需要基于 span 树做区间运算，目标一没实现，是目标二的核心工作 |

### 结论

目标一的产出对目标二"token 层"的覆盖度很高（8 项里 6 项已有，1 项部分，1 项需要离线 join），"episode 层"也基本够用（6 项里 5 项已有，1 项需要外部定价配置）。真正的缺口集中在"model invocation 层"——这跟 `inspect_ai_roadmap.md` 之前的纸面判断一致，但这次多了一个意外惊喜：**cache hit/miss 这一项，靠 provider 自己上报的 `ModelUsage.input_tokens_cache_read` 字段，目标一已经顺手拿到了，不需要专门接 vLLM 指标**。真正还完全空白、需要额外基础设施的，缩小到 queueing/tokenization/prefill/decode 时间拆分和 GPU 显存/batch size 这几项，跟之前的判断一致，工作量估计不变。

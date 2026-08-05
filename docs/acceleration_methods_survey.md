# 加速方法调研：insight 与 method 分析（目标五）

这是 [`efficient-harness.md`](./efficient-harness.md) 目标五的实际产出——持续积累的清单，不是一次性写完的文档。每次新分析一篇论文，就在"已完成分析"里加一条，同时把它从"候选清单"挪走（或者标记完成）。

## 方法论：怎么分析一篇论文

不是写摘要。每条分析拆成这几块：

| 字段 | 要回答的问题 |
|---|---|
| Insight | 论文观察到的、支撑其方法成立的底层事实/假设是什么？（不是"用了什么算法"，是"为什么这个算法有存在的必要"） |
| Method | Insight 怎么落地成具体机制？ |
| 对应到我们 harness 的哪里 | 落在目标一哪条需求 / 目标二哪一层 / 目标三哪个干预点（context/generation/runtime/tool）/ 目标四 online 还是 offline？ |
| 我们自己数据的印证情况 | 用已经跑过的真实数据检验论文的 insight 是否成立——**成立 / 不成立 / 测的是不同指标（不可比）/ 现有数据测不了**，四选一，如实标注，不能默认"论文说的都对" |
| 要验证需要什么 | 现成可用（已有字段/已有实验能直接算）还是需要新数据（新 benchmark、新干预实现） |

第一条实战案例是 SPORK——起因是日常讨论时怀疑"论文说的问题我们好像没观察到"，结果发现是测的不是同一个指标，不是论文错了或者我们错了。这个过程本身就是目标五要做的事，收录进来当模板参考。

## 已完成分析

### SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference

[arXiv:2607.03333](https://arxiv.org/abs/2607.03333)（[baihuajun24/spork](https://github.com/baihuajun24/spork)，开源）

**Insight**：Agent 的 Thought-Action-Observation 循环通常是串行的——模型推理、吐出 tool call、然后 GPU 空闲等待外部工具执行完才能继续 decode。论文测出这段空闲墙钟时间在他们的 workload 里占 16-37%（引用的既有文献里更高，35-61%）。更进一步的 insight：模型自己就能预测自己接下来要调用的 tool 名字——在生成开始处 fork 一个探测分支，对 Qwen3-32B 预测下一个 tool 名字的准确率能到 74.6%-99.6%（五个 benchmark 上）。

**Method**：训练free 的轻量 controller，在正式生成的同时 fork 一个探测分支预测 tool 名字，提前把预测出的 tool call 派发去执行，跟剩余的 CoT decode 重叠；一个代价模型判断什么时候投机划算；prefix-cache fork 降低探测本身的代价；置信度 gate 过滤误判；被拒绝的探测结果还能退化成 token 级投机解码的 draft，不浪费。部署形态是一层薄 controller，架在标准 completion API 之上，不需要重新训练、不需要辅助模型、不需要离线 trace，跟 token 级投机解码正交。GAIA 上把 Qwen3-32B 的 P95 latency 从 131.9s 降到 108.1s（-18%）。

**对应到我们 harness 的哪里**：
- 目标一需求三（执行拓扑）——SPORK 的"model waiting for tool"这个空闲区间，正是 `execution_topology` 检测器要重建的东西之一。
- 目标二 episode 层——`total_busy_seconds` vs `end_to_end_latency_seconds` 的差值，就是 SPORK 论文里说的那 16-37% 空闲时间的直接测量口径，我们已经有这个字段。
- 目标三 generation/runtime 干预点——SPORK 本身就是一种 runtime 层干预（提前派发 tool 执行），如果要在我们 harness 上实现类似机制，落点在 `on_before_model_generate` 之外还需要一个能"提前触发 tool 执行、之后再对齐结果"的新钩子，现有干预点还不够。

**我们自己数据的印证情况**：**测的是不同指标（不可比），且用我们已有数据能进一步说明"为什么在我们的 workload 上测不出这个问题"**——完整分析过程见对话记录，摘要：
- 我们说"tool-call 占生成 token 比例小"（真实数字见下面 ToolSpec 条目：BFCL 200 样本上是 17.5%），SPORK 说的是"等待 tool 执行结果的墙钟时间占比"——一个测 token 数、一个测时间，字面上不是同一个命题，也不矛盾。
- 用能直接对应的口径重新测（`total_busy_seconds` vs `end_to_end_latency_seconds`）：`goal2_real_validation_findings.md` 里真实样本 `multi_turn_base_0` 两者只差 5ms/16.288s（约 0.03%），远低于 SPORK 的 16-37%。
- 原因找到了：我们目前用的 BFCL `multi_turn_*` 和 tau2-bench `mock` domain，tool 执行都是**进程内 mock 函数**，调用即返回，没有真实网络往返；SPORK 测的是 GAIA 这类 **real-tool benchmark**（真实 web 搜索、代码执行沙箱），tool 执行本身要几百 ms 到几秒。SPORK 要解决的问题在我们当前测的对象里结构性地不存在，不是我们的检测器有问题。

**要验证需要什么**：`total_busy_seconds`/`end_to_end_latency_seconds` 两个字段现成可用，不需要新代码。真正缺的是**一个真实调用慢速外部工具的 benchmark/agent**（真实网络 API、真实代码执行沙箱）——没有这个，SPORK 的核心命题在我们的 harness 上永远测不出正的信号，不管代码写得多好。这也是下面"优先级建议"里反复出现的一个共性缺口，不止 SPORK 一篇论文受影响。

### ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding

[arXiv:2604.13519](https://arxiv.org/abs/2604.13519)

**Insight**（论文原话论证）：tool-call 的生成本身是高度结构化的——固定 schema（工具名、参数名、类型都是预先定义好的），而且真实调用轨迹里同一个工具的调用参数经常跟历史调用高度相似（API-Bank 上平均每个工具被重复调用 10.95 次）。论文的落脚点是**延迟占比，不是 token 占比**：原话是"tool-calling generation is the dominant bottleneck in tool-calling pipelines"，并给出两个具体数字——tool-calling 生成耗时"roughly 4× larger than the time spent executing the tools"，占端到端延迟"up to 96%"。这跟 SPORK 完全是另一个问题：SPORK 关心的是"tool 执行完之前 GPU 在空等"（执行占大头），ToolSpec 关心的是"生成 tool-call 这段文本本身要多久"（生成占大头）——两者在各自的 workload 上给出的"谁是瓶颈"结论，方向是相反的，见下面"我们自己数据的印证情况"。

**一个论文没写但推理上站得住的补充论证**（团队自己的推断，不是论文原话，标注清楚以免误引用）：投机解码本身要求 draft 分布贴近 target 分布，而 agent 一次生成里 reasoning（自然语言、高熵）和 tool-call JSON（schema 约束、近确定性、低熵）是统计特性完全不同的两段——一个通用 draft 模型很难同时贴合两边的分布。这能解释论文里"schema 部分用确定性 FSM 填充、只对真正变化的字段做投机"这个设计选择为什么比通用 training-free 投机解码（比如 prompt lookup）更好，但论文本身没有用"跨 segment 分布异质"这个框架去论证，是我们自己补的一层理论解释,不能当成论文的主张转述给别人。

**Method**：一个有限状态机在"确定性 schema token 填充"（工具名、参数 key、括号引号这类固定不变的部分，不需要模型采样，直接按 schema 摆上去）和"投机生成"（参数值这类变量字段，用 draft 模型或历史相似调用当草稿，一次验证多个 token）之间切换；额外用检索增强，把历史上相似的真实调用取出来当草稿，进一步提高 draft 命中率。即插即用，不需要重新训练。论文报告在多个 benchmark 上相比现有 training-free 投机解码方法最高 4.2x 加速（针对 tool-call 生成这一段，不是整个 episode）。

**对应到我们 harness 的哪里**：
- 目标一需求一（token 级记录）——`inspect_trace/src/inspect_trace/segment_tokens.py`/`token_attribution.py` 已经把每次模型调用的输出拆成 `reasoning_tokens_estimate`/`tool_calling_tokens_estimate`/`final_response_tokens_estimate` 三类，这正是判断"ToolSpec 那 4.2x 能换来多少整体收益"所需要的分母。
- 目标二 token 层——`token_layer.summarize_run()` 直接能算出 tool-call token 占总输出 token 的比例（见下）。
- 目标三 generation 干预点——ToolSpec 是纯生成侧的投机解码机制，不涉及 tool 执行/context 干预，落点明确，但需要 serving 侧支持 draft/verify（vLLM 的 speculative decoding 或 prompt-lookup decoding 能力），不是 inspect_ai Hooks 能单独实现的。

**我们自己数据的印证情况**：**用真实数据算出了 ToolSpec 论文里没给出的一个数字，并做了一个粗略但诚实的整体收益估算**——BFCL 200 样本全量 run（`runs/goal1_bfcl_multi_turn_base_full`）上用 `token_layer.summarize_run()` 现算：

| 输出 token 分类 | token 数 | 占总输出 token 比例 |
|---|---|---|
| 总计费 output tokens | 86,639 | 100% |
| `tool_calling_tokens_estimate` | 15,143 | **17.5%** |
| `final_response_tokens_estimate` | 55,332 | 63.9% |
| `reasoning_tokens_estimate` | 0 | 0.0%（Qwen2.5-3B-Instruct 非推理模型，符合预期） |
| 未归类（segment 解析覆盖不到的部分） | 16,164 | 18.7% |

ToolSpec 论文没有报告"tool-call token 占总输出的比例"这个数字，我们的数据补上了这一块：**17.5%，不是"可忽略不计"，但确实是少数**，`final_response` 占大头（63.9%）。用一个简化的 Amdahl's-law 估算（假设每 token 生成耗时大致均匀，只对 tool-call 这部分应用论文报的 4.2x）：整体 decode 阶段的理论加速比 ≈ 1 / (0.175/4.2 + 0.825) ≈ **1.15x**，即约 15% 的整体 decode 时间缩短——跟论文标题党式的"4.2x"数字差距很大，但这不是论文错了，是论文报告的加速比本来就是"只对 tool-call 这一段"，我们这里如实换算成了"对整个 episode 的实际预期收益"。**这个估算本身很粗糙**（假设了 tool-call token 和其他 token 的单 token 生成耗时相同，忽略了 TTFT/prefill、忽略了 draft/verify 本身的开销，忽略了 SPORK 案例里已经确认的"tool 执行等待时间"这个可能占比更大的因素），只作为量级参考，不是精确预测。

**跟论文"tool-calling 是延迟大头（最高 96%）"这个具体断言直接冲突的一条真实数据**：用同一份数据算平均单次长度而不是总量——`tool_calling_tokens_estimate` 总量 / 目标一阶段统计的真实 tool call 次数（857 次，`goal1_real_benchmark_findings.md`）≈ **17.7 token/次**；`final_response_tokens_estimate` 总量 / 粗估的 final-response 类调用次数（每 episode 平均 7.03 次调用中刨除平均 4.285 次 tool call）≈ **100.8 token/次**。也就是说在我们这个"小模型 + BFCL 简单工具集"的场景下，tool-call 段无论是总 token 数还是单次长度都不是大头，`final_response` 反而更长——这跟 ToolSpec 论文的核心断言方向相反。合理解释是 workload 差异（论文测的 API-Bank 这类场景工具/参数结构更复杂），不代表论文错，但如实说明：**"tool-calling 是不是延迟瓶颈"本身是 workload-dependent 的经验问题，不能默认套用到任何一个新 benchmark 上**，用之前必须先用自己的数据测一遍再下结论。（另外要注意一个已知的度量口径问题：`goal1_r3_r4_real_benchmark_findings.md` 记录过 `final_response_tokens_estimate` 有时会把"tool_calls 解析失败、模型其实想调用工具但格式错了"的输出也计入，所以这里的 `final_response` 总量可能略微偏高，不是纯粹"模型主动选择直接回答"的部分——如实标注这个混淆源，不假装数字绝对干净。）

**这条数据顺带解释了它跟 SPORK 的矛盾从哪来**：SPORK 说执行等待占大头（16-37%+），ToolSpec 说生成占大头（最高 96%）——两个论断逻辑上互斥，合理推断是他们测的 workload 里"工具执行速度"截然不同（ToolSpec 大概率是快/mock 工具，生成主导；SPORK 是 GAIA 这类真实慢工具，执行主导）。这是评估任何"加速 tool calling"类论文时都该先问的第一个问题：**这篇论文的收益前提，是工具执行快还是慢？**——决定了它能不能在我们当前的 mock-tool workload 上体现出来，还是需要换一个真实工具的 benchmark。

**要验证需要什么（已完成）**：读完源码后直接读了 ToolSpec 自己的仓库（`/home/liuyingen/code/ToolSpec`），原样跑通了它的官方复现（baseline/pld/recycling/samd/toolspec 五种方法，Qwen2.5-3B-Instruct，API-Bank 100 条），又把它的核心机制迁移进了我们自己的 `inspect_ai` harness（新项目 `toolspec_adapter/`，一个从零实现的自定义 `ModelAPI`，因为 ToolSpec 是原始 HF `transformers` 生成循环、不是 OpenAI-compatible 服务，跟 tau2-bench 的接入方式完全不同），逐 token 精确复现了原生仓库的行为（包括它"并非严格 lossless"这个意外发现的真实特性）。完整过程、真实速度数字、以及一个"四种独立实现的投机解码方法在同样 11/100 个问题上偏离 greedy baseline"的交叉验证发现，见 [`toolspec_integration_findings.md`](./toolspec_integration_findings.md)。另外还跟 vLLM 服务自带的通用 ngram 投机解码做了真实对比——ToolSpec 的领域特定方法在这个任务上明显更好（速度快约 60%、偏离率更低），见 [`toolspec_vllm_speculative_comparison.md`](./toolspec_vllm_speculative_comparison.md)。

## 候选论文清单（待分析）

按主题分类；每条只给了初步一句话定位，还没有做完整的四栏分析。**用途**：下次要分析新论文时先看这里,不用每次现从头搜。

### 并行化 / 投机执行

- **LLMCompiler**——[An LLM Compiler for Parallel Function Calling](https://arxiv.org/abs/2312.04511)（ICML 2024）。Insight 初步定位：多数 agent 框架把"要不要并行调用"这件事留给逐步 ReAct 循环，天然串行；LLMCompiler 用一个 Planner 先规划出带依赖关系的 DAG，再并行派发执行，论文报告延迟最多降 3.7x、成本最多降 6.7x。直接对应目标一需求三"执行拓扑"——我们已经确认 BFCL 现有工具集因为没设 `parallel=True` 从未观测到真并行（见 `goal1_r3_r4_real_benchmark_findings.md`），LLMCompiler 的规划器正是"主动制造并行"的方法，值得作为验证需求三检测器"真的能测出并行"这个缺口的现成候选。
- **ReWOO**——[Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323)。Insight 初步定位：交替式 reasoning-tool call 会导致大量冗余 prompt 重复发送（每一步都要把之前的 observation 原样带进下一次调用）；ReWOO 把流程拆成 Planner（一次性规划所有步骤和占位符）/Worker（执行）/Solver（用真实结果替换占位符再统一推理），减少 LLM 调用次数和重复 token。直接对应目标二 token 层（能不能减少"复用 context token"这部分我们已经测过是重复 prefill 的大头）和目标一需求三（从交替执行改成"先规划再执行"，拓扑结构完全变了）。
- **Speculative Actions**——[A Lossless Framework for Faster Agentic Systems](https://arxiv.org/abs/2510.04371)（上次讨论 SPORK 时顺带搜到，还没细读）。跟 SPORK 同一个方向（投机预测 agent 下一步动作），需要读完确认跟 SPORK 的具体差异点（比如是否也依赖模型自己做预测器、是否也需要真实慢速工具才能体现收益）。

### Prefix / context 复用调度

- **SGLang / RadixAttention**——[Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)。Insight：多次生成调用之间共享的 KV cache 可以用 radix tree 结构自动识别复用，不需要用户手动管理。跟我们已经发现的真实问题直接相关——`model_dataset_comparison_findings.md` 记录过本地 vLLM 部署当时 `enable_prefix_caching=False`，同样调用量下被多计费了 12 万+ token，这是一个现成的、已经写进目标三对照实验清单的"prefix caching 开 vs 关"实验（`framework-selection.md` 里提过），RadixAttention 是这个方向更进一步的调度算法。
- **Preble**——[Efficient Distributed Prompt Scheduling for LLM Serving](https://arxiv.org/abs/2407.00023)。Insight：agent/工具调用类 workload 的 prompt 前缀重复率高，调度器如果同时考虑"KV cache 复用"和"计算负载均衡"两个目标（E2 算法），比单纯轮询快 1.5x-14.5x。跟目标二 model invocation 层的 `queue_depth_running/waiting` 字段直接相关——但我们现在这些字段全是 0（因为一直是 `MAX_CONNECTIONS=1` 串行跑的，`goal2_real_validation_findings.md` 里明确标注过这是已知限制），要检验 Preble 的调度收益，前提是先跑一次真并发实验。

### KV cache 压缩/淘汰

- **StreamingLLM**——[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)（ICLR 2024）。Insight：模型对最初几个 token 有异常强的 attention（"attention sink"现象，不代表语义重要），只要保留这几个 sink token + 最近的滑动窗口，就能让有限窗口训练的模型稳定处理远超训练长度的流式输入。
- **H2O**——[Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)（NeurIPS 2023）。Insight：attention 矩阵高度稀疏，累积 attention score 高的"heavy hitter" token 贡献了大部分输出质量，动态淘汰低贡献 token 能在几乎不掉点的情况下把 KV cache 内存降到 1/5。
- 这两篇都对应目标二 model invocation 层的 `peak GPU memory`/`KV cache size` 字段——但这两个字段本身在我们锁定的 vLLM 版本上就拿不到绝对值（`goal2_design.md`/`goal2_real_validation_findings.md` 已经如实标注），要检验这两篇论文的收益，需要先补上这层观测能力，不是直接能测的。

### Prompt / context 压缩

- **LLMLingua**——[Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736)（EMNLP 2023），以及后续 [LongLLMLingua](https://arxiv.org/abs/2310.06839)。Insight：prompt 里不是所有 token 对下游任务同等重要，用一个小模型算 token 级重要性、迭代式压缩，最高 20x 压缩比几乎不掉点。直接对应目标三 context 干预点（compaction/compression 这一类），也是 `inspect_ai_docs_examples_audit.md` 里提到过的 `docs/compaction.qmd`（inspect_ai 自带的 context compaction 机制）的一个可对比的外部方法。

### 模型级联/路由

- **FrugalGPT**——[How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176)。Insight：不同 LLM API 的定价差两个数量级，多数查询不需要最贵的模型就能答对；按置信度级联（先问便宜模型，不够自信再升级）能在几乎不掉点的情况下降本最多 98%。对应目标三 runtime 干预点（模型选择/路由）和目标二 episode 层的 `cost per successful episode`——是目前候选清单里唯一直接以"钱"为优化目标、而不是以延迟/token 为目标的方法，值得跟前面几篇对照着看优化目标的差异。

## 优先级建议

不是按论文影响力排，是按"现有 harness 不用大改就能测出信号"排：

1. **LLMCompiler / ReWOO 优先**——都能直接在现有 BFCL `multi_turn_*`/tau2-bench workload 上验证，不需要真实慢速外部工具（这点上跟 SPORK 不同），而且跟已经做完的目标一需求三（执行拓扑）关系最近，复用现成检测器就能看出差异。
2. **SGLang RadixAttention / Preble 次优先**——"prefix caching 开 vs 关"这个对照实验在 `framework-selection.md` 里已经被标记为"现成可做的目标三实验"，直接延伸到这两篇就是水到渠成的事；Preble 那部分需要先有一次真并发实验（`MAX_CONNECTIONS>1`），目标二阶段验证时因为本地 vLLM 并发不稳定被搁置过，重新捡起来时可以顺带把这篇也测了。
3. **KV cache 淘汰（StreamingLLM/H2O）、prompt 压缩（LLMLingua）、模型级联（FrugalGPT）暂缓**——不是不重要，是我们当前的 benchmark（BFCL 单条 episode 平均 7 次调用、context 不算长；tau2-bench mock domain 同样规模有限）本来就没有把这几篇论文要解决的问题（超长上下文、天量 prompt、多模型选择）真正 stress 出来，先分析也测不出有意义的对照结果。等接入一个 context 明显更长/调用链更深的 benchmark，或者目标三/四基础设施更成熟之后再回头做。
4. **SPORK 本身、以及后续任何"投机执行覆盖 tool 等待时间"类方法**——都需要先有一个真实调用慢速外部工具的 benchmark（不是 mock 函数），这是比"分析哪篇论文"更优先的一个基础设施缺口，值得单独当一项任务考虑（可能是这个项目要不要接入 GAIA 或类似 real-tool benchmark 的问题）。
5. **ToolSpec：已完成**——不依赖工具执行快慢的纯生成侧优化这个判断是对的，"要补服务端能力"这个缺口也已经补上：没有走 vLLM（vLLM 的投机解码/prompt-lookup 接口跟 ToolSpec 自己 patch 过的 KV cache 树形验证代码对不上），而是直接给 ToolSpec 的原始 HF `transformers` 生成循环写了一个自定义 `ModelAPI`（`toolspec_adapter/`），复现出真实 3.05x 加速，且逐 token 精确对齐原生仓库输出。详见 [`toolspec_integration_findings.md`](./toolspec_integration_findings.md)。

# Agent + vLLM 系统的分层：每层负责什么、对应项目里哪部分、加速方法都在哪层

背景知识文档，整理"一次 agent 任务从发起到拿到结果，中间经过了哪几层，每一层各自能做什么优化"这个概念模
型。跟 [`vllm_request_concepts.md`](./vllm_request_concepts.md)（单次 vLLM 请求内部）是互补关系：那篇讲
一次请求"进了 vLLM 之后"发生了什么，这篇讲一次请求"从 agent 发起、到最终落在哪张卡上的哪个 kernel 执行"
中间经过的完整链路，以及每一环分别是谁的责任、这个项目目前在哪一层做了/没做什么。

## 五层总览

```
L1 Harness / Agent 层
      │  agent 决定问什么、问几次、怎么问
      ▼
L2 部署 / 编排层
      │  决定这次请求该被送到哪个 vLLM 实例
      ▼
L3 vLLM 引擎层（单实例）
      │  调度、continuous batching、KV cache 管理、投机解码
      ▼
L4 Model / 权重层
      │  定义"要算什么"：架构（多少层、多大）+ 权重数值
      ▼
L5 执行运行时层（Kernel / 算子）
         定义"怎么算"：PyTorch/CUDA/FlashAttention 这些具体执行代码
```

五层不是凭空分的，是这次对话里从"vLLM 是不是黑盒"这个问题一路往下追、又亲手踩过其中几层的真实 bug 之后
理出来的——每一层都有这个项目里能对上号的真实文件或真实事件，不是纯理论框架。

**L4/L5 顺序纠正**：最初的草稿把这两层写反了（Kernel 放在 Model 上面）。正确顺序是 Model 在上、Kernel
在下——Model（架构定义 + 权重数值）说的是"要执行哪些运算、用哪些数字"，Kernel（算子）说的是"这些运算具体
怎么在硬件上跑起来"，前者更抽象、后者更贴近硬件，跟"应用层在上、驱动/硬件在下"是同一个方向。这次崩溃的
调用链正好能验证这个方向：`model_runner.py`（模型的前向计算逻辑）往下调用 `sampler.py`，`sampler.py` 再
往下调用 `flashinfer_sample()`——是模型的计算逻辑在调用具体的 kernel 实现，不是反过来，顺序不能颠倒。不
过要老实说一句：这两层在真实代码里不是干净分开、按调用栈严格排列的两个模块，而是在模型的每一步前向计算
里高度交织——权重本身也会作为参数直接传进 kernel 调用里，不是"Model 处理完了再交给 Kernel"这种顺序执
行的关系，这里的"上下"更多是"更抽象 vs 更贴近硬件"的概念顺序，不是一次调用的时间顺序。

## L1 Harness / Agent 层

**负责什么**：agent 要不要调用模型、调用几次、每次问什么、把多少历史上下文塞进去、工具调用要不要并行
发出去。这一层完全不碰模型内部，只决定"发送什么样的请求、发几个"。

**对应项目里哪部分**：`inspect_trace` 包的目标一/二相关代码——`prefill_diff.py`（追踪哪些 prompt 内容
是新的、哪些是重复的）、`token_attribution.py`（token 花在系统提示/工具 schema/对话历史/推理/最终回复
哪个部分）、`execution_topology.py`（工具调用有没有真的并行发出去）、`action_parsing.py`（工具调用解析失
败要不要重试）。`toolspec_adapter/`、`tau2_adapter/` 这些自定义 agent loop 也在这一层。

**这层常见的加速方法**：
- **减少调用次数**：一次多步的 agent 循环里，能不能把几步合并成一次调用、减少来回；批量处理多个独立子任
  务而不是一个个串行问。
- **利用/配合下层的 prefix cache**：把不变的内容（系统提示、工具 schema）固定放在 prompt 最前面、不变的
  就不改——这本身不产生加速，但决定了 L3 的 prefix cache 命不命中，是"上层纪律决定下层收益"的典型例子。
- **精简 prompt**：工具描述写短一点、历史对话做摘要压缩而不是全量保留、`max_tokens` 设得跟实际需求匹配
  （不要留一大截无意义的生成空间）。
- **客户端缓存**：完全相同的请求直接复用上次结果，不用真的再发一次——`vllm_metrics.py` 文档里提到的
  `"no_new_observation"`（inspect_ai 自己的本地缓存命中）就是这个的真实例子。
- **ToolSpec 是这层的代表性案例**：它不是 vLLM 内置能力，是针对工具调用场景专门设计的一套生成循环（原生
  跑在 HF transformers 上，不经过 vLLM），本质是"agent 层自己实现的领域特定投机执行"，这也是为什么这个
  项目要单独写一个自定义 `ModelAPI` 才能接进来（见 `toolspec_integration_findings.md`）——跟 L3 的通用投
  机解码是两条不同的路线，`toolspec_vllm_speculative_comparison.md` 就是拿这两条路线做的真实对比。

## L2 部署 / 编排层

**负责什么**：这次请求该被送去哪个 vLLM 实例（如果有不止一个的话）、怎么在多实例之间做负载均衡、要不要
把 prefill 和 decode 拆成两拨独立实例、一个模型要不要切开放到多张卡上。

**对应项目里哪部分**：几乎没有——`nscc_model_server/scripts/serve.sh` 里的 `TENSOR_PARALLEL_SIZE`（目前
锁定 1，注释里提到过如果 OOM 可以试 2，但没试过）、响应体里的 `kv_transfer_params` 字段（这次真实抓到的
响应里一直是 `null`，说明没有配置成分离式部署）。目前就是"一个实例、不切、没有负载均衡器"这种最简单的
形态。

**这层常见的加速方法**（如实说明：这些是这层"理论上"该做的事，不是这个项目现在需要做的事——参考前面对
话的结论，没有多实例/多用户场景之前动这层是过度设计）：
- **横向扩容**：多个实例 + 负载均衡器，提升整体吞吐，不改善单次请求延迟。
- **负载感知路由**：均衡器根据每个实例当前的排队/KV cache 占用选最闲的那个去发——这需要读到 L3 暴露出
  来的指标才能做，直接依赖需求 B 这批可观测性工作。
- **Session affinity / KV-cache 感知路由**：多轮对话尽量固定路由到同一个实例，让 prefix cache 能跨轮次
  复用；naive 轮询会系统性拉低多实例场景下的 prefix cache 命中率。
- **Prefill/Decode 分离部署**：prefill（算力密集、短时爆发）和 decode（显存带宽密集、长时间占用）资源画
  像差异很大，拆成两拨独立实例池分别调优，`kv_transfer_params` 就是这个机制的真实接口。
- **张量并行（TP）**：把一个模型切到多张卡，跟"横向扩容"是不同的事——TP 是让一个逻辑实例变大，不是复制
  出更多实例。上次真实碰到过的应用场景：KV cache 只剩 4.71 GiB 那次，`TENSOR_PARALLEL_SIZE=2` 是现成的
  解法（还没试过）。

## L3 vLLM 引擎层（单实例）

**负责什么**：一个 vLLM 进程内部的事——continuous batching 怎么调度、KV cache 怎么分配和复用、要不要用
投机解码、guided decoding 怎么处理。这是"vLLM 是不是黑盒"那个问题里，我们这几天一直在打开看的那一层。

**对应项目里哪部分**：这是当前项目投入最多的一层——`local-model-server/`/`nscc_model_server/`（起服务）、
`vllm_metrics.py`/`vllm_per_request_metrics.py`（对这一层做可观测性，需求 B 的核心）、
`toolspec_vllm_speculative_comparison.md`（ngram 投机解码真实测过 1.87x）、
`nscc_h100_speculative_decoding_plan.md`（EAGLE-3 的计划，发布数据 3.0-3.4x）。需求 B（观测闭环）和需求
C（标准化加速干预接口，还没做）都是瞄准这一层设计的。

**这层常见的加速方法**：
- **Continuous batching**：不是一个"可选加速项"，是 vLLM 的默认架构本身，前面已经讲过原理。
- **投机解码（speculative decoding）**：草稿模型/方法先猜一串 token，目标模型批量验证——ngram/prompt
  lookup（不需要额外模型，纯靠上下文找重复片段，已测：1.87x，23/100 输出偏离）、EAGLE-2/EAGLE-3（需要
  专门训练的草稿模型，发布数据 2.4-2.7x/3.0-3.4x）、Medusa、lookahead decoding 都是这条路线的不同变种。
- **Prefix caching（自动前缀缓存）**：重复的 prompt 前缀不用重新算 prefill，直接复用之前算好的 KV
  cache——这正是需求 B4 新加的 `prefix_cache_hit_rate` 在测的东西，也是 agent 场景（工具 schema/系统提
  示反复出现）里理论上收益最大的一类优化。
- **Chunked prefill**：长 prompt 的 prefill 不一次性算完，切成小块跟别的请求的 decode 步骤交替进行，改
  善混合负载下短请求被长 prompt 卡住的问题。
- **量化 KV cache（如 FP8 KV cache）**：压缩 KV cache 本身占用的显存，同样显存能撑更多并发——直接对应
  这次 KV cache 只剩 4.71 GiB 那个真实瓶颈，是除了 TP 之外的另一条解法路线，这个项目也还没试过。
- **Guided/structured decoding**：约束解码只能生成合法 token（比如合法 JSON），有真实开销，这就是需求
  B4 新加的 `guided_decoding_overhead_seconds` 那条。

## L4 Model / 权重层

**负责什么**：模型本身多大、什么精度存的、什么架构（稠密 vs MoE）——定义"要执行哪些运算、用哪些具体数
字"。这一层的选择决定了下面 Kernel 层要跑多大的矩阵、以及其他层能有多少显存余量（比如权重多大直接决定
L3 能分给 KV cache 多少空间）。

**对应项目里哪部分**：没有——模型选择（Qwen2.5/Qwen3 系列、EAGLE-3 草稿模型要不要匹配到具体某个尺寸）
是这个项目的**给定输入**，不是研究变量。`docs/nscc_h100_speculative_decoding_plan.md` 里纠结"要不要从
Qwen2.5 换到 Qwen3"是唯一沾边的地方，而且纠结点是"跟已有结果可比性"，不是"哪个模型更快"。

**这层常见的加速方法**（这个项目不做，纯背景知识）：
- **量化**：权重本身用 INT4/INT8/FP8 存储（不是 L3 的 KV cache 量化，是模型参数本身），常见方法
  AWQ/GPTQ。
- **蒸馏**：训一个更小的模型去模仿大模型的行为。
- **剪枝/稀疏化**：去掉不重要的权重/结构。
- **换模型/架构**：同样任务用一个本来就更小更快的模型；MoE 架构（响应体里 `routed_experts` 字段就是给
  MoE 模型准备的，Qwen3-32B 是稠密模型，这个字段恒为 `null`）用"每个 token 只激活一部分参数"的方式在同
  样激活量下塞进更多总参数。

## L5 执行运行时层（Kernel / 算子）

**负责什么**：Model 层定义好"要算什么"之后，真正把矩阵运算发到 GPU 上执行的那一层——PyTorch、CUDA、以
及 FlashAttention/FlashInfer 这类专门写的融合 kernel、`torch.compile` 编译出来的代码，回答的是"这些运算
具体怎么在硬件上跑起来"。vLLM 自己不直接写底层 kernel，是在调用这些库。

**对应项目里哪部分**：这个项目不研究这一层，但**这几天真实撞上过这一层的 bug**——NSCC 节点没有可发现的
CUDA toolkit（找不到 `nvcc`），导致 FlashInfer 的融合采样 kernel 现场编译崩溃（用
`VLLM_USE_FLASHINFER_SAMPLER=0` 绕开了，退回没有 JIT 编译需求的原生实现），`deep_gemm`（一个 FP8 GEMM
加速库）也是同样原因导致导入失败。启动日志里 `torch.compile took 33.80s`/`Compiling a graph... takes
18.35s` 这些也是这一层的真实活动——每次冷启动都要重新编译一遍（除非有缓存命中）。

**这层常见的加速方法**：
- **融合 kernel**：FlashAttention/FlashInfer 这类把多个算子合并成一个 kernel，减少显存读写和 kernel 启
  动开销。
- **量化计算 kernel**：不只是压缩权重/KV cache 的存储（那是 L3/L4 的事），是用 INT8/FP8 精度直接做矩阵
  乘法，`deep_gemm` 就是这类库的例子。
- **CUDA Graph 捕获**：把重复出现的固定 shape 计算图录下来，之后直接重放，省掉每次单独发 kernel 的调度
  开销。
- **`torch.compile` 图编译优化**：算子融合、内存规划，这次日志里能看到它确实在跑，也确实要花几十秒编译
  时间（一次性成本，之后复用缓存）。

这一层对这个项目的意义更多是"要知道它存在、知道它可能是某个诡异 bug 的真正原因"（这次的 CUDA toolkit 问
题就是活生生的例子），不是一个要主动去优化的目标——除非以后真的要精调具体 kernel，那已经超出"agent 加速
方法研究"的范畴，是纯 GPU 系统工程的课题了。

## 一张表总结

| 层 | 负责什么 | 这个项目对应 | 现状 |
|---|---|---|---|
| L1 Harness | 问什么、问几次 | `inspect_trace` 目标一/二 | 已有较完整观测，未系统性做加速 |
| L2 部署/编排 | 送去哪个实例 | `TENSOR_PARALLEL_SIZE`、`kv_transfer_params` | 几乎未涉及，明确判断为当前范围外 |
| L3 vLLM 引擎 | 单实例怎么调度 | 需求 B（观测）+ 需求 C（干预接口，未做） | 本轮工作重心 |
| L4 Model | 模型本身 | 无，给定输入 | 范围外 |
| L5 执行运行时 | kernel 怎么跑 | 无主动优化，但真实踩过环境 bug | 只作为故障排查知识，非研究目标 |

## 相关文档

- [`vllm_request_concepts.md`](./vllm_request_concepts.md) —— L3 内部一次请求的完整生命周期
- [`serving_observability_b_howto.md`](./serving_observability_b_howto.md) —— 需求 B 怎么对 L3 做可观测
  性
- [`next_phase_requirements.md`](./next_phase_requirements.md) —— 需求 B/C/D/E 的完整定义，C 是标准化
  L3 层加速干预接口
- [`nscc_h100_speculative_decoding_plan.md`](./nscc_h100_speculative_decoding_plan.md) —— L3 层投机解
  码方法的选型背景

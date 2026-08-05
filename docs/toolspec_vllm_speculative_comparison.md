# vLLM 自带投机解码 vs ToolSpec：真实对比

延续 [`toolspec_integration_findings.md`](./toolspec_integration_findings.md) 的问题：ToolSpec 论文自己的方法（schema-aware + retrieval-augmented speculative decoding）比起我们已经在用的 vLLM 服务本身自带的投机解码功能，效果差多少？这份文档记录真实跑出来的对比数据。

## 背景：为什么不能直接拿 vLLM 跟 ToolSpec 的 tokens/s 比

vLLM（HTTP 服务，continuous batching + paged attention）和 ToolSpec 的原始 HF `transformers` 生成循环是两套完全不同的 serving 栈，各自的 baseline（不开任何加速）吞吐本身就不一样：

| baseline | tokens/s |
|---|---|
| ToolSpec 原生仓库（HF 直接 generate，见 `toolspec_integration_findings.md`） | 25.35 |
| ToolSpec 适配器（同一套 HF 代码，走 inspect_ai） | 25.41 |
| **vLLM（HTTP 服务，本文档新测）** | **27.82** |

所以公平的比较方式是**各自的加速比**（speculative 模式 tokens/s ÷ 各自 baseline tokens/s），不是直接比原始 tokens/s 数字。

## vLLM 自带投机解码：ngram 模式

这台机器只有 `Qwen/Qwen2.5-3B-Instruct`（HF 缓存已有，无需下载），没有缓存任何更小的同系列草稿模型，所以先测的是 vLLM **不需要额外草稿模型**的 n-gram/prompt-lookup 投机解码（`--speculative-model "[ngram]"`），概念上接近 ToolSpec 自己拿来对比的 `pld`（Prompt Lookup Decoding）baseline，只是这次是在 serving 层而不是 ToolSpec 的原始 HF 循环里实现的。

启动命令（已经封装进 `local-model-server/scripts/serve.sh`，见下面复现命令）：

```bash
SPECULATIVE_MODE=ngram NUM_SPECULATIVE_TOKENS=5 NGRAM_PROMPT_LOOKUP_MAX=4 NGRAM_PROMPT_LOOKUP_MIN=1 \
  ./local-model-server/scripts/serve.sh
```

同样的 100 条 API-Bank 问题，通过 `toolspec_adapter` 的 dataset/task（复用不改，只是把 `--model` 指向 vLLM 而不是 ToolSpec 的自定义 provider）：

```bash
VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
  uv run --project toolspec_adapter inspect eval task.py -T limit=100 \
  --model "openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" --max-connections 1
```

## 真实结果

速度（`ModelEvent.working_time`/`output.usage.output_tokens`，逐题算比值再取平均，跟 ToolSpec 自己 `evaluation/speed.py` 同一套方法论——原因和踩过的坑见 `toolspec_integration_findings.md` 的"速度对比"一节）：

| | tokens/s | 各自的 speedup |
|---|---|---|
| vLLM baseline（无加速） | 27.82 | 1.00x |
| **vLLM + ngram 投机解码** | **52.03** | **1.87x** |
| ToolSpec 原生仓库 | 77.33（baseline 25.35） | 3.05x |
| ToolSpec 适配器（我们 harness） | 76.16（baseline 25.41） | 3.00x |

正确性（跟各自baseline 比，不是跟 ToolSpec 的 baseline 比——vLLM 的 baseline 本身跟 ToolSpec 的 baseline 就有 20/100 的差异，这是两套 serving 引擎数值路径不同导致的正常现象，不代表哪边错，混着比较没有意义）：

| | 跟自己 baseline 的偏离数 |
|---|---|
| vLLM + ngram vs vLLM baseline | **23/100** |
| ToolSpec (adapter) vs ToolSpec baseline | 11/100 |

## 结论

**在这个具体任务（API-Bank，Qwen2.5-3B-Instruct）上，ToolSpec 的 schema-aware + retrieval-augmented 方法明显比 vLLM 自带的通用 ngram 投机解码更好——速度上快 60%（3.00x vs 1.87x），正确性上偏离率还只有一半（11% vs 23%）。**

这个结果是符合预期的，不是巧合：vLLM 的 ngram 方法是完全通用的——只在当前 prompt/已生成内容里找重复的 n-gram 当草稿，不知道这是个 tool-calling 任务，不知道输出要符合什么 JSON schema。ToolSpec 是专门为这类高度结构化、参数经常重复的 tool-calling 场景设计的——schema-aware FSM 直接知道 `{"name": "..."` 这类固定片段该怎么写，retrieval 机制还能跨请求复用历史相似调用。**通用方法 vs 领域特定方法，在领域特定信息很强的任务上，后者赢是符合直觉的**，这次是把它量化出来了。

正确性偏离率更高这一点也提供了一个新的角度：ngram 方法完全基于"字面重复"猜测下一个 token，缺少 ToolSpec 那种 schema 约束/verification 的额外结构信息，更容易在参数值这类变化点上瞎猜出跟真实 baseline 不同的候选并被意外接受。

**没有测的部分（如实标注）**：vLLM 也支持基于**独立草稿模型**的投机解码（`--speculative-model <model_path>`），这次没测——这台机器的 HuggingFace 缓存里没有比 Qwen2.5-3B-Instruct 更小的同系列模型（比如 Qwen2.5-0.5B-Instruct），需要额外下载。草稿模型型的投机解码在通用文本生成上通常比 ngram 效果更好，但对 tool-calling 这种任务，ToolSpec 论文自己的实验（EAGLE3 对比）已经说明专门做过 schema 适配的方法仍然占优——这个方向值得作为后续工作补充，不是这次的核心结论会因此改变。

## 一个真实发现：`inspect_trace` 的 vLLM metrics 采集器在这次场景下完全没有产出数据

**如实记录一个过程中发现的、跟"ToolSpec vs vLLM"这个核心问题无关，但值得单独记下来的真实 bug。** 跑这次对比时想用 `inspect_trace/vllm_metrics.py`（`goal2_design.md`/`goal2_real_validation_findings.md` 里那套 TTFT/ITL 实时采集机制，之前的验证声称 49/49 次调用 100% 精确归因）拿更细的 model-invocation 层数据，结果这次的两个 vLLM run（`runs/toolspec_vllm_baseline/`、`runs/toolspec_vllm_ngram/`）的 `.inspect_trace/` 目录里**一条 `vllm_metrics` 记录都没有**——`prefill_diff`/`segment_tokens`/`token_attribution`/`attempt_group`/`execution_topology` 五种记录都正常各 100 条，唯独 `vllm_metrics` 完全缺失。

**排查过程（用临时调试日志逐步定位，调试代码已经清理干净，不在最终代码里）**：
1. 确认 `/metrics` 端点本身工作正常、有真实数据（`curl http://localhost:8000/metrics` 能看到 `vllm:time_to_first_token_seconds_count` 随请求数增长）。
2. 确认 `TraceHooks.on_before_model_generate` 阶段的 `VLLMMetricsTracker.before_model_generate()` 正常执行、正常抓到快照、正常存入 `_before_by_key`。
3. 确认 `on_sample_event` 阶段 `VLLMMetricsTracker.sample_event()` 被调用、correlation key **正确匹配**（`before` 不是 None）。
4. 但紧接着 `await fetch_snapshot(self._client)`（第二次抓取 `/metrics`）这一步**既不返回、也不抛出任何异常**——加了 `try/except Exception` 包裹也没捕获到任何东西，换一个全新的 `httpx.AsyncClient()` 现场创建也没用，问题依旧。
5. 一个真实的中间发现（虽然不是根因，但过程中必须先解决才能继续调试）：`toolspec_adapter` 用 `uv sync` 装的 `inspect_trace` **是编译打包复制进 `site-packages` 的，不是 editable 安装**——改 `inspect_trace` 源码后必须 `uv sync --reinstall-package inspect_trace` 才会生效，直接改源码重跑没有任何变化，一度误以为是"改了没生效"。

**没能在这次任务的时间预算内查到根因**——推测跟 inspect_ai 自己的 hook 派发机制有关：`inspect_ai/src/inspect_ai/hooks/_hooks.py::drain_sample_events()` 用 `anyio.move_on_after(5)` 限制等待 hook 处理完成的时间，且 sample event 是通过一个独立的后台任务（`_emit_loop`，`active.tg.start_soon` 启动）异步消费的，跟 `before_model_generate` 触发时所在的调用栈不是同一个执行路径——如果这两处 hook 回调运行在不同的 anyio 任务/取消作用域下，`httpx.AsyncClient` 的连接池在跨作用域场景下卡住是有先例的一类问题，但这只是一个有根据的猜测，不是已经验证的结论。

**这次任务里的应对**：改用 inspect_ai 自己在 `ModelEvent.working_time` 上暴露的、更朴素但可靠的字段（每次模型调用的真实端到端耗时）配合 `output.usage.output_tokens` 计算 tokens/s，绕开这个坏掉的采集器，本文档所有速度数字都是这样算出来的，没有依赖 `vllm_metrics`。

**这跟 `goal2_real_validation_findings.md` 里"49/49 100% attribution confidence"的结论是否矛盾**：不确定，如实标注为待查——可能是这次的调用模式（`toolspec_adapter` 的 provider 组合、或者这个具体版本的运行环境）触发了一个那次验证没有覆盖到的边界情况，也可能是这中间某次改动引入了回归。**这是一个需要单独立项调查的真实 bug，不应该被当作"目标二基础设施仍然可靠"的既有结论继续沿用而不重新验证**——已经如实记进这里，等专门的时间来查。

## 复现命令

```bash
# 1. vLLM baseline（无加速）
./local-model-server/scripts/serve.sh
VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
  uv run --project toolspec_adapter inspect eval task.py -T limit=100 \
  --model "openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" --max-connections 1 \
  --log-dir runs/toolspec_vllm_baseline/logs
./local-model-server/scripts/stop.sh

# 2. vLLM + ngram 投机解码
SPECULATIVE_MODE=ngram ./local-model-server/scripts/serve.sh
VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
  uv run --project toolspec_adapter inspect eval task.py -T limit=100 \
  --model "openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" --max-connections 1 \
  --log-dir runs/toolspec_vllm_ngram/logs
./local-model-server/scripts/stop.sh
```

原始数据：`runs/toolspec_vllm_{baseline,ngram}/logs/*.eval`（用 `inspect view` 或 `inspect_ai.log.read_eval_log` 直接读，方法见 `toolspec_integration_findings.md`"怎么亲自核对"一节）。

## 相关文档

- [`toolspec_integration_findings.md`](./toolspec_integration_findings.md) —— ToolSpec 原生复现 + 迁移进 harness
- [`acceleration_methods_survey.md`](./acceleration_methods_survey.md) —— 目标五，ToolSpec 的 insight/method 分析
- [`goal2_design.md`](./goal2_design.md) / [`goal2_real_validation_findings.md`](./goal2_real_validation_findings.md) —— `vllm_metrics` 采集器的设计和之前的验证结果（本文档发现的 bug 跟这两篇的结论存在未解决的冲突）

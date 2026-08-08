# 需求 B（Serving 观测闭环）：怎么跑 + 怎么核对验收标准

配套 `docs/next_phase_requirements.md` 的需求 B（B1-B5）。这些脚本是**面向 NSCC H100 部署写的**（新版
vLLM，见 `../nscc_model_server/`），不是给这台本地开发机（老 vLLM、无 EAGLE-3）用的——这台机器上没法端到
端跑通任何一步，如实说明，不假装验证过。

## 现状：哪些字段是真的、哪些是待验证的猜测

在写这批脚本时发现一个之前没人记录过的真实约束（细节见
`inspect_trace/src/inspect_trace/vllm_per_request_metrics.py` 模块 docstring）：**inspect_ai 默认只保留
每个模型前 5 次调用的原始请求/响应数据**（`DEFAULT_LOG_MODEL_API_CALLS=5`），超过这个数量后
`ModelEvent.call` 变成 `None`，逐请求指标 collector 就拿不到任何东西。`run_b5_matrix.sh` 已经设置了
`INSPECT_EVAL_LOG_MODEL_API=true` 来关掉这个截断——**如果你自己另外写脚本调用
`run_bfcl_benchmark.sh`，必须自己设这个环境变量，否则 B1/B2 的逐请求指标只有前 5 次调用有数据，其余全部
是 `confidence=unattributed`，看起来像是"vLLM 版本不支持"，实际上是这个截断在起作用**。

vLLM 逐请求 metrics 响应里具体字段叫什么名字，官方文档页（docs.vllm.ai）这次抓取一直被限流（429），
2026-08-07 换成直接读 vLLM 在 GitHub 上的真实源码（`main` 分支）拿到了字段名：`ChatCompletionResponse`
自带一个 `metrics: PerRequestTimingMetrics | None` 字段，真实字段名是
`time_to_first_token_ms`/`queue_time_ms`/`generation_time_ms`/`mean_itl_ms`/`tokens_per_second`（**都是
毫秒**，之前代码假设的是秒，已经改成自动 ÷1000 换算）；这个类里**没有** `prefill_time`、也没有投机解码/
guided decoding 相关字段。同样的办法确认了 Prometheus `/metrics` 那边
`vllm:prefix_cache_queries`/`vllm:prefix_cache_hits`（不是猜的 `gpu_prefix_cache_*_total`）、
`vllm:num_preemptions`（不带 `_total`）、`vllm:kv_cache_usage_perc`（不带 `gpu_` 前缀）、
`vllm:iteration_tokens_total` 这几个真实名字，都已经改成候选表里优先尝试的名字，旧的猜测降级成兜底候选。

**当时猜错的一点，2026-08-08 用真实 NSCC 服务器（`vllm-0.26.0-8cfe525c`）纠正了**：原来以为 `metrics`
"不需要任何请求参数开启，自动填充"——实际打了真请求发现响应里 `"metrics"` 字段确实存在，但值是 `null`；
往下查 vLLM 真实 serving 代码（`vllm/entrypoints/openai/chat_completion/serving.py`）才发现，这是一个
**服务端启动参数** `--enable-per-request-metrics` 控制的，不开就永远是 `null`，跟请求本身无关。之前
`inspect_per_request_metrics.sh` 里猜的请求级 `return_metrics` 参数已经被真实验证证明什么都不做，脚本
里那部分已经删掉了。`serve.sh` 现在默认带上这个参数了。

**2026-08-08 二次确认**：`serve.sh` 带上 `--enable-per-request-metrics` 重新起服务后，再跑一次
`inspect_per_request_metrics.sh`，`metrics` 这次真的填充了，五个字段名（`time_to_first_token_ms`/
`generation_time_ms`/`queue_time_ms`/`mean_itl_ms`/`tokens_per_second`）**跟源码读到的完全一致**，没有
一个猜错，代码里的 `_FIELD_CANDIDATES`/`_MS_FIELDS` 不需要再改。B1 这条从"有源码依据、未验证"正式变成
"已验证"，"现场验证进度"一节有完整记录。

## 需求条款 ↔ 实现对应表（B1-B5 逐条，如实标状态）

下面照 `next_phase_requirements.md` 里 B1-B5 的原文逐条列，不是笼统地说"做了"——状态分三种：
**✅ 已实现**（代码存在，逻辑对应需求，但还没有真实硬件数据验证过，因为这台机器跑不了新版 vLLM）、
**⚠️ 部分实现**（做了一部分，或者字段/机制存在但不保证真的能读到值）、**❌ 未实现**（这次没做，如实
承认）。写这张表的过程中自己核对出几处之前文字描述里含糊带过的缺口，已经顺手在代码里补了（下面标注了
"2026-08-07 补"的就是这次连着一起修的），仍然没做的也如实留在表里，不假装完整。

### B1：逐请求指标优先

| 条款 | 状态 | 说明 |
|---|---|---|
| queue time / TTFT / decode time / mean ITL / output tokens/s | ✅ 已实现，**真实 NSCC 响应验证过** | `vllm_per_request_metrics.py` 提取 `response["metrics"]` 里的 `queue_time_ms`/`time_to_first_token_ms`/`generation_time_ms`/`mean_itl_ms`/`tokens_per_second`。2026-08-08 二次跑 `inspect_per_request_metrics.sh`（`serve.sh` 已带上 `--enable-per-request-metrics`）拿到真实填充的响应，五个字段名**完全对上**、没有一个猜错，也确认了这版本里确实没有 `prefill_time`/投机解码相关字段（不是没开，是真的没有） |
| "是否启用、响应结构必须现场验证" | ✅ 已验证 | 机制（服务端 `--enable-per-request-metrics` flag，不是请求参数）和字段名（见上一行）2026-08-08 都用真实 NSCC 响应确认过了；之前猜的请求级 `return_metrics` 参数已经证实什么都不做，从脚本里删掉了 |
| Prometheus 服务级观测：running/waiting requests | ✅ 已实现 | `service_metrics_sampler.py`，字段名沿用 `vllm_metrics.py` 里已经在这台机器老版本 vLLM 上验证过的名字（`vllm:num_requests_running` 等，新版本 V1 metrics 源码确认这两个名字没变），但**没有在新版本上实测过** |
| prefill/decode/queue **histogram**（分布，不只是均值） | ❌ 未实现 | `service_metrics_sampler.py` 目前只读 gauge/counter 类字段，没有解析这几个 histogram 本身（bucket 分布），只有 invocation 层单次值的 prefill/decode time（见 B4） |
| preemption / KV Cache | ✅ 已实现，字段名**有源码依据** | `vllm:num_preemptions`（不带 `_total`，跟老版本的 `vllm:num_preemptions_total` 不一样）、`vllm:kv_cache_usage_perc`（不带 `gpu_` 前缀，是从老版本 `vllm:gpu_cache_usage_perc` 改名来的）——都是从 vLLM V1 metrics 源码读到的真实名字，候选表已更新为优先尝试这两个 |
| prefix cache 命中率 | ⚠️ 部分实现，字段名**有源码依据** | 源码里没有现成的"命中率"字段，是两个真实计数器 `vllm:prefix_cache_queries`/`vllm:prefix_cache_hits`（不是最初猜的 `gpu_prefix_cache_*_total`）的比值，`service_metrics_sampler.py` 已更新为优先用这两个真实名字算 |
| speculative acceptance（服务级） | ❌ 未实现 | 只在 B4 invocation 层的逐请求候选字段里猜了 `speculative_acceptance_rate`（读源码确认 `PerRequestTimingMetrics` 里其实没有这个字段，纯粹是没依据的长线候选），服务级（Prometheus 累计值）没做 |
| MFU（Model FLOPs Utilization） | ❌ 未实现 | 这次完全没碰，`service_metrics_sampler.py`/`vllm_per_request_metrics.py` 都没有任何 MFU 相关字段，读过的源码文件里也没见到 |

### B2：统一请求关联

| 条款 | 状态 | 说明 |
|---|---|---|
| `run_id → eval_id → sample_uuid/episode_id → model_event_uuid/invocation_id → serving_request_id` | ✅ 已实现 | 前四层本来就有（`TraceEnvelope`/`model_event_uuid`），这次只新加了最后一层 `serving_request_id` 字段（`schema.py` 的 `VLLMPerRequestMetricsRecord`） |
| 不允许只靠时间窗口匹配 | ✅ 已实现 | 新 collector 完全不做前后窗口比较，直接读这次调用自己的响应数据，架构上就不存在"窗口匹配"这回事 |
| 缺少 `serving_request_id` 时标记 `unattributed` | ✅ 已实现 | `extract_per_request_metrics()` 的 `confidence` 判定逻辑 |

### B3：指标证据等级

| source | 状态 | 说明 |
|---|---|---|
| `server_per_request` | ✅ 已实现 | `VLLMPerRequestMetricsRecord.source`，新记录自带 |
| `server_prometheus` | ✅ 已实现（2026-08-07 补） | 最初漏了——`VLLMMetricsRecord`（老的 Prometheus 差值记录）原来没有 `source` 字段，写文档核对这张表时发现的，已经补上（默认值 `"server_prometheus"`，向后兼容，不影响已有 3 个消费者） |
| `client_stream` | ❌ 未实现 | 没有任何 record kind 显式标了这一档——概念上最接近的是"客户端观测的首/相邻 chunk 时间"，这个项目目前没有单独采集这个 |
| `inspect_event` | ❌ 未实现 | `working_time`/`usage` 这类数据本来就在 `.eval` 日志里（`TokenAttributionRecord` 等会用到），但没有一个字段显式标出"这条数据的 source 是 inspect_event 这一档" |
| `estimated` | ❌ 未实现 | `prefill_diff.py`/`token_attribution.py` 里一堆 `*_tokens_estimate` 字段本质就是这一档，但同样没有显式 `source` 标签字段 |

**如实说明**：B3 字面要求"每个指标都要记 source"，这次只把新写的和顺手回补的这两种记录做到了，另外三档
（`client_stream`/`inspect_event`/`estimated`）对应的数据在代码库里是存在的，只是没有一个统一的 `source`
字段把它们标出来——不是数据缺失，是"贴标签"这件事没做完。要做完需要给好几个已有 record kind（`schema.py`
里另外 5 个）都加字段，这次没做，如实留到这里。

### B4：必须采集的指标

**Invocation 层**——全部 9 个字段都已经在 `VLLMPerRequestMetricsRecord` 里定义了，但"定义了" ≠
"保证能读到"：

| 字段 | 状态 |
|---|---|
| prompt/generated/cached token、finish/cancel/error reason | ✅ 已实现，标准 OpenAI 响应字段，vLLM 现在就在填，不依赖这次的新功能 |
| queue time、TTFT、decode time、mean ITL、request tokens/s | ⚠️ 部分实现，字段名**有 vLLM 源码依据**（`PerRequestTimingMetrics`，2026-08-07 读源码确认，见 B1 表），已按真实字段名（含毫秒→秒换算）重写，未在 NSCC 实际版本上确认 |
| prefill time | ❌ 读源码后确认现在这个字段**不存在**于 `PerRequestTimingMetrics` 里——B4 原文"服务端版本可用时"这个限定词现在看是对的，不是随口一说 |
| speculative draft/accepted/acceptance rate、guided decoding 开销 | ⚠️ 部分实现，候选字段名**没有任何源码依据**（读过的 `PerRequestTimingMetrics` 源码里没有这些），纯粹是"万一某个版本/某条路径有"的长线候选 |

**服务层**——这次审这张表之前漏了"batch/iteration token"，已经补上（`_iteration_tokens()`，
2026-08-07）：

| 字段 | 状态 |
|---|---|
| running/waiting requests、preemption、KV Cache 使用率 | ✅ 已实现，字段名**有源码依据**（running/waiting 沿用老版本就对的名字；preemption/KV Cache 按 V1 源码改名后的真实名字重写，见 B1 表） |
| prefix cache 命中率、batch/iteration token | ⚠️ 部分实现，字段名**有源码依据**（真实计数器名字，见 B1 表），未在 NSCC 实际版本确认 |
| GPU utilization/显存/功耗 | ✅ 已实现，走 `nvidia-smi`，不依赖 vLLM 版本 |
| 指标采集本身开销 | ⚠️ 部分实现，`service_metrics_sampler.py` 有 `poll_duration_seconds`；`vllm_metrics.py`（老 collector）自己的两次 `/metrics` 请求耗时没有单独测量，这是已知缺口 |

**Episode 层**：

| 字段 | 状态 |
|---|---|
| P50/P95（P99 样本量门槛） | ✅ 已实现，`episode_layer.py` |
| critical path、model/tool/waiting 时间、success rate、cost per successful episode | ✅ 已实现，这次之前就有 |
| 单调用 speedup 到 episode speedup 转化率 | ⚠️ 部分实现，`speedup_conversion.py` 算法写好了，但**没有接进任何脚本自动调用**——要用的话得自己拿两个 run（baseline + 方法）各自的逐请求 tokens/s 列表和 `episode_layer.summarize_run()` 结果手动传进去 |

### B5：最小稳健性实验矩阵

| 维度 | 状态 | 说明 |
|---|---|---|
| concurrency（1/4/8，8 干净才加 16） | ✅ 已实现 | `run_b5_matrix.sh` |
| workload（真实 trace，保留长度分布） | ✅ 已实现 | 复用 `run_bfcl_benchmark.sh` 对接的真实 BFCL 数据集切片，长度分布是"真实数据集的自然分布"，不是刻意保持的——因为没有做任何重新采样/加权，分布不会被破坏，但也没有专门验证/统计过这批切片的分布跟全集像不像 |
| method（baseline + 可选一个已有方法验证兼容性） | ⚠️ 部分实现 | 脚本本身不区分 baseline/方法，只跑 `MODEL` 指向的那个服务；要满足这条得手动跑两次——先对着 `serve_baseline.sh` 跑一遍矩阵，再对着 `serve_eagle3.sh` 跑一遍，不是脚本自动做比较 |
| arrival（固定并发，Poisson/QPS 是扩展） | ✅ 已实现 | 只做了固定并发（`MAX_CONNECTIONS`），Poisson/burst/open-loop QPS 按需求原文本来就是扩展实验，没做 |
| repetition（≥3，记录样本数和离散程度） | ✅ 已实现（2026-08-07 补） | `REPETITIONS=3` 默认值本来就有；"样本数和离散程度"最初只有 `manifest.jsonl` 的 pass/fail，没有真正算离散度——写文档核对这张表时发现的缺口，已经补上：每个并发度的所有 rep 跑完后，汇总该并发度下所有 episode 的延迟，算 n/mean/stddev，写到 `concurrency_<N>/dispersion_summary.json` |

## 这次做了什么、为什么这么做

这一节讲代码本身：改了哪些文件、每处为什么这么设计，不只是怎么调用。按依赖顺序讲（先讲底层的数据怎么拿到，
再讲上层怎么用它）。

### 整体思路：B3 要求的"多个来源并列、不覆盖"，靠两个独立 record kind 实现，不是一张大表

需求 B3 说同名指标如果来自不同来源（比如 TTFT 既可以从新版 vLLM 的逐请求 API 拿，也可以从 Prometheus 直方
图差值法拿），必须并列保存、可以交叉验证，不能互相覆盖。最直接的错误做法是做一张大表，把 5 种 source 的
值都塞进同一条记录的同名字段——这样后写的会覆盖先写的，正是 B3 明确禁止的。

实际做法：新逐请求指标是**完全独立的一种新 record kind**（`vllm_per_request_metrics`），跟原来就有的
`vllm_metrics`（Prometheus 差值法，`inspect_trace/src/inspect_trace/vllm_metrics.py`，这次完全没改）并列
存在，两条记录靠共享的 `model_event_uuid` 字段 join 在一起。这样"多来源并列"这个要求不需要额外机制，两个
独立 record kind 本身就是并列的，天然可以交叉验证，谁也不覆盖谁。原来的 `vllm_metrics` 一行代码都没动，
`kind` 名也没改——它已经被 3 个脚本消费（两份 dashboard + `run_concurrency_validation.sh`），改名字的影响
面没必要承担。

### 逐请求指标怎么拿到的：`inspect_trace/src/inspect_trace/vllm_per_request_metrics.py`（新文件）

这是这次的核心新代码。原来的 `vllm_metrics.py` 靠"在模型调用前后各查一次 Prometheus `/metrics`，用直方图
计数的差值猜这次调用的数字"，这个方法**必须串行执行**才准（`_count` 差值恰好是 1 才能确定这次差值就是这
次调用的），一旦真并发（B5 要测的 concurrency=4/8）就大概率失真，被标成 `ambiguous`。这正是 B1 说的"不允
许只靠时间窗口匹配"要解决的问题。新版 vLLM 的逐请求 API 每个请求自带真实 ID，不需要猜——这是唯一真正的解
法，所以新写了这个文件，而不是在老文件上修修补补。

**怎么读到数据**：不是像 `vllm_metrics.py` 那样自己再发一次 HTTP 请求去问 vLLM，而是直接读 inspect_ai 已
经拿到手的东西——每次模型调用完成后，inspect_ai 会把原始请求/响应存在 `ModelEvent.call.response`（一个
`dict`）里。`extract_per_request_metrics(event)` 是个纯函数，只读这一个字段，没有任何网络调用。

**写这个文件时踩到的一个真坑，值得单独说清楚**：读 inspect_ai 源码（`inspect_ai/log/_transcript.py`）才发
现，**它默认只保留每个模型前 5 次调用的 `call.response`**（常量 `DEFAULT_LOG_MODEL_API_CALLS = 5`），第 6
次调用开始 `event.call` 直接变成 `None`，而且这个截断在 hook 层面就生效了（不是"日志文件里没存"，是"运行
时这个 hook 拿到手的 event 本身就是 None"）。这意味着：如果不特意设置
`INSPECT_EVAL_LOG_MODEL_API=true`（或者 `eval()` 的 `log_model_api=True` 参数），任何一个模型调用超过 5
次的 benchmark，新逐请求指标从第 6 次调用起全部是 `confidence=unattributed`——表面看像是"这个 vLLM 版本不
支持逐请求指标"，实际上是这个截断在起作用，跟 vLLM 版本毫无关系。`run_b5_matrix.sh` 已经设置了这个环境变
量；如果以后你自己另外写脚本跑 benchmark 并期望拿到逐请求指标，必须记得也设它。

这个发现顺带解释了一个这个项目里之前一直没搞清楚的旧疑点：`build_toolspec_dashboard.py` 的
`load_vllm_run()` 函数原来的注释说"`ModelCall.response` 只在 toolspec_adapter 的自定义 ModelAPI 里才
有"，我把这条注释改成了准确的说法——真实机制是这个"前 5 次"截断，跟用哪个 ModelAPI 无关，toolspec_adapter
的那几次运行大概率只是调用次数没超过 5 次，所以之前一直"看起来正常"。

**字段怎么解析出来的、什么是确定的、什么是猜的**：官方 vLLM 逐请求 metrics 文档页（docs.vllm.ai）这次抓
取一直被限流，2026-08-07 改成直接读 vLLM 在 GitHub 上的真实源码，分三个可信度处理：

- **确定能读到的**（标准 OpenAI chat-completion 响应格式，vLLM 现在就在填，跟这次的新功能无关）：`id`
  （当作 `serving_request_id` 候选）、`choices[0].finish_reason`、`usage.prompt_tokens`、
  `usage.completion_tokens`、`usage.prompt_tokens_details.cached_tokens`。
- **有 vLLM 源码依据、未在 NSCC 实际版本确认**：`response["metrics"]` 这个容器 key 本身，加上里面的
  `time_to_first_token_ms`/`queue_time_ms`/`generation_time_ms`/`mean_itl_ms`/`tokens_per_second`——这些
  名字是从 vLLM `main` 分支源码的 `PerRequestTimingMetrics` 类读到的（`vllm/entrypoints/openai/engine/
  protocol.py`），不是编的，`_FIELD_CANDIDATES` 已经改成把这些真实名字放在候选列表最前面，单位是毫秒，
  提取时会自动 ÷1000 换算成秒（`_MS_FIELDS`）。`tokens_per_second` 是 vLLM 自己算好的，直接用，不再自己
  拿 decode_time 除 token 数重新算一遍。
- **没有任何依据的长线候选**（读过的源码里确认没有）：`prefill_time`、投机解码 draft/accepted token 数
  和接受率、guided decoding 开销——`PerRequestTimingMetrics` 这个类本身就没有这几个字段，候选表里还留着
  它们纯粹是"万一某条路径/某个版本有"，不代表有依据。

不管哪一类，找不到都如实记进 `raw_fields_missing` 列表，不会拿 0 或者猜测值顶替；容器 key 本身也是
`metrics` 排第一（源码确认），`vllm_metrics`/`timing` 降级成兜底（`_METRICS_CONTAINER_CANDIDATES`）。

**`confidence` 怎么判定**：`event.call` 是 `None`（前面说的截断，或者调用本身失败）→
`confidence="unattributed"`；能读到响应但没有 `serving_request_id`（连 `id` 字段都没有）→ 也是
`unattributed`；只要有一个能站得住脚的 ID → `confidence="exact"`。这对应 B2"缺少 serving_request_id 时
必须标记 unattributed，不得静默归因"。

### schema 改动：`inspect_trace/src/inspect_trace/schema.py`

新增 `VLLMPerRequestMetricsRecord`（`kind="vllm_per_request_metrics"`），字段基本是上面那些提取出来的值
一一对应，另外两个字段是设计上专门加的：`source`（固定值 `"server_per_request"`，标出这条记录属于 B3 五
级证据里的哪一级）、`raw_fields_missing`（前面说的"这些字段这次没找到"清单，让下游一眼看出这条记录里哪
些是真数据、哪些是空的）。加进了 `TraceRecord` 联合类型，其余现有 record 一个字段都没动。

### 接进现有流程：`inspect_trace/src/inspect_trace/hooks.py`

`TraceHooks` 原来的模式是"每种指标一个 tracker，在 `on_sample_event` 里产出一个 payload dict，包一层
schema record，写盘"——`vllm_metrics.py` 已经是照这个模式接的。新 tracker 完全复用同一个模式，在原有
`vllm_metrics` 那段代码后面追加了几行，没有改动 writer/context 这些底层机制。唯一多加的判断是
`if "vllm" in event.model.lower()`：因为这个新 collector 不像老的 `vllm_metrics.py` 有"探测 `/metrics`
是否可达"这个天然的开关，纯读 event 数据的话对托管模型（Claude/GPT 等）调用也会"成功"提取出一堆 `None`，
所以用模型名字符串里有没有"vllm"这个粗糙但够用的信号，避免给非 vLLM 调用也写一条内容全空的
`vllm_per_request_metrics` 记录。

### episode 层百分位数：`inspect_trace/src/inspect_trace/analysis/episode_layer.py`

`EpisodeLayerRunSummary` 原来只有 `mean_end_to_end_latency_seconds`。加了
`p50_end_to_end_latency_seconds`/`p95_end_to_end_latency_seconds`/`p99_end_to_end_latency_seconds`，用一
个标准的线性插值百分位函数 `_percentile()` 算（跟 numpy 默认的 `interpolation="linear"` 结果一致）。P99
单独加了一个门槛：样本数少于 `_P99_MIN_SAMPLES`（=100）时不计算，直接是 `None`——B 类验收标准原文写"样本
不足时不报告 P99"，如果样本只有几个，P99 跟"最大值"没有本质区别，报出来会误导人，不如明确留空。这条改动
没碰任何数据来源，纯粹是在已有的 per-episode 延迟列表上加统计。

### 单调用到 episode 的 speedup 转化率：`inspect_trace/src/inspect_trace/analysis/speedup_conversion.py`（新文件）

一个方法让单次模型调用变快，不代表整个 episode 按同样倍数变快——工具调用时间、等待时间这些不会因为解码变
快而缩短。这个模块算的就是"单调用的 speedup 里有多少比例真的体现到了 episode 级别"：`call_speedup` 是两
个 run（baseline vs 加速方法）的平均 tokens/s 之比，`episode_speedup` 是两个 run 的平均 episode 延迟之
比，`conversion_rate = (episode_speedup - 1) / (call_speedup - 1)`（用"倍数减一"而不是原始倍数，因为要问
的是"变快的部分里有多少传导下去了"，不是倍数本身的比值）。

这里有个特意的选择：`call_speedup` 用的是"两个 run 各自的平均 tokens/s 求比值"（ratio-of-means），不是
"每次调用配对求比值再平均"（mean-of-ratios）。后者需要能把两个 run 里的调用一一配对，这个项目目前没有跨
run 的调用级配对机制（episode 有稳定 ID 能配对，单次调用没有），所以老实选了做得到的那种算法，而不是假装
能做更精细的配对。

这次没有去改 `build_tau2_dashboard.py`/`build_toolspec_dashboard.py` 两份已有 dashboard 脚本去接这个新
模块——它们各自的两 run 比较逻辑（一个是按 reward 做三路分类，一个是按补全文本做 mismatch 对比）跟这里要
算的东西领域细节不一样，本来重复的部分很薄，强行抽象反而是给能跑的代码找麻烦，所以没动。

### 服务层独立采样器：`inspect_trace/scripts/service_metrics_sampler.py`（新文件）

B4 的"服务层"那组指标（排队数、KV Cache 使用率、prefix cache 命中率、GPU 利用率/显存/功耗）本质上是服务
状态随时间变化的曲线，不是某一次调用的属性。原有的 hook 架构是"每次调用前后各测一次"，这在 B5 要测的高并
发场景下会漏掉调用之间发生的排队/抢占——所以这是个完全独立于 inspect_ai hook 生命周期的后台进程，按固定
时间间隔（默认 1 秒）轮询 vLLM `/metrics`（复用了 `vllm_metrics.py` 里现成的 `parse_metrics_text`，没有
重复造解析逻辑）和 `nvidia-smi`，一行一条 JSON 写盘，每写一行就 flush（这样 `kill` 掉它最多丢一次正在进
行的轮询，不会丢整个文件）。每条记录自带 `poll_duration_seconds`——这次轮询本身花了多久，对应 B4"指标采
集本身的开销要单独测量"这一条。prefix cache 命中率、preemption、KV Cache、batch/iteration token 这几个
Prometheus 字段名同样是 2026-08-07 读 vLLM V1 metrics 源码（`vllm/v1/metrics/loggers.py`）确认的真实名
字（`vllm:prefix_cache_queries`/`vllm:prefix_cache_hits`、`vllm:num_preemptions`、
`vllm:kv_cache_usage_perc`、`vllm:iteration_tokens_total`），已经作为候选表里优先尝试的名字，最初凭空猜
的那批（`gpu_prefix_cache_*_total` 之类）降级成兜底候选；命中哪个就在 `prefix_cache_hit_rate_source`/
`iteration_tokens_source` 里如实标出（比如 `"counter_ratio"` 表示是从两个计数器现算的，不是现成的
gauge），都找不到就是 `None`。跟前面 `vllm_per_request_metrics.py` 一样，这是"有源码依据"不是"验证过"——
`main` 分支不等于 NSCC 上会装到的具体版本。

### B5 矩阵跑批：`inspect_trace/scripts/run_b5_matrix.sh`（新文件）

对每个并发度（1/4/8，8 如果全部干净再追加 16）跑 `REPETITIONS`（默认 3）次，每次调用现有的
`run_bfcl_benchmark.sh`（沿用它的参数约定，没有重新发明跑 benchmark 的方式），跑之前顺带在后台拉起一个
`service_metrics_sampler.py`，跑完就 kill 掉。每个 cell 独立记成功/失败到 `manifest.jsonl`，失败了不重
试、不中断整个矩阵——"concurrency=8 有没有把服务器打崩"本身就是 B5 要跑出来的答案，不是要绕过去的麻烦。

写的时候踩了一个 bash 的坑，写完立刻自己发现并改掉了：一开始想用 `CONCURRENCIES="$CONCURRENCIES 16"` 在
循环里动态给并发度列表追加 16（"8 干净就加 16"），但 `for x in $CONCURRENCIES` 这种写法在循环一开始就把
字符串按空格切好了，循环中途再改这个变量根本不会影响已经在跑的循环——16 永远不会被真正跑到。改成了用数组
（`concurrency_queue`）配合下标遍历，这样循环中途 `concurrency_queue[${#concurrency_queue[@]}]=16` 追加
的新元素才能在后续迭代里真的被访问到。

`run_concurrency_validation.sh`（B5 的雏形，只有两个写死的并发条件）没有动——它记录的"老硬件上真并发直接
把 vLLM 崩了"是真实的历史证据，不该因为有了新脚本就删掉，只在文件头加了一行指向 `run_b5_matrix.sh`。

### 探测脚本：`nscc_model_server/scripts/inspect_per_request_metrics.sh`（新文件）

这个脚本的作用是先在真实服务器上打一个请求，把原始响应（含 headers）如实打印出来给人看，而不是等
`run_b5_matrix.sh` 跑完一大堆 cell 之后才发现字段名对不上。写这个脚本的时候（第一版）还没读 vLLM 源码，
先发一个普通请求、再发一个带猜测的 `return_metrics: true` 参数的请求、对比两次响应差异，是当时"完全不知
道机制是什么"下的探索写法。2026-08-07 读到 vLLM 真实源码后（见前面"逐请求指标怎么拿到的"一节），已经知
道 `metrics` 字段是自动填充、不需要任何 opt-in 参数的，所以脚本头部注释和步骤 [1]/[2] 的说明文字已经更
新——[1] 变成主检查项（直接看 `response["metrics"]` 是不是长这样：
`time_to_first_token_ms`/`queue_time_ms`/`generation_time_ms`/`mean_itl_ms`/`tokens_per_second`），[2]
的 `return_metrics` 参数留着只是顺手交叉确认，不再是主要假设。风格仍然照抄
`nscc_model_server/scripts/verify_eagle3.sh` 的"如实报告实际看到什么"写法。跑完之后应该把输出发回来，
用来确认这批从源码读来的字段名在这个具体部署上到底对不对，而不是假设 GitHub `main` 分支就是最终答案。

## 跑的顺序

```bash
cd efficient-harness

# 0. 先起服务（EAGLE-3 或 baseline 都行，取决于你想测哪个）
cd nscc_model_server && ./scripts/serve_eagle3.sh   # 或 ./scripts/serve_baseline.sh
cd ..

# 1. 探测真实的逐请求 metrics 响应长什么样，把输出发回来，用来修正 vllm_per_request_metrics.py
#    的字段候选表（当前那份是没验证过的第一版）
cd nscc_model_server && ./scripts/inspect_per_request_metrics.sh
cd ..

# 2. 跑 B5 最小矩阵（concurrency x repetition，含逐请求指标 + 服务层采样）
cd inspect_trace
VLLM_BASE_URL="http://localhost:8000/v1" ./scripts/run_b5_matrix.sh
```

`run_b5_matrix.sh` 每个 cell 的输出在 `runs/b5_matrix/concurrency_<N>/rep_<R>/`：
- `logs/*.eval` —— 原始 inspect_ai 日志
- `.inspect_trace/**/*.jsonl` —— 派生记录，含 `vllm_metrics`（老的 Prometheus 差值法）、
  `vllm_per_request_metrics`（新的逐请求法）、`token_attribution` 等
- `service_metrics.jsonl` —— B4 服务层时间序列（running/waiting/KV/prefix cache/GPU）
- `stdout.log`/`stderr.log` —— 这个 cell 的完整输出，失败时看这里
- 顶层 `runs/b5_matrix/manifest.jsonl` —— 每个 cell 的 pass/fail 一行汇总
- `concurrency_<N>/dispersion_summary.json` —— 该并发度下所有 rep 汇总的 episode 延迟 n/mean/stddev（B5 的"记录样本数和离散程度"）

## B 类验收标准逐条核对

| 验收标准 | 怎么核对 |
|---|---|
| concurrency=1 下逐请求指标与现有客户端/Prometheus 方法交叉验证 | `concurrency_1/rep_*/.inspect_trace/` 里同一个 `model_event_uuid` 应该同时有 `vllm_metrics` 和 `vllm_per_request_metrics` 两条记录——按这个 key join，比较 TTFT/e2e 是否接近 |
| concurrency=4/8 下每个 invocation 能通过 request ID 关联，不能依赖 `_count delta == 1` | 看 `vllm_per_request_metrics` 记录的 `confidence` 字段分布（应该看 `serving_request_id` 是否非空，不是看 `vllm_metrics` 那条老记录的 `attribution_confidence`——高并发下老记录大概率是 `ambiguous`，这是预期内的，不代表 B 失败，B 的答案在新记录里） |
| baseline 在相同 1/4/8 受控负载下均能完成逐请求关联 | `manifest.jsonl` 里对应 cell 的 `outcome` 是否为 `success`，且该 cell 下逐请求记录的 `confidence=="exact"` 占比 |
| queue、batch、KV Cache 和 preemption 如实报告 | 直接读 `service_metrics.jsonl`，不要手动过滤/平滑——`vllm_metrics_reachable=false` 或某字段为 `null` 就如实是 `null`，不要拿别的值顶替 |
| 报告包含 P50/P95、样本量和离散程度 | `inspect_trace.analysis.episode_layer.summarize_run(trace_dir)` 现在返回 `p50_end_to_end_latency_seconds`/`p95_end_to_end_latency_seconds`/`n_episodes`；`p99_end_to_end_latency_seconds` 样本量不足 100 时是 `None`，这是设计如此，不是 bug |
| 指标采集开销单独测量并报告 | `service_metrics.jsonl` 每行的 `poll_duration_seconds` 就是这次采集本身花的时间；**已知缺口**：`vllm_metrics.py`（老的 Prometheus 差值 collector）本身的两次 `/metrics` 请求耗时目前没有单独测量，`vllm_per_request_metrics.py` 是纯内存读取（不额外发请求），本身开销可忽略不计，但这一条还没有补上，如实标注在这里 |
| vLLM 指标不可用时运行继续但显式记录缺失原因 | `vllm_per_request_metrics` 记录的 `raw_fields_missing` 列表、`service_metrics.jsonl` 的 `vllm_metrics_reachable`/`prefix_cache_hit_rate_source` 字段都是为这个设计的 |

## 现场验证进度（2026-08-08 更新，第一批真实 NSCC 数据）

NSCC 节点真实跑起来了（`a2ap-dgx037`），确认了几件事，如实记录：

- **真实 vLLM 版本**：`vllm-0.26.0-8cfe525c`（响应的 `system_fingerprint` 字段里带的），比 `nscc_model_server` 锁的 `vllm>=0.9.0` 下限新得多。
- **两个真实环境问题，都已经修复**：(1) 这个节点没有可发现的 CUDA toolkit（找不到 `nvcc`，`/usr/local/cuda` 不存在，只有驱动）——第一次导致 FlashInfer 融合采样 kernel 现场编译崩溃，已经用 `VLLM_USE_FLASHINFER_SAMPLER=0` 绕开；(2) `serve.sh` 自己的启动失败检测逻辑有 bug，把 vLLM 捕获住的良性 WARNING（里面恰好包含 "Traceback" 字样）误判成致命错误，已经修（只有不带 `WARNING` 前缀的才算真失败）。
- **B1 逐请求指标——容器 key、触发机制、字段名，三件事全部验证完成**：`response["metrics"]` 默认是 `null`，要传 `--enable-per-request-metrics`（服务端启动参数）才会填，已经加进 `serve.sh`；加上之后二次请求拿到的真实响应里，`time_to_first_token_ms`/`generation_time_ms`/`queue_time_ms`/`mean_itl_ms`/`tokens_per_second` 五个字段名跟 vLLM 源码读到的**完全一致**，代码不需要再改。真实数值例子：TTFT 35ms、decode time 158ms、queue time 0.02ms（单请求无排队）、mean ITL 8.3ms、103 tokens/s。
- **一个跟本文档主题相关但还没处理的真实约束**：默认参数（`GPU_MEMORY_UTILIZATION=0.9`、`MAX_MODEL_LEN=16384`）下，32B 模型权重 + EAGLE-3 草稿头占用 63.94 GiB，KV cache 只剩 4.71 GiB（18,976 tokens），日志里"Maximum concurrency for 16,384 tokens per request: 1.16x"——这对 B5 要测的 concurrency=4/8 是个真实瓶颈，大概率需要调低 `MAX_MODEL_LEN` 或调高 `GPU_MEMORY_UTILIZATION` 才跑得动，还没验证过调整后的效果。

**还没验证/还没做的**（原样保留，没有因为上面这些进展就假装解决了）：

- `service_metrics_sampler.py` 里 Prometheus 字段名（`vllm:prefix_cache_queries`/`vllm:num_preemptions`/`vllm:kv_cache_usage_perc`/`vllm:iteration_tokens_total`）——B1 逐请求指标那条已经验证过了，但服务级这几个字段还没有单独跑 `service_metrics_sampler.py` 对着这台真实服务器确认过，源码依据仍然只是 vLLM `main` 分支；running/waiting 是唯一在老版本和新版本源码里都确认没变名的。
- `nvidia-smi` 在 PBS 分配到的节点上是否可直接调用（多卡/MIG 切分场景下 `--query-gpu` 的行为未验证）。
- concurrency=8/16 是否会把服务器打崩——这正是 B5 要跑出来的答案，不是跑之前就该知道的。
- KV cache 余量不够高并发这件事，具体调整到什么参数才够用，还没试。

上面这份是"猜了但没验证"的清单；另外还有几项是这次**压根没做**（不是猜错了，是没碰），已经在上面"需求条款 ↔
实现对应表"里逐条标了 ❌：Prometheus 服务级的 prefill/decode/queue histogram、服务级 speculative
acceptance、MFU、B3 五档 source 里的 `client_stream`/`inspect_event`/`estimated` 三档没有显式打标签、B5
的 baseline/方法对比需要手动跑两次矩阵而不是脚本自动做。

## 相关文档

- [`next_phase_requirements.md`](./next_phase_requirements.md) —— 需求 B 的完整定义
- [`nscc_h100_speculative_decoding_plan.md`](./nscc_h100_speculative_decoding_plan.md) —— NSCC 部署背景
- `inspect_trace/src/inspect_trace/vllm_per_request_metrics.py`、`inspect_trace/src/inspect_trace/vllm_metrics.py` —— 两个 collector 的完整设计说明都在各自模块 docstring 里

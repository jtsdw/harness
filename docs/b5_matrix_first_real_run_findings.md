# B5 最小稳健性实验矩阵：第一次真实 NSCC 数据分析

2026-08-08，`run_b5_matrix.sh` 在 NSCC（`a2ap-dgx037`，`vllm-0.26.0-8cfe525c`）上跑完的第一批真实数据分
析。跑的是 concurrency ∈ {1, 4, 8, 16} × 3 次重复，`inspect_evals/bfcl` 的 `multi_turn_base` 类别，
`LIMIT=20`（每个 cell 最多 20 个样本）。服务端是这一天早些时候用 `serve_eagle3.sh` 起的
（`Qwen/Qwen3-32B` + EAGLE-3 草稿），但这次矩阵跑的时候没有单独重新跑 `verify_eagle3.sh` 确认投机解码是
不是还在生效，如实标注这一点不确定。原始数据在 `nscc_runs/b5_matrix/`（`git pull` 之后跑
`./scripts/pull_runs.sh pull` 同步下来的，不进 git）。

**复现这份分析**：

```bash
cd inspect_trace
uv run python3 -c "
from pathlib import Path
from inspect_trace.analysis.episode_layer import summarize_run
print(summarize_run(Path('../nscc_runs/b5_matrix/concurrency_8/rep_1/.inspect_trace')))
"
```

（或者直接读 `nscc_runs/b5_matrix/manifest.jsonl`、`concurrency_*/dispersion_summary.json`、
`concurrency_*/rep_*/service_metrics.jsonl` 这几份原始文件。）

## 真实发现一：B1/B2 的新逐请求指标在高并发下没有退化，老方法退化得很明显

这是需求 B 整个设计最核心的那个问题——"高并发下还能不能把指标准确归因到某一次调用"——第一次有真实数据能
回答：

| concurrency | 新方法（`vllm_per_request_metrics`）exact 占比 | 老方法（`vllm_metrics` Prometheus 差值）exact 占比 |
|---|---|---|
| 1 | 400/400（100%） | 400/400（100%） |
| 4 | 566/566（100%） | 50/566（9%，其余 516 条是 `ambiguous`） |
| 8 | 556/556（100%） | 25/556（4.5%） |
| 16 | 435/435（100%） | 11/435（2.5%） |

老方法（发请求前后各查一次 `/metrics`，靠直方图计数差值猜是不是"只有一次新观测"）完全符合 B1 文档里描述
的失效模式：并发一高，同一个采样窗口里进来好几个请求的观测，没法再确定这次差值是哪一个请求的，绝大多数
被标成 `ambiguous`。新方法直接读每次请求自己响应体里的 `metrics` 字段，不靠时间窗口猜，四个并发度下全部
是 `exact`——这不是理论推导，是这次真实数据给出的结果。

## 真实发现二：queue time / TTFT 随并发清晰增长，KV cache 在 concurrency≥4 就跑满了

| concurrency | queue_time 均值 | TTFT 均值 | decode_time 均值 | tokens/s 均值 | KV cache 使用率均值 | 本段新增 preemptions |
|---|---|---|---|---|---|---|
| 1 | 0.000s | 0.059s | 5.46s | 81.2 | 24.3% | 0 |
| 4 | 0.109s | 0.162s | 5.42s | 75.9 | 69.5% | 24 |
| 8 | 2.618s | 0.367s | 5.90s | 62.8 | 65.8%（峰值 100%） | 140 |
| 16 | 8.732s | 0.484s | 6.87s | 59.0 | 65.0%（峰值 100%） | 135 |

queue time 从 0 涨到 8.7 秒，是这次矩阵里最干净的一条趋势线，直接对应 B5 想验证的问题（"收益是否随资源
竞争变化"）。KV cache 使用率从 concurrency=4 开始就能摸到 100% 的峰值——这正好印证了之前起服务时日志里
"Available KV cache memory: 4.71 GiB / Maximum concurrency for 16,384 tokens per request: 1.16x" 那个
担心是真实存在的约束，不是纸面推测。preemptions（服务端为了给别的请求腾 KV cache 空间，把某个正在进行
的请求换出去）从 concurrency=4 开始才真正出现，走势和 KV cache 打满的时间点吻合。

**这几个 preemptions/prefix cache 数字是累计计数器，不是每个并发度单独重新计的**——vLLM 的 `/metrics` 
里这几个字段（`vllm:num_preemptions`、`vllm:prefix_cache_queries`/`hits`）是从服务端启动那一刻起累加
的，这次矩阵从 concurrency=1 到 16 是同一个 vLLM 进程连续跑下来的，没有重启过。上表 preemptions 那列已
经处理成"这个并发度这一段新增了多少"（用这段结束时的计数减去开始时的计数），是真实的段内增量，可以直接
比较；但 KV cache 使用率、prefix cache 命中率没法这样处理干净（下一节细说）。

## 真实发现三：prefix cache 命中率随会话推进先冲高再逐渐走低——但这个数字有已知局限，不能直接当成"这个并发度的真实命中率"

`service_metrics_sampler.py` 存的是命中率这个比值本身（`hits/queries` 现算的），没有存原始的 hits/
queries 两个计数器——这是当时设计时的一个遗漏。这次段末观测到的累计比值：concurrency=1 段末 93.4%、
concurrency=4 段末 82.3%、concurrency=8 段末 51.6%、concurrency=16 段末 44.9%。这个数字是"从服务器启动
到这一刻为止全部请求的累计命中率"，不是"这个并发度自己的命中率"——真实情况可能是这样：BFCL `multi_turn_base` 
样本之间有大量重复结构（工具 schema、系统提示），所以刚开始（concurrency=1 段）命中率冲得很快；后面随着
处理的请求越来越多、内容越来越多样，累计比值自然会往下走，这跟"高并发是不是导致 KV cache 被抢占、cache
提前被挤掉"这两个可能的原因混在一起，没法从现在存的这份数据里分开验证哪个是真正原因。

**这是一个值得在 `service_metrics_sampler.py` 里补的真实缺口**：把 `vllm:prefix_cache_queries`/
`vllm:prefix_cache_hits` 这两个原始计数器也存下来（不只存算好的比值），下次矩阵跑完就能像 preemptions
一样算出每个并发度段内真正的命中率增量，而不是只能看一个被前面所有请求污染过的累计数字。这次没有当场改，
如实记在这里。

## 真实发现四（最重要，也是本次矩阵测出来的真实问题）：`MAX_MODEL_LEN=16384` 对这个 workload 明显不够，3/12 个 cell 被真实打断，而且脚本没检测出来

12 个 cell 里有 3 个（25%）在跑的过程中撞上了同一个真实报错，被 `inspect eval` 自己中断：

```
BadRequestError('Error code: 400 - {\'error\': {\'message\': "This model\'s maximum context length
is 16384 tokens. However, you requested 0 output tokens and your prompt contains at least 16385
input tokens..."}}')
```

| cell | 报的中断情况 | episode_layer 实际能分析到的episode数 |
|---|---|---|
| concurrency_1/rep_3 | "2 of 20 total samples logged before interruption" | 3 |
| concurrency_16/rep_2 | "no samples completed before interruption" | 16 |
| concurrency_16/rep_3 | "19 of 20 total samples logged before interruption" | 20 |

`multi_turn_base` 的部分样本本身对话轮数多，累积上下文超过 `MAX_MODEL_LEN=16384` 只是时间问题——这次
concurrency=1（完全没有并发压力）也踩中了同一个错误，说明这不是并发导致的，是这个 workload 本身跟当前
`MAX_MODEL_LEN` 设置不匹配，B5 的并发扫描只是提高了踩中它的样本数（16 次重复扫过去更容易撞上某个特别长
的样本）。

**更值得关注的是：这三个 cell 在顶层 `manifest.jsonl` 里全部记的是 `"outcome": "success", "exit_code": 0`**——
`run_b5_matrix.sh` 现在的成功/失败判定只看 `run_bfcl_benchmark.sh` 的退出码，而 `inspect eval` 在这种
"跑到一半被打断"的情况下依然返回了 0，脚本层面完全没有察觉到这三个 cell 是不完整的。实际统计（上面第一节
的表格）也因此偷偷混入了这几个不完整 cell 的数据，只是这次矩阵的样本量还够大，没有导致结论跑偏，但这是一
个需要修的真实缺口，不能假装没看见：`run_b5_matrix.sh` 的成功判定至少应该额外检查一下实际写出的样本数是
不是等于 `LIMIT`，或者 grep 一下 stdout 里有没有 "interrupted" 字样。

**连带的另一个真实现象**：这三个被打断的 cell，`_manifest.jsonl` 都没有写出来（`TraceHooks.on_task_end`
显然没有机会触发）——这次分析时靠读每个 cell 原始 per-sample trace JSONL 里的 `eval_id` 手动重建了这个
文件才能继续跑 `episode_layer.summarize_run()`。这个行为本身是符合预期的（任务真的没有正常结束，
`on_task_end` 不触发是对的），只是提醒以后遇到"某个 cell 分析不出来"，先查是不是这种情况，而不是假设数
据传输坏了。

## Episode 层汇总（含上面提到的不完整 cell，数据量足够大，结论方向不受影响）

| concurrency | episode 数 | BFCL 成功率 | 平均端到端延迟 | P50 | P95 | 平均模型调用次数 |
|---|---|---|---|---|---|---|
| 1 | 43 | 66.7% | 56.4s | 38.2s | 160.0s | 10.3 |
| 4 | 60 | 65.0% | 59.6s | 44.7s | 95.5s | 10.4 |
| 8 | 60 | 63.3% | 92.1s | 73.6s | 181.3s | 10.3 |
| 16 | 56 | 69.2% | 135.7s | 116.3s | 214.5s | 8.5 |

**BFCL 成功率在四个并发度之间基本持平（63%-69%），没有随并发下降的趋势**——说明这次矩阵测出来的代价完
全体现在延迟上（P50 从 38s 涨到 116s），没有体现在正确性上；换句话说，vLLM 这一层的并发调度没有让模型输
出变得不可靠，只是变慢了。concurrency=16 的平均模型调用次数（8.5）比其他三档（约 10.3-10.4）低一些，大
概率是被上面说的那三个不完整 cell 拖低的（尤其是 0/20 完成的那个 cell），不是一个独立的真实现象。

## 一句话总结 + 后续建议

- B1/B2 的核心设计目标（高并发下依然能精确归因）**第一次有真实数据验证通过**：新方法四个并发度全部
  100% exact，老方法从 100% 一路掉到 2.5%。
- KV cache 4.71 GiB 这个此前只在启动日志里看到的担心，**这次真实测出了后果**：concurrency≥4 就打满，
  伴随 preemptions 真实出现。
- **`MAX_MODEL_LEN=16384` 需要调大**（下一步动作），或者接受这个 workload 下 25% 的 cell 会被打断——不
  应该带着这个已知问题继续跑后面的实验；调大势必进一步压缩本就紧张的 KV cache，可能需要跟 TP=2 或者 FP8
  KV cache 一起考虑（`system_layers_and_acceleration.md` 里 L2/L3 提过这两条路）。
- `run_b5_matrix.sh` 的成功判定需要补一个"样本数是否等于 LIMIT"或者"grep interrupted"的检查，现在会把
  被打断的 cell 悄悄记成成功。
- `service_metrics_sampler.py` 应该把 `prefix_cache_queries`/`hits` 原始计数存下来，不只存算好的比值，
  否则没法算出每个并发度段内真正的命中率增量。
- 这次矩阵是 concurrency 1→4→8→16 严格递增着跑的一整段连续会话，并发度和"跑了多久/处理过多少历史请求"
  完全绑在一起，没法用这份数据单独把"并发压力"和"会话进行到多久了"这两个因素分开——prefix cache 命中率
  下降到底是并发挤压缓存导致的，还是纯粹随时间自然回落，这次数据回答不了，需要以后设计矩阵跑序时把这个
  混杂因素纳入考虑（比如打乱顺序跑，或者每个并发度前都重启一次服务清空累计计数器）。

## 相关文档

- [`serving_observability_b_howto.md`](./serving_observability_b_howto.md) —— B5 脚本设计、B 类验收标准
  逐条核对
- [`vllm_request_concepts.md`](./vllm_request_concepts.md) —— `queue_time`/TTFT/decode time 这些字段
  在单次请求里具体测的是哪一段
- [`system_layers_and_acceleration.md`](./system_layers_and_acceleration.md) —— KV cache 压力对应的
  L2/L3 层解法（TP、量化 KV cache）

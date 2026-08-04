# 目标二三层 profiling：实现与真实数据验证结果

延续目标一的方法论：先设计（见 [`goal2_design.md`](./goal2_design.md)），再用真实数据验证，如实报告结果——包括限制和缺口。

## 复现命令

Token 层 / episode 层（纯离线分析，读已有的任意 run）：

```python
from pathlib import Path
from inspect_trace.analysis import token_layer, episode_layer

# 相对 efficient-harness/ 仓库根目录；runs/ 是 inspect_trace/ 的同级目录，不在它内部。
trace_dir = Path("runs/goal1_bfcl_multi_turn_base_full/.inspect_trace")
token_layer.summarize_run(trace_dir)     # TokenLayerRunSummary
episode_layer.summarize_run(trace_dir)   # EpisodeLayerRunSummary
```

Model invocation 层（需要本地 vLLM 正在跑，且用 `MAX_CONNECTIONS=1`）：

```bash
cd /home/liuyingen/code/efficient-harness/local-model-server && ./scripts/serve.sh
cd /home/liuyingen/code/efficient-harness/inspect_trace
MODEL="openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" MODEL_ARGS="emulate_tools=true" \
VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
CATEGORIES="multi_turn_base" LIMIT=8 MAX_CONNECTIONS=1 \
OUTPUT_DIR="/home/liuyingen/code/efficient-harness/runs/goal2_vllm_metrics_validation" \
./scripts/run_bfcl_benchmark.sh
```

## Token 层：真实结果（复用目标一 200 样本全量 run）

| 指标 | 数值 |
|---|---|
| episodes | 200 |
| 真实计费 input tokens 总计 | 7,061,285 |
| 真实计费 output tokens 总计 | 86,639 |
| 复用（重复）对话消息 tokens | 233,306 |
| 复用（重复）工具 schema tokens | 8,624,262 |
| 重试浪费的 output tokens | 0 |

跟之前 benchmark 输出的 `I: 7,061,285 O: 86,639` 完全一致——离线聚合代码跟 inspect_ai 自己统计的真实计费数字对得上，不是凭空算出来的。工具 schema 的重复量（862 万 token）比对话历史重复量（23 万）大 37 倍，再次印证目标一"工具 schema 是重复 prefill 的大头"这个结论——现在有了 token 层的正式统计口径，不再只是个案观察。重试浪费为 0，跟目标一"从未观测到真实重试"的结论一致。

## Episode 层：真实结果（同一份 200 样本数据）

| 指标 | 数值 |
|---|---|
| success_rate | 4.5%（9/200，跟 benchmark 自己报告的 accuracy 一致） |
| 平均 end-to-end latency | 21.67s |
| 平均 LLM 调用次数 | 7.03 |
| 平均 tool 调用次数 | 4.285（跟目标一"857 次 tool call / 200 样本"一致） |
| 总重试次数 | 0 |
| 观测到并行的 episode 数 | 0 / 200 |
| 总成本（本地 vLLM） | $0.00 |

`critical_path_latency_seconds` 逐条核对过，跟 `end_to_end_latency_seconds` **完全相等**（不是约等于）——这是设计上的必然结果（见 `episode_layer.py` 模块 docstring：单条 episode 的"关键路径耗时"定义上就是"这条 episode 完成所用的总时间"，不存在另一套算法能算出不同的数）。真正需要另外算的是 `total_busy_seconds`（去重后的忙碌时间）——真实数据上略小于 `end_to_end`（比如 `multi_turn_base_0`：16.288s vs 16.283s，差的 5ms 就是真实的 model-waiting-for-tool/tool-waiting-for-model 空档），`concurrency_savings_seconds` 全部为 0（跟"从未观测到真并行"一致，用另一个专门写的并行 mock 场景测试过这个字段在真的有重叠时能正确算出正数）。

## Model invocation 层：真实结果（8 样本验证 run，49 次真实 model 调用）

| 指标 | 结果 |
|---|---|
| `attribution_confidence == "exact"` 占比 | **49/49（100%）** |
| TTFT | 0.57s ～ 1.13s，随输入 token 数增长而增长（2997 输入 token → 0.571s；5873 输入 token → 1.065s） |
| ITL（平均每输出 token 耗时） | 稳定在 ~33.3ms/token |
| queue_depth_running / queue_depth_waiting | 全部 0.0 |
| preemptions_delta | 全部 0.0 |

**两个交叉验证，都对上了**：
1. ITL ~33.3ms/token，对照目标一阶段用"真实 working_time + 真实 token 数"做线性回归估出的 decode 吞吐（~29 tokens/s ≈ 34.5ms/token）——两个完全独立的方法（vLLM 自己的 histogram vs. 我们自己的回归）算出几乎一样的数，互相印证都是对的。
2. TTFT 随输入 token 数增长（2997→0.571s，5873→1.065s，大致线性）——符合"TTFT 主要由 prefill 决定，prefill 时间正比于输入长度"这个预期，不是随机噪声。

**`queue_depth`/`preemptions` 全部是 0，如实报告**：这是 `MAX_CONNECTIONS=1` 串行执行的必然结果——没有第二个请求在排队，也就不可能有排队深度或抢占。这些字段的设计目的是给未来真正并发的场景用的，当前数据只能证明"字段在无排队场景下正确显示 0"，不能证明"字段在真排队场景下工作正确"（需要一次故意开高并发的验证，但会撞上目标一阶段已经发现的本地 vLLM 并发崩溃问题，这次没有为了验证这一个字段去冒这个风险）。

### 一个已发现的限制：`gpu_cache_usage_perc_at_end` 采样时机不对

49 条记录里，这个字段**全部是 0.0**，没有一条例外。原因：当前设计只在 `on_before_model_generate`（调用前）和 `on_sample_event` 的 `ModelEvent` 分支（调用完全结束后）各拉一次 `/metrics`——但因为是串行执行，一次调用结束的瞬间它占用的 KV cache 就已经被释放了，"调用后"这个采样点其实已经错过了真正的峰值占用。想拿到真实的峰值 KV cache 占用，需要在调用**进行中**高频轮询（就像最初做 spike 验证时那样单独起一个轮询协程），而不是只在前后各拉一次——这是下一步如果要认真做这个字段应该改的地方，当前版本如实标注为已知限制，不假装这个字段现在是准的。

## 结论

- Token 层、episode 层：全部用真实数据验证通过，数字跟 inspect_ai 自己报告的、目标一阶段已经建立的结论完全吻合，没有发现新问题。
- Model invocation 层：**比 `inspect_ai_roadmap.md` 最初预估的更可行**——TTFT/ITL/e2e-latency 三个 histogram 在我们锁定的 vLLM 版本上就有，不需要等更新版本；100% 精确归因（在 `MAX_CONNECTIONS=1` 前提下）；两处交叉验证都通过。唯一的真实限制是 `gpu_cache_usage_perc_at_end` 的采样时机，已如实记录，不影响 TTFT/ITL/e2e-latency 三个核心字段的可信度。
- `queueing time`（数值化的排队耗时）、`prefill time`/`decode time` 的严格拆分、`batch size`、`peak GPU memory`（绝对值）——`goal2_design.md`里已经说明这些字段在当前 vLLM 版本上确实拿不到，这次真实验证没有推翻这个判断。

详见代码：`inspect_trace/vllm_metrics.py`（实时采集）、`inspect_trace/analysis/{token_layer,episode_layer}.py`（离线分析）、`inspect_trace/tests/test_vllm_metrics.py` + `test_analysis_layers.py`（测试）。

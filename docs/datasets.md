# 本地缓存的数据集

两个知名 benchmark，都已经本地缓存过，不需要联网就能重跑。

## BFCL（Berkeley Function-Calling Leaderboard）

- 缓存位置：`~/.cache/inspect_evals/BFCL/`（26 个 category 的 JSON 文件，共 13MB）。
- 加载方式：`inspect_evals.bfcl.bfcl`，见 `inspect_trace/scripts/run_bfcl_benchmark.sh`。
- 选它的原因：这是我们目标一真实验证（`goal1_real_benchmark_findings.md`）用的主 benchmark——多轮、真实工具调用密集，最能体现 agent 执行轨迹的复杂度，`multi_turn_base`/`multi_turn_miss_func`/`multi_turn_miss_param` 等类别都已经在缓存里，`live_parallel`/`parallel_multiple` 这两个并行工具调用类别也在，是后续补 Q6 验证的现成数据源。
- 重新触发缓存（如果换机器）：跑一次 `eal run`/`inspect eval inspect_evals/bfcl --limit 1` 就会自动下载，不需要手动操作。

### `multi_turn_base` vs `live_parallel`：两种完全不同的测试范式

`goal1_r3_r4_dashboard.html`（[`goal1_r3_r4_real_benchmark_findings.md`](./goal1_r3_r4_real_benchmark_findings.md) 的配套可视化）里同时跑了这两个 category，不是随便挑的，是刻意留的一组结构性对照，值得在这里说清楚区别：

| | `multi_turn_base` | `live_parallel` |
|---|---|---|
| BFCL 版本 | v3（`multi_turn`） | v2（`live`） |
| solver（`inspect_evals/bfcl/solve/`） | `multi_turn_solver.py` | `single_turn_solver.py` |
| 是否真的执行 tool | **是**——调用真实 `execute_tools()`，工具是一个内存模拟的假后端（假文件系统/假股票交易/假消息 app），会产生真实 `ToolEvent` | **否**——从不调用 `execute_tools()` |
| 轮数 | 多轮，数据集里预先写好一系列脚本化的后续用户指令（平均每条 sample 6-9 次 model call） | 单轮，一次 `generate()` 就结束 |
| 怎么打分 | 比较"模型操作后的后端最终状态"跟"期望状态"是否一致（打分结果里的 `State mismatch` 解释） | 直接对模型输出的 `tool_calls` 结构做 AST 匹配，看是不是同一个函数、同样的参数 |
| 名字里的"parallel"指什么 | 不适用 | 指"这条数据期望模型在**一次**回复里**同时**提议多个函数调用"（AST 匹配维度的并行意图），不是"tool 真的并发执行" |

**为什么两个都要跑**：最初选 `live_parallel`是想直接验证目标一需求三"并行 tool call"这件事，结果真实跑出来发现它**结构性地不可能产生任何 `ToolEvent`**——即便模型确实一次提议了两个 `tool_calls`（真实见过 `get_current_weather` 被同时提议查北京和上海），也连不上 `execution_topology`/`action_parsing` 这两个检测器。如果只跑 `multi_turn_base`、不留着 `live_parallel` 做对比，没法判断"某类样本全部 `total_stages=0`"到底是检测器坏了还是这类数据本来就没有 tool 执行。留着 `live_parallel` 当**阴性对照组**，才能证明检测器没问题，是这一类 BFCL 数据的实现方式导致它天然没有 tool 执行可看——详见 `goal1_r3_r4_real_benchmark_findings.md` 的"需求三·发现一"。

## GSM8K

- 缓存位置：`~/.cache/huggingface/datasets/openai___gsm8k/` 和 `~/.cache/huggingface/hub/datasets--openai--gsm8k/`（HuggingFace `datasets` 库的标准缓存路径）。
- 加载方式：`inspect_evals.gsm8k.gsm8k`。
- 选它的原因：跟 BFCL 形成对照——单轮、无工具调用、纯文本推理的数学题。作为"简单场景"基线很有用：可以验证 `inspect_trace` 在完全没有工具调用的轨迹上（`tools_total=0`、`reused_messages` 全程为 0，因为只有一步）是否还能正确工作、不会因为"没有工具"这种边界情况而报错，是 BFCL 复杂轨迹之外的一个健全性对照组。

## 关于 GAIA（暂未缓存）

之前在 `framework-selection.md`/`inspect_ai_roadmap.md` 里提到的 GAIA 数据集在 HuggingFace 上标记为 `gated: auto`——需要先在 HF 网站接受使用协议、再配置 `HF_TOKEN` 才能下载，这台环境目前没有配置 HF 账号认证，所以没有强行缓存。如果之后需要跑 GAIA，需要先手动完成 HF 授权：

```bash
huggingface-cli login   # 或者 export HF_TOKEN=...
```

授权完成后，`inspect eval inspect_evals/gaia --limit 1` 会自动触发下载缓存，跟 BFCL/GSM8K 的流程一样。GAIA 本身还需要 Docker 沙箱（`compose.yaml`），是三个候选里 setup 最重的，其余两个都不需要 Docker。

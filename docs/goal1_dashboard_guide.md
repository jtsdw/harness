# 目标一实测面板：需求回顾 + 构建理念 + 使用方法

配套产物：[`goal1_r3_r4_dashboard.html`](./goal1_r3_r4_dashboard.html)（本地打开即可，不需要联网/起服务）。本文档记录它对应哪四条需求、为什么长成现在这样（迭代过程里每一次改版都是回应一个具体问题）、怎么用、数据从哪来、怎么重新生成。

## 原始四条需求（回顾）

完整原文见 [`efficient-harness.md`](./efficient-harness.md) 目标一部分，这里摘要成对照面板设计时反复引用的版本：

- **需求一：完整的 token 级记录**——每次模型调用都要有真实（计费）prefill/decode token 数，加上按 pipeline 阶段（system template / tool schema / reasoning / tool call / final response）的归因，覆盖 model 层面（原始输入 vs 输出拆分）和 agent 层面（ReAct 阶段拆分）。
- **需求二：每一步的新增 vs 复用上下文**——判定每一步 context 里哪些是新增、哪些是历史内容原样重发，system template 这类静态内容要能单独统计。
- **需求三：执行拓扑**——重建一次 episode 是线性执行，还是存在并行 tool call、等待（model waiting for tool / tool waiting for model）、重试。
- **需求四：action parsing 与 observation 回填追踪**——模型 tool-call 的解析/校验失败要留痕，工具结果写回 context（observation 回填）的过程要能追溯。

面板的四个"01-04"分区，跟这四条是严格一一对应的，不是自己发挥出来的分类。

面板后来又加了第五个分区"目标二：三层成本归因"，对应 [`goal2_design.md`](./goal2_design.md) 定义的三层——token 层（per-episode token/成本聚合）、episode 层（端到端延迟/并发节省/成本）、model invocation 层（单次真实 vLLM 调用的 TTFT/ITL/队列深度）。这部分不是目标一需求的一部分，单独成节，见下面"新增：目标二三层数据"一节。

## 构建理念

面板不是一次性设计出来的，是跟着一连串具体追问一步步改出来的——每条理念背后都有一个真实被指出的问题，不是凭空定的规范。

### 1. 每个面板都先引用需求原文，再放数据

最早的版本只有图表，没有对照需求文字。用户要求"严格按照我们的需求文档来"之后，改成每个 01-04 分区顶部先放一段浅底色的原文引用（`req-quote`），把需求文档的具体措辞（比如"model 层面（原始的输入 vs 输出拆分）"）直接印在数据上方——这样看图表时能明确知道"这一块对应需求文档的哪一句话"，而不是自己猜。

### 2. 只用真实数据，拒绝合成/演示样例

面板里没有一个数字是编出来演示效果用的。所有内容都来自真实跑出来的 `.eval` 日志和 `inspect_trace` 落盘的 JSONL——包括真实的模型输出文本、真实的解析失败信息、真实的 `bfcl_scorer` 打分结果。看到"0 次重试""system_template 全部是 0"这类空结果，也如实展示，不用别的数据顶替、不隐藏。

### 3. 两级结构：数据集总览 → 样本详情

最初版本只深挖了一个样本，用户问"为什么只有一个样本"。改成两级结构：进面板先看到数据集总览（`multi_turn_base`/`live_parallel` 各一张表，含真实 `score`/steps/错误数/拓扑标记），点一行样本才进入需求一至四的完整详情——这样既能看全局分布，又能追到单条样本的具体细节，不用二选一。

### 4. 需求一"具体上下文"：不能只有聚合数字

只有 token 计数的柱状图无法回答"agent 到底是怎么执行的"。加了一段可展开的真实转录（每一步的每条消息，标了 new/reused，附带模型这一步真实产生的 reasoning/tool call/文本），点开就是原始对话，不是脱水后的统计量。

### 5. 需求三：一条统一时间线，不要拆开看

最早版本把"并行 tool call"（chip 列表）和"等待时间"（单独的统计卡片）分开展示。用户明确要求"tool call、等待、model generate 应该在同一条时间线上",于是重做成一条按真实时间比例绘制的时间线：tool call 和 model generate 都是时间线上的色块，色块之间的空隙**就是**等待时间，不需要另外配数字解释；如果真的出现两个 tool call 时间窗口重叠，会自动分泳道并排显示在同一条时间线里——并行现象直接从时间线的视觉重叠读出来，不是另一套机制。

### 6. 如实报告"没有发生"的现象，并追问为什么

`live_parallel` 类别 6 个样本全部 `total_stages=0`，`multi_turn_base` 从未观测到真并行、从未观测到重试——面板和配套的 [`goal1_r3_r4_real_benchmark_findings.md`](./goal1_r3_r4_real_benchmark_findings.md) 都没有回避这些空结果，而是深挖了具体原因（`live_parallel` 走的是不执行 tool 的 single-turn solver；工具没开 `parallel=True`），把"检测器没问题、是被测对象结构性没有这个现象"讲清楚，而不是让人误以为检测器坏了。

### 7. 目标二不是另起一个面板，而是接到目标一已有的数据结构上

目标二实现完之后（`vllm_metrics.py` + `analysis/{token_layer,episode_layer}.py`）一开始完全没有可视化——数据只存在于 pytest 断言和一份文字版验证报告里。用户问"我该怎么直观感受到这些数据"，倒逼出这条理念：token/episode 层的数字不是孤立指标，是已有样本的**延伸属性**，所以没有新开一个页面，而是（a）把 episode 层结果（`concurrency_savings_seconds`/`cost_usd`）直接并入需求三时间线的统计行——看时间线的同时就能看到这条 episode 省了多少并发时间、花了多少钱；（b）在数据集总览页新增"目标二"分区，用两张按数据集分列的卡片展示 token/episode 层的聚合统计，跟需求一至四的分区平级并列；（c）model invocation 层（TTFT vs 计费 input token 的散点图）单独成一块——它的数据来自另一个专门跑的验证 run（8 样本，见下），不能假装成跟主数据集是同一批，所以面板上明确标注了数据来源和样本量，不混着展示制造"全量都测过"的错觉。

## 使用方法

1. **打开**：本地文件路径 `docs/goal1_r3_r4_dashboard.html`，双击或者浏览器地址栏输入 `file:///home/liuyingen/code/efficient-harness/docs/goal1_r3_r4_dashboard.html`。完全自包含（字体、数据全部内嵌），不需要联网、不需要起服务。

2. **数据集总览页**（打开后默认看到的）：
   - 顶部四个卡片：需求一至四的整体验证结论速览（是否通过、有什么限制）。
   - 两张表：`multi_turn_base`（真实执行 tool 的样本，看得到执行拓扑/action parsing）和 `live_parallel`（不执行 tool 的对照组，两者的区别见 [`datasets.md`](./datasets.md)"两种测试范式"一节）。每行是真实评测结果（`score` ✓/✗）+ 关键统计。

3. **点一行样本进入详情页**：
   - 需求一：model 层面真实计费条形图 → pipeline 阶段归因堆叠图（输入侧/agent 层面输出侧）→ 可展开的逐步真实转录。
   - 需求二：new/reused 消息与工具 schema 复用的逐步表格。
   - 需求三：统一执行时间线（往上翻看第 5 条理念）。
   - 需求四：解析/校验失败统计与错误表格 + "observation 回填追溯"三张真实链路卡片（正常调用 / 报错但 id 仍真实 / JSON 错到 id 都恢复不出来）。
   - `live_parallel` 样本详情页更简单：一段结构性说明（为什么没有 tool 执行数据）+ 模型真实提议的 tool_calls。
   - 详情页顶部有"← 返回数据集总览"。

4. **目标二分区**（数据集总览页顶部导航"目标二：三层成本归因"）：
   - Token 层 / Episode 层卡片：`multi_turn_base`（200 样本全量）和 `live_parallel`（15 样本全量）各一张，聚合统计（平均 token 数、平均成本、平均并发节省秒数等）都来自这两个数据集本身，不是另外跑的。
   - Model invocation 层：散点图（横轴计费 input token 数，纵轴 TTFT），数据来自单独的 `goal2_vllm_metrics_validation` 验证 run（8 样本，不是 200/15 全量），面板上有明确的样本量和数据来源标注，散点按 `attribution_confidence`（`exact`/`ambiguous`）区分颜色，hover 能看到每次调用的精确数值。
   - 点进任意 `multi_turn_base` 样本详情页，需求三时间线下方的统计行会多出两条：并发节省秒数、本地 vLLM 成本（当前本地模型定价为 $0，所以这一项目前恒为 $0——见 [`analysis/pricing.py`](../inspect_trace/src/inspect_trace/analysis/pricing.py) 里只收录了本地模型这一条定价，没有编造别的模型价格）。

## 数据来源与重新生成

数据来自三个真实 run（命令见 [`goal1_r3_r4_real_benchmark_findings.md`](./goal1_r3_r4_real_benchmark_findings.md)"背景"一节和 [`goal2_real_validation_findings.md`](./goal2_real_validation_findings.md)"复现命令"一节）：

- `runs/goal1_bfcl_multi_turn_base_full/`（`.eval` 日志 + `inspect_trace` JSONL，200 样本全量，需求一至四 + 目标二 token/episode 层的数据源；对应 `MULTI_TURN_RUN_DIR` 环境变量）
- `runs/goal1_bfcl_live_parallel_full/`（同上，15 样本全量；对应 `LIVE_PARALLEL_RUN_DIR`）
- `runs/goal2_vllm_metrics_validation/`（8 样本，唯一带 `vllm_metrics` 记录的 run，只喂目标二的 model invocation 层；对应 `GOAL2_VLLM_METRICS_RUN_DIR`）

重新生成面板（比如跑了新的 run、想更新数据）：

```bash
cd /home/liuyingen/code/efficient-harness/inspect_trace
uv run python scripts/build_r3_r4_dashboard.py
```

脚本会读上面三个 run 目录（可以用 `MULTI_TURN_RUN_DIR`/`LIVE_PARALLEL_RUN_DIR`/`GOAL2_VLLM_METRICS_RUN_DIR` 环境变量指向别的 run），提取真实转录/token 归因/执行拓扑/action parsing/时间线/评测分数/目标二三层聚合，套用 `_dashboard_template.html`（同目录下的页面骨架）和系统自带的 Latin Modern Sans 字体，输出到 `OUTPUT_PATH`（默认就是 `docs/goal1_r3_r4_dashboard.html`）。整个脚本只读已有数据，不会自己触发新的 eval 运行。

脚本和模板都在 `/home/liuyingen/code/efficient-harness/inspect_trace/scripts/`：`build_r3_r4_dashboard.py`（数据提取 + 组装）+ `_dashboard_template.html`（HTML/CSS/JS 骨架，把 `__LM_REGULAR__`/`__LM_BOLD__`/`__DATA_JSON__` 三个占位符替换掉就是最终页面）。

## 相关文档

- 需求原文：[`efficient-harness.md`](./efficient-harness.md)
- 完整真实数据发现（这份面板呈现的所有结论的详细文字版）：[`goal1_r3_r4_real_benchmark_findings.md`](./goal1_r3_r4_real_benchmark_findings.md)
- 两个 BFCL category 的区别：[`datasets.md`](./datasets.md)
- 目标一整体现状：[`inspect_ai_roadmap.md`](./inspect_ai_roadmap.md)
- 目标二三层设计：[`goal2_design.md`](./goal2_design.md)
- 目标二真实数据验证（面板"目标二"分区呈现的所有结论的详细文字版）：[`goal2_real_validation_findings.md`](./goal2_real_validation_findings.md)

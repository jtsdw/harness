# Efficient Harness 项目文档索引

这个文件夹装的是"用 inspect_ai 实现 `efficient-harness.md` 四个目标"这个项目积累下来的全部文档。按**阅读顺序**组织，不是字母序——跟着顺序走一遍，就是这个项目从"选型"到"现状"的完整故事线。

这份文档索引现在是 `efficient-harness/` 仓库（`/home/liuyingen/code/efficient-harness/`）的一部分，跟代码（`inspect_trace/`/`local-model-server/`）同仓库、不同目录。2026-08-04 之前它曾短暂放在 `/home/liuyingen/code/doc/efficient-harness/`（一个跟代码分离的独立仓库），那个位置现在已经并入这里，见下方"我们写的代码/脚本在哪"一节。

## 推荐阅读顺序

| # | 文档 | 一句话 | 读它之前需要什么背景 |
|---|---|---|---|
| 1 | [`efficient-harness.md`](./efficient-harness.md) | 项目的原始需求文档：四个目标是什么、为什么要做、要防的三个风险 | 无，从这里开始 |
| 2 | [`framework-selection.md`](./framework-selection.md) | 为什么最终选 inspect_ai 做底座（对比过 AgentCompass/EfficientAgentLab/pydantic-ai/smolagents），含两轮独立分析的分歧记录和第三轮补充（pydantic-ai 桥接发现） | 读完 1 |
| 3 | [`inspect_ai_quickstart.md`](./inspect_ai_quickstart.md) | 手把手上手：不需要 API Key，跑通第一个 eval，看懂 `.eval` 产出，理解 `inspect_trace` 怎么挂上去，含 uv 环境管理和 conda 对比 | 读完 2，准备开始动手 |
| 4 | [`source_code_reading_guide.md`](./source_code_reading_guide.md) | 源码阅读指南：从 `Sample.input` 到模型输出的完整链路，每一段代码对照真实数据 | 读完 3，跑过至少一次 eval |
| 5 | [`eval_log_format.md`](./eval_log_format.md) | `.eval` 文件逐字段详解，对照真实样本数据 | 可以跟 4 一起读，互相引用 |
| 6 | [`goal1_real_benchmark_findings.md`](./goal1_real_benchmark_findings.md) | 目标一在真实 benchmark（BFCL）上的验证：发现的 bug（工具 schema 未追踪）、修复、以及一个消息级去重 vs 真实 prefix-cache 边界的科研发现 | 读完 4-5 |
| 6b | [`goal1_r3_r4_real_benchmark_findings.md`](./goal1_r3_r4_real_benchmark_findings.md) | 目标一重新梳理为四条需求后，需求三（执行拓扑）/需求四（action parsing）在真实 BFCL `live_parallel` 上的验证结果 | 读完 6 |
| 6c | [`goal1_dashboard_guide.md`](./goal1_dashboard_guide.md) | 配套可视化面板 `goal1_r3_r4_dashboard.html` 的使用指南：四条需求怎么对应到面板分区、构建理念（为什么长成这样）、怎么重新生成 | 读完 6b，想直观看数据就看这篇 |
| 7 | [`local_model_deployment.md`](./local_model_deployment.md) | 本地 vLLM 部署全过程（含 GPU 驱动/CUDA 版本坑、三个损坏依赖的排查修复） | 独立可读，跟 6 没有强依赖 |
| 8 | [`datasets.md`](./datasets.md) | 本地缓存了哪些数据集、为什么选它们 | 读完 7 更有上下文 |
| 9 | [`model_dataset_comparison_findings.md`](./model_dataset_comparison_findings.md) | 真实模型 × 真实数据集的对照实验结果分析，**含目标一/目标二契合度对照表**（逐指标核对现有产出能不能覆盖目标二） | 读完 6-8 |
| 10 | [`inspect_ai_docs_examples_audit.md`](./inspect_ai_docs_examples_audit.md) | inspect_ai 自带 docs/examples 里哪些对我们四个目标有复用价值的审计结果（含几处对之前判断的修正） | 读完 9，准备规划后续目标 |
| 11 | [`inspect_ai_roadmap.md`](./inspect_ai_roadmap.md) | 现状总览：四个目标各自的完成度、可行性、工作量估计，是这个项目当前的"我们在哪、下一步去哪"结论 | 放在最后读，前面 10 篇的结论都汇总在这里 |
| 12 | [`goal2_design.md`](./goal2_design.md) | 目标二（三层 profiling 成本归因）实现设计：真实 vLLM `/metrics` spike 结果、token/episode/model invocation 三层各自怎么做、明确不做的部分 | 读完 11 |
| 12b | [`goal2_real_validation_findings.md`](./goal2_real_validation_findings.md) | 目标二真实数据验证结果：token/episode 层跟 benchmark 报告数字完全吻合，model invocation 层 49/49 次调用 100% 精确归因，两处独立方法交叉验证通过 | 读完 12 |
| 13 | [`deployment_migration_guide.md`](./deployment_migration_guide.md) | 迁移到新服务器（H100 80GB）+ 团队协作指南：版本控制补救、CUDA 13 下版本锁定怎么变、原生 tool-calling 能否替代 emulate_tools、MIG 分区 vs 真并发两种多人共用方案 | 需要迁移/加人的时候看 |
| 14 | [`tau2_bench_integration_findings.md`](./tau2_bench_integration_findings.md) | 接入外部 benchmark 框架 tau2-bench（双控 agent 评测）的真实全链路记录：同步/异步桥接方案、三个真实 bug 的排查修复、Hooks 触发验证、原生 CLI vs 我们适配器的逐任务结果对比，以及"同一个 bench 能不能跑出同样效果"的实证结论 | 独立可读，想知道能不能接别的 benchmark 框架时看 |

读完这 14 篇，应该能达到"看得懂现在的代码、说得清楚下一步该做什么"的程度。如果只有十分钟，只读 1、2、11——分别是"要做什么"、"为什么用这个底座"、"现在做到哪了"。

## 按用途查找

- **我想跑一遍试试** → 3（quickstart）
- **我想读源码** → 4（source_code_reading_guide）+ 5（eval_log_format）
- **我想搭本地模型环境** → 7（local_model_deployment），或者直接看 [`environment_checklist.md`](./environment_checklist.md) 的速查清单
- **我想知道现在还缺什么、接下来该干什么** → 11（roadmap）
- **我想知道某个 inspect_ai 官方机制能不能直接抄** → 10（audit）
- **我想读 inspect_ai 自己的官方文档** → 见下面 [`inspect_ai_essential_docs.md`](./inspect_ai_essential_docs.md)
- **忘了某个词是什么意思** → [`glossary.md`](./glossary.md)
- **要在新机器上从零搭环境** → [`environment_checklist.md`](./environment_checklist.md)

## 我们写的代码/脚本在哪（不在这个文件夹里，列出来方便对照）

这份文档所在的 `docs/` 和下面提到的代码，现在都在同一个仓库 `/home/liuyingen/code/efficient-harness/` 下（2026-08-04 从原来分散在 `inspect_ai` 克隆内部/独立目录的三处东西整合而来，见 [`deployment_migration_guide.md`](./deployment_migration_guide.md)）：

- `inspect_trace` 包（目标一二实现）：`../inspect_trace/`，自带 `README.md`
- BFCL/GSM8K 复现脚本：`../inspect_trace/scripts/run_bfcl_benchmark.sh`、`run_gsm8k_benchmark.sh`
- `goal1_r3_r4_dashboard.html` 生成脚本：`../inspect_trace/scripts/build_r3_r4_dashboard.py` + `_dashboard_template.html`，用法见 [`goal1_dashboard_guide.md`](./goal1_dashboard_guide.md)
- 目标二三层 profiling 实现：`inspect_trace/vllm_metrics.py`（model invocation 层实时采集）+ `inspect_trace/analysis/{token_layer,episode_layer,pricing}.py`（token/episode 层离线分析），自带测试 `tests/test_vllm_metrics.py`/`test_analysis_layers.py`
- 本地模型服务：`../local-model-server/`，自带 `README.md` + `scripts/{setup,serve,stop}.sh`
- 实验原始数据（gitignored，不会被清理）：`../runs/`
- 上游 inspect_ai 本体：**不在这个仓库里**，只读参考克隆仍在 `/home/liuyingen/code/inspect_ai/`（读框架自身源码/走官方教程用，不是我们项目的一部分）

## 建议后续补充的文档

- [x] **术语表**——[`glossary.md`](./glossary.md)，已完成。
- [x] **环境/基础设施清单**——[`environment_checklist.md`](./environment_checklist.md)，已完成。
- [x] **目标二实现设计文档**——[`goal2_design.md`](./goal2_design.md) + [`goal2_real_validation_findings.md`](./goal2_real_validation_findings.md)，已完成并用真实数据验证。
- [ ] **踩坑合集（Troubleshooting）**——目前真实踩过的坑（`anyio.get_current_task()` 不稳定、`pyairports` 空壳包、CUDA 驱动版本不匹配、nvidia wheel 缓存损坏）分散在 `goal1_real_benchmark_findings.md` 和 `local_model_deployment.md` 里。现在坑还不算多，值得等自然积累到一定量、且分散在各篇里确实开始难查的时候再抽取，不急。
- [ ] **目标三（标准化加速干预接口）实现**——`inspect_ai_roadmap.md` 判断可行性高、工作量相对最小，是目标二完成之后下一个建议推进的目标，还没开始。

## inspect_ai 官方文档精选

见 [`inspect_ai_essential_docs.md`](./inspect_ai_essential_docs.md)——这是从 `inspect_ai/docs/` 93 篇文档里挑出来的、面向"快速建立对 inspect_ai 整体的理解"这个目的的精选阅读清单，跟 [`inspect_ai_docs_examples_audit.md`](./inspect_ai_docs_examples_audit.md) 不是一回事：审计文档回答的是"哪些机制对我们四个目标有复用价值"（很多基础文档因为跟目标不直接相关而没有被覆盖），这份精选回答的是"要理解 inspect_ai 这个框架本身，应该按什么顺序读"。

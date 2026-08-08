# Efficient Harness 项目文档索引

这个文件夹装的是"用 inspect_ai 实现 `efficient-harness.md` 五个目标"这个项目积累下来的全部文档（2026-08-05 从四个目标扩展为五个，新增目标五"分析成熟加速方法的 insight 与 method"）。按**阅读顺序**组织，不是字母序——跟着顺序走一遍，就是这个项目从"选型"到"现状"的完整故事线。

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
| 12b | [`goal2_real_validation_findings.md`](./goal2_real_validation_findings.md) | 目标二真实数据验证结果：token/episode 层跟 benchmark 报告数字完全吻合，model invocation 层 49/49 次调用 100% 精确归因，两处独立方法交叉验证通过；2026-08-05 补充了一个后续发现的真实 bug（`vllm_metrics` 采集器在另一个场景下完全不产出记录），如实标注为待查、不再假设这套机制现在整体可靠 | 读完 12 |
| 13 | [`deployment_migration_guide.md`](./deployment_migration_guide.md) | 迁移到新服务器（H100 80GB）+ 团队协作指南：版本控制补救、CUDA 13 下版本锁定怎么变、原生 tool-calling 能否替代 emulate_tools、MIG 分区 vs 真并发两种多人共用方案 | 需要迁移/加人的时候看 |
| 14 | [`benchmark_integration_playbook.md`](./benchmark_integration_playbook.md) | 接入新 benchmark 的操作手册：三层难度分类、前置条件检查、核心架构原则、实现步骤划分、验证顺序、会反复遇到的坑分类总结 | 下次要接一个新 benchmark 时，先看这篇 |
| 14b | [`tau2_bench_integration_findings.md`](./tau2_bench_integration_findings.md) | 上面那篇手册的真实案例来源：接入 tau2-bench（双控 agent 评测）的完整全链路记录——同步/异步桥接方案、三个真实 bug 的排查修复、Hooks 触发验证、原生 CLI vs 我们适配器（修复前后两个版本）的逐任务结果对比 | 读完 14，想看具体案例细节时看 |
| 15 | [`remote_compute_workflow.md`](./remote_compute_workflow.md) | 本地 agent 节点 + 远程 NSCC 计算节点的日常工作流：代码同步（git 为主）、结果拉取、PBS 两种使用范式（交互式常驻 vLLM / 批处理长任务）、计算节点友好格式检查清单、待现场验证清单 | 要往计算节点部署/跑实验时看 |
| 16 | [`team_collaboration.md`](./team_collaboration.md) | 两人协作方案：git 分支+PR 规范、共享 NSCC 账号下的计算资源协调、项目管理约定 | 读完 15，两人协作时看 |
| 17 | [`acceleration_methods_survey.md`](./acceleration_methods_survey.md) | 目标五的持续积累清单：已完成分析 SPORK（论文说的问题为什么在我们数据里没出现）+ ToolSpec（用真实 token 层数据算出论文没给的 tool-call token 占比 17.5%，换算出更保守的整体收益估算）+ 9 篇候选论文（LLMCompiler/ReWOO/SGLang/Preble/StreamingLLM/H2O/LLMLingua/FrugalGPT/Speculative Actions）分类和优先级建议 | 读完 11（roadmap），想知道外部方法能不能落进我们的干预接口时看 |
| 17b | [`toolspec_integration_findings.md`](./toolspec_integration_findings.md) | ToolSpec 原生复现（五种方法真实速度对比，发现"并非严格 lossless"的真实特性）+ 迁移进 harness（自定义 `ModelAPI` provider，因为它是原始 HF transformers 生成循环不是 HTTP 服务）+ 二次复现，逐 token 精确对齐原生仓库输出 | 读完 17，想看具体接入案例时看 |
| 17c | [`toolspec_vllm_speculative_comparison.md`](./toolspec_vllm_speculative_comparison.md) | ToolSpec vs vLLM 自带 ngram 投机解码的真实对比：ToolSpec 领域特定方法比通用方法快约 60%、偏离率更低；顺带发现一个 `inspect_trace` 的 `vllm_metrics` 采集器在某些场景下完全不产出记录的真实 bug，跟 `goal2_real_validation_findings.md` 的既有结论有未解决的冲突 | 读完 17b，想知道跟 vLLM 自带能力比怎么样时看 |
| 17d | [`nscc_h100_speculative_decoding_plan.md`](./nscc_h100_speculative_decoding_plan.md) | NSCC H100 节点投机解码策略：为什么本地开发机的 vLLM 版本约束在那边不成立、为什么升级能用上 EAGLE-3（真实发布数据 3.0-3.4x）、为什么要换模型才有现成草稿 checkpoint——**设计阶段产出，`nscc_model_server/` 还没在真实硬件上跑通**，如实标注 | 要在 NSCC 上做投机解码实验时看，注意"待现场验证"清单 |
| 18 | [`next_phase_requirements.md`](./next_phase_requirements.md) | 下一阶段需求文档：优先完成 vLLM 逐请求观测和标准化干预接口，用小规模受控并发做稳健性验证，再推进 Replay 和案例研究，最后完成可复现与面试交付收口 | 读完 11-17d，准备开始下一轮实现时读 |
| 19 | [`serving_observability_b_howto.md`](./serving_observability_b_howto.md) | 需求 B（Serving 观测闭环）配套脚本怎么跑、每条 B 类验收标准怎么核对；含一个新发现的真实约束（inspect_ai 默认只保留每个模型前 5 次调用的原始响应，超过就拿不到逐请求指标）和字段名待现场验证清单 | 读完 18，要实际跑需求 B 的脚本时看 |
| 20 | [`vllm_request_concepts.md`](./vllm_request_concepts.md) | vLLM 单次请求背景知识：为什么是 HTTP/URL 请求、为什么能持续运行并处理并发（continuous batching）、一次请求从进队列到返回的完整生命周期，配一个真实抓到的响应逐字段详解 | 想搞懂 TTFT/queue time/KV cache 这些指标具体对应请求的哪个阶段时看，跟 19 配合读 |

读完这份清单，应该能达到"看得懂现在的代码、说得清楚下一步该做什么"的程度。如果只有十分钟，只读 1、2、11——分别是"要做什么"、"为什么用这个底座"、"现在做到哪了"。

## 按用途查找

- **我想跑一遍试试** → 3（quickstart）
- **我想读源码** → 4（source_code_reading_guide）+ 5（eval_log_format）
- **我想搭本地模型环境** → 7（local_model_deployment），或者直接看 [`environment_checklist.md`](./environment_checklist.md) 的速查清单
- **我想知道现在还缺什么、接下来该干什么** → 11（roadmap）
- **我想知道某个 inspect_ai 官方机制能不能直接抄** → 10（audit）
- **我想读 inspect_ai 自己的官方文档** → 见下面 [`inspect_ai_essential_docs.md`](./inspect_ai_essential_docs.md)
- **忘了某个词是什么意思** → [`glossary.md`](./glossary.md)
- **要在新机器上从零搭环境** → [`environment_checklist.md`](./environment_checklist.md)
- **想接入一个新的外部 benchmark 框架** → 14（benchmark_integration_playbook）
- **要往 NSCC 计算节点部署/跑实验、或者两人协作有疑问** → 15（remote_compute_workflow）+ 16（team_collaboration）
- **看到一篇 agent 加速相关论文，想知道跟我们有没有关系** → 17（acceleration_methods_survey）
- **准备下一轮开发，想知道必须做什么、做到什么才算完成** → 18（next_phase_requirements）
- **要实际跑需求 B 的脚本、核对验收标准** → 19（serving_observability_b_howto）
- **搞不清楚 vLLM 请求/响应里某个字段是什么意思、为什么服务能一直跑着处理并发** → 20（vllm_request_concepts）

## 我们写的代码/脚本在哪（不在这个文件夹里，列出来方便对照）

这份文档所在的 `docs/` 和下面提到的代码，现在都在同一个仓库 `/home/liuyingen/code/efficient-harness/` 下（2026-08-04 从原来分散在 `inspect_ai` 克隆内部/独立目录的三处东西整合而来，见 [`deployment_migration_guide.md`](./deployment_migration_guide.md)）：

- `inspect_trace` 包（目标一二实现）：`../inspect_trace/`，自带 `README.md`
- BFCL/GSM8K 复现脚本：`../inspect_trace/scripts/run_bfcl_benchmark.sh`、`run_gsm8k_benchmark.sh`
- `goal1_r3_r4_dashboard.html` 生成脚本：`../inspect_trace/scripts/build_r3_r4_dashboard.py` + `_dashboard_template.html`，用法见 [`goal1_dashboard_guide.md`](./goal1_dashboard_guide.md)
- 目标二三层 profiling 实现：`inspect_trace/vllm_metrics.py`（model invocation 层实时采集）+ `inspect_trace/analysis/{token_layer,episode_layer,pricing}.py`（token/episode 层离线分析），自带测试 `tests/test_vllm_metrics.py`/`test_analysis_layers.py`
- 本地模型服务（这台开发机专用，vLLM 锁死在 `0.6.3.post1`）：`../local-model-server/`，自带 `README.md` + `scripts/{setup,serve,stop}.sh`
- NSCC H100 节点的模型服务（独立项目，不锁旧版本 vLLM，EAGLE-3 投机解码；**还没在真实硬件上跑通**，见 [`nscc_h100_speculative_decoding_plan.md`](./nscc_h100_speculative_decoding_plan.md)）：`../nscc_model_server/`
- tau2-bench 适配器（第三方 benchmark 接入示例，OpenAI-compatible 服务型）：`../tau2_adapter/`，自带 `scripts/{setup_tau2_bench,run_native_baseline,run_adapter}.sh`
- ToolSpec 适配器（第三方加速方法接入示例，原始 HF transformers 生成循环型，自定义 `ModelAPI`）：`../toolspec_adapter/`，自带 `scripts/{setup_toolspec,run_native_repro,run_adapter}.sh`
- ToolSpec 可视化面板生成脚本：`../inspect_trace/scripts/build_toolspec_dashboard.py`，产出 `toolspec_dashboard.html`，每次运行都现场从原始 run 数据重新计算
- 跨子项目脚本（NSCC 同步/PBS）：`../scripts/{pull_runs,nscc_interactive_gpu_session,pbs_vllm_server_job}.sh`，用法见 [`remote_compute_workflow.md`](./remote_compute_workflow.md)
- 实验原始数据：本地跑的（gitignored，不会被清理）：`../runs/`；从 NSCC 拉回来的：`../nscc_runs/`（两者故意分开，见 `nscc_runs/README.md`）
- 上游 inspect_ai 本体：**不在这个仓库里**，只读参考克隆仍在 `/home/liuyingen/code/inspect_ai/`（读框架自身源码/走官方教程用，不是我们项目的一部分）

## 建议后续补充的文档

- [x] **术语表**——[`glossary.md`](./glossary.md)，已完成。
- [x] **环境/基础设施清单**——[`environment_checklist.md`](./environment_checklist.md)，已完成。
- [x] **目标二实现设计文档**——[`goal2_design.md`](./goal2_design.md) + [`goal2_real_validation_findings.md`](./goal2_real_validation_findings.md)，已完成并用真实数据验证。
- [ ] **踩坑合集（Troubleshooting）**——目前真实踩过的坑（`anyio.get_current_task()` 不稳定、`pyairports` 空壳包、CUDA 驱动版本不匹配、nvidia wheel 缓存损坏）分散在 `goal1_real_benchmark_findings.md` 和 `local_model_deployment.md` 里。现在坑还不算多，值得等自然积累到一定量、且分散在各篇里确实开始难查的时候再抽取，不急。
- [ ] **目标三（标准化加速干预接口）实现**——`inspect_ai_roadmap.md` 判断可行性高、工作量相对最小，是目标二完成之后下一个建议推进的目标，还没开始。
- [ ] **目标五（加速方法 insight/method 分析）**——[`acceleration_methods_survey.md`](./acceleration_methods_survey.md)，2026-08-05 新增目标，目前完成了 SPORK、ToolSpec 两篇的完整分析，候选清单里还有 9 篇没细读，按文档里的优先级建议逐步推进。
- [x] **下一阶段需求收敛**——[`next_phase_requirements.md`](./next_phase_requirements.md)，按“逐请求观测（受控并发仅做稳健性验证）→ 干预接口 → Replay → 案例研究 → 可复现与面试交付收口”的顺序写成 M0-M5 验收需求。

## inspect_ai 官方文档精选

见 [`inspect_ai_essential_docs.md`](./inspect_ai_essential_docs.md)——这是从 `inspect_ai/docs/` 93 篇文档里挑出来的、面向"快速建立对 inspect_ai 整体的理解"这个目的的精选阅读清单，跟 [`inspect_ai_docs_examples_audit.md`](./inspect_ai_docs_examples_audit.md) 不是一回事：审计文档回答的是"哪些机制对我们四个目标有复用价值"（很多基础文档因为跟目标不直接相关而没有被覆盖），这份精选回答的是"要理解 inspect_ai 这个框架本身，应该按什么顺序读"。

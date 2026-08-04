# inspect_ai 官方文档精选阅读清单

`inspect_ai/docs/` 有 93 篇文档，不可能全读。这里挑出对"快速建立对 inspect_ai 整体理解"最有效的一批，按层级分组、组内按建议阅读顺序排列。每条给的介绍是从文档自己的 `llms-description` 元数据里摘的（官方自己写的一句话摘要，不是我总结的），保证准确。

跟 [`inspect_ai_docs_examples_audit.md`](./inspect_ai_docs_examples_audit.md) 的区别：那份文档回答"哪些机制对我们四个目标有复用价值"，覆盖的是偏进阶/专项的主题（tracing、checkpointing、intervention、caching、analysis、hooks 示例……），下面这份覆盖的是**框架基础概念**——两份加起来才是对 inspect_ai 相对完整的官方文档地图。

## 第一层：不读这几篇看不懂任何代码

| 文档 | 摘要 |
|---|---|
| `docs/index.qmd` | Welcome and overview of Inspect AI |
| `docs/tutorial.qmd` | Step-by-step walkthrough of Inspect, from a first eval through agents, analysis, and the broader feature set |
| `docs/tasks.qmd` | Tasks bring together datasets, solvers, and scorers to define an evaluation. Strategies for creating flexible, re-usable tasks and for configuring and overriding them at runtime |
| `docs/solvers.qmd` | Solvers encompass prompt engineering and other elicitation strategies. Using built-in solvers and creating your own |
| `docs/models.qmd` | Models provide a uniform API for evaluating a variety of large language models and using models within evaluations |
| `docs/tools.qmd` | Tools extend the capabilities of models by registering Python functions for them to call. How to create custom tools |
| `docs/scorers.qmd` | Scorers evaluate the work of solvers and aggregate scores into metrics. Overview of the built-in scorers and pointers to custom scorers, metrics, and the scoring workflow |
| `docs/datasets.qmd` | Datasets provide samples to evaluation tasks. How to adapt various data sources for use with Inspect, including multi-modal data |

读完这 8 篇，[`source_code_reading_guide.md`](./source_code_reading_guide.md) 里的"全局地图"（`Sample → TaskState → plan(solver 链) → generate() → model.generate() → execute_tools()`）就有了对应的概念背景，不再是抽象的代码流程图。

## 第二层：Agent 相关（我们的实验直接用到）

| 文档 | 摘要 |
|---|---|
| `docs/agents.qmd` | Agents combine planning, memory, and tool usage to pursue complex, longer horizon tasks |
| `docs/react-agent.qmd` | Using and customizing the built-in ReAct agent（我们测试里用的 `basic_agent()`/`react()` 就是这个） |
| `docs/multi-agent.qmd` | Composing agents together in multi-agent architectures（对应目标一 Q6 的并行 subagent 部分） |

## 第三层：跑得更规范、更大规模

| 文档 | 摘要 |
|---|---|
| `docs/running.qmd` | Running evaluations reliably and at scale with eval sets, parallelism, error handling, limits, early stopping, and tracing |
| `docs/eval-sets.qmd` | Describing, running, and analysing larger sets of evaluation tasks（以后要跑一整套 benchmark 对比而不是单个 task 时看这个） |
| `docs/parallelism.qmd` | Running multiple tasks and models in parallel, sandbox concurrency, and writing parallel custom code |
| `docs/options.qmd` | Covers the various options available for evaluations as well as how to manage model credentials |
| `docs/handling-errors.qmd` | Techniques for dealing with runtime errors and recovering from crashes during evaluation |

## 第四层：日志与扩展（跟我们 `inspect_trace` 直接相关）

| 文档 | 摘要 |
|---|---|
| `docs/eval-logs.qmd` | Getting the most out of evaluation logs for developing, debugging, and analyzing evaluations（比 [`eval_log_format.md`](./eval_log_format.md) 更全面的官方视角，我们那篇更偏"逐字段配真实数据"） |
| `docs/extensions.qmd` | Extending Inspect with new Model APIs, tool execution environments, and storage platforms（扩展机制总览） |
| `docs/extensions-hooks.qmd` | Extending Inspect with Hooks（`inspect_trace` 的实现基础，之前已经深度调研过，细节见 `inspect_ai_docs_examples_audit.md`） |

## 待读：`docs/compaction.qmd`

`llms-description`: "Compacting message histories for long-running agents that exceed the context window"——这篇在 `inspect_ai_docs_examples_audit.md` 里被标记为"这次审计没有细读、下次优先级不低于已读项目"的线索，跟目标一"上下文膨胀"和目标三"context 层干预接口"直接相关。建议读完第一、二层建立基础概念后，紧接着读这篇。

## 不在这份清单里，但你可能会搜到的

`tools-custom.qmd`（自定义工具的沙箱/错误处理等进阶特性）、`standard-scorers.qmd`/`custom-scorers.qmd`（scorer 细节）、`multimodal.qmd`/`structured.qmd`/`reasoning.qmd`（多模态/结构化输出/推理模型专项）——这些是"用到再查"型文档，不需要现在通读，读完上面四层再按需检索即可。

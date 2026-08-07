# τ²-bench Qwen3.6-27B 全量运行：分支改动与实机问题记录

> 记录 `feat/tau2-qwen27b-full-adapter` 分支相对**原始仓库**（`jtsdw/harness` 的
> `main` 分支，即本分支 fork 起点）的所有更改：改了什么、为什么改、碰到了什么问题。
> 面向后续在 workstation 上做细致检查与探析的开发者。
>
> 时间范围：2026-08-03 ~ 2026-08-07（NSCC ASPIRE2 实机验证期）。
> 配套文档：`docs/nscc_tau2_full_run_agent_guide.md`（操作 runbook，本文件为
> 改动溯源/问题记录，两者互补）。

---

## 1. 分支与原始仓库基线

- 上游仓库：`https://github.com/jtsdw/harness.git`
- 原始分支：`main`（本分支 fork 起点 = `main` 上的 `dc44b6d` 之前的线性历史）
- 本分支：`feat/tau2-qwen27b-full-adapter`
- 分支提交（已 commit，2 个）：
  - `2f0497c` feat: make tau2 adapter portable for NSCC（2026-08-06 19:23）
  - `efe712a` docs: add NSCC tau2 full-run agent guide（2026-08-06）
- 未提交的工作区改动（随本记录一起同步）：
  - `tau2_adapter/src/tau2_adapter/agent.py`：空输出重试（§3.1）
  - `docs/nscc_tau2_full_run_agent_guide.md`：runbook 维护更新
  - 本文件（新增）

---

## 2. 已提交改动：2f0497c「feat: make tau2 adapter portable for NSCC」

**为什么**：原始仓库的 tau2_adapter 深度绑定作者本地环境——硬编码
`/home/liuyingen/code/...` 路径、只支持 `mock` 域、依赖本地 vLLM 启动脚本
（`local-model-server/scripts/serve.sh`）。要在 NSCC ASPIRE2 上跑五域全量，必须
去硬编码、通用化、可复现。

**改动清单**（13 文件，+484/-109）：

| 文件 | 改动 | 原因 |
|---|---|---|
| `.gitignore` | 新增 `/.deps/` | 上游 tau2-bench 源码以 path 依赖引入，不应入库 |
| `tau2_adapter/README.md` | 重写为可移植说明（五域、setup、TAU2_DATA_DIR 外部化） | 原 README 指向作者本地路径、只提 mock |
| `tau2_adapter/pyproject.toml` | `inspect_trace`/`tau2` 改为 **editable path 依赖**（`../inspect_trace`、`../.deps/tau2-bench`） | 原指向 `/home/liuyingen/code/tau2-bench`；editable 使 NSCC 上改上游源码即生效（空输出修复、NL assertions 修复都依赖这一点） |
| `tau2_adapter/scripts/run_adapter.sh` | 通用化：`TAU2_DOMAIN`/`TAU2_TASK_SET`/`TAU2_TASK_SPLIT`/`NUM_TASKS`/`RUN_NAME`、`USER_MODEL_NAME`/`JUDGE_MODEL_NAME` 分离、`TAU2_USER_LLM_ARGS`/`TAU2_JUDGE_LLM_ARGS`、`INSPECT_TRACE_VLLM_METRICS_URL` 推导、`LITELLM_LOCAL_MODEL_COST_MAP`、`TAU2_LLM_NL_ASSERTIONS(_ARGS)` 透传、`TAU2_DATA_DIR` 校验、入口改 `task.py@tau2` | 原脚本只跑 mock、硬编码作者路径、不区分 user/judge 模型、不传 NL assertions 配置（直接导致 retail 域 Missing credentials，见 §3.2） |
| `tau2_adapter/scripts/run_native_baseline.sh` | 同样通用化 | 同上 |
| `tau2_adapter/scripts/setup_tau2_bench.sh` | 从固定 commit `a1e85084` sparse-checkout 上游 tau2-bench 到 `.deps/`，**自动应用 Bug2 patch**，uv sync 双环境 | 可复现：锁死上游 commit + 兼容补丁入库 |
| `tau2_adapter/src/tau2_adapter/agent.py` | 注释去硬编码路径 | 仅注释清理 |
| `tau2_adapter/src/tau2_adapter/dataset.py` | `mock_dataset()` → 通用 `tau2_dataset(domain, task_set, task_split)` | 只支持 mock → 支持全部五域 |
| `tau2_adapter/src/tau2_adapter/runtime.py` | **新增**：`resolve_task_set`/`resolve_task_split`/`load_domain_tasks`/`resolved_selection`/`build_domain_environment`/`json_object_from_env` | 通过 tau2 registry 解析任意域的任务集与 split（`auto`→优先 base） |
| `tau2_adapter/src/tau2_adapter/solver.py` | 适配通用 dataset 接口 | 同上 |
| `tau2_adapter/src/tau2_adapter/task.py` | 支持 `-T domain/task_set/task_split` 参数 | 让一个 Inspect task 跑五域 |
| `tau2_adapter/tests/test_runtime.py` | **新增**：runtime 解析逻辑测试 | 保证 registry 解析正确 |
| `tau2_adapter/uv.lock` | 更新 | 依赖路径变更 |

---

## 3. 未提交工作区改动

### 3.1 `tau2_adapter/src/tau2_adapter/agent.py`：空输出重试（核心修复）

**碰到的问题（实机 2026-08-06 晨检发现）**：

- 现象：首轮全量（20260803_232435 起跑）中，telecom 43/114、retail 27/114、
  telecom-workflow 33/114 处中断。**外层 wrapper 仍返回 rc=0**（假成功），实际
  整域被 inspect 中断。
- 根因链：
  1. Qwen3.6-27B 开启 thinking 时，偶发只返回 reasoning 内容，`content` 为空且
     `tool_calls` 为空（模型侧空生成，非网络问题）。
  2. `convert.py:79`（tau2 上游）把空生成构造为**空 `AssistantMessage`**。
  3. `message.py:288`（tau2 上游）校验抛 `ValueError: AssistantMessage must have
     either content or tool_calls`。
  4. inspect_ai 捕获异常 → **中断整域**，后续任务全部丢失。
- 修复：`agent.py` 在把模型输出转成 `AssistantMessage` **之前**检查原始
  `ModelOutput`，空生成时重试同一 prompt（默认最多重试 3 次，首次请求不计入重试；
  `TAU2_AGENT_MAX_EMPTY_RETRIES` 可覆盖）。必须在转换前检查，因为 tau2 的模型校验会在
  `AssistantMessage` 构造期间立即抛错，构造后检查永远无法执行。重试耗尽后抛出含明确
  重试次数的 `RuntimeError`，且不把无效消息写入 state。
- 测试：新增 `tau2_adapter/tests/test_agent.py`，覆盖首次空输出后成功、耗尽预算、以及
  `TAU2_AGENT_MAX_EMPTY_RETRIES=0` 时只请求一次。
- 为什么改在 agent.py 而不 patch 上游：上游 `convert.py`/`message.py` 的校验是
  合理防御，问题在模型侧空生成；重试是「容忍模型偶发故障」的适配层职责。

### 3.2 依赖内改动：`.deps/tau2-bench`（独立 git 仓库，被 `/.deps/` ignore，不入 harness git）

> `.deps/tau2-bench` 是 setup 脚本从上游 `sierra-research/tau2-bench` 固定 commit
> `a1e85084` 拉取的独立仓库。此处两处改动是**实机运行必需**，但**不在 harness 的
> git 历史里**——重跑 `setup_tau2_bench.sh` 会丢失 config.py 改动（见 §5 风险）。

#### 3.2a `src/tau2/config.py`：NL assertions 支持环境变量（修复 Missing credentials）

- **碰到的问题**：retail 112/114 任务含 NL_ASSERTION 判定，跑起来报
  `Missing credentials`，任务直接失败。
- **根因**：上游硬编码 `DEFAULT_LLM_NL_ASSERTIONS="gpt-4.1-2025-04-14"`，**不读
  环境变量**；`run_adapter.sh` 传的 `TAU2_LLM_NL_ASSERTIONS` 被无视，litellm 找不到
  gpt-4.1 的凭据（NSCC 上只有本地 vLLM 服务）。
- **修复**：`config.py` 改为 `os.environ.get("TAU2_LLM_NL_ASSERTIONS", "gpt-4.1-2025-04-14")`
  （默认值不变，行为向后兼容）；新增 `TAU2_LLM_NL_ASSERTIONS_ARGS`（JSON）支持
  temperature 等参数覆盖。
- **验证**：basis=['DB','NL_ASSERTION']、reward=1.0、0 次 Missing credentials。

#### 3.2b `src/tau2/utils/llm_utils.py`：移除 tool_calls 顶层 `name` 字段（Bug 2 fix）

- **碰到的问题**：原生 tool-calling 时 vLLM 拒绝 tau2 构造的 tool_calls 消息。
- **根因**：`to_litellm_messages()` 在 tool_calls 对象顶层放了 `"name": tc.name`，
  而 OpenAI API 格式只允许 `function.name`（顶层 `name` 是非法字段）。
- **修复**：删除顶层 `name`，保留 `function.name`。
- **注意**：此改动已由 `setup_tau2_bench.sh` 通过入库的
  `tau2_adapter/scripts/tau2_bench_bug2_fix.patch` **自动应用**，重跑 setup 不会丢。

---

## 4. 实机运行问题清单（按时间线，2026-08-03 ~ 08-07）

### 4.1 环境搭建（vLLM 部署，全部已解决）

1. **torch CPU 版**：env 里 torch 是 `2.11.0+cpu`，GPU 加载缺 `libtorch_cuda.so`
   → 重装 `torch==2.11.0 --index-url .../whl/cu130`。
2. **无 nvcc**：flashinfer JIT 报 `Could not find nvcc` → env 内装 cuda-toolkit。
3. **`cannot find -lcuda`**：conda stub 在 `lib/stubs/`，flashinfer 找 `lib64/stubs/`
   → 软链 `lib64/stubs/libcuda.so`。
4. **GLIBCXX_3.4.32 not found**：conda GCC 编译产物运行时找不到新 libstdc++
   → `LD_LIBRARY_PATH=env/lib`。
5. **ninja not found**：flashinfer JIT 子进程 PATH 缺 env/bin → `export PATH=env/bin:$PATH`。
6. **KV cache 溢出**：默认 `max-model-len 262144` 超单 H100 → `--max-model-len 65536`。
7. **Mamba cache blocks 不足**：默认 `max-num-seqs 1024` > 316 → `--max-num-seqs 300`。
8. **`VLLM_USE_FLASHINFER=0` 无效**：Qwen3.6 GDN 线性注意力 kernel 必须 flashinfer JIT。

### 4.2 评测链路（全部已解决）

9. **tool-call parser 未启用**：tau2 每任务秒挂
   `litellm.BadRequestError: "auto" tool choice requires --enable-auto-tool-choice
   and --tool-call-parser`（results 显示 `termination_reason: infrastructure_error`）
   → vLLM 加 `--enable-auto-tool-choice --tool-call-parser qwen3_xml`。
10. **litellm cost 噪音**：`This model isn't mapped yet ... model_prices_and_context_window.json`
    → `LITELLM_LOCAL_MODEL_COST_MAP=True` + `register_model`（后者在
    qwen-taobench-deploy 的 `src/tau2/__init__.py`，见该仓库记录）。
11. **`Address already in use`**：8000 端口被残留 Qwen2.5-32B 服务占用（双次出现）
    → 杀旧进程再启；第二张卡用 8001。
12. **telecom 单任务空 trace**：user simulator 可自行 `USER→ENV` 操作
    （`user_msg.is_tool_call()` 路径），该任务 agent 零模型调用
    （`model_usage={}`，basis=ENV_ASSERTION）——**合法设计，非故障**。

### 4.3 运行管理（已解决/有对策）

13. **inspect 中断 rc=0 假成功**：wrapper 成功码不可信 → 验收一律数 sample 数（§9 脚本）。
14. **双卡并行**：串行 ~14h 超 12h walltime → 同节点双 GPU，8000/8001 隔离，按域分工。
15. **C-c 误杀**：外层 bash 循环 C-c 杀不死（wrapper 继续跑下一个域），需 kill 进程树。
16. **脚本 API key 行污染**：`VLLM_API_KEY=***` 行经 write_file/管道传输被污染成字面
    `***` → 上传前 base64 校验真实字节，定向 patch 修复。
17. **conda activate 静默失败**：GPU 节点 `conda activate vllm` 无效 → 一律绝对路径
    `~/scratch/envs/vllm/bin/python`。

---

## 5. 当前状态与风险提示

### 运行状态（2026-08-07 上午）

- 首轮全量 airline 50/50 完整保留；telecom/retail/telecom-workflow 三域因空输出 bug
  中断（43/27/33），已用修复后 agent.py 在两张新卡（a2ap-dgx035，jobs 200065/200066）
  重跑补跑。
- 用户指示「抢到卡先别拉起真实任务」→ 补跑已停止（评测进程已 kill，GPU 作业保留中）。
- 待用户决定：继续补跑、或在 workstation 上检查代码后再说。

### 风险提示

1. **config.py 改动未入库**：只存在于 NSCC `.deps/tau2-bench` 工作区。重跑
   `setup_tau2_bench.sh` 会丢失（Bug2 patch 不会，因为它入库了）。**建议**：将
   config.py 改动做成 `tau2_bench_nl_assertions_env.patch` 入库并在 setup 脚本中
   一并应用，或合入上游。
2. **agent.py 空输出重试未 commit**：本次同步后仍在工作区。建议在 workstation 上
   检查确认后 commit（连同本记录与 guide）。
3. **workstation 无 `.deps/`**：workstation checkout 未跑过 setup，若要在 workstation
   上复现/调试，需先 `./scripts/setup_tau2_bench.sh` 再应用 config.py patch。
4. **NL judge 默认值仍为 gpt-4.1**：仅本地 vLLM 场景需要 env 覆盖；无 env 时行为
   与上游一致。
5. **多 trial 未验证**：当前全量为 1 trial（inspect_ai epochs=1 默认）；多 trial
   稳定性对比需单独跑（4 trials 必然超 12h walltime，需拆域）。
6. **运维信息暴露**：runbook 含 NSCC 用户名、项目号、节点/job ID 与绝对路径。它们不是
   API 密钥，但若仓库可公开访问，建议改为环境变量/占位符，并把实机值放到不入库的
   私有运维记录中。

---

## 6. 结论

本分支相对原始仓库的改动可归纳为三组：
1. **可移植化**（commit 2f0497c）：去硬编码、五域支持、可复现 setup——已完成并入库。
2. **适配层容错**（工作区 agent.py）：容忍 Qwen3.6 偶发空生成，重试而非崩溃——待 commit。
3. **上游兼容修复**（.deps config.py + Bug2 patch）：NL assertions 环境变量化、
   tool_calls 格式合规——Bug2 已入库，config.py 待入库。

## 7. Workstation 代码审查结论（2026-08-07）

- **已修正阻断问题**：NSCC 同步版先构造 `AssistantMessage`、再检查是否为空；但空消息会
  在构造期间校验失败，所以重试循环不可达。现改为先检查 `ModelOutput.message`，确认存在
  text 或 tool calls 后才转换。
- **已修正重试语义**：配置值表示额外重试次数；默认 3 即最多 4 次模型请求。负数在模块
  加载时直接报配置错误，0 表示不重试。耗尽后不写入无效 state。
- **已修正 runbook 命令**：`TAU2_USER_LLM_ARGS` 的 `printf` 原本只有一个 `%s` 却传两个
  参数，bash 会重复使用 format，产生两个相连的 JSON 对象。现恢复 `api_base`/`api_key`
  两个占位符，并避免 smoke 命令覆盖已导出的 key。
- **新增回归测试**：覆盖空后成功、重试耗尽、零重试预算。当前 workstation 尚未创建
  `tau2_adapter/.venv` 且系统 Python 未安装 pytest，故本轮只能完成语法/静态检查；安装固定
  tau2 依赖后仍需运行 `tau2_adapter/.venv/bin/pytest -q tau2_adapter/tests`。
- **仍需处理的高风险项**：NL assertions 的 `config.py` 修复仍只存在于被忽略的 `.deps`
  工作区，无法由干净 checkout 复现。合并前应将其做成受版本控制的 patch，并由
  `setup_tau2_bench.sh` 幂等应用。

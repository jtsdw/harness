# NSCC τ²-bench 全量运行 Agent Runbook（NSCC 实机验证版）

> 给执行型 agent 的操作手册。目标是在 NSCC ASPIRE2 上，从
> `feat/tau2-qwen27b-full-adapter` 分支启动 OpenAI-compatible 模型后端，先完成五域
> smoke test，再运行 τ²-bench core 全量，并保留 Inspect `.eval` 与 `inspect_trace`
> 探针数据。
>
> **本版为 NSCC 实机验证版（2026-08-07 修订）**：全部路径、项目号、backend 配方均来自
> 已在 NSCC 真机跑通的实践（2026-08-03/04 Qwen3.6-27B × τ²-bench），并补充了
> 2026-08-06/07 双卡补跑、空输出 bug 修复、Missing credentials 修复的实测结论。
> 不再是占位符。

## 0. NSCC 实机事实（全部已验证）

| 项目 | 实际值 |
|---|---|
| 用户 | `n2505716` |
| **PBS 项目号** | **`12004380`** |
| 共享 scratch | `/home/users/ntu/n2505716/scratch/`（login/计算节点共享） |
| **本仓库 checkout** | **`/home/users/ntu/n2505716/scratch/wly/agent-research/harness`** |
| vLLM conda 环境 | `/home/users/ntu/n2505716/scratch/envs/vllm`（vllm 0.25.1, torch 2.11.0+cu130） |
| 模型 | `/home/users/ntu/n2505716/scratch/model/Qwen3.6-27B`（config.json: model_type=qwen3_5） |
| **vLLM 启动脚本** | **`~/scratch/wly/start_vllm.sh`**（真机跑通，见 §5.2） |
| vLLM 日志 | `~/scratch/wly/vllm_qwen36.log`（仅 start_vllm.sh 启动才写） |
| **tau2 数据** | **`/home/users/ntu/n2505716/scratch/wly/agent-research/qwen-taobench-deploy/external/tau2-bench/data`**（已对齐 harness 固定 commit a1e85084 的 926 个文件） |
| 队列 | `normal` / `aidev`（实测 aidev 拿到 a2ap-dgx037 H100 80GB, CUDA 13.2） |
| **已修复的上游 bug** | 见 §1.5「已知问题与修复」（agent.py 空输出重试、tau2 config.py NL assertions、llm_utils.py Bug2） |

## 1. 不可违反的边界

1. 只使用远端分支 `feat/tau2-qwen27b-full-adapter`（必须包含提交 `2f0497c`；开始长任务后记录 commit，不再 `git pull`）。
2. 不要寻找、复制或调用 `tau2_qwen27b_local/`（本地 5090 配置，已从 Git 排除）。
3. `tau2_adapter/scripts/setup_tau2_bench.sh` 只部署固定提交的上游 Python 源码；benchmark 数据位于仓库外，用绝对路径 `TAU2_DATA_DIR` 指定（见 §0）。
4. 不得把 tasks/DB/policy/模型权重/`.eval`/JSONL/控制台日志加入 Git。运行结果只写入被忽略的 `runs/`。
5. 被测 agent 必须使用原生 tool-calling 路径 `tau2-agent-vllm/vllm/<served-model>`，不要退回 `emulate_tools`。
6. 固定 `--max-connections 1 --max-samples 1`。
7. 五域 smoke 未全部通过之前，不得提交全量 PBS 作业。
8. Reward 为 0 不等于基础设施失败；HTTP 错误、sample error、空 trace、工具协议错误、metrics 缺失才是阻断性故障。
9. **验收必须数 sample 数，不能只看退出码**：inspect_ai 中断时可能仍返回 rc=0（假成功，见 §1.5-3）。
10. **vLLM 与评测必须在同一计算节点**：两者都访问 `127.0.0.1`，跨节点不可达。

## 1.5 已知问题与修复（2026-08-03 ~ 08-07 实机确认）

> 这些是本分支相对上游的**必要修改**，改动清单与理由见
> `docs/tau2_qwen27b_branch_changes_log.md`（同一仓库 docs/ 下）。

### 1.5.1 空输出 bug（Qwen3.6 只输出思维链，评测整域中断）——已修

- **现象**：Qwen3.6-27B 开启 thinking 时，偶发只返回 reasoning 内容、`content` 为空且
  `tool_calls` 为空。tau2 的 `convert.py:79` 会构造一个空的 `AssistantMessage`，
  `message.py:288` 校验抛 `ValueError: AssistantMessage must have either content or tool_calls`，
  导致 inspect 中断**整域**（已跑任务保留，后续任务全挂）。
- **根因**：模型侧空生成，不是网络/协议问题。
- **修复**：`tau2_adapter/src/tau2_adapter/agent.py` 增加空输出检测 + 重试同一 prompt
  （默认最多 3 次，可用 `TAU2_AGENT_MAX_EMPTY_RETRIES` 覆盖）。重试而非 nudge 提示，
  避免污染 trace 上下文。
- **验证方式**：`grep -n MAX_EMPTY_RETRIES tau2_adapter/src/tau2_adapter/agent.py`
  （editable 安装，改源码即生效，无需重装）。
- **注意**：低概率事件，修复后仍需观察是否复现；若重试耗尽会抛出明确的
  `RuntimeError`。当前 Inspect/orchestrator 是否只标记单个 sample 失败仍需实机验证，
  不能假定它一定不会中断整域，因此验收仍须核对 sample 数。

### 1.5.2 Missing credentials（retail 域 NL assertions 硬编码 gpt-4.1）——已修

- **现象**：retail 112/114 任务含 NL_ASSERTION 判定，跑起来报 `Missing credentials`，
  任务直接失败。
- **根因**：tau2 上游 `config.py` 硬编码 `DEFAULT_LLM_NL_ASSERTIONS="gpt-4.1-2025-04-14"`，
  **不读环境变量**；`run_adapter.sh` 传的 `TAU2_LLM_NL_ASSERTIONS` 被忽略，litellm 找不到
  gpt-4.1 的 key。
- **修复**：patch `.deps/tau2-bench/src/tau2/config.py` 支持
  `TAU2_LLM_NL_ASSERTIONS` 与 `TAU2_LLM_NL_ASSERTIONS_ARGS`（JSON）环境变量，默认值不变。
  实测 basis=['DB','NL_ASSERTION']、reward=1.0、0 次 Missing credentials。
- **注意**：该改动在 `.deps/tau2-bench`（独立 git 仓库，被 `/.deps/` ignore，不进 harness
  git）。`setup_tau2_bench.sh` 重跑会**覆盖**此改动——若重装依赖需重新应用
  （见变更记录文档 §4.2）。

### 1.5.3 inspect 中断返回 rc=0（假成功）——验收陷阱

- **现象**：评测被中断（C-c / 异常）后，外层 wrapper 仍可能以 rc=0 结束，`domain-status.tsv`
  显示成功，实际只跑了一部分任务。
- **修复**：无代码修复；**验收一律用 §9 的 sample 计数脚本**，以 `.eval` 中 sample 数
  与 402 目标对比为准，不信退出码。

### 1.5.4 双卡并行（12h walltime 约束下的加速）——实践配方

- **背景**：串行 402 任务估算 ~14h，超 12h `qsub -I` walltime 上限；必须并行。
- **做法**：同节点双 GPU 开两个 `qsub -I` 会话，vLLM 用 **8000 / 8001** 端口隔离，
  两卡分别跑不同 domain（如卡1=telecom→retail，卡2=telecom-workflow）。
- **启动**：`start_vllm.sh` 复制一份改 `--port 8001`，日志分开（`vllm_qwen36_8001.log`）。
- **注意**：每卡都是独立 inspect 进程、独立 `RUN_NAME`、独立 `.eval`；双卡并行时
  vLLM 的 metrics 归因仍为 `exact`（串行请求）。

## 2. "全量"定义（402 任务）

| Domain | `task_split=auto` 实际选择 | 任务数 |
|---|---:|---:|
| mock | all（无 base split） | 10 |
| airline | base | 50 |
| retail | base | 114 |
| telecom | base | 114 |
| telecom-workflow | base | 114 |
| 合计 | | 402 |

> **telecom-workflow 数据说明（实机核实）**：registry 中它注册为复用 telecom 的 task
> loader（`telecom_domain_get_tasks`），任务数据即 `data/tau2/domains/telecom/tasks.json`
> （base=114）。`ls data/tau2/domains/` 看不到 `telecom-workflow/` 目录是正常现象，不是数据缺失。
>
> 不要将 core 全量与穷举任务文件（telecom 两域 `task_split=all` 各 2,285，五域 4,744）混为
> 一谈；未经明确授权不跑 exhaustive。`banking_knowledge` 不属于默认 core 全量。

## 3. 登录节点：固定代码状态（真实路径）

```bash
CHECKOUT=/home/users/ntu/n2505716/scratch/wly/agent-research/harness
cd "$CHECKOUT"
git fetch origin
git switch feat/tau2-qwen27b-full-adapter
git pull --ff-only

git status --short
git merge-base --is-ancestor 2f0497c HEAD
export HARNESS_COMMIT="$(git rev-parse HEAD)"
printf 'harness_commit=%s\n' "$HARNESS_COMMIT"
```

验收：`git status --short` 无输出（**或仅含已说明的 agent.py 空输出重试改动**）；
ancestry 退出码 0；分支正确。
若 checkout 不存在（NSCC 上已 clone 到上述路径，正常跳过）：

```bash
git clone --branch feat/tau2-qwen27b-full-adapter \
  https://github.com/jtsdw/harness.git \
  /home/users/ntu/n2505716/scratch/wly/agent-research/harness
```

## 4. 登录节点：绑定外部 benchmark 数据（真实路径）

`TAU2_DATA_DIR` 指向 data 这一层（含 `tau2/domains`）：

```bash
export TAU2_DATA_DIR=/home/users/ntu/n2505716/scratch/wly/agent-research/qwen-taobench-deploy/external/tau2-bench/data
test -d "$TAU2_DATA_DIR/tau2/domains"
test -f "$TAU2_DATA_DIR/tau2/domains/mock/tasks.json"
test -f "$TAU2_DATA_DIR/tau2/domains/airline/tasks.json"
test -f "$TAU2_DATA_DIR/tau2/domains/retail/tasks.json"
test -f "$TAU2_DATA_DIR/tau2/domains/telecom/tasks.json"
```

> ⚠️ 数据已与 harness 固定的 tau2 commit（`a1e85084`）对齐（926 个文件）。若需重新
> 拉取：`cd /home/users/ntu/n2505716/scratch/wly/agent-research/qwen-taobench-deploy/external/tau2-bench && git fetch origin a1e85084a3960281cb06997594133e8f39ea42a7 && git checkout a1e85084a3960281cb06997594133e8f39ea42a7 -- data/`。

安装固定版本 tau2 源码 + adapter 环境：

```bash
cd "$CHECKOUT/tau2_adapter"
./scripts/setup_tau2_bench.sh
```

> setup 脚本会 `git fetch --depth 1 --filter=blob:none` 固定 commit `a1e85084...`
> （tau2-bench 上游）并 sparse-checkout 到 `harness/.deps/tau2-bench`。需 GitHub 可达
> （NSCC 登录节点已验证可达）。
>
> ⚠️ **setup 脚本会应用 Bug2 patch（llm_utils.py 移除 tool_calls 顶层 `name`），但不会
> 应用 config.py 的 NL assertions env patch（§1.5.2）**。重跑 setup 后必须手动重新应用
> config.py 改动，否则 retail 域会再次出现 Missing credentials。具体 patch 内容见
> `docs/tau2_qwen27b_branch_changes_log.md` §4.2。

用 registry 核对任务选择，输出必须为 `10, 50, 114, 114, 114`：

```bash
cd "$CHECKOUT"
TAU2_DATA_DIR="$TAU2_DATA_DIR" tau2_adapter/.venv/bin/python - <<'PY'
from tau2_adapter.runtime import load_domain_tasks, resolved_selection
domains = ["mock", "airline", "retail", "telecom", "telecom-workflow"]
for domain in domains:
    _, split = resolved_selection(domain, task_split="auto")
    count = len(load_domain_tasks(domain, task_split="auto"))
    print(domain, split or "all", count)
PY
```

若数量不同，停止并报告（数据快照与代码版本不一致）。

## 5. PBS 与模型后端边界（真实实践配方）

### 5.1 交互式 GPU allocation（项目号 12004380）

在登录节点 tmux 内（防断线）：

```bash
tmux attach -t qwen36 2>/dev/null || tmux new -s qwen36
qsub -I -q aidev -l select=1:ngpus=1 -P 12004380 -l walltime=12:00:00
```

分配成功后：

```bash
hostname        # 应为 a2ap-dgx0XX
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

> 仓库里 `scripts/nscc_interactive_gpu_session.sh` 需要 `PBS_PROJECT`，默认队列 normal；
> 上面是实测命令，直接用即可。walltime 上限 12h。aidev 满员（3/3）时改
> `-q normal` 路由到 aiq1/aiq2 也成功。

### 5.2 启动 vLLM 后端（真实跑通的 start_vllm.sh）

```bash
bash ~/scratch/wly/start_vllm.sh
```

脚本**精确内容**（已按 NSCC 真机验证，2026-08-03 跑通；实机日志确认参数
`enable_auto_tool_choice=True, tool_call_parser=qwen3_xml, max_model_len=65536,
max_num_seqs=300`）：

```bash
#!/bin/bash
export PATH=$HOME/scratch/envs/vllm/bin:$PATH
export CUDA_HOME=$HOME/scratch/envs/vllm
export LD_LIBRARY_PATH=$HOME/scratch/envs/vllm/lib:$LD_LIBRARY_PATH
mkdir -p $HOME/scratch/wly
nohup $HOME/scratch/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/users/ntu/n2505716/scratch/model/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B \
  --gpu-memory-utilization 0.9 \
  --max-model-len 65536 \
  --max-num-seqs 300 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --port 8000 \
  > $HOME/scratch/wly/vllm_qwen36.log 2>&1 &
echo "vllm started pid $!"
```

等待就绪（首次约 5-8 分钟编译 GDN kernel，之后 1-3 分钟）：

```bash
tail -f ~/scratch/wly/vllm_qwen36.log    # 看到 "Application startup complete"
```

> **为什么这些 flag 是必需的（NSCC 实机踩坑）**：
> - Qwen3.6-27B 是 attention+GDN 线性注意力混合架构，GDN kernel **必须** flashinfer JIT
>   编译（`VLLM_USE_FLASHINFER=0` 无效），因此 PATH/CUDA_HOME/LD_LIBRARY_PATH 三件套缺一不可
> - 默认 max-model-len 262144 超过单 H100 KV cache → `--max-model-len 65536`
> - 默认 max-num-seqs 1024 超过 Mamba cache blocks（316）→ `--max-num-seqs 300`
> - 无 tool-call parser 时 tau2 每任务秒挂 `infrastructure_error` → `--enable-auto-tool-choice --tool-call-parser qwen3_xml`

服务就绪后设置：

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=***
export MODEL_NAME=Qwen3.6-27B        # 以 /v1/models 返回的准确 ID 为准
export INSPECT_TRACE_VLLM_METRICS_URL=http://127.0.0.1:8000/metrics

curl -fsS "$VLLM_BASE_URL/models"
```

> ⚠️ 若 vLLM 不是用 `start_vllm.sh` 启动（如手动起），`~/scratch/wly/vllm_qwen36.log`
> 可能停留在旧失败记录（mtime 冻结）。以 `ss -tlnp | grep 8000` + `ps aux | grep EngineCore`
> 为准判断服务真实状态。

Qwen3.6-27B：agent-under-test 保持 thinking 开启；user simulator 与 NL assertion judge
用非 thinking 模式：

```bash
export TAU2_USER_LLM_ARGS="$(printf \
  '{"temperature":0,"max_tokens":2048,"api_base":"%s","api_key":"%s","input_cost_per_token":0,"output_cost_per_token":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  "$VLLM_BASE_URL" "$VLLM_API_KEY")"
export TAU2_JUDGE_LLM_ARGS="$TAU2_USER_LLM_ARGS"
```

> ⚠️ 若 NSCC 后端不接受 `chat_template_kwargs`，先用最小请求查清服务端支持方式；
> 不得静默删除配置后直接跑全量。

## 6. 五域 smoke gate

同一 GPU allocation、同一 backend，每域 1 任务：

```bash
cd "$CHECKOUT/tau2_adapter"

for domain in mock airline retail telecom telecom-workflow; do
  safe_domain="${domain//-/_}"
  TAU2_DATA_DIR="$TAU2_DATA_DIR" \
  VLLM_BASE_URL="$VLLM_BASE_URL" \
  VLLM_API_KEY="$VLLM_API_KEY" \
  MODEL_NAME="$MODEL_NAME" \
  INSPECT_TRACE_VLLM_METRICS_URL="$INSPECT_TRACE_VLLM_METRICS_URL" \
  TAU2_USER_LLM_ARGS="$TAU2_USER_LLM_ARGS" \
  TAU2_JUDGE_LLM_ARGS="$TAU2_JUDGE_LLM_ARGS" \
  TAU2_DOMAIN="$domain" \
  TAU2_TASK_SPLIT=auto \
  NUM_TASKS=1 \
  RUN_NAME="nscc_tau2_smoke_${safe_domain}" \
    ./scripts/run_adapter.sh native
done
```

逐域验收：
- 退出码 0；`runs/nscc_tau2_smoke_<domain>/logs/` 存在 `.eval`；状态非 error 且含 1 sample
- `.inspect_trace/**/sample-*.jsonl` 非空；出现 `token_attribution`、`attempt_group`、`vllm_metrics`
- 串行时 `vllm_metrics.attribution_confidence` 应为 `exact`
- 无持续 connection error、HTTP 4xx/5xx、tool schema 400、JSON judge 解析错误

> 已知结构性缺失（非故障）：tau2 的工具由 Environment 自身执行，不经 Inspect ToolEvent，
> 所以 `execution_topology`/`action_parsing` 的 tool-call 统计不完整；user simulator 和 NL
> judge 的调用也不由 inspect_trace 追踪。探针只要求被测 agent 的模型调用正确落盘。

## 7. 提交 core 全量作业

全量必须在 PBS batch job 或受保护的长时交互 allocation 中执行。controller 核心同原文档
（按 domain 建立独立 `.eval`、失败继续、退出码写入 `domain-status.tsv`）。PBS wrapper 生命周期：

1. 加载 NSCC 已验证的 CUDA/container module（本项目用 conda env，无需额外 module）
2. 导出全部环境变量（含 `TAU2_DATA_DIR`、`VLLM_BASE_URL`、`MODEL_NAME` 等）
3. 启动 27B backend（`bash ~/scratch/wly/start_vllm.sh`），轮询 `/v1/models` 与 `/metrics` 直到 ready
4. 运行 controller
5. 无论成败用 `trap` 停止 backend
6. 保留 PBS stdout/stderr、backend 日志、controller status、全部 `runs/` 输出

walltime 预算：按 smoke 单任务耗时 × 402 + 模型加载 + ≥20% 余量；不够则按 domain 拆
五个 job，不要提高 `--max-samples` 并发压缩时间。

> ⚠️ **实测提醒（2026-08-04）**：`qsub -I` 12h 内 4-trials 全量（800 次 simulation）会被
> walltime 打断。core 全量 402 任务若 1 trial 约 5-8h（视单任务耗时），12h 内可完成；
> 若要多 trial 必须拆 domain 或多个会话。
>
> ⚠️ **双卡并行（2026-08-06 实测）**：串行 ~14h 超 12h walltime，改为同节点双 GPU 并行
> （8000/8001 端口隔离，卡1=airline+telecom、卡2=retail+telecom-workflow）。双卡时每卡
> 独立 wrapper、独立 RUN_NAME、独立 `.eval`。

## 8. 监控与恢复

```bash
qstat -u n2505716
tail -F <pbs-output-file>
tail -F "$CHECKOUT/runs/<FULL_RUN_ID>"/*.console.log
```

每次状态汇报包含：job ID、节点、commit、模型 ID、已完成 domain、成功 sample 数、
当前 domain、最近 HTTP/tool error、剩余 walltime。恢复规则同原文档。

> **静默 watchdog（2026-08-06 配置）**：Hermes cron 每 30 分钟检查一次状态文件，仅当
> 域完成/异常才通知；平时静默。状态文件：`~/scratch/wly/tau2_full_status.txt` 与
> `rerun2` 变体。主动查看用 `qstat -u n2505716` + `ls runs/`。

## 9. 最终验收

1. `domain-status.tsv` 五行 exit code 全为 0（**但见 §1.5.3：不信退出码，信 sample 数**）
2. **五域 sample 总数为 402**（mock 10 + airline 50 + retail 114 + telecom 114 + telecom-workflow 114）
3. 每个 sample 有 tau2 reward、termination reason、duration
4. 每域非空 trace
5. agent model calls 的 `vllm_metrics` 存在，串行归因为 `exact`
6. 无 sample error、持续 HTTP 重试风暴、schema 400
7. 记录实际 commit、数据路径、模型 ID、backend 启动参数、PBS 资源、job ID

核对 `.eval`（**sample 计数是唯一可信验收**）：

```bash
cd "$CHECKOUT"
tau2_adapter/.venv/bin/python - <<'PY'
from pathlib import Path
from inspect_ai.log import read_eval_log
total = 0
for path in sorted(Path("runs").glob("nscc_tau2_core_full_*_*/logs/*.eval")):
    log = read_eval_log(str(path))
    count = len(log.samples or [])
    total += count
    print(path, log.status, count)
print("total_samples", total)
PY
```

## 10. 结果回收

在本地分析机 checkout 中执行（**注意默认 REMOTE_SUBDIR 是 `harness-<username>`，本仓库
实际在 `scratch/wly/agent-research/harness`，必须显式覆盖**）：

```bash
NSCC_REMOTE_SUBDIR=wly/agent-research/harness ./scripts/pull_runs.sh preview
NSCC_REMOTE_SUBDIR=wly/agent-research/harness ./scripts/pull_runs.sh pull
```

> ⚠️ 也可直接 rsync 整个 `runs/`：
> `rsync -avz n2505716@aspire2pntu.nscc.sg:/home/users/ntu/n2505716/scratch/wly/agent-research/harness/runs/ ./nscc_runs/`
> （workstation/本地无 sshpass 时经 hk 中继或用 NSCC 密码）。

拉回后结果位于本地 `nscc_runs/`。先保留原始 `.eval`、JSONL、PBS/backend/console 日志，
再生成报告；不要只保存汇总分数。

## 11. Agent 停止条件

同原文档（无真实 PBS project code / 无法获得 GPU allocation / 数据缺失或任务数不符 /
`/v1/models` 的 served ID 与 `MODEL_NAME` 不一致 / 原生 tool-calling 最小请求失败 /
五域 smoke 任一失败 / inspect_trace 为空或 vllm_metrics 缺失 / 归因持续 ambiguous /
walltime 明显不足且无法拆分 / checkout 运行中被修改）。

报告 blocker 时附可复现命令、退出码、日志绝对路径和最近一段错误输出；不泄露 API key、
密码或集群凭据。

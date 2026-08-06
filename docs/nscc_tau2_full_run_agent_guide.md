# NSCC τ²-bench 全量运行 Agent Runbook

这是一份给执行型 agent 的操作手册。目标是在 NSCC ASPIRE2 上，从
`feat/tau2-qwen27b-full-adapter` 分支启动一个 OpenAI-compatible 模型后端，先完成五域
smoke test，再运行 τ²-bench core 全量，并保留 Inspect `.eval` 与 `inspect_trace`
探针数据。

本文只描述共享代码和 NSCC 操作。不得上传 benchmark 数据，不得使用或移植本地
RTX 5090 runner，也不得把本地 Docker 镜像、模型缓存或运行结果加入 Git。

## 1. 不可违反的边界

执行 agent 必须遵守以下规则：

1. 只使用远端分支 `feat/tau2-qwen27b-full-adapter`。该分支必须包含提交
   `2f0497c`；开始长任务后记录 commit，不再在运行中途 `git pull`。
2. 不要寻找、复制或调用 `tau2_qwen27b_local/`。它是本地 5090 单卡执行配置，已从
   Git 中排除，不是 NSCC 依赖。
3. `tau2_adapter/scripts/setup_tau2_bench.sh` 只部署固定提交的上游 Python 源码。
   Benchmark 数据必须位于仓库外，并通过绝对路径 `TAU2_DATA_DIR` 指定。
4. 不得把 tasks、DB、policy、模型权重、`.eval`、JSONL 或控制台日志加入 Git。
   运行结果只能写入被忽略的 `runs/`，之后用 `scripts/pull_runs.sh` 拉回本地。
5. 被测 agent 必须使用原生 tool-calling 路径
   `tau2-agent-vllm/vllm/<served-model>`。不要退回 `emulate_tools` 来掩盖后端错误。
6. 固定 `--max-connections 1 --max-samples 1`。后者尤其重要：Inspect 默认会并行所有
   samples，而 tau2 user simulator 的 LiteLLM 请求不完全受 Inspect 连接限制；并行会
   污染逐请求的 vLLM metrics 归因。
7. 五个 domain 的 smoke 未全部通过之前，不得提交全量 PBS 作业。
8. Reward 为 0 可能只是模型能力问题，不等于基础设施失败；HTTP 错误、sample error、
   空 trace、工具调用协议错误和 metrics 缺失才是阻断全量运行的故障。

## 2. 本文所说的“全量”

默认目标是可与 τ²-bench 标准 base split 对比的 core 全量，共 402 个任务：

| Domain | `task_split=auto` 实际选择 | 任务数 |
|---|---:|---:|
| `mock` | all（该域没有 base split） | 10 |
| `airline` | base | 50 |
| `retail` | base | 114 |
| `telecom` | base | 114 |
| `telecom-workflow` | base | 114 |
| 合计 |  | 402 |

不要把“core 全量”和“穷举任务文件”混为一谈。在固定数据快照中，若把 telecom 两个
domain 都设为 `task_split=all`，它们各有 2,285 个任务，五域合计为 4,744。除非用户
明确要求 exhaustive run，否则不要运行这套更大的集合。

`banking_knowledge` 不属于本文默认 core 全量；它还需要 retrieval 配置和额外依赖，
不得未经明确授权混入本次作业。

## 3. 登录节点：固定代码状态

所有 Git 操作和依赖准备优先在登录节点完成。计算节点可能没有外网。

```bash
cd /home/users/ntu/n2505716/scratch/<your-checkout>
git fetch origin
git switch feat/tau2-qwen27b-full-adapter
git pull --ff-only

git status --short
git merge-base --is-ancestor 2f0497c HEAD
export HARNESS_COMMIT="$(git rev-parse HEAD)"
printf 'harness_commit=%s\n' "$HARNESS_COMMIT"
```

验收条件：

- `git status --short` 没有输出；
- ancestry 命令退出码为 0；
- 当前分支是 `feat/tau2-qwen27b-full-adapter`。

如果 checkout 不存在，在登录节点 clone 该分支；不要在 PBS 计算节点里临时 clone：

```bash
git clone --branch feat/tau2-qwen27b-full-adapter \
  https://github.com/jtsdw/harness.git \
  /home/users/ntu/n2505716/scratch/<your-checkout>
```

## 4. 登录节点：绑定外部 benchmark 数据

设置现有数据快照的绝对路径。`TAU2_DATA_DIR` 必须指向 `data/` 这一层，而不是
`data/tau2/`。

```bash
export TAU2_DATA_DIR=/absolute/nscc/path/to/tau2-bench/data
test -d "$TAU2_DATA_DIR/tau2/domains"
test -f "$TAU2_DATA_DIR/tau2/domains/mock/tasks.json"
test -f "$TAU2_DATA_DIR/tau2/domains/airline/tasks.json"
test -f "$TAU2_DATA_DIR/tau2/domains/retail/tasks.json"
test -f "$TAU2_DATA_DIR/tau2/domains/telecom/tasks.json"
```

若找不到数据，停止并报告缺失的绝对路径。不要下载进 `harness/`，不要把数据加入
分支。数据传输和代码 Git 流程必须分开。

安装固定版本的 tau2 Python 源码和 adapter 环境：

```bash
cd /home/users/ntu/n2505716/scratch/<your-checkout>/tau2_adapter
./scripts/setup_tau2_bench.sh
```

随后用 registry 实际读取外部数据，核对选择结果。以下输出必须为
`10, 50, 114, 114, 114`：

```bash
TAU2_DATA_DIR="$TAU2_DATA_DIR" uv run python - <<'PY'
from tau2_adapter.runtime import load_domain_tasks, resolved_selection

domains = ["mock", "airline", "retail", "telecom", "telecom-workflow"]
for domain in domains:
    _, split = resolved_selection(domain, task_split="auto")
    count = len(load_domain_tasks(domain, task_split="auto"))
    print(domain, split or "all", count)
PY
```

若数量不同，说明数据快照与代码版本不一致。停止，不要靠修改 expected count 继续跑。

## 5. PBS 与模型后端边界

仓库中的 `scripts/pbs_vllm_server_job.sh` 是旧 BFCL/3B 示例，不是 τ²/Qwen 27B
全量入口。可以参考它的 PBS 头和“启动服务—运行评估—停止服务”生命周期，但不得直接
提交该脚本来冒充本次作业。

执行 agent 必须先在交互式 GPU allocation 中验证 NSCC 实际支持的 backend 启动命令：

```bash
cd /home/users/ntu/n2505716/scratch/<your-checkout>
PBS_PROJECT=<real-project-code> PBS_WALLTIME=24:00:00 \
  ./scripts/nscc_interactive_gpu_session.sh
```

分配成功后：

1. 运行 `nvidia-smi`，记录节点名、GPU 型号、显存和驱动。
2. 使用 NSCC 支持的 module/container/runtime 启动目标量化模型。
3. 开启 Qwen 对应的原生自动 tool choice 和正确的 tool-call parser。
4. 不要把本地 `qwen36-offline-package`、5090 Docker image 或 Compose launcher 复制过来。
5. 将服务绑定到计算节点本地可访问的端口，并保留服务日志。

后端启动方式取决于 NSCC 当前容器和模型存储，本文不伪造一个未经现场验证的命令。
如果 agent 尚不知道正确的 NSCC backend 命令，这是需要用户或集群管理员提供信息的
真实 blocker，不能用旧 3B 脚本替代。

服务就绪后设置：

```bash
export VLLM_BASE_URL=http://127.0.0.1:<port>/v1
export VLLM_API_KEY=<api-key-or-nonempty-placeholder>
export MODEL_NAME=<exact-served-model-id>
export INSPECT_TRACE_VLLM_METRICS_URL=http://127.0.0.1:<port>/metrics

curl -fsS "$VLLM_BASE_URL/models"
curl -fsS "$INSPECT_TRACE_VLLM_METRICS_URL" | head
```

`MODEL_NAME` 必须取 `/v1/models` 返回的准确 ID，不要填写 Hugging Face 目录名来猜。

对于 Qwen3.6-27B，agent-under-test 保持 thinking 开启；不要给 agent 请求传
`enable_thinking=false`。User simulator 和 NL assertion judge 使用非 thinking 模式，
避免把推理文本混入严格 JSON/tool-call 输出：

```bash
export TAU2_USER_LLM_ARGS="$(printf \
  '{"temperature":0,"max_tokens":2048,"api_base":"%s","api_key":"%s","input_cost_per_token":0,"output_cost_per_token":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  "$VLLM_BASE_URL" "$VLLM_API_KEY")"
export TAU2_JUDGE_LLM_ARGS="$TAU2_USER_LLM_ARGS"
```

如果 NSCC 后端不接受 `chat_template_kwargs`，先用最小请求查清服务端支持方式；不得静默
删除配置后直接跑全量。

## 6. 五域 smoke gate

在同一个 GPU allocation 和同一个 backend 上，每个 domain 跑一个任务：

```bash
cd /home/users/ntu/n2505716/scratch/<your-checkout>/tau2_adapter

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

- 命令退出码为 0；
- `runs/nscc_tau2_smoke_<domain>/logs/` 中存在 `.eval`；
- `.eval` 状态不是 error，且包含 1 个 sample；
- `.inspect_trace/**/sample-*.jsonl` 非空；
- 至少出现 `token_attribution`、`attempt_group` 和 `vllm_metrics`；
- 串行运行时 `vllm_metrics.attribution_confidence` 应为 `exact`；
- 无持续的 connection error、HTTP 4xx/5xx、tool schema 400 或 JSON judge 解析错误。

不要把以下现象误判成 probe 故障：tau2 的工具实际由自己的 Environment 执行，不经过
Inspect `ToolEvent`，所以 `execution_topology`/`action_parsing` 中的 tool-call 统计存在
结构性缺失；user simulator 和 NL judge 的调用也不由 `inspect_trace` 追踪。探针只要求
被测 agent 的模型调用正确落盘。

可以用以下只读脚本汇总 smoke trace：

```bash
cd /home/users/ntu/n2505716/scratch/<your-checkout>
tau2_adapter/.venv/bin/python - <<'PY'
import collections
import json
from pathlib import Path

for run_dir in sorted(Path("runs").glob("nscc_tau2_smoke_*")):
    records = []
    for path in run_dir.glob(".inspect_trace/**/sample-*.jsonl"):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    kinds = collections.Counter(record.get("kind") for record in records)
    confidence = collections.Counter(
        record.get("attribution_confidence")
        for record in records
        if record.get("kind") == "vllm_metrics"
    )
    print(run_dir.name, dict(kinds), dict(confidence))
PY
```

任何一个 domain 未过 gate，就停止全量并保留现场日志。不要通过提高重试次数、切换
emulate mode 或跳过失败 domain 来制造“完成”。

## 7. 提交 core 全量作业

全量必须在 PBS batch job 或受保护的长时交互 allocation 中执行。不要直接在登录节点
运行模型，也不要依赖普通 SSH 会话存活。

建议在被忽略的 `runs/<FULL_RUN_ID>/` 中保存一个 NSCC 专用 controller 和 PBS 输出，
不要把 project code、模型路径或集群 module 写进共享 Git。下面是 controller 的核心；
它按 domain 建立独立 `.eval`，某个 domain 失败后仍继续，并把退出码写入状态表：

```bash
#!/usr/bin/env bash
set -uo pipefail

: "${CHECKOUT_DIR:?set CHECKOUT_DIR}"
: "${TAU2_DATA_DIR:?set TAU2_DATA_DIR}"
: "${VLLM_BASE_URL:?set VLLM_BASE_URL}"
: "${VLLM_API_KEY:?set VLLM_API_KEY}"
: "${MODEL_NAME:?set MODEL_NAME}"
: "${INSPECT_TRACE_VLLM_METRICS_URL:?set metrics URL}"
: "${TAU2_USER_LLM_ARGS:?set user args}"
: "${TAU2_JUDGE_LLM_ARGS:?set judge args}"
: "${FULL_RUN_ID:=nscc_tau2_core_full_$(date +%Y%m%d_%H%M%S)}"

controller_dir="${CHECKOUT_DIR}/runs/${FULL_RUN_ID}"
status_file="${controller_dir}/domain-status.tsv"
mkdir -p "$controller_dir"
printf 'domain\ttask_split\texpected_tasks\texit_code\n' >"$status_file"

domains=(mock airline retail telecom telecom-workflow)
expected=(10 50 114 114 114)
overall=0

cd "${CHECKOUT_DIR}/tau2_adapter"
for index in "${!domains[@]}"; do
  domain="${domains[$index]}"
  safe_domain="${domain//-/_}"
  run_name="${FULL_RUN_ID}_${safe_domain}"
  console_log="${controller_dir}/${safe_domain}.console.log"

  set +e
  TAU2_DATA_DIR="$TAU2_DATA_DIR" \
  VLLM_BASE_URL="$VLLM_BASE_URL" \
  VLLM_API_KEY="$VLLM_API_KEY" \
  MODEL_NAME="$MODEL_NAME" \
  INSPECT_TRACE_VLLM_METRICS_URL="$INSPECT_TRACE_VLLM_METRICS_URL" \
  TAU2_USER_LLM_ARGS="$TAU2_USER_LLM_ARGS" \
  TAU2_JUDGE_LLM_ARGS="$TAU2_JUDGE_LLM_ARGS" \
  TAU2_DOMAIN="$domain" \
  TAU2_TASK_SET="$domain" \
  TAU2_TASK_SPLIT=auto \
  NUM_TASKS="" \
  RUN_NAME="$run_name" \
    ./scripts/run_adapter.sh native 2>&1 | tee "$console_log"
  rc=${PIPESTATUS[0]}
  set -e

  printf '%s\tauto\t%s\t%s\n' "$domain" "${expected[$index]}" "$rc" \
    >>"$status_file"
  if ((rc != 0)); then
    overall=1
  fi
done

cat "$status_file"
exit "$overall"
```

PBS wrapper 必须完成以下生命周期：

1. 加载 NSCC 已验证的 CUDA/container module；
2. 导出上面的全部环境变量；
3. 启动 27B backend，并轮询 `/v1/models` 和 `/metrics` 直到 ready；
4. 运行 controller；
5. 无论 controller 成败都停止 backend；
6. 保留 PBS stdout/stderr、backend 日志、controller status 和所有 `runs/` 输出。

使用 `trap` 停止 backend，不能只在成功路径清理。walltime 应根据 smoke 的单任务耗时乘以
402，再加模型加载和至少 20% 余量。若单个 PBS walltime 不够，按 domain 拆成五个 job；
不要通过提高 `--max-samples` 并发来压缩时间。

## 8. 监控与恢复

监控使用 PBS 自身和日志文件：

```bash
qstat -u "$USER"
tail -F <pbs-output-file>
tail -F <checkout>/runs/<FULL_RUN_ID>/*.console.log
```

Agent 每次状态汇报至少包含：job ID、节点、commit、模型 ID、已完成 domain、成功 sample
数、当前 domain、最近一次 HTTP/tool error 和剩余 walltime。

恢复规则：

- `domain-status.tsv` 中 exit code 为 0 的 domain 不重跑；
- 只为失败或未开始的 domain 创建新的唯一 `RUN_NAME`；
- 不覆盖已有 `.eval` 或 `.inspect_trace`；
- 后端配置发生变化时使用新的 `FULL_RUN_ID`，不能把不同配置的结果拼成一个 run；
- walltime 到期不是代码成功，必须明确标记 incomplete。

## 9. 最终验收

全量完成必须同时满足：

1. `domain-status.tsv` 五行 exit code 全为 0；
2. 五个 domain 的 sample 总数为 402；
3. 每个 sample 都有 tau2 reward、termination reason 和 duration；
4. 每个 domain 都存在非空 trace；
5. agent model calls 对应的 `vllm_metrics` 存在，串行条件下归因应为 `exact`；
6. 没有 sample error、持续 HTTP 重试风暴或 schema 400；
7. 记录实际 commit、数据路径/快照标识、模型 ID、backend 启动参数、PBS 资源和 job ID。

Reward 低、任务 `max_steps` 结束或不同重复之间 reward 波动，不自动判定为基础设施失败。
τ² 是双控模拟，temperature 0 也不保证轨迹完全确定。

可用 Inspect API 核对 `.eval` 状态和 sample 数：

```bash
cd /home/users/ntu/n2505716/scratch/<your-checkout>
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

结果不通过 Git 传输。在本地分析机器的 harness checkout 中执行：

```bash
./scripts/pull_runs.sh preview
./scripts/pull_runs.sh pull
```

如果 NSCC checkout 目录名不是脚本默认值，显式设置：

```bash
NSCC_REMOTE_SUBDIR=<remote-checkout-directory> ./scripts/pull_runs.sh preview
NSCC_REMOTE_SUBDIR=<remote-checkout-directory> ./scripts/pull_runs.sh pull
```

拉回后结果位于本地 `nscc_runs/`。先保留原始 `.eval`、JSONL、PBS/backend/console 日志，
再生成报告；不要只保存汇总分数。

## 11. Agent 停止条件

遇到以下任一情况必须停止并报告，不得猜测：

- 没有真实 PBS project code 或无法获得 GPU allocation；
- 不知道 NSCC 上正确的 27B backend/container 启动方式；
- 外部 `TAU2_DATA_DIR` 缺失或任务数与固定快照不符；
- `/v1/models` 的 served ID 与 `MODEL_NAME` 不一致；
- 原生 tool-calling 最小请求失败；
- 五域 smoke 任一失败；
- `inspect_trace` 为空或预期的 `vllm_metrics` 缺失；
- 归因在串行设置下持续为 ambiguous；
- PBS walltime 明显不足且无法拆分 domain job；
- checkout 在运行中被修改。

报告 blocker 时附上可复现命令、退出码、日志绝对路径和最近一段错误输出，但不要泄露
API key、密码或集群凭据。

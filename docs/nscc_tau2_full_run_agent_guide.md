# NSCC τ²-bench 运行指南

本指南只记录稳定的集群运行接口。账号、项目号、checkout 路径、模型路径和队列名属于
机器配置，不写入仓库；通过 PBS 脚本或环境变量提供。

## 1. 准备代码与数据

在登录节点记录本次运行的代码版本，然后准备固定版本的 tau2 源码：

```bash
export CHECKOUT=/path/to/harness
export TAU2_DATA_DIR=/path/to/tau2-bench/data
cd "$CHECKOUT/tau2_adapter"
git -C "$CHECKOUT" rev-parse HEAD
./scripts/setup_tau2_bench.sh
test -d "$TAU2_DATA_DIR/tau2/domains"
```

`TAU2_DATA_DIR` 必须指向与固定 tau2 版本匹配的数据目录。源码由 setup 脚本固定，
数据 revision 会由 runner 尽可能写入 `run-manifest.txt`；如果数据不在 Git checkout
中，应另行记录其校验值。

## 2. PBS 与模型后端

vLLM 和评测必须在同一个计算节点上运行。仓库提供
`scripts/nscc_tau2_qwen36_job.sh`，其中保留了 NSCC 上验证过的 Qwen3.6 参数：
`qwen3_xml` tool parser、65,536 上下文、300 sequences，以及启动健康检查。
模板不会覆盖集群 module 已设置的 `CUDA_HOME`；如集群要求显式 CUDA module，应在提交
前加载，或按本机模块名称加到 PBS 脚本中。上述 vLLM 参数都可用同名环境变量覆盖。

首次使用时，在脚本头部填写 PBS project code；账号、checkout、数据、模型和 Python
路径通过 `qsub -v` 提供：

```bash
cd "$CHECKOUT"
qsub -v \
CHECKOUT_DIR="$CHECKOUT",\
TAU2_DATA_DIR="$TAU2_DATA_DIR",\
MODEL_PATH=/path/to/Qwen3.6-27B,\
VLLM_PYTHON=/path/to/vllm-env/bin/python \
  scripts/nscc_tau2_qwen36_job.sh
```

模板会启动 backend、验证 served model 和 metrics、执行五域 smoke、运行可恢复的 full
controller，并通过 `trap` 停止 backend。walltime 不足时使用单域作业，例如追加
`TAU2_DOMAINS=airline`；多个作业必须使用不同 GPU、端口和 `FULL_RUN_ID`。

交互式调试时也可以自行启动 backend，然后导出：

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=local
export MODEL_NAME=served-model-name
export USER_MODEL_NAME="$MODEL_NAME"
export JUDGE_MODEL_NAME="$MODEL_NAME"
export INSPECT_TRACE_VLLM_METRICS_URL=http://127.0.0.1:8000/metrics
export TAU2_EMPTY_RESPONSE_RETRIES=3

export TAU2_USER_LLM_ARGS="$(printf \
  '{\"temperature\":0,\"api_base\":\"%s\",\"api_key\":\"%s\"}' \
  "$VLLM_BASE_URL" "$VLLM_API_KEY")"
export TAU2_JUDGE_LLM_ARGS="$TAU2_USER_LLM_ARGS"
```

旧 NSCC wrapper 中的 `TAU2_AGENT_MAX_EMPTY_RETRIES` 暂时仍可使用，但会输出弃用警告。
不要把 API key、模型权重、benchmark 数据或结果文件提交到 Git。

## 3. 五域 smoke test

全量运行前，在同一个 backend 上逐域运行一个任务：

```bash
cd "$CHECKOUT/tau2_adapter"
for domain in mock airline retail telecom telecom-workflow; do
  TAU2_DOMAIN="$domain" \
  TAU2_TASK_SPLIT=auto \
  NUM_TASKS=1 \
  RUN_NAME="nscc_tau2_smoke_${domain//-/_}" \
    ./scripts/run_adapter.sh
done
```

每域应满足：一个 sample 完成、没有 sample error、tau2 reward 元数据存在，并且被测
agent 的 trace 非空。Reward 为 0 是模型结果，不等于基础设施失败。部分任务可能由
user simulator 直接操作环境，因此 agent trace 为空时应先核对该任务是否发生了 agent
模型调用。

## 4. 五域全量

core 选择共 402 个任务：mock 10、airline 50、retail 114、telecom 114、
telecom-workflow 114。公共 controller 为每个域建立独立输出，失败后继续记录；使用相同
`FULL_RUN_ID` 重跑时，会验证并跳过已经完整成功的域：

```bash
FULL_RUN_ID="nscc_tau2_core_$(date +%Y%m%d_%H%M%S)"
FULL_RUN_ID="$FULL_RUN_ID" ./scripts/run_full_core.sh

# walltime 不够时可以按域拆分：
FULL_RUN_ID="$FULL_RUN_ID" TAU2_DOMAINS="airline telecom" \
  ./scripts/run_full_core.sh
```

不要通过提高 `--max-samples` 来挤压单个 backend；walltime 不足时按 domain 拆 PBS job
或使用相互隔离的 GPU、端口和输出目录。

## 5. 验收与结果回收

Inspect 在部分中断场景下可能仍返回成功退出码，因此 controller 不只看进程退出码；它会
调用 `validate_run.py` 检查最新 `.eval` 的状态、sample error 和预期样本数。最终查看：

```bash
cat "$CHECKOUT/runs/$FULL_RUN_ID/domain-status.tsv"
```

五行都应为 `result=ok`，合计 402 个 sample、零 sample error。每域还应有 `.eval`、
`.inspect_trace` 和 `run-manifest.txt`。结果目录不进入 Git；使用仓库的
`scripts/pull_runs.sh` 或 rsync 回收到 `nscc_runs/`，并与 PBS/backend 日志一起保存。

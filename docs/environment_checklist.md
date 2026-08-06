# 环境清单

单页参考：项目涉及的所有环境、密钥、缓存位置放在哪。想知道"这台机器要跑通整个项目需要什么"，看这一篇就够，不用翻 quickstart/local_model_deployment/datasets 三篇拼凑。

## 五个独立环境总览

| | 位置 | 管理方式 | 用途 |
|---|---|---|---|
| `inspect_trace`（真正跑 benchmark 用这个） | `/home/liuyingen/code/efficient-harness/inspect_trace/.venv` | uv（`uv sync --extra dev`） | `inspect_ai`（PyPI）+ `inspect_evals`（PyPI）+ `inspect_trace` 本体，跑 benchmark、看日志、生成面板都在这个环境里 |
| local-model-server | `/home/liuyingen/code/efficient-harness/local-model-server/.venv` | uv（`./scripts/setup.sh`） | vLLM 本地推理服务，跟 `inspect_trace` 环境完全隔离（vLLM 的 torch/CUDA 依赖很重，故意分开，见下"为什么分两个环境"） |
| `tau2_adapter`（可选，只在接入 tau2-bench 时需要） | `/home/liuyingen/code/efficient-harness/tau2_adapter/.venv` | uv（`./scripts/setup_tau2_bench.sh`） | 跑 tau2-bench 原生复现 + 迁移进我们 harness 的对比，详见 [`tau2_bench_integration_findings.md`](./tau2_bench_integration_findings.md)——路径依赖指向 `/home/liuyingen/code/tau2-bench`，换机器要先把那个仓库也 clone 过去 |
| `toolspec_adapter`（可选，只在接入 ToolSpec 时需要） | `/home/liuyingen/code/efficient-harness/toolspec_adapter/.venv` | uv（`./scripts/setup_toolspec.sh`） | 跑 ToolSpec 原生复现 + 迁移进我们 harness 的对比，详见 [`toolspec_integration_findings.md`](./toolspec_integration_findings.md)——同样是运行时通过 `TOOLSPEC_REPO_DIR` 指向 `/home/liuyingen/code/ToolSpec` 这个外部仓库，换机器要先 clone |
| `inspect_ai` 上游参考克隆（只读，可选） | `/home/liuyingen/code/inspect_ai/.venv` | uv（`uv sync --extra dev`，仓库自带） | **不是我们项目的运行环境**——2026-08-04 重构之后，`inspect_trace` 已经改成对 PyPI 上的 `inspect-ai` 声明普通依赖，不再需要这个克隆才能跑。留着纯粹是为了读 inspect_ai 框架自身源码/走官方教程（`inspect_ai_quickstart.md` 里大部分内容用的是这个环境），不想读源码可以完全不装 |

这五个是**完全独立的 uv 项目**，互不干扰，也不需要互相知道对方存在——`inspect_trace`/`toolspec_adapter`/`tau2_adapter` 跟 `local-model-server` 之间只通过 HTTP（`http://localhost:8000`）通信，不共享 Python 环境。`tau2_adapter`/`toolspec_adapter` 是**可选的**：只有在需要接入 tau2-bench 或 ToolSpec 这两个具体第三方项目、跑对比实验时才需要装，平时跑 BFCL/GSM8K 之类的 benchmark 完全用不到，不用一开始就装全。

## 常见用法速查

环境装好之后实际怎么跑——每一行都是可以直接复制执行的真实命令，不是需要手改的模板。

### 跑 BFCL / GSM8K（最常用，只需要 `inspect_trace` + `local-model-server` 两个环境）

```bash
cd local-model-server
./scripts/serve.sh   # 或 serve_baseline.sh / serve_native_tool_calling.sh / serve_ngram_speculative.sh，见下方 vLLM 投机解码一节

cd ../inspect_trace/scripts
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct" ./run_bfcl_local_vllm.sh
# 换成 run_gsm8k_local_vllm.sh 跑 GSM8K，接口一样；MODEL_NAME 换成你实际在跑的模型

cd ../../local-model-server && ./scripts/stop.sh    # 跑完记得停，释放显存
```

`run_bfcl_local_vllm.sh`/`run_gsm8k_local_vllm.sh` 是薄包装脚本——只负责把连本地 vLLM 需要的那些环境变量（`VLLM_BASE_URL`/`VLLM_API_KEY`/`MAX_CONNECTIONS`/`MODEL_ARGS`）设好并正确 `export`，然后调用真正的 `run_bfcl_benchmark.sh`/`run_gsm8k_benchmark.sh`。**不要直接改这两个共享脚本本身去写死模型名**——改了要么被下次 `git pull` 覆盖，要么冲突，而且脚本内部用 `: "${VAR=value}"` 这种写法设的值不会自动 `export` 给子进程，这两个坑都真实踩过。想连托管 API（不是本地 vLLM）或者要用 `CATEGORIES`/`LIMIT`/`OUTPUT_DIR` 这些参数，直接调用底层的 `run_bfcl_benchmark.sh`/`run_gsm8k_benchmark.sh`，用法见它们各自的头部注释。

### tau2-bench（需要先装 `tau2_adapter` 环境）

```bash
cd tau2_adapter/scripts
./setup_tau2_bench.sh          # 装 tau2-bench 依赖 + 打补丁，幂等，重复跑会自动跳过已完成的步骤
./run_native_baseline.sh       # 用 tau2-bench 自己的原生 CLI 跑一遍，作为对照基线
./run_adapter.sh native        # 迁移进我们 inspect_ai harness 再跑一遍（用 tau2-agent-vllm provider，真原生 tool-calling）
./run_adapter.sh emulate       # 迁移进我们 harness 的另一个变体（emulate_tools=true，client 端模拟工具调用）
```

结果对比、发现的三个真实 bug、Hooks 触发验证，见 [`tau2_bench_integration_findings.md`](./tau2_bench_integration_findings.md)。

### ToolSpec（需要先装 `toolspec_adapter` 环境）

```bash
cd toolspec_adapter/scripts
./setup_toolspec.sh            # 装 ToolSpec 依赖（含 torch==2.5.1 cu121 wheel），幂等
NUM_QUESTIONS=100 METHODS="baseline pld recycling samd toolspec" ./run_native_repro.sh   # 原生仓库五种方法复现
./run_adapter.sh baseline      # 迁移进我们 harness：不开加速
./run_adapter.sh toolspec      # 迁移进我们 harness：ToolSpec 的 schema-aware + retrieval-augmented 投机解码
```

真实速度数字、"并非严格 lossless"这个真实发现、原生仓库 vs 适配器的逐 token 对比，见 [`toolspec_integration_findings.md`](./toolspec_integration_findings.md)。

### vLLM 自带投机解码（只需要 `local-model-server` + `inspect_trace`，不需要额外装环境）

```bash
cd local-model-server
./scripts/serve_ngram_speculative.sh   # 起服务，n-gram/prompt-lookup 模式，不需要额外草稿模型

cd ../inspect_trace/scripts
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct" ./run_bfcl_local_vllm.sh

cd ../../local-model-server && ./scripts/stop.sh
```

### 验证某个 `serve_*.sh` 模式真的生效了（不只是"起来了"）

`serve.sh`/`serve_baseline.sh` 是等价的（后者就是不加任何开关调用前者），验证过一个就不用重复验证另一个。剩下两个各有专门的验证脚本，不需要手动拼 curl/Python：

```bash
# serve_native_tool_calling.sh：确认服务端真的返回结构化 tool_calls，不是纯文本
cd local-model-server
./scripts/serve_native_tool_calling.sh
MODEL=Qwen/Qwen2.5-3B-Instruct ./scripts/verify_native_tool_calling.sh   # 打印 PASS/FAIL
./scripts/stop.sh

# serve_ngram_speculative.sh：跑一遍 baseline 和 ngram 各自的真实 BFCL eval，对比真实 tokens/s
# 这个脚本自己管理两次服务的起停，不需要你手动交替 serve/stop
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct" LIMIT=20 ./scripts/verify_ngram_speculative.sh
```

`verify_ngram_speculative.sh` 在仓库根目录的 `scripts/` 下（跨 `local-model-server`/`inspect_trace` 两个项目的编排脚本都放这里，参考 `scripts/pull_runs.sh`）。`LIMIT` 建议不要设太小——样本太少测出来的速度差异全是噪声，看不出真实加速比。

跟 ToolSpec 的真实对比（速度、正确性）见 [`toolspec_vllm_speculative_comparison.md`](./toolspec_vllm_speculative_comparison.md)。

## 为什么分两个环境

`local-model-server` 的依赖（`vllm`/`torch`/CUDA runtime 库）被这台机器的 GPU 驱动（535.230.02，最高支持 CUDA 12.2）严格限定了版本范围，`torch==2.4.0`（CUDA 12.1 wheel）是能用的上限。如果跟 `inspect_trace` 装进同一个环境，任何一边升级依赖都可能连带把这套精确版本组合搞坏。分开之后，`inspect_trace` 环境可以自由升级，不用管 GPU 驱动这件事。完整踩坑记录见 [`local_model_deployment.md`](./local_model_deployment.md)。

## 环境变量 / API Key

| 变量 | 用于 | 从哪来 | 什么时候需要 |
|---|---|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 连 hosted 模型（当前用 DeepSeek，`openai/deepseek-chat`，通过 OpenAI-compatible 接口） | 自己的 DeepSeek API key，`OPENAI_BASE_URL="https://api.deepseek.com/"` | 跑任何用 `openai/...` 模型字符串的 benchmark 时 |
| `VLLM_BASE_URL` / `VLLM_API_KEY` | 连本地 vLLM 服务 | `VLLM_BASE_URL="http://localhost:8000/v1"`，`VLLM_API_KEY` 随便填一个非空字符串（本地服务不校验） | 跑 `openai-api/vllm/...` 模型字符串的 benchmark 时 |
| `INSPECT_TRACE_DIR` | 控制 `inspect_trace` 衍生 JSONL 落盘位置 | 自己指定路径 | 不设默认是 `./.inspect_trace`，一般不需要手动设 |
| `PATH` 里要有 `$HOME/.local/bin` | 找到 `uv` 命令 | `python3 -m pip install --user uv` 装的时候会装到这 | 每次开新终端跑 `uv` 相关命令前（除非已经写进 `~/.bashrc`） |

密钥不要直接写进任何会被读进对话/日志的地方——用 `export` 或者 `source` 一个只有自己能读的 env 文件（`chmod 600`），具体做法见 [`inspect_ai_quickstart.md`](./inspect_ai_quickstart.md) 第 2 节。

## 数据缓存位置

| 内容 | 位置 |
|---|---|
| BFCL 数据集 | `~/.cache/inspect_evals/BFCL/`（26 个 category JSON，约 13MB） |
| GSM8K 数据集 | `~/.cache/huggingface/datasets/openai___gsm8k/` + `~/.cache/huggingface/hub/datasets--openai--gsm8k/` |
| `Qwen2.5-3B-Instruct` 模型权重 | `~/.cache/huggingface/hub/`（vLLM/HuggingFace 标准缓存路径，首次 `serve.sh` 会自动下载，约 6GB） |
| uv 包缓存（所有 uv 项目共享，硬链接去重） | `~/.cache/uv/` |
| 实验原始产出（`.eval` 日志 + `inspect_trace` JSONL） | `/home/liuyingen/code/efficient-harness/runs/`（已 gitignore，不会被清理脚本删） |

详见 [`datasets.md`](./datasets.md)。

## 硬件

```
NVIDIA RTX 2000 Ada Generation，16GB 显存，驱动 535.230.02（最高支持 CUDA 12.2）
```

本地模型选型（`Qwen2.5-3B-Instruct`）和 vLLM 版本锁定（`0.6.3.post1`）都是照着这个约束定的，换机器/换卡要重新核对，见 [`local_model_deployment.md`](./local_model_deployment.md) 的"硬件/驱动约束"一节。

## 从零重建（换新机器时照这个顺序走）

```bash
# 1. uv（如果没有）
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"

# 2. inspect_trace 环境（inspect_ai/inspect_evals 都是它 pyproject.toml 里声明的普通 PyPI 依赖，
#    uv sync 会自动装，不需要单独 clone/pip install 上游 inspect_ai）
cd /home/liuyingen/code/efficient-harness/inspect_trace
uv sync --extra dev

# 3. 本地模型环境（如果这台机器有 GPU，想跑本地模型的话）
cd /home/liuyingen/code/efficient-harness/local-model-server
./scripts/setup.sh      # 会自检 GPU 驱动/CUDA 兼容性，装坏了会告诉你哪里坏

# 4. 验证
cd /home/liuyingen/code/efficient-harness/inspect_trace
uv run pytest tests -q      # 应该全部 passed（迁移前是 13 个）

# 5.（可选）接入 tau2-bench 时才需要——先 clone tau2-bench 本体，再装适配器环境
git clone <tau2-bench 仓库地址> /home/liuyingen/code/tau2-bench
cd /home/liuyingen/code/efficient-harness/tau2_adapter
./scripts/setup_tau2_bench.sh

# 6.（可选）接入 ToolSpec 时才需要——同样先 clone ToolSpec 本体
git clone <ToolSpec 仓库地址> /home/liuyingen/code/ToolSpec
cd /home/liuyingen/code/efficient-harness/toolspec_adapter
./scripts/setup_toolspec.sh
```

第 3 步是可选的——如果这台机器没有兼容的 GPU，或者只打算用 hosted API（DeepSeek 之类），可以跳过，只做 1、2、4。第 5、6 步也是可选的——只有要接入 tau2-bench 或 ToolSpec 这两个具体第三方项目时才需要，两者互不依赖，可以只装其中一个。

## 当前状态（写这份文档时的快照，会过时，仅供参考）

- `inspect_trace` `.venv`（新位置 `efficient-harness/inspect_trace/.venv`）：待 2026-08-04 重构后首次 `uv sync` 验证
- local-model-server `.venv`：存在，vLLM 服务正在跑（`http://localhost:8000`），可以 `curl http://localhost:8000/v1/models` 确认
- BFCL 缓存：13MB；GSM8K 缓存：4.6MB
- 如果发现服务没在跑：`cd local-model-server && ./scripts/serve.sh`；不需要了记得 `./scripts/stop.sh` 释放显存

# 环境清单

单页参考：项目涉及的所有环境、密钥、缓存位置放在哪。想知道"这台机器要跑通整个项目需要什么"，看这一篇就够，不用翻 quickstart/local_model_deployment/datasets 三篇拼凑。

## 三个独立环境总览

| | 位置 | 管理方式 | 用途 |
|---|---|---|---|
| `inspect_trace`（真正跑 benchmark 用这个） | `/home/liuyingen/code/efficient-harness/inspect_trace/.venv` | uv（`uv sync --extra dev`） | `inspect_ai`（PyPI）+ `inspect_evals`（PyPI）+ `inspect_trace` 本体，跑 benchmark、看日志、生成面板都在这个环境里 |
| local-model-server | `/home/liuyingen/code/efficient-harness/local-model-server/.venv` | uv（`./scripts/setup.sh`） | vLLM 本地推理服务，跟 `inspect_trace` 环境完全隔离（vLLM 的 torch/CUDA 依赖很重，故意分开，见下"为什么分两个环境"） |
| `inspect_ai` 上游参考克隆（只读，可选） | `/home/liuyingen/code/inspect_ai/.venv` | uv（`uv sync --extra dev`，仓库自带） | **不是我们项目的运行环境**——2026-08-04 重构之后，`inspect_trace` 已经改成对 PyPI 上的 `inspect-ai` 声明普通依赖，不再需要这个克隆才能跑。留着纯粹是为了读 inspect_ai 框架自身源码/走官方教程（`inspect_ai_quickstart.md` 里大部分内容用的是这个环境），不想读源码可以完全不装 |

`inspect_trace` 和 `local-model-server` 是**两个独立的 uv 项目**，互不干扰，也不需要互相知道对方存在——它们之间只通过 HTTP（`http://localhost:8000`）通信，不共享 Python 环境。

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
```

第 3 步是可选的——如果这台机器没有兼容的 GPU，或者只打算用 hosted API（DeepSeek 之类），可以跳过，只做 1、2、4。

## 当前状态（写这份文档时的快照，会过时，仅供参考）

- `inspect_trace` `.venv`（新位置 `efficient-harness/inspect_trace/.venv`）：待 2026-08-04 重构后首次 `uv sync` 验证
- local-model-server `.venv`：存在，vLLM 服务正在跑（`http://localhost:8000`），可以 `curl http://localhost:8000/v1/models` 确认
- BFCL 缓存：13MB；GSM8K 缓存：4.6MB
- 如果发现服务没在跑：`cd local-model-server && ./scripts/serve.sh`；不需要了记得 `./scripts/stop.sh` 释放显存

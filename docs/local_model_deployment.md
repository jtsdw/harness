# 本地模型部署（vLLM）

在 16GB 显存的 RTX 2000 Ada 上用 vLLM 起了一个本地 OpenAI-compatible 服务（`Qwen2.5-3B-Instruct`），
用来在不依赖任何 hosted API 的情况下跑通 inspect_ai + `inspect_trace` 的完整链路。

## 三行版本

```bash
cd /home/liuyingen/code/efficient-harness/local-model-server
./scripts/setup.sh   # 环境搭建 + GPU 自检，幂等，可反复跑
./scripts/serve.sh   # 起服务，首次会自动下载模型，等到就绪或失败才返回
```

详细的踩坑记录和依赖版本选择理由在 `/home/liuyingen/code/efficient-harness/local-model-server/README.md`，这里只给结论和怎么接进 inspect_ai。

## 为什么不能直接 `uv add vllm`

这台机器的 NVIDIA 驱动（535.230.02）最高只支持 CUDA 12.2。不指定版本直接装最新 vLLM 会连带装上为 CUDA 12.8+ 编译的 torch，服务启动时直接报错"驱动版本太老"。定位到 `vllm==0.6.3.post1` 是最新的、torch 依赖（`2.4.0`）默认解析出的 wheel 仍然是 CUDA 12.1（能被 12.2 驱动兼容）的版本——再往上一个版本 `0.6.4.post1` 就跳到 torch `2.5.1`/CUDA 12.4，直接跨过驱动上限。

代价：`vllm==0.6.3.post1` 还没有 `--enable-auto-tool-choice`/`--tool-call-parser` 这套原生工具调用解析机制，所以本地这个服务**不能原生解析结构化 tool call**。解决方式是让 inspect_ai 自己做 prompt 层面的工具调用模拟（`emulate_tools=true`），细节见下面"接入 inspect_ai"。

跑通过程中还踩了三个损坏/过期依赖的坑（`transformers`/`torchaudio` 版本漂移导致要 CUDA 13 的库、`pyairports` 在 PyPI 上是个空壳包、部分 `nvidia-*-cu12` wheel 从坏缓存装出来是空的），完整细节和修复方式写在 `local-model-server/README.md` 里，`setup.sh` 已经把这些检查和修复都自动化了。

## 接入 inspect_ai

```bash
VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
  uv run --project /home/liuyingen/code/efficient-harness/inspect_trace inspect eval <task> \
  --model "openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" -M emulate_tools=true
```

`run_bfcl_benchmark.sh`（`inspect_trace/scripts/`）已经支持这个模式，不用另外写命令：

```bash
MODEL="openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" MODEL_ARGS="emulate_tools=true" \
VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
  /home/liuyingen/code/efficient-harness/inspect_trace/scripts/run_bfcl_benchmark.sh
```

已经用 `multi_turn_base` 跑通验证过：真实生成、`inspect_trace` 三个检测器全部正常记录（`prefill_diff`/`segment_tokens`/`attempt_group`），链路没有问题。3B 模型 + 模拟工具调用这个组合本身的任务准确率不高（这是模型能力/prompt 格式的问题，不是 harness 的 bug）——如果需要更强的本地模型能力，`local-model-server/README.md` 里给了升级到 `Qwen2.5-7B-Instruct-AWQ` 的路径。

## 停止服务

```bash
cd /home/liuyingen/code/efficient-harness/local-model-server
./scripts/stop.sh
```

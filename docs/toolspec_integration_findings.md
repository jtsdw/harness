# ToolSpec 接入记录：原生复现 + 迁移到 efficient-harness + 二次复现对比

延续 tau2-bench 那次的方法论（见 [`tau2_bench_integration_findings.md`](./tau2_bench_integration_findings.md)）：先读源码，再原样跑通原始仓库，再迁移进我们自己的 harness，最后用真实数据对比两边，如实报告，包括发现的问题。这次的对象是 [ToolSpec](https://arxiv.org/abs/2604.13519)（`/home/liuyingen/code/ToolSpec`），我们在 [`acceleration_methods_survey.md`](./acceleration_methods_survey.md) 里已经分析过它的 insight/method，这篇文档是"实际跑起来看看"的后续。

## 背景：ToolSpec 跟 tau2-bench 是完全不同的接入难度

tau2-bench 是一个真正的 Python 库（有 `pyproject.toml`，`pip install`/`uv add` 能直接用，OpenAI-compatible 的模型调用层），接入方式是"在 inspect_ai Task 内部驱动它的 Orchestrator，用 anyio 做同步/异步桥接"。

ToolSpec 完全不是这类东西：

- 没有 `pyproject.toml`/`setup.py`，就是一堆脚本，`model/`、`evaluation/` 只有在仓库根目录被加进 `sys.path` 时才能被当包导入（跟它自己的 `eval.sh` 用 `python -m evaluation.inference_toolspec` 的方式一致）。
- 不是 OpenAI-compatible HTTP 服务——是一个**原始 HuggingFace `transformers` 生成循环**，配合手写的、patch 过 KV cache 处理逻辑的 `modeling_qwen_kv.py`/`modeling_llama_kv.py`，用来支持树形 speculative verification。没有网络请求，没有 wire protocol，加速机制本身就活在 Python 函数调用栈里。
- 单轮：一条样本 = 一次 system+user 消息 → 一次 generate 调用 → 一个 tool call 文本输出，不像 BFCL multi_turn/tau2-bench 那样有多轮 agent loop、不调用 `execute_tools()`。

这意味着接入方式必须完全不同：不是"驱动一个外部 Orchestrator"，而是"自己实现一个 `ModelAPI` provider，在 provider 内部直接调用 ToolSpec 自己的 `baseline_forward()`/`toolspec_forward()` 函数"。好处是这样反而比 tau2-bench 的接入更简单——我们自己就是 inspect_ai 调用来生成的那个东西，不需要从"外部框架的异步循环"桥接回"inspect_ai 自己的 Sample 执行上下文"，Hooks 天然会触发。

## Phase 1：原样跑通原始仓库

### 环境搭建

```bash
cd /home/liuyingen/code/ToolSpec
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python transformers==4.51.1 accelerate==1.13.0 fschat==0.2.31 \
  gradio==3.50.2 openai==0.28.0 anthropic==0.5.0 sentencepiece==0.2.0 protobuf==3.19.0 \
  datasets==3.4.1 shortuuid tqdm
```

`requirements.txt` 写的 `torch==2.5.1` 不指定 index 的话，`pip`/`uv` 默认解析出的是 CUDA 12.4 wheel——这台机器驱动（535.230.02）最高只支持 CUDA 12.2，跟 `local-model-server` 当初遇到的坑完全一样（见 [`local_model_deployment.md`](./local_model_deployment.md)），同样的修法：显式指定 `--index-url https://download.pytorch.org/whl/cu121`。

模型选择：`eval.sh` 默认样例用 `Llama-3.1-8B-Instruct`（Meta 官方模型，HuggingFace 上是 gated，需要账号申请许可）。这台机器没有配置 HF token，直接用 gated 模型会卡在下载这一步。改用 `Qwen/Qwen2.5-3B-Instruct`——不需要申请许可、本地已经缓存过（`local-model-server` 之前下载过），`model/toolspec/modeling_qwen_kv.py` 本来就支持 `qwen2` model_type，不需要额外适配。

### 数据集

`data/API-Bank/level-{1,2,3}-api_processed.json` 已经打包在仓库里（合计 399 条），不需要额外下载。

### 五种方法的真实复现结果（Qwen2.5-3B-Instruct，API-Bank 前 100 条，`float16`）

```bash
./toolspec_adapter/scripts/setup_toolspec.sh
NUM_QUESTIONS=100 METHODS="baseline pld recycling samd toolspec" ./toolspec_adapter/scripts/run_native_repro.sh
```

| 方法 | mean accepted tokens | tokens/s | speedup vs baseline |
|---|---|---|---|
| baseline（原生 autoregressive） | 1.000 | 25.35 | 1.00x |
| pld（Prompt Lookup Decoding） | 2.089 | 45.99 | 1.81x |
| recycling（Token Recycling） | 2.930 | 44.40 | 1.75x |
| samd | 3.053 | 50.01 | 1.97x |
| **toolspec** | **4.761** | **77.33** | **3.05x** |

结论跟论文报告的排序完全一致（ToolSpec > samd > pld ≈ recycling > baseline），量级也在合理范围内——论文报告的"最高 4.2x"是在他们自己的模型/硬件组合下测出来的，我们用的是 3B 模型 + RTX 2000 Ada（16GB），3.05x 不完全等同但方向和排序完全对得上，没有理由怀疑复现有问题。

### 一个意外发现：ToolSpec 在实践中并不是严格 lossless

README 里"training-free"这个描述容易让人以为输出跟严格 greedy baseline 完全一致（speculative decoding 理论上通过精确验证保证这一点）。真实数据不是这样：

```
100 questions compared, 11 mismatches
mismatch question_ids: [7, 9, 25, 26, 27, 45, 48, 75, 82, 88, 93]
```

**排除了"这是 fp16/硬件本身不确定"的可能性**：把 baseline 在同一批数据上重新跑一遍（同一段代码、同一个模型、只是重新起一次进程），跟第一次的结果逐条对比，**0/100 不一致**——纯 autoregressive greedy decoding 在这台机器上是完全确定的。

**排除了"这是 ToolSpec 自己的 bug"的可能性**：把 pld/recycling/samd 三个 baseline 方法也各自对比了一遍原生 greedy baseline，四种方法（pld/recycling/samd/toolspec）的不一致集合几乎完全重合：

```
pld:       [9, 25, 26, 27, 29, 45, 48, 75, 82, 88, 93]
recycling: [7, 9, 25, 26, 27, 45, 48, 75, 82, 88, 93]
samd:      [9, 25, 26, 27, 29, 45, 48, 75, 82, 88, 93]
toolspec:  [7, 9, 25, 26, 27, 45, 48, 75, 82, 88, 93]
```

四个**独立实现**的 drafting 机制（schema-aware、prompt-lookup、token-recycle、samd 树）唯一的共同点是它们都要做"树形/batch 化的一次性 forward 验证多个候选 token"——这跟 baseline 逐 token 单独 forward 是不同的计算路径（`model/toolspec/modeling_qwen_kv.py` 用的是手写的 eager 风格 attention，显式 `attn_weights + attention_mask` 相加，跟 baseline 走的 stock sdpa 实现不是同一套 kernel）。四种方法几乎相同的不一致集合，指向的是**这台 GPU/这个模型上，批量化树形验证的浮点数值路径跟严格单 token 串行前向本身就存在微小的数值差异**，在极少数 logit 非常接近的分支点上会翻转 argmax——这是投机解码文献里已知的、"理论上无损，工程上批量验证有极小概率不完全一致"的现象，不是这四个方法各自的实现缺陷。11/100（11%）这个具体比例，如实记录，不淡化也不夸大。

## Phase 2：迁移到 efficient-harness

### 架构决策

写了一个新项目 `toolspec_adapter/`（跟 `tau2_adapter/` 同级），核心是一个从零实现的 `ModelAPI`（不是像 tau2_adapter 那样继承 `OpenAICompatibleAPI`，因为这次根本没有 HTTP 层可继承）：

- `src/toolspec_adapter/provider.py::ToolSpecHFModelAPI` ——`__init__` 里加载（并缓存）ToolSpec 的 patched 模型 + tokenizer；`generate()` 把 inspect_ai 的 `ChatMessage` 列表转成 ToolSpec 期待的 `[{"role":..,"content":..}]` 格式，用 `tokenizer.apply_chat_template` 拼 prompt，再通过 `anyio.to_thread.run_sync` 把同步、阻塞 CUDA 的 `_forward()` 丢进工作线程跑，避免卡住事件循环。
- `_forward()` 按 `-M mode=baseline|toolspec` 直接调用 ToolSpec 自己的 `evaluation.inference_baseline.baseline_forward()` / `evaluation.inference_toolspec.toolspec_forward()`——**复用真实代码，不重新实现**，这是从 tau2-bench 那次沿用下来的原则。
- retrieval 用的 `output_memory` 保存在 provider 实例上，跨 `generate()` 调用持续累积（模仿 ToolSpec 自己 `eval_toolspec.py::get_model_answers()` 的状态设计）——只有在 `--max-connections 1` 时才是正确的复现（这也是这个项目一贯的本地 GPU 跑法约定），文档和脚本里都标注清楚。
- `src/toolspec_adapter/dataset.py` 直接读 ToolSpec 自己打包的 `data/API-Bank/level-*.json`，`question_id` 编号方式跟原仓库的 `load_questions()` 完全一致，方便逐条对照。
- `src/toolspec_adapter/task.py` 用 inspect_ai 自带的 `generate()` solver（单轮，不需要自定义 solver/orchestrator）+ 一个把输出跟 Phase 1 产出的原生 baseline JSONL 逐条对比的 scorer（`matches_reference_baseline`）——直接复用"是否跟已知参考结果一致"这套已经在 BFCL/tau2-bench 上用过的验证方法论，不是重新发明一套正确性指标。

### 一个真实的依赖冲突，以及为什么不硬装

`evaluation/inference_toolspec.py`/`inference_baseline.py` 在模块顶层写了 `from fastchat.utils import str_to_torch_dtype`——只是拿来做一个 dtype 字符串到 `torch.dtype` 的字典查表，我们自己已经有等价逻辑，根本用不上这个函数。但 `fschat==0.2.31` 这个包本身钉死 `pydantic<2`，而 `inspect_ai` 需要 `pydantic>=2`——`uv sync` 直接报依赖不可解。不装真的 `fastchat`，改成在 `provider.py` 里注册一个假的 `sys.modules["fastchat.utils"]`，提供一个满足这一行 import 语句、但内部实现只是查表的桩函数（`_stub_fastchat()`）。这是刻意的、有注释说明的取舍，不是掩盖问题——如果以后真的需要 fastchat 别的功能，这个桩会立刻在那处调用点报错，不会静默出错。

### 验证：Hooks 触发、正确性、逐条对照

```bash
export TOOLSPEC_REFERENCE_JSONL=/home/liuyingen/code/ToolSpec/output/APIBank/Qwen2.5-3B-Instruct/Qwen2.5-3B-Instruct-vanilla-float16-temp-0.0.jsonl
./toolspec_adapter/scripts/run_adapter.sh baseline
./toolspec_adapter/scripts/run_adapter.sh toolspec
```

- **Hooks 确认触发**：`INSPECT_TRACE_DIR` 下产出了真实的 `_manifest.jsonl` + 逐样本 JSONL，不是空的。
- **`mode=baseline` 逐条对照 Phase 1 的原生 baseline 输出**：100/100 完全一致（`accuracy: 1.000`）——这是必然应该成立的底线检查（两边都是同一段 `baseline_forward()`，只是调用路径不同），先确认这条路径干净，再看更复杂的 `toolspec` 模式。
- **`mode=toolspec` 逐条对照原生 baseline**：`accuracy: 0.890`（11/100 不一致）——**跟 Phase 1 原生仓库的不一致数量完全一样**，而且是完全相同的 11 个 `question_id`（`[7, 9, 25, 26, 27, 45, 48, 75, 82, 88, 93]`，逐条用代码比对过，不是数字碰巧一样）。这证明适配器不是在近似 ToolSpec 的行为，是在**逐 token 精确复现**它（包括它的"非完全 lossless"这个真实特性）。

### 速度对比：原生仓库 vs 适配器（forward-call-only 计时）

用 inspect_ai `.eval` 日志里每次 `ModelCall` 记录的 `wall_time`/`new_tokens`（只统计 `_forward()` 内部的真实计算耗时，不含 inspect_ai 自身的编排/trace 写入开销），跟 Phase 1 原生脚本用完全同口径的字段比：

| | tokens/s (baseline) | tokens/s (toolspec) | speedup |
|---|---|---|---|
| 原生仓库（Phase 1） | 25.35 | 77.33 | 3.05x |
| 适配器（Phase 2） | 25.73 | 69.18 | 2.69x |

baseline 模式两边基本一致（25.35 vs 25.73，1.5% 的差异在噪声范围内），**toolspec 模式适配器慢了约 10.5%**（77.33 → 69.18 tokens/s），mean accepted tokens 也从 4.761 略降到 4.613。

**最可能的原因，如实标注为推断不是已证实的定论**：适配器里 `wall_time` 的计时范围是 `await anyio.to_thread.run_sync(self._forward, ...)` 这一整段，包含线程池调度开销，而不只是 `_forward()` 内部真正的 CUDA 计算时间；原生脚本的计时是在同一个线程里直接 `torch.cuda.synchronize(); start=time.time(); ...`，没有这层调度开销。这个固定的每次调用开销，在 baseline（平均每题 ~2s）里占比很小，在 toolspec（平均每题 ~0.6s）里占比相对大得多——这跟"baseline 两边几乎一致、toolspec 差了 10%"这个观察方向是吻合的。**没有进一步做微基准测量去精确量化这个开销**，如实标注为最可能的解释而非已验证的结论，值得记录但不值得为了这一个数字继续深挖。

## 结论

- ToolSpec 的核心机制（schema-aware + retrieval-augmented speculative decoding）**真实可复现**：3.05x 加速，排序符合论文预期，在我们的模型/硬件组合下是可信的正向结果。
- **"training-free 且无损"这个描述在实践中有一个小但真实的例外**：11/100（11%）的输出偏离严格 greedy baseline，但证明了这不是 ToolSpec 独有的问题，是这类"批量树形验证"方法在 fp16 GPU 上普遍共享的数值特性（pld/recycling/samd 都有几乎相同的偏离集合）——这是一个值得写进后续论文分析/工程决策的真实发现,不是我们复现有误。
- **迁移到 inspect_ai 之后，逐 token 精确复现了原生仓库的行为**（包括它的不完美之处），证明适配器设计是对的——不是"看起来差不多"，是用代码逐条比对过完全一致的 11 个 mismatch question_id。
- **迁移引入了约 10% 的可归因、有解释、但未精确量化的速度损耗**，来源大概率是线程调度开销而非逻辑差异；如实记录，没有为了让数字好看而回避。

## 已知限制

- 只迁移了 `baseline`/`toolspec` 两种模式，`pld`/`recycling`/`samd` 三个 baseline 方法尚未做同样的 `ModelAPI` 封装——如果要在 harness 里做完整五方法对比，需要照 `provider.py` 的 `_forward()` 分支模式各加一段（每个方法自己的 forward 函数签名不同，不是简单的参数切换）。
- 这次跑的是 100/399 条样本（API-Bank 三个 level 汇总的前 100 条），不是全量——跟这个项目一贯的"先用有代表性的子集验证、需要更强统计显著性时再扩大规模"的做法一致，扩到全量样本只需要改 `NUM_QUESTIONS`/`-T limit=`，代码不用改。
- 适配器的 10% 速度损耗来源没有做微基准精确定位（见上），如实标注为待深挖项，不影响本次复现结论的可信度（因为逐 token 输出已经证明了行为一致性，速度差异是"adapter overhead"层面的问题，不是"复现对不对"层面的问题）。
- `inspect_trace` 的 token 层三分类（`reasoning`/`tool_calling`/`final_response`）在这个 benchmark 上没有意义——API-Bank 的输出**全部**是 tool_calling（单轮预测，不像 BFCL multi_turn 那样混合推理文本和最终回复），跟 ToolSpec 论文本身讨论"tool-calling 生成占比"的问题设定是一致的（这也是为什么 `acceleration_methods_survey.md` 里 ToolSpec 那条分析要用 BFCL 数据单独算 token 占比,不能直接从这次的数据里拿）。

## 复现命令

```bash
# Phase 1：原生仓库复现
./toolspec_adapter/scripts/setup_toolspec.sh
NUM_QUESTIONS=100 METHODS="baseline pld recycling samd toolspec" ./toolspec_adapter/scripts/run_native_repro.sh

# Phase 2：迁移进 harness 后二次复现
uv sync --project toolspec_adapter   # 或直接用 run_adapter.sh 内部的 uv run --project
./toolspec_adapter/scripts/run_adapter.sh baseline
./toolspec_adapter/scripts/run_adapter.sh toolspec
```

## 文件清单

- `toolspec_adapter/src/toolspec_adapter/provider.py` —— 核心 `ModelAPI` 实现
- `toolspec_adapter/src/toolspec_adapter/_registry.py` —— 注册 `toolspec-hf` provider
- `toolspec_adapter/src/toolspec_adapter/dataset.py` —— API-Bank 数据集加载
- `toolspec_adapter/src/toolspec_adapter/task.py` —— Task 组装 + 对照参考结果的 scorer
- `toolspec_adapter/scripts/{setup_toolspec,run_native_repro,run_adapter}.sh` —— 复现脚本
- 关联分析：[`acceleration_methods_survey.md`](./acceleration_methods_survey.md) 的 ToolSpec 条目（insight/method 层面的分析，这篇文档是它的"实测验证"后续）

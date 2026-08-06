# NSCC H100 投机解码策略：为什么升级 vLLM + 换模型

2026-08-06，确认了 NSCC 那台 DGX 节点（`a2ap-dgx037`）的真实硬件/软件情况后拟定的计划。**这篇文档记录的是设计阶段的决策和理由，`nscc_model_server/` 里的具体脚本还没有在真实硬件上跑过、验证过**——如实标注，不要当成已经跑通的复现步骤。

## 真实发现：NSCC 上的 vLLM 版本约束不成立

`local-model-server/pyproject.toml` 把 vLLM 锁在 `0.6.3.post1`，唯一的原因是这台本地开发机（RTX 2000 Ada）的驱动只支持到 CUDA 12.2，更新版本的 vLLM 解析出的 torch/CUDA 组合会被驱动拒绝（详见 `local-model-server/README.md`"硬件/驱动约束"一节）。

2026-08-06 现场核实 NSCC 节点：

```
Driver Version: 595.71.05, CUDA Version: 13.2
GPU: NVIDIA H100 80GB HBM3
```

这个约束在这台机器上**根本不存在**。目前那边跑的也是 `0.6.3.post1`，大概率是复制/沿用了本地开发机这份 pyproject.toml 里的 pin，没有意识到这个 pin 本身是为了适配一张完全不同的卡才定的，不是项目本身的硬性要求。

## 为什么要升级：EAGLE-3

`vllm==0.6.3.post1` 只支持 `[ngram]`（prompt lookup）投机解码——这是我们已经在 `docs/toolspec_vllm_speculative_comparison.md` 里真实测过的：1.87x 加速，23/100 输出偏离 baseline。

vLLM 0.9+ 原生支持 EAGLE-3，查到的真实发布数据（H100，FP8）是 decode 阶段 **3.0-3.4x** 加速（EAGLE-2 在同样硬件上是 2.4-2.7x）——比我们测出的 ngram 效果明显更强，也是 ToolSpec 论文自己拿来对比的方法之一（`acceleration_methods_survey.md` 里还没细读的候选论文列表里就有）。

Sources:
- [EAGLE-3 Speculative Decoding on AMD Instinct GPUs: Training and Serving with vLLM and AMD Quark | vLLM Blog](https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct)

## 为什么要换模型

EAGLE-3 需要一个**针对具体目标模型训练过的草稿 checkpoint**，不是随便哪个模型都能配——这跟 `[ngram]` 完全不同（`[ngram]` 不需要任何额外模型，纯粹基于当前上下文找重复片段）。

查过 HuggingFace 上现成的 Qwen 系列 EAGLE-3 checkpoint，目前团队在跑的 32B（或者 commit 里写的"27b"，具体是哪个模型还需要你确认）**没有找到现成匹配的草稿**。但确认了一组真实存在、尺寸对得上的组合：

| 目标模型 | EAGLE-3 草稿 |
|---|---|
| `Qwen/Qwen3-32B` | `RedHatAI/Qwen3-32B-speculator.eagle3` |
| `Qwen/Qwen3-8B` | `RedHatAI/Qwen3-8B-speculator.eagle3` |

Sources:
- [RedHatAI/Qwen3-32B-speculator.eagle3 · Hugging Face](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3)
- [RedHatAI/Qwen3-8B-speculator.eagle3 · Hugging Face](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3)

`nscc_model_server/`（新建的独立项目，见其 README）默认用 `Qwen/Qwen3-32B` + `RedHatAI/Qwen3-32B-speculator.eagle3` 这一组——尺寸跟团队现在用的 32B 级别模型相当，只是换成了 Qwen3 这一代而不是 Qwen2.5。

**这是需要你确认的一个真实取舍**：如果团队已经跑的 tau2/ToolSpec 实验都是基于 Qwen2.5 系列、换到 Qwen3 会导致跟已有结果不可比，需要重新考虑——要么接受这个代际差异（Qwen3 通常是同尺寸下更强的模型，不是倒退），要么找 Qwen2.5-32B 专属的 EAGLE-3 草稿（目前没查到现成的），要么退回 `[ngram]` 继续用 Qwen2.5（放弃 EAGLE-3 的加速上限，但保持模型代际一致）。

## 具体方案：`nscc_model_server/`

新建的独立 uv 项目，**不是**改 `local-model-server/`（那个必须保持 `vllm==0.6.3.post1` 给本地开发机用，改了会直接搞坏本地环境）。

```bash
cd nscc_model_server
./scripts/setup.sh              # 装 vllm>=0.9.0，打印真实装到的版本号
./scripts/serve_eagle3.sh        # Qwen3-32B + EAGLE-3 草稿
./scripts/verify_eagle3.sh       # 确认投机解码真的生效了，不是服务起来了就算数
./scripts/stop.sh

./scripts/serve_baseline.sh      # 同一个模型，不开投机解码，对比基线
```

## 待现场验证清单（如实列出，这次没法验证）

- `uv sync` 装出来的 vllm 具体是哪个版本——`>=0.9.0` 只是下限，具体解析出什么、`--speculative-config` 这个 JSON 语法在那个具体版本上是否完全一致，都要实测确认。
- `RedHatAI/Qwen3-32B-speculator.eagle3` 这个 checkpoint 能不能正常下载/加载、维度是否真的跟 `Qwen/Qwen3-32B` 匹配——只是从 HuggingFace 页面上看着型号对得上，没有实际验证过。
- 单张 H100 80GB 装不装得下 32B 模型（bf16 权重约 64GB）+ KV cache + EAGLE-3 草稿头——粗略算过应该够，但没有实测，OOM 的话需要 `TENSOR_PARALLEL_SIZE=2`（如果这个 PBS 作业分到了不止一张卡的话）。
- `verify_eagle3.sh` 检查的两个信号（启动日志关键字、`/metrics` 里有没有 spec-decode 相关字段）都是基于"猜测新版本大概会这样"，不是已经见过的真实输出格式。
- 老版本 vLLM 那个 `CUDA_VISIBLE_DEVICES` UUID 崩溃的 bug（`docs/local_model_deployment.md`/`local-model-server/README.md` 记录过）在新版本上是否还存在——`nscc_model_server/scripts/serve.sh` 保留了同样的防御性修复，但没验证新版本是否已经自己修了这个 bug。

## 相关文档

- [`toolspec_vllm_speculative_comparison.md`](./toolspec_vllm_speculative_comparison.md) —— 本地这张卡上 ngram 投机解码的真实结果，作为对照基线
- [`remote_compute_workflow.md`](./remote_compute_workflow.md) —— NSCC 连接/同步方式
- `nscc_model_server/README.md` —— 这个新项目本身的说明

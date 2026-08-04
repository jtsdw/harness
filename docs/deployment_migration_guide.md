# 部署迁移指南：新服务器（H100 80GB）+ 团队协作

写作背景：现在这台机器（RTX 2000 Ada，16GB，驱动 535.230.02，最高 CUDA 12.2）上积累的一整套依赖锁定（`vllm==0.6.3.post1`、`emulate_tools=true`、`pyairports` 本地 shim 等，见 [`local_model_deployment.md`](./local_model_deployment.md)）全部是**反推自这台机器的硬件限制**的。新服务器（NVIDIA H100 80GB HBM3，驱动 595.71.05，CUDA 13.2）硬件约束完全不同，不能照抄旧配置——这份指南分三部分：先解决"我们的东西现在没有被版本控制保护"这个前置问题，再讲硬件升级带来的具体变化，最后讲团队共用一张 GPU 的方案。

**如实说明本文档的局限**：我没有新服务器的直接访问权限，下面所有涉及"新服务器上该怎么装"的内容，能查到官方文档确认的会标"已核实（查文档）"，没法在新服务器上实际跑一遍验证的会标"待现场验证"——这跟我们一路坚持的"如实报告、不臆测"是同一个原则，只是这次验证者要换成你自己在新服务器上做。

## 第一部分：版本控制现状（2026-08-04 已解决，跟硬件无关）

这部分原来记录的是"`inspect_trace` 嵌在上游 `inspect_ai` 克隆里，不知道该推到哪个 remote"这个悬而未决的问题——2026-08-04 已经通过一次重构彻底解决：把 `inspect_trace`/`local-model-server`/`doc/efficient-harness` 三处东西整合进一个新的、我们完全拥有的项目 `/home/liuyingen/code/efficient-harness/`（不再嵌在上游 `inspect_ai` 克隆内部；`inspect_trace/pyproject.toml` 本来就把 `inspect_ai` 声明成普通 PyPI 依赖，验证过零代码改动就能对着 PyPI 版本跑通全部测试，嵌在上游克隆里从来不是硬性依赖）。

现状：

| 位置 | 状态 |
|---|---|
| `/home/liuyingen/code/efficient-harness/`（`inspect_trace`/`local-model-server`/`docs`/`runs` 四合一） | ✅ 全新 git 仓库，`main` 分支，单次干净初始提交（不含旧的三个来源仓库各自的历史——用户已确认不需要保留），尚未配置远程仓库 |
| 旧的 `inspect_ai/src/inspect_trace/`、`local-model-server/`、`doc/efficient-harness/` | 内容已复制到新项目，**旧目录本身未删除**（留作过渡期安全网），后续确认新项目跑通之后再单独决定要不要清理 |
| `/home/liuyingen/code/inspect_ai/`（上游克隆本体） | 继续保留，现在的角色是"只读参考副本"（读框架自身源码/走官方教程用），不再是我们项目的家，也不需要跟着搬到新服务器 |

**远程仓库地址仍然待你提供**——`git init` + 本地提交到此为止，我不会自己创建远程仓库；确认目标仓库（自己 fork 的 GitHub？团队自建的 GitLab/Gitea？）之后可以直接 `git remote add`/`push`。

## 第二部分：硬件升级带来的具体变化

### CUDA 约束完全解除，之前的版本锁定应该重新评估

旧机器整条依赖链（`vllm==0.6.3.post1` → `torch==2.4.0`/CUDA 12.1 → `transformers==4.46.3` → `pyairports` 本地 shim）都是因为驱动最高只支持 CUDA 12.2。新服务器 CUDA 13.2，这条约束不存在了。

已核实（查 vLLM 官方文档，2026-08 时间点）：
- vLLM 稳定版本默认编译目标包含 CUDA 13.0 兼容二进制，用 `VLLM_PRECOMPILED_WHEEL_VARIANT`（如 `cu130`）或 `VLLM_MAIN_CUDA_VERSION` 可以显式指定变体，一般会根据系统 CUDA 版本自动探测。
- 安装方式：`uv pip install vllm --torch-backend=auto`（自动探测本机 CUDA 版本选对应 wheel），不再需要像旧机器那样手工试探哪个版本能装。
- CUDA 13.x 跟 CUDA 12.x 的库不兼容，装的时候要确认拿到的确实是 13.x 变体，不是不小心装了 12.x 的默认 wheel。

**待现场验证**：具体该锁定哪个 vLLM 版本号，需要在新服务器上实际 `uv add vllm` 看它解析到哪个版本、`import vllm` 能不能正常跑通——不要照抄这份文档里查到的任何具体版本号（版本更新很快，等你真正装的时候大概率已经有更新的稳定版）。`pyairports` 那个坏 wheel 的 bug 是否还存在也需要重新触发验证一次（大概率新版本 vLLM 已经不依赖那条路径，但没有实测过不能打包票）。

### 原生 tool-calling：`emulate_tools=true` 这条路径大概率可以退休了

旧机器踩的一大堆坑（`<tool_call>` 标签解析失败、标签互相污染、静默丢失——见 `goal1_r3_r4_real_benchmark_findings.md` 的"附加发现"部分）根源都是 `emulate_tools=true`：因为旧版 vLLM 没有 `--enable-auto-tool-choice`/`--tool-call-parser`，逼着模型把整个调用写成裸文本再让 inspect_ai 自己用正则解析。

已核实（查 vLLM 官方文档 + Qwen 官方部署文档）：Qwen2.5 系列的 chat template 原生支持 Hermes 风格的 tool-calling 格式，vLLM 对应的启动参数是：

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
```

这样跑起来之后，inspect_ai 那边应该可以去掉 `-M emulate_tools=true`，直接走原生 `tool_calls` 字段——**理论上会让需求四那一整类"标签解析失败"的 bug 消失**，因为不再需要客户端自己拼接/解析裸文本标签。

**待现场验证**：这个推测需要重新跑一遍目标一的真实 benchmark 验证（复用 `run_bfcl_benchmark.sh`，去掉 `emulate_tools=true`）才能确认——如果真的验证通过，`goal1_r3_r4_real_benchmark_findings.md` 里那几条标签解析 bug 的结论就要标注"仅适用于 emulate_tools 路径，原生 tool-calling 下不复现"，不能不验证就直接下结论说问题解决了。

### 显存从 16GB 到 80GB：值得考虑换更大的模型，但这是独立决定

当前用 `Qwen2.5-3B-Instruct` 完全是被 16GB 显存逼的。80GB 显存打开了用更大模型的可能性（`multi_turn_base` 4.5% 的准确率有相当一部分是模型太小导致的真实任务执行能力不足，见之前的分析）。**这个决定我不替你做**——换模型会让所有已有的 benchmark 结果失去可比性，需要你确认是要"在新硬件上复现同样的小模型实验"还是"顺便升级模型重新建立基线"。

## 第三部分：多人共用一台服务器/一张 GPU

现有的所有真实 benchmark 脚本和 `inspect_trace/vllm_metrics.py` 的关联逻辑都假设 `MAX_CONNECTIONS=1`（单用户串行）——这个假设在多人共用场景下会被打破。两个方案，各有取舍：

### 方案 A：MIG 硬件分区（推荐，如果团队规模不大）

已核实（查 NVIDIA/vLLM 相关资料）：H100 支持 MIG，最多可以切成 7 个硬件级隔离的实例，每个有独立显存/算力配额，互不干扰（"hard QoS"，不是"尽力而为"的共享）。对我们的场景很合适：每个团队成员/每类工作负载分到一个固定的 MIG 实例，各自跑自己的 vLLM，互相看不见、抢不到对方的资源，`inspect_trace/vllm_metrics.py` 的按请求精确关联（依赖 `MAX_CONNECTIONS=1` 串行）在每个实例内部继续成立，不需要改代码。

已知的两个限制（查到的资料里明确提到）：
- MIG 隔离的是算力和显存，**不隔离 PCIe 带宽**——如果多个实例同时做大量 PCIe 密集操作（比如同时加载模型），彼此还是会有一定干扰。
- 切分粒度和"大模型"之间有矛盾：切成 7 份意味着每份显存变小（80GB/7 ≈ 11GB，跟我们现在这张 16GB 卡差不多），如果决定顺便升级模型（见上一节），可能需要切成更少、更大的份（比如 2-4 份），需要按实际团队人数和模型大小权衡。

配置需要 root/admin 权限（待现场验证是否有这个权限），大致命令：

```bash
sudo nvidia-smi -mig 1                                    # 开启 MIG 模式（可能需要重启 GPU/服务）
sudo nvidia-smi mig -cgi <profile_id> -C                  # 创建 GPU 实例，profile_id 决定每份大小
nvidia-smi -L                                              # 确认切分结果，每个实例会有自己的 UUID
```

每个团队成员用 `CUDA_VISIBLE_DEVICES=<MIG-UUID>` 指定自己的实例起 `local-model-server`，其余流程（`serve.sh`/`stop.sh`）不用改。

### 方案 B：单个大 vLLM 实例 + 真并发

不切分 GPU，起一个 vLLM 实例，把 `MAX_CONNECTIONS` 调到大于 1，让 vLLM 自己的连续批处理（continuous batching）去调度多个并发请求。H100 的显存和算力余量比旧的 RTX 2000 Ada 大得多，之前那次并发崩溃（`MQLLMEngine already dead`）有很大概率是旧版 vLLM 在资源紧张的消费级卡上的稳定性问题，换新硬件+新版本后可能已经不再复现——但**这纯粹是推测，没有验证过，不能当结论用**。

这个方案的代价：`inspect_trace/vllm_metrics.py` 现在的设计（`attribution_confidence: "exact"` 要求 histogram `_count` 差值恰好是 1）在真并发下会大量退化成 `"ambiguous"`——按请求精确关联 TTFT/ITL 这件事，本质上需要串行才能做到，这是设计上的固有权衡，不是 bug。如果选方案 B，需要接受"多人同时用的时候，目标二 model invocation 层的数据变得不精确"这个后果，或者只在低峰期/单人使用时段跑真正需要精确 profiling 的实验。

### 顺便做一件跟"多人共用"无关、但同样需要真并发的事：第一次真正验证目标二的排队/并发字段

这不是团队协作本身要求的，是趁着这次迁移**必须**要做的一次验证：`concurrency_savings_seconds`/`queue_depth_running`/`queue_depth_waiting`/`preemptions_delta` 这几个字段，在旧机器上积累的全部真实数据里**恒为 0**——不是因为算法错了（`concurrency_savings_seconds` 在专门写的 mock 场景里验证过能正确算出正数），是因为旧机器上的测试条件（`MAX_CONNECTIONS=1` + BFCL 工具默认不开 `parallel=True`）结构性地从没触发过并发/排队。这几个字段"在真的有并发/排队时是否正确"至今没有被验证过，只验证了"没有并发时正确显示 0"。详细复盘见 [`inspect_ai_roadmap.md`](./inspect_ai_roadmap.md)"目标一/二现状批判性复盘"一节。

新服务器是第一次有条件打破这个限制的机会。不管最终选方案 A 还是 B，建议单独跑一次刻意制造负载的验证实验：给 BFCL 的部分工具显式标 `@tool(parallel=True)`（或者换一个原生会并行调用工具的 agent/benchmark）+ `MAX_CONNECTIONS>1`，确认这几个字段能正确从 0 变成非零值。这一步应该在"团队开始日常使用"之前做，而不是顺带做——一旦多人开始用，就很难再干净地制造"故意让它排队/并发"这种受控条件了。

### 两个方案都需要的：网络访问方式

vLLM 现在只绑定 `localhost`。多人从各自机器访问同一台服务器，需要选一种方式（我不会替你选，涉及网络安全配置，需要你确认）：
- SSH 端口转发（`ssh -L 8000:localhost:8000 user@server`）——最简单，不改任何代码，每人各自转发到自己习惯的本地端口。
- 绑定到内网接口 + 防火墙规则——如果需要多人同时长期访问，SSH 隧道会比较麻烦，但这个需要新服务器的网络管理权限，而且要考虑要不要加认证（vLLM 本身的 API key 校验很弱，本地环境下"随便填一个非空字符串"就能过，见 `environment_checklist.md`，暴露到内网前应该重新评估）。

## 迁移清单：哪些东西要复制、哪些重新生成

| 内容 | 处理方式 |
|---|---|
| 源代码（`efficient-harness/` 一整个仓库：`inspect_trace`/`local-model-server`/`docs`） | git（见第一部分），clone 到新服务器；`runs/` 不在这个仓库内，见下一行 |
| `.venv` | 不用复制，新服务器上 `uv sync`/`./scripts/setup.sh` 重新生成（而且新服务器上装的依赖版本会不一样，复制旧的没意义） |
| HuggingFace 模型权重缓存（~15GB） | 可选：如果不想重新下载，`rsync -avP ~/.cache/huggingface/ newserver:~/.cache/huggingface/`；不复制的话首次 `serve.sh` 会自动下载 |
| uv 全局缓存（~25GB） | 不用复制，新服务器上会自己按需建立，而且旧缓存里很多是给旧 CUDA 版本编译的 wheel，对新服务器没用 |
| `runs/`（实验原始数据，28MB，已 gitignore） | **必须手动复制**，不会跟着 git clone 走——这是所有 findings 文档的证据基础，建议 `rsync` 或打包成 tarball 一起搬，不要遗漏 |
| API key / 密钥 | 不要复制任何文件，新服务器上按 `environment_checklist.md` 的方法重新 `export`/建私有 env 文件 |

## 待现场验证清单（汇总）

- [ ] `local-model-server` 在新服务器上用最新 vLLM 能否顺利装起来（不要照抄旧版本号）
- [ ] `pyairports` 那个坏 wheel 的问题是否还存在
- [ ] 原生 tool-calling（`--enable-auto-tool-choice --tool-call-parser hermes`）能否替代 `emulate_tools=true`，替代后目标一那几条标签解析 bug 是否真的消失
- [ ] 是否有 root/admin 权限配置 MIG
- [ ] 如果选方案 B（真并发），`MAX_CONNECTIONS>1` 是否还会复现旧机器那次崩溃
- [ ] 网络访问方式（SSH 隧道 vs 内网绑定）由谁决定、要不要加认证
- [ ] 团队开始日常使用前，跑一次刻意制造并发/排队负载的实验，确认 `concurrency_savings_seconds`/`queue_depth_*`/`preemptions_delta` 这几个至今恒为 0 的字段在真负载下能正确产生非零值

Sources:
- [GPU - vLLM Documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Quickstart - vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
- [Tool Calling - vLLM](https://docs.vllm.ai/en/stable/features/tool_calling/)
- [vLLM - Qwen](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- [Practical NVIDIA-SMI + MIG partitioning to run multiple vLLM model servers on the same GPU](https://medium.com/@leonardo.haubrich98/slice-the-h100-not-your-slos-01013818a98d)

# 本地 agent 节点 + 远程计算节点：日常开发工作流

这份文档回答一个具体问题：本地这台机器有 Claude Code agent，用来写代码、debug、跑小规模验证；远程计算节点（NSCC ASPIRE2 集群，`aspire2pntu.nscc.sg`，PBS 作业调度）**没有 agent**，跑真正的大规模实验。两边怎么保持代码同步、结果怎么拉回来分析、以及"计算节点没有 agent 兜底"这件事对我们平时写代码/提交代码有什么要求。

## 连接信息

| 项目 | 值 |
|---|---|
| SSH 别名 | `nscc`（`~/.ssh/config` 已配置） |
| 主机地址 | `aspire2pntu.nscc.sg` |
| 账号 | `n2505716`（**团队共用同一个账号**，见 [`team_collaboration.md`](./team_collaboration.md)） |
| 作业调度 | PBS（`qsub`/`qstat`），不是直接 SSH 上去跑东西 |

连接：`ssh nscc`。这次尝试过 SSH key 认证，被拒绝（`Permission denied (password)`）——这个集群至少需要密码，很可能还有额外的 2FA/OTP（国家级超算中心的常见做法），具体登录方式列进了文末"待现场验证"清单，如实标注没有在这次对话里验证成功。

**这份文档跟 [`deployment_migration_guide.md`](./deployment_migration_guide.md) 的关系**：那份文档写的是"拿到一个 GPU 节点之后，CUDA/vLLM 版本怎么锁、原生 tool-calling 能不能用、MIG 怎么配"——这些内容在 PBS 分配到的节点上应该原样适用（大概率之前讨论的"H100 服务器"就是这个集群里通过 PBS 分配到的一个 GPU 节点，只是当时没意识到是 PBS 集群）。这份文档补的是**更上一层**：怎么从本地把代码送过去、怎么申请到那个节点、怎么把结果拉回来——`deployment_migration_guide.md` 完全没覆盖这一层（它当时假设的是直接 SSH 上去，不是走作业调度）。

## 代码同步：用 git，不用 rsync 推代码

本地改完 → commit → push 到 GitHub（`git@github.com:jtsdw/harness.git`）→ 计算节点侧 `git pull`。这是**主路径**，理由：

- 两人共用同一个 NSCC 账号、同一片 `scratch/` 空间——如果用 rsync 整树推代码（`/home/liuyingen/code/doc/sync-guide.md` 描述的那种方式），两人的改动会互相覆盖，没有冲突提示、没有历史。git 天然解决这个问题。
- `/home/liuyingen/code/doc/sync-guide.md` 和 `/home/liuyingen/code/quant/sync_to_aspire2pntu.sh` 用的 rsync push 方式仍然有效、可以用（比如临时应急、单人快速迭代不想等 git 那一套），但不是这个项目的主路径——保留作为备选，不重复写一遍。

具体操作：

```bash
# 本地
git add -A && git commit -m "..." && git push

# 计算节点（登录节点，不需要先进 PBS 会话）
cd ~/scratch/harness-<你的用户名>   # 见下"各自独立 checkout"
git pull
```

**关键细节：`git pull` 在登录节点做，不要指望在 PBS 分配到的计算节点里做**——超算集群的计算节点通常不能直接访问外网（登录节点才有），这条没有在这次对话里现场验证过（见文末待验证清单），但按这类集群的常见架构假设是成立的,先按这个流程走,真遇到计算节点也能连外网的话再简化。所以正确顺序是：先在登录节点 `git pull` 把代码更新好，再 `qsub -I` 进计算节点跑——不要在计算节点会话里再指望能联网装东西/拉代码。

## 结果拉取：`scripts/pull_runs.sh`

远程计算节点上的 `runs/`（真实实验数据：`.eval` 日志 + `inspect_trace` JSONL）已经 gitignore，不会跟着 git 走——这部分照抄 `/home/liuyingen/code/quant/nscc2local.sh` 的 rsync 风格单独处理，拉到本地的 **`nscc_runs/`**（不是本地自己的 `runs/`——两者故意分开，本地自己跑的实验放 `runs/`，从 NSCC 拉回来的放 `nscc_runs/`，避免本地临时跑的东西和真实跑在大卡上的结果混在一个目录分不清哪个是哪个，详见 [`nscc_runs/README.md`](../nscc_runs/README.md)）：

```bash
./scripts/pull_runs.sh preview   # 先看看会传什么，不真的传
./scripts/pull_runs.sh pull      # 真的拉，落到 nscc_runs/
./scripts/pull_runs.sh delete    # 拉 + 删掉本地有但远程没有的（远程数据被清理过时用）
```

默认按你的本地用户名拼出对应的远程 checkout 路径（`NSCC_REMOTE_SUBDIR` 环境变量可以覆盖）。拉回来之后本地跑 `inspect_trace/scripts/build_*.py`/`build_eval_report.py`（换成指向 `nscc_runs/<name>/`）生成可视化面板——生成面板这一步一直是本地做的，不需要在计算节点上跑。

## 各自独立 checkout，不共享同一份工作目录

两人用**同一个** NSCC 账号登录、同一片 `scratch/` 空间——如果共享同一份 `scratch/harness/` checkout，一个人 `git pull` 或者在跑长 job 时，会直接干扰另一个人正在用的代码状态。约定：各自在 `scratch/` 下建独立子目录：

```
scratch/harness-<你的用户名>/   ← 你自己 git clone 的一份
scratch/harness-<对方用户名>/   ← 对方自己 git clone 的一份
```

`scripts/pull_runs.sh`/`scripts/nscc_interactive_gpu_session.sh`/`scripts/pbs_vllm_server_job.sh` 都是按这个约定写的（默认用 `$(whoami)` 拼路径），不需要额外配置。更多协调细节见 [`team_collaboration.md`](./team_collaboration.md)。

## PBS 使用范式：两种方式，看场景选

### 方式一：交互式长 walltime 会话（真正需要人在场调试的时候用）

```bash
PBS_PROJECT=xxxxxxxx ./scripts/nscc_interactive_gpu_session.sh
```

`PBS_PROJECT` 没有默认值，脚本会直接报错拒绝瞎猜——项目代码用 `project -list` 查（这条也在待验证清单里，没有现场跑过）。拿到分配之后：

1. **先在登录节点开 `tmux`/`screen`，再跑这个脚本**——交互式会话是绑在这条 SSH 连接上的，连接一断，GPU 分配和里面跑着的东西都没了。
2. 进去之后 `cd` 到你自己的 checkout，`cd local-model-server && NATIVE_TOOL_CALLING=true ./scripts/serve.sh`（或不加这个变量，走默认的 `emulate_tools=true` 路径）起 vLLM。
3. `nvidia-smi` 确认真的分到了 GPU。
4. 剩下的操作（跑 benchmark、跑 tau2-bench 适配器等）跟本地开发时完全一样，用的是同一批已经验证过的脚本（见下）。

### 方式二：批处理长任务（不需要盯着、丢一个任务跑一晚上）

```bash
qsub scripts/pbs_vllm_server_job.sh
```

这个脚本本身照抄了 `/home/liuyingen/code/quant/unimq/bash/run_native.sh` 的 `#PBS` 头风格——提交前**必须**把脚本里 `#PBS -P REPLACE_ME_WITH_YOUR_PROJECT_CODE` 改成真实项目代码（不能用环境变量,PBS 头必须是静态文本）。跑完自动起 vLLM、跑一次 benchmark、关掉 vLLM，不需要人在场。默认跑的是 `multi_turn_base`，200 条——改 `BENCHMARK_*` 那几个变量（或者 `qsub -v VAR=value` 传参）换成别的组合。

## 已有的自包含脚本，在计算节点上原样能用

这些脚本在这次项目搭建过程中已经反复验证过，是自包含的（`set -euo pipefail`、清晰 usage、参数校验），git pull 下来之后在计算节点上直接就能跑，不需要改：

| 脚本 | 干什么 |
|---|---|
| `local-model-server/scripts/setup.sh` | 环境搭建 + GPU 自检，幂等 |
| `local-model-server/scripts/serve.sh` | 起 vLLM（`NATIVE_TOOL_CALLING=true` 开原生 tool-calling） |
| `local-model-server/scripts/stop.sh` | 停 vLLM |
| `inspect_trace/scripts/verify.sh` | 一次跑完 pytest/ruff/mypy |
| `inspect_trace/scripts/run_bfcl_benchmark.sh` / `run_gsm8k_benchmark.sh` | 跑真实 benchmark |
| `inspect_trace/scripts/run_concurrency_validation.sh` | 并发/排队字段验证（见 `deployment_migration_guide.md`"顺便验证目标二"一节） |
| `tau2_adapter/scripts/setup_tau2_bench.sh` | 装 tau2-bench 依赖 + 应用本地 bug 补丁，幂等 |
| `tau2_adapter/scripts/run_native_baseline.sh` | tau2-bench 原生 CLI 基线跑法 |
| `tau2_adapter/scripts/run_adapter.sh emulate\|native` | tau2-bench 适配器跑法（两个变体） |

## 提交代码到计算节点友好格式：检查清单

计算节点没有 agent 兜底——脚本写错了不会有人现场帮你调，得一次写对。提交前过一遍：

- [ ] **`set -euo pipefail`**——每个新 `.sh` 文件都要有，没有的话某一步真的失败了脚本会假装成功继续跑下去。
- [ ] **换行符必须是 LF，不能是 CRLF**——这是"本地某个编辑器/工具意外存成 Windows 换行、远程 Linux 节点跑不了"这类场景最经典的坑（`$'\r': command not found`）。这次写的所有脚本都用 `grep -c $'\r' <file>` 确认过是 0；以后新写的 `.sh` 文件提交前也过一下这个检查。
- [ ] **参数校验 + 清晰的 usage**——没传必需参数（比如 `PBS_PROJECT`）要直接报错退出并告诉人怎么填，不要往下跑然后在某个深层步骤莫名其妙失败。
- [ ] **环境变量集中在脚本顶部，给出默认值**——方便一个不熟悉这套代码的人（哪怕是熟悉 shell 但不懂 Claude Code 的人）直接改脚本顶部几行就能用,不需要读懂整个脚本逻辑。
- [ ] **不依赖本机专属的东西**——不要硬编码只在本地装了的工具路径、不要假设某个只在这台开发机上存在的环境变量。
- [ ] **幂等，能重复跑**——`setup_tau2_bench.sh`/`setup.sh` 这类脚本重复跑第二次应该直接跳过已完成的步骤，不要报错或者重复消耗资源。

## 待现场验证清单（如实列出，这次没法验证）

- [ ] `nscc` 的真实登录方式（是否需要 2FA/OTP，密码认证是否已经足够）——用户已经在对话里贴过一次真实密码,已建议尽快去 NSCC 门户改掉,这次没有用它做任何自动化登录尝试。
- [ ] PBS 项目代码（`project -list` 查出来填进 `PBS_PROJECT`/`pbs_vllm_server_job.sh` 的 `#PBS -P`）。
- [ ] `qsub -I` 交互式会话在这个集群/队列上是否真的开放（`quant` 项目的脚本目前只看到批处理 `qsub`，没看到交互式用法的先例）。
- [ ] 计算节点是否有外网访问（决定了 `git pull`/`uv sync` 从 PyPI 装包这类操作能不能在计算节点内部直接做，还是必须都在登录节点上先准备好）。
- [ ] `deployment_migration_guide.md` 里"具体该锁哪个 vLLM 版本号"仍然待验证，跟这份文档是同一个"待现场核实"的性质。
- [ ] BFCL 在原生 tool-calling 下需求四那几条标签解析 bug 是否真的消失——`deployment_migration_guide.md` 已经补了具体验证命令，还没有真的跑过。

## 相关文档

- [多人协作方案](./team_collaboration.md)
- [部署迁移指南（CUDA/vLLM 版本、MIG）](./deployment_migration_guide.md)
- [环境搭建速查](./environment_checklist.md)
- 本地-远程整树同步的备选方式：`/home/liuyingen/code/doc/sync-guide.md`

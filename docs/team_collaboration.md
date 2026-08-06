# 两人协作方案

现状：两人在同一台本地机器上用各自独立的 Unix 账号（都有自己的 Claude Code agent），但最终都连到**同一个** NSCC 账号（`n2505716`）——也就是说本地互不干扰，远程计算资源是共享的。这份文档定下代码协作、计算节点资源协调、项目管理这三方面的具体约定，目标是让两人不互相踩脚。

## Git 协作：哪怕两人也走分支 + PR

**规则**：

- `main` 分支任何时候都必须是能跑通的（至少 `inspect_trace/scripts/verify.sh` 能过）。不直接在 `main` 上改代码。
- 每个改动开一个短生命周期的 feature 分支，改完发 PR，哪怕只有对方一个人看。**为什么哪怕两人也要走 PR，不直接 push `main`**：这个项目已经有过好几次"看似正确、实测才发现是错的"的教训（比如这次对话里自己发现的算错的 `6/10 → 7/10`、以为是版本不兼容实际是 tau2-bench 自己代码里的 bug）——多一双眼睛看 diff，比多一次自我复核更可靠；而且 PR 描述本身就是变更记录，比 commit message 更适合写清楚"为什么这么改"。
- 开始新工作前先 `git pull`，避免在过期代码上改了半天最后一堆冲突。
- 涉及计算节点脚本（`scripts/`、`local-model-server/scripts/`、`tau2_adapter/scripts/`、`inspect_trace/scripts/`）的 PR，按 [`remote_compute_workflow.md`](./remote_compute_workflow.md) 的"计算节点友好格式检查清单"自查一遍再提。

**不需要**：复杂的分支保护规则、CI 强制门禁——两人规模，靠 PR 走查 + `verify.sh` 手动跑一遍就够，不为此另外搭基础设施。

## 计算节点资源协调：共享账号 = 共享配额

同一个 NSCC 账号意味着：两人的 PBS 任务共用同一份配额/allocation，长时间占用 GPU 的任务会直接影响对方能不能申请到资源。约定：

- **各自独立 checkout**：`scratch/harness-<你的用户名>/`、`scratch/harness-<对方用户名>/`——见 [`remote_compute_workflow.md`](./remote_compute_workflow.md#各自独立-checkout不共享同一份工作目录)，避免一个人 `git pull`/跑长 job 时干扰另一个人的工作目录。
- **起长时间占用 GPU 的任务前，跟对方说一声**（哪怕只是一句话）——尤其是交互式常驻 vLLM 会话（`nscc_interactive_gpu_session.sh`），这类会话会一直占着分配直到手动结束，容易在忘记的情况下长期占用共享配额。
- **`qstat` 先看一眼再提交**——提交新任务前确认没有已经在跑的、可能被误以为"没人用"而重复起的任务（比如两人都各自起了一个 vLLM 常驻服务，浪费配额）。
- 结果数据（远程各自的 `runs/`）已经按各自 checkout 分开存放，不会互相覆盖；`scripts/pull_runs.sh` 默认拉自己那份到本地的 `nscc_runs/`，需要看对方的结果时改 `NSCC_REMOTE_SUBDIR` 指过去即可。

## 项目管理：复用现有 docs/ 体系，不新开一套

- 架构级决策（选型、放弃某个方案的理由、真实踩过的坑）落成 `docs/` 下的文档，不靠口头/聊天记录传递——这个项目目前的 `docs/` 已经是这么做的（`framework-selection.md`、`tau2_bench_integration_findings.md` 都是这个模式），继续保持。
- 不新建看板/任务管理工具——两人规模，`docs/README.md` 的索引 + 各自开工前读一遍相关文档，足够同步状态。
- 谁在做什么：开始一项有一定规模的工作前，在 PR 描述或者对方能看到的地方说一句在做什么，避免两人同时改同一块东西产生大冲突。

## 相关文档

- [本地-远程同步工作流](./remote_compute_workflow.md)

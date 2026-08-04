# inspect_ai 快速上手 + 目标一（`inspect_trace`）实践指南

这份指南面向"要亲自把 inspect_ai 跑起来看看情况"的场景，目标是让你在不依赖任何真实模型 API Key 的前提下，从零理解 inspect_ai 的核心概念，跑通第一个 eval，看懂它产出的执行轨迹，再理解 `inspect_trace`（我们做的目标一实现）是怎么挂在它上面工作的。

配套文档：
- 四个目标的原始定义：[`efficient-harness.md`](./efficient-harness.md)
- 目标一到四的现状/可行性/工作量分析：[`inspect_ai_roadmap.md`](./inspect_ai_roadmap.md)
- 框架选型的完整调研过程：[`framework-selection.md`](./framework-selection.md)
- 目标一在真实 benchmark 上的验证结果、发现的 bug 与结论：[`goal1_real_benchmark_findings.md`](./goal1_real_benchmark_findings.md)
- 本地模型部署（vLLM）：[`local_model_deployment.md`](./local_model_deployment.md)
- 本地缓存的数据集：[`datasets.md`](./datasets.md)
- 模型 × 数据集对照实验 + 目标一/目标二契合度对照表：[`model_dataset_comparison_findings.md`](./model_dataset_comparison_findings.md)
- `.eval` 文件格式逐字段详解：[`eval_log_format.md`](./eval_log_format.md)
- inspect_ai 自带 docs/examples 复用价值审计：[`inspect_ai_docs_examples_audit.md`](./inspect_ai_docs_examples_audit.md)
- 源码阅读指南（从 prompt 输入到模型输出的完整链路，对照真实数据逐段拆解）：[`source_code_reading_guide.md`](./source_code_reading_guide.md)
- `inspect_trace` 自身的说明：`/home/liuyingen/code/efficient-harness/inspect_trace/README.md`

**关于本文出现的路径**：这份指南教的是 inspect_ai 这个框架本身怎么用（`mockllm`、uv 项目模型、Hooks 机制概念），跟 `inspect_trace` 具体怎么跑是两件事。前者用的是上游参考克隆 `/home/liuyingen/code/inspect_ai/`（只读，用来读源码/走官方教程，不是我们项目的一部分）；后者（第 5-7 节里凡是牵涉到 `inspect_trace` 本身——跑它的测试、读它的代码、用它的 hooks）一律指向我们自己的项目 `/home/liuyingen/code/efficient-harness/inspect_trace/`。两套路径都在下文原样出现，读的时候留意区分。

---

## 1. 五分钟概念图

不看代码，先建立心智模型。inspect_ai 的核心对象只有这几个：

| 概念 | 是什么 | 类比 |
|---|---|---|
| `Task` | 一次评测的完整定义：用什么数据、用什么方式让 agent 跑、怎么打分 | 一份实验配置 |
| `Sample` | 数据集里的一条数据（一个输入 + 一个期望目标） | 一条测试用例 |
| `Solver` | 决定 agent **怎么跑**的策略，比如"直接问一次"还是"允许多轮工具调用直到 submit" | agent 的行为逻辑 |
| `Model` | 模型抽象层，屏蔽 OpenAI/Anthropic/vLLM/mock 等不同后端的差异 | LLM 的统一接口 |
| `Tool` | 模型可以调用的函数 | agent 的手脚 |
| `Scorer` | 判断这条 sample 跑得对不对 | 打分器 |
| `eval()` | 把上面几样东西组装起来真正跑一遍的入口函数 | "Run" 按钮 |
| `EvalLog` / `.eval` 文件 | 一次 `eval()` 运行的完整产出：结果 + 逐步执行轨迹 | 运行日志 + 录像 |
| `Event` / `Transcript` | `.eval` 文件里逐步记录的"发生了什么"，包括每次模型调用（`ModelEvent`）、每次工具调用（`ToolEvent`）等 | 录像的每一帧 |
| `Hooks` | 官方提供的、不改 inspect_ai 源码就能挂进去观测/干预执行过程的插件机制 | 挂钩子 |

一句话理解 inspect_ai 的运行时形状：

```
Task(dataset, solver, scorer) --eval()--> 反复执行 [Solver 决定下一步 -> Model 生成 -> Tool 执行（如果有）] --> EvalLog(.eval 文件)
                                                                    │
                                                                    └── 过程中触发的每个 Hooks 回调
```

`inspect_trace`（目标一的实现）就是挂在最下面那个 Hooks 层上的一个观测者，不改动上面任何一层。

---

## 2. 环境搭建

这个仓库自己声明了官方的开发环境管理方式是 **uv**：根目录有已提交的 `uv.lock`（锁定了每个依赖的精确版本，保证任何人在任何机器上 `uv sync` 出来的环境完全一致），`pyproject.toml` 里也有专门的 `[tool.uv]` 配置段。README 里的原话是"uv sync --extra dev syncs the development environment from the checked-in lockfile"。所以这里不用 conda、也不用裸 `pip install`，直接用 uv。

如果还没装 uv：

```bash
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"   # 建议写进 ~/.bashrc 或 ~/.zshrc，避免每次都要 export
```

装好环境（会按 `uv.lock` 精确复现依赖版本，包含 pytest/mypy/ruff 等开发工具和各家 provider SDK）：

```bash
cd /home/liuyingen/code/inspect_ai
uv sync --extra dev
uv pip install -e src/inspect_trace        # 目标一实现，会自动注册 Hooks

# 确认能正常导入，且 inspect_trace 的 hooks 已经注册
uv run python -c "import inspect_ai; print(inspect_ai.__version__)"
uv run python -c "from importlib.metadata import entry_points; print(list(entry_points(group='inspect_ai')))"
```

第二条命令的输出里应该能看到 `EntryPoint(name='inspect_trace', value='inspect_trace._registry', ...)`——这代表 `inspect_trace` 的 Hooks 已经生效，之后任何 `eval()` 调用都会自动被它观测到，不需要额外传参启用。

`uv sync` 底层还是在项目根目录建一个普通的 `.venv`，所以下面两种跑命令的方式是等价的，看个人习惯：

```bash
# 方式一：每条命令前面加 uv run（不需要手动激活，本指南后续统一用这种写法）
uv run python hello_eval.py

# 方式二：像平时用 venv 一样先激活，再直接跑
source .venv/bin/activate
python hello_eval.py
```

注意：`uv sync --extra dev` 会连 OpenAI/Anthropic/Google 等各家 provider SDK 一起装（体积较大，第一次跑要花几分钟）。这份指南全程只用 inspect_ai 内置的 `mockllm/model`，不需要任何 provider SDK，也不需要任何 API Key——多装的这些依赖不影响这份指南的内容，只是因为跟着仓库官方推荐的 `--extra dev` 走会连带装上。

### 2.1 从 conda 过来的人：uv 怎么做环境隔离

用惯 conda 的人习惯"一个项目建一个 conda env，靠自己记住去 activate 哪个"，容易好奇 uv 是不是也有版本冲突问题、是不是也要手动一个项目配一个环境。答案是：**uv 默认就是"一个项目一个环境"，而且是自动的，不需要手动管理**——跟 conda 比，隔离效果等价甚至更严格，但操作模型完全不同。

- **conda 的模型**：环境是全局命名的资源（`conda create -n foo`，装在 `~/miniconda3/envs/foo`），conda 本身不知道"项目"这个概念，全靠你自己按约定记住"这个目录该 activate 哪个 env"，忘了 activate 是常见事故来源。
- **uv 的模型**：环境是项目的附属物。只要目录里有 `pyproject.toml`/`uv.lock`，这个目录就是一个"uv project"，`uv sync`/`uv run` 会自动在**这个目录下**建一个 `.venv`（就是本节看到的 `/home/liuyingen/code/inspect_ai/.venv`），只服务于这个项目。你 `cd` 到别的项目目录再跑 `uv run`，用的是那个项目自己的 `.venv`，两者互不相干、自动切换，不存在"忘了切环境导致装错依赖"的问题。

**磁盘会不会因此爆炸（N 个项目 = N 份完整依赖）？** 不会。uv 有一个全局 cache（`~/.cache/uv`），不同项目如果依赖了同一个包的同一个版本，磁盘上只存一份，各项目 `.venv` 里放的是**硬链接（hardlink）**，不是拷贝。实测验证过（拿 inspect_ai 里的 `tenacity` 包举例）：

```bash
$ ls -i /home/liuyingen/code/inspect_ai/.venv/lib/python3.13/site-packages/tenacity/__init__.py
28598334 ...
$ find ~/.cache/uv -path "*tenacity*__init__.py" | xargs ls -i
28598334 ...   # inode 号完全一样，确认是硬链接而非拷贝
```

也就是说即使给每个项目都建独立 `.venv`，实际磁盘占用大致只是"去重后的包体积"，不是"项目数 × 依赖体积"——这一点 conda 一般做不到（conda env 之间通常是各自独立的拷贝）。

**uv 真正做不到的事**：`.venv` 本质是纯 Python 虚拟环境，**只能管 Python 包**。conda 的差异化优势从来不是"环境隔离"，而是能管非 Python 的系统级依赖（特定版本 CUDA、MKL、编译好的 C/C++ 库）。如果某个项目需要固定系统库版本，uv 解决不了，那种场景还是得靠 conda 或 Docker。uv 自己也能管 Python 解释器版本（`uv python install 3.11`、`uv python pin 3.11`，会按项目 `requires-python` 自动下载/选用），这一块跟 conda "一个 env 一个 python 版本"的能力是对等的，只是不涉及非 Python 依赖。

实践上：像 inspect_ai 这种纯 Python 项目，uv 完全够用；如果你有别的项目涉及编译型/GPU 依赖，继续用 conda 管那部分没问题——一台机器上同时有 conda envs 和多个 uv 项目的 `.venv` 是常见组合，互不干扰，不需要二选一。

---

## 3. 第一次运行：跑一个不需要 API Key 的 smoke eval

新建一个脚本 `hello_eval.py`（放哪都行，比如 `$HOME/scratch/hello_eval.py`）：

```python
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import basic_agent

# mockllm/model 是 inspect_ai 内置的假模型，custom_outputs 按顺序弹出预设的回复,
# 不发任何真实网络请求，天然适合用来学习/调试/写测试。
model = get_model(
    "mockllm/model",
    custom_outputs=[
        ModelOutput.for_tool_call(
            model="mockllm/model",
            tool_name="submit",
            tool_arguments={"answer": "42"},
        ),
    ],
)

task = Task(
    dataset=[Sample(input="What is 6*7?", target="42")],
    solver=basic_agent(tools=[]),   # basic_agent = inspect_ai 内置的 ReAct 循环 solver
    scorer=includes(),
)

logs = eval(task, model=model)
print("status:", logs[0].status)
print("log 文件路径:", logs[0].location)
```

运行：

```bash
cd /home/liuyingen/code/inspect_ai
uv run python /path/to/hello_eval.py
```

你会看到一个终端进度面板（Rich 渲染的表格），跑完打印 `status: success` 和一个 `.eval` 文件的路径（默认在当前目录的 `./logs/` 下）。这就是你的第一次 inspect_ai 运行。

---

## 4. 读懂产出：`.eval` 文件里到底有什么

`.eval` 文件是二进制/压缩格式，不要直接 `cat`。三种读取方式：

**方式一：CLI 转成 JSON 看**

```bash
uv run inspect log dump ./logs/xxxxx.eval | less
```

**方式二：CLI 图形化 viewer（推荐第一次用这个建立直觉）**

```bash
uv run inspect view --log-dir ./logs
```

会起一个本地网页（默认 `http://localhost:7575`），可以逐 sample、逐 event 点开看，是理解"轨迹长什么样"最直观的方式。

**方式三：Python API 读进来做程序化分析**（`inspect_trace` 内部和它的测试都是这么读的）

```python
from inspect_ai.log import read_eval_log

log = read_eval_log("./logs/xxxxx.eval")
sample = log.samples[0]
for event in sample.events:
    print(event.event, getattr(event, "model", None))
```

`sample.events` 是一个扁平的事件流，`event.event` 是判别字段（`"model"`/`"tool"`/`"sample_init"`/`"span_begin"`/... 等等）。每一次模型调用对应一个 `ModelEvent`，关键字段：

- `event.input`：这次调用送进模型的**完整消息列表**（不是增量，是全量快照——这是我们做"重复 prefill 检测"的基础）
- `event.output`：模型的回复，`event.output.usage` 里有真实计费的 token 数
- `event.error` / `event.retries`：这次调用有没有出错、重试了几次
- `event.uuid`：这个事件的唯一 id

跑一遍第 3 节的脚本，用方式三读一下 `sample.events`，能看到至少一个 `ModelEvent`——这就是 `inspect_trace` 观测的原始数据源。

`.eval` 文件里其他每个字段（`eval`/`plan`/`results`/`stats` 顶层结构，`samples[i]` 里除 `events` 外的字段，`attachment://` 去重机制）的完整、逐字段讲解见 [`eval_log_format.md`](./eval_log_format.md)。

---

## 5. Hooks 机制：不改源码怎么插进去

inspect_ai 提供了一套官方插件机制，叫 `Hooks`。核心是三件事：

1. 继承 `inspect_ai.hooks.Hooks`，重写你关心的回调方法（比如 `on_sample_event`）。
2. 用 `@hooks(name=..., description=...)` 装饰一个返回你的 `Hooks` 子类的函数。
3. 在你的 Python 包的 `pyproject.toml` 里声明一个 `[project.entry-points.inspect_ai]` 入口，`uv pip install -e`（或 `pip install -e`）之后 inspect_ai 启动时会自动扫描发现，**不需要在 `eval()` 调用里传任何参数去启用**。

`inspect_trace` 本身就是这套机制的一个完整示例，代码不长，建议直接读：

```bash
$EDITOR /home/liuyingen/code/efficient-harness/inspect_trace/src/inspect_trace/_registry.py   # 入口：entry_points 指向这里
$EDITOR /home/liuyingen/code/efficient-harness/inspect_trace/src/inspect_trace/hooks.py         # TraceHooks 主类，回调怎么编排
```

如果你想写一个自己的最小 hook 练手（比如只是在每个 sample 开始/结束时打印一行日志），照着 `_registry.py` 的模式建一个新包即可，不需要碰 inspect_ai 或 `inspect_trace` 的任何一行代码。

Hooks 里最值得记住的几个回调（`inspect_trace` 全部用到了）：

| 回调 | 触发时机 | 用途 |
|---|---|---|
| `on_before_model_generate` | 每次模型调用**前**（含每次重试） | 观测/修改即将发出的请求 |
| `on_model_retry` | 一次调用失败、即将重试前 | 统计 retry 成本 |
| `on_sample_event` | 任意一个 event **完成**时（pending 状态的不会触发） | 最主要的观测点，能拿到完整的 `ModelEvent`/`ToolEvent` |
| `on_sample_end` | 一个 sample 彻底跑完（成功或最终失败） | 清理该 sample 的状态，避免长跑内存泄漏 |

---

## 6. 目标一实战：`inspect_trace` 怎么工作

### 6.1 它解决的四个缺口

inspect_ai 自带的 event 体系已经覆盖了 `efficient-harness.md` 目标一六个问题里的四个（上下文快照、invalid action 留痕、并行/等待/回滚、HTTP retry 留痕）。`inspect_trace` 补的是剩下四类**衍生事实**，一一对应一个模块：

| 模块 | 解决什么 |
|---|---|
| `prefill_diff.py` | 这一步的哪些消息在更早的步骤里已经出现过（重复 prefill），按工具名聚合"哪类 observation 被反复复用" |
| `segment_tokens.py` | 把模型输出按 reasoning / tool-call / 最终文本拆开估算 token 数，跟真实计费值并列展示、不覆盖 |
| `attempt_groups.py` | 把同一个逻辑请求的多次重试尝试串成一组，算出retry 浪费了多少等待时间 |
| `writer.py` | 上面三者的产出落盘成 JSONL，通过 `sample_uuid`/`model_event_uuid` 回链原始 `.eval` 日志，不拷贝、不修改原始内容 |

### 6.2 跑一遍，看真实产出

复用 `inspect_trace` 自带的端到端测试作为例子（它们本身就是很好的"如何用"的示范代码）：

```bash
cd /home/liuyingen/code/efficient-harness/inspect_trace
uv run pytest tests -v
```

或者手动跑一次、自己去看产出的 JSONL 长什么样：

```bash
mkdir -p /tmp/trace_demo && cd /tmp/trace_demo
INSPECT_TRACE_DIR=./.inspect_trace uv run --project /home/liuyingen/code/efficient-harness/inspect_trace python /path/to/hello_eval.py   # 用第 3 节的脚本
find .inspect_trace -type f
cat .inspect_trace/*/*/sample-*.jsonl | python3 -m json.tool
```

产出目录结构：

```
.inspect_trace/
  <run_id>/
    <eval_id>/
      sample-<sample_uuid>.jsonl   # 每行一条 prefill_diff / segment_tokens / attempt_group 记录
  _manifest.jsonl                  # eval_id -> 原始 .eval 日志路径的映射
```

### 6.3 一个值得记住的坑

实现 `attempt_groups.py` 时踩过一个坑，值得写下来：想用 `anyio.get_current_task()` 的返回值当"当前协程"的稳定标识（同一个协程内的多次重试应该映射到同一个 key），但 `anyio.get_current_task()` **每次调用都返回一个新分配的 wrapper 对象**，直接对这个 wrapper 取 `id()` 是不稳定的（只是恰好复用了刚释放的内存地址，测试跑起来时好时坏）。正确做法是取 wrapper 自带的 `.id` 属性（`anyio.get_current_task().id`），这个值才是跨调用稳定的。

这个坑的教育意义：**任何"看起来是稳定标识"的东西，如果没有明确的接口契约说它稳定，都需要写一个最小复现脚本验证一下，而不是想当然。** 复现脚本就两行：

```bash
uv run python -c "
import anyio

async def check():
    a, b = anyio.get_current_task(), anyio.get_current_task()
    print('id() 相等？', id(a) == id(b))   # False —— 陷阱
    print('.id 相等？', a.id == b.id)       # True —— 应该用这个

anyio.run(check)
"
```

---

## 7. 常用命令速查

```bash
cd /home/liuyingen/code/efficient-harness/inspect_trace

# 跑 inspect_trace 的测试
uv run pytest tests -v

# 用 CLI 图形界面看某次 eval 的完整轨迹
uv run inspect view --log-dir ./logs

# 把 .eval 日志转成 JSON 看
uv run inspect log dump path/to/xxx.eval | less

# 查看当前环境里已注册的 inspect_ai 扩展（含 inspect_trace 的 hooks）
uv run python -c "from importlib.metadata import entry_points; print(list(entry_points(group='inspect_ai')))"

# 用环境变量控制 inspect_trace 的输出目录（默认 ./.inspect_trace）
export INSPECT_TRACE_DIR=/path/to/somewhere

# 环境/依赖有变化时重新同步（读 uv.lock，保证可复现）
uv sync --extra dev

# 在别的目录下跑，但用这个项目的环境（uv run 默认只认当前目录下的 uv 项目）
uv run --project /home/liuyingen/code/efficient-harness/inspect_trace python your_script.py
```

---

## 8. 下一步

目标一已经实现并验证。往下走建议顺序（详细依据见 [`inspect_ai_roadmap.md`](./inspect_ai_roadmap.md)）：

1. 先读一遍 `inspect_trace` 产出的真实 JSONL（第 6.2 节），对着 `efficient-harness.md` 目标一的六个问题逐条核对，建立"这份数据能不能回答那个问题"的直觉。
2. 目标二（三层 profiling）里 token 层和 episode 层可以直接在 `inspect_trace` 产出的数据上做离线聚合，量级不大，适合下一步先做；model invocation 层（TTFT/KV cache 这类）需要额外接 vLLM，建议先做一个小规模验证 spike 再决定要不要投入。
3. 目标三（标准化干预接口）设计时，建议参考 pydantic-ai 的 `AbstractCapability`（`wrap_*` 系列）设计思路，细节见 `framework-selection.md`。

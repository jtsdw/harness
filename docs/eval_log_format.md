# `.eval` 文件格式详解

`.eval` 文件本质是一个 `EvalLog` 对象序列化后的结果，定义在 `inspect_ai/src/inspect_ai/log/_log.py`。本文逐字段拆解，全部对照真实跑出来的一份数据（`runs/compare/deepseek_gsm8k/logs/*.eval`，DeepSeek-chat 跑 GSM8K 的一条样本）——不是照搬文档瞎编，每个字段后面标的都是真实值。

查看方式见 `inspect_ai_quickstart.md` 第 4 节：`inspect log dump` 转 JSON、`inspect view` 图形界面、或 Python API `read_eval_log`。

## 整体骨架

```
EvalLog
├── version, status          # 文件格式版本、这次跑得怎么样
├── eval        (EvalSpec)   # 这次评测"是什么"——身份、配置、数据集、模型
├── plan        (EvalPlan)   # 用了哪些 solver、怎么串起来的
├── results     (EvalResults)# 打分结果汇总
├── stats       (EvalStats)  # 耗时、token 用量汇总
├── error                    # 整个 eval 中途炸了才有值
├── samples     (list[EvalSample])  # 每条数据的完整轨迹，这是最大、最重要的部分
└── reductions               # 多 epoch 场景下打分归约结果
```

## 1. 顶层字段

| 字段 | 真实值 | 含义 |
|---|---|---|
| `version` | `2` | 日志文件格式版本号，不是你评测跑了几次 |
| `status` | `"success"` | 整个 eval 的最终状态：`started`/`success`/`error`/`cancelled` |
| `error` | `null` | 只有 `status=="error"` 时才有值，装的是导致整个 eval 中断的异常 |
| `invalidated` | `false` | 有没有 sample 被标记失效（比如手动编辑过日志） |
| `tags`/`metadata` | `[]`/`{}` | 你自己给这次评测打的标签和元数据 |

## 2. `eval`（EvalSpec）——这次评测的"身份证"

这部分回答"这是哪次评测、用什么数据、什么模型、什么配置"：

```json
{
  "eval_id": "ZuTRxKzas5RtuqCrSC4r2A",   // 这次 task 执行的全局唯一 id
  "run_id": "Dy6HjSmXRWWbdgjTGzwPws",     // 一次 eval()/eval_set() 调用的 id（可能包含多个 task）
  "task": "inspect_evals/gsm8k",         // task 名字
  "task_version": 2,                      // task 定义本身的版本号
  "task_args": {"fewshot": 10, "fewshot_seed": 42, "shuffle_fewshot": true},  // 跑这个 task 时用的完整参数（含默认值）
  "task_args_passed": {},                 // 你自己显式传的参数（这次全用默认值，所以是空）
  "dataset": {
    "name": "openai/gsm8k",
    "samples": 1319,                      // 数据集总共有多少条
    "sample_ids": ["gsm8k_4b7e54d8", ...] // 这次实际跑的是哪几条（因为我们 --limit 5）
  },
  "model": "openai/deepseek-chat",        // 用的模型字符串
  "model_generate_config": {},            // temperature/max_tokens 这类生成参数
  "config": {"limit": 5, "epochs": 1, "fail_on_error": true, ...},  // CLI/API 传的运行时配置
  "revision": {"type": "git", "commit": "1ea01a9e1", "dirty": true},  // 跑这次评测时 inspect_ai 自己的代码版本（连 dirty 状态都记了）
  "packages": {"inspect_ai": "0.3.252...", "inspect_evals": "0.16.0"},  // 关键依赖包版本
  "scorers": [{"name": "match", "options": {"numeric": true}, "metrics": [...]}]  // 用了哪个 scorer、什么参数
}
```

这一段的设计意图是"可复现性凭证"——`revision`+`packages`+`task_args`+`dataset.sample_ids` 加起来足够精确复现出"这次到底是怎么跑的"，哪怕是 `dirty: true`（本地有未提交改动）这种情况也如实记录，不会假装是干净状态。

## 3. `plan`（EvalPlan）——怎么跑的

```json
{
  "name": "plan",
  "steps": [
    {"solver": "system_message", "params": {"template": "...少样本示例..."}},
    {"solver": "prompt_template", "params": {"template": "Solve the following math problem..."}}
  ],
  "config": {}
}
```

`steps` 是这次评测串起来的 solver 链条，每一步记录了 solver 名字和它被实例化时用的完整参数——GSM8K 这里是"先塞系统消息（含十个 few-shot 示例）→ 再套一层 prompt 模板"，这两步就是把原始数据集的一道题目变成实际发给模型的 prompt 的过程。`finish` 字段（这次是 `None`）如果有值，是"不管前面 solver 链怎么走，最后一定会跑的一步"。

## 4. `results`（EvalResults）——打分结果

```json
{
  "total_samples": 5,
  "completed_samples": 5,        // 没出错正常跑完的样本数，正常应该等于 total_samples
  "scores": [{
    "name": "match", "scorer": "match",
    "scored_samples": 5, "unscored_samples": 0,
    "metrics": {
      "accuracy": {"name": "accuracy", "value": 1.0},
      "stderr": {"name": "stderr", "value": 0.0}
    }
  }]
}
```

`total_samples` 和 `completed_samples` 不相等时说明有 sample 中途失败了（比如触发了 `--fail-on-error`）。`scores` 是数组是因为一次评测可以挂多个 scorer，每个 scorer 下面又能算多个 metric（比如这里的 `accuracy`/`stderr`）。

## 5. `stats`（EvalStats）——耗时和用量汇总

```json
{
  "started_at": "2026-08-02T07:26:08+00:00",
  "completed_at": "2026-08-02T07:26:11+00:00",
  "model_usage": {
    "openai/deepseek-chat": {"input_tokens": 9955, "output_tokens": 454, "total_tokens": 10409, "reasoning_tokens": 0}
  },
  "role_usage": {}
}
```

这是**整个 eval 跑下来**、所有 sample 加总的模型用量（目标一/目标二分析里用的 episode 层数据就是从这里 + 每条 sample 自己的那一份算出来的）。`role_usage` 是按 model role（比如 grader 模型 vs 主模型）拆分的用量，本次没用到多角色所以是空。

## 6. `samples`（`list[EvalSample]`）——真正的主体

每条 sample 是一个完整对象，字段远比 summary 视图丰富：

| 字段 | 这条样本的真实值 | 含义 |
|---|---|---|
| `id` | `"gsm8k_2bcc778b"` | 数据集里这条数据的 id |
| `epoch` | `1` | 第几次重复跑这条数据（多 epoch 场景下同一条数据会跑好几遍） |
| `uuid` | `"AwC442S2rZZeqEzXJ5WMQH"` | 这次**执行**的全局唯一 id（`inspect_trace` 用它当 join key，跟 `id` 不是一回事——同一条数据换个 epoch 跑，`id` 一样但 `uuid` 不一样） |
| `target` | `"540"` | 数据集给的标准答案 |
| `messages` | 3 条：`system`/`user`/`assistant` | 这条样本**最终定格**的完整对话历史（跟下面 events 里 `model` 事件的 `input` 不是一回事——见下） |
| `output` | `ModelOutput` | 模型最后一次生成的完整输出（多轮场景下是最后一轮） |
| `scores` | `{"match": {"value": "C", "answer": "540", "explanation": "...", "history": []}}` | 打分结果；`value` 是分数（这里 `"C"` 是 correct 的缩写编码），`answer` 是从输出里提取出的答案，`explanation` 是打分依据 |
| `model_usage` | `{"openai/deepseek-chat": {...}}` | 只属于这一条样本的 token 用量 |
| `started_at`/`completed_at`/`total_time`/`working_time` | `1.387`秒 / `1.217`秒 | `total_time` 是墙钟时间，`working_time` 是刨除了排队/等待信号量之后"真正在干活"的时间——这俩不等的差值就是纯粹的等待开销 |
| `error_retries` | `null` | 这条样本本身如果被重跑过（sample 级重试，不是 HTTP 重试），会在这留痕 |
| `attachments` | `{...}` | 见下面"attachment 去重" |
| `events` | 20 条事件 | **`inspect_trace` 真正消费的原始数据源**，见下一节 |

**一个容易搞混的点**：`sample.messages` 是这条样本**最终**的消息列表，而 `inspect_trace` 分析的其实是 `events` 里**每一次** `ModelEvent.input`——同一条样本如果有 5 轮工具调用，`sample.messages` 只有 1 份（最终态），但 `events` 里会有 5 条 `ModelEvent`，每条都有自己那一刻的 `input` 快照。做"重复 prefill 检测"用的是后者，不是前者。

## 7. `events`——逐步发生了什么

这条 GSM8K 样本一共 20 个事件，类型分布：`span_begin`/`span_end`（成对出现，标记一段执行的起止）、`sample_init`（样本初始化）、`state`（消息历史发生了变化）、`model`（一次模型调用）、`score`（打分）。每个事件都继承了这几个公共字段（定义在 `event/_base.py`）：

| 公共字段 | 含义 |
|---|---|
| `uuid` | 这个事件自己的唯一 id |
| `span_id` | 这个事件属于哪个 span（span 是"一段有起止的执行"的分组，比如"这是 system_message 这个 solver 步骤内发生的"） |
| `timestamp` | 发生的绝对时间 |
| `working_start` | 这个事件发生时，"工作时间"计时器走到了多少（用来算耗时，不受排队等待影响） |
| `pending` | 是不是还没执行完（`inspect_trace` 只处理 `pending` 不为 true 的，也就是已完成的事件） |

### `span_begin` / `span_end`

```json
{"event": "span_begin", "id": "Zsyd7BTq4oRa7QQnigcEKb", "parent_id": "k9rMrQj67r3bDYjasaC4EU", "type": "solver", "name": "system_message"}
```

标记"从这里到对应的 `span_end` 之间，是 `system_message` 这个 solver 在跑"。`parent_id` 让 span 能嵌套成一棵树（`solvers` 大 span 底下套着 `system_message`/`prompt_template` 两个小 span）。这就是分析"并行/等待"结构（目标一 Q6）时会用到的树状结构。

### `state`

```json
{"event": "state", "changes": [
  {"op": "replace", "path": "/messages/0/content", "value": "attachment://1c06524818e1a522a54b9aaa3380e298", "replaced": "James decides to run..."}
]}
```

不是完整快照，是**JSON Patch 格式的增量变更**——记录 `system_message` 这个 solver 把消息内容从原始文本换成了什么。要重建某一时刻的完整状态，得从 `sample_init` 的初始状态开始，把 `state` 事件按顺序应用下去。

### `model`——最关键的一类，`inspect_trace` 全部三个检测器都靠它

```json
{
  "event": "model",
  "model": "openai/deepseek-chat",
  "input": [ {消息1}, {消息2} ],       // 这次调用发送的完整消息列表——重复 prefill 检测的原始数据
  "tools": [],                          // 这次调用带的工具定义——之前修的那个 bug 就是这个字段
  "output": {
    "choices": [{"message": {...}, "stop_reason": "stop"}],
    "usage": {"input_tokens": 1969, "output_tokens": 68, "reasoning_tokens": 0},  // 真实计费值
    "time": 1.2011209893971682          // 这次调用本身耗时
  },
  "call": {"request": {...}, "response": {...}}  // 发给 provider 的原始请求 + provider 原始返回，一字不改
}
```

`call.request`/`call.response` 是"黑盒之外"的最后一道保险——就算 inspect_ai 自己对字段的解析理解有偏差，这两个字段保留了 provider 真实收发的原始 JSON，可以拿去交叉核对。这条真实数据里 `call.request` 长得像 OpenAI 的 Responses API 格式（`role: "developer"`、`type: "message"`）而不是经典 Chat Completions 格式——这是 inspect_ai 的 `openai` provider 默认走的接口，不影响分析，只是说明"provider 层的具体协议格式"和"inspect_ai 对外统一暴露的 ChatMessage 抽象"是两层不同的东西。

### `score`

打分这一步产生的事件，内容跟 `sample.scores` 里的结构一致，只是多了时间戳信息，标记"打分这件事是什么时候发生的"。

## 8. Attachment 去重机制

dump 出来的 JSON 里内容经常是 `"attachment://1c06524818e1a522a54b9aaa3380e298"` 这种占位符，而不是真实文本：

```json
"attachments": {
  "1c06524818e1a522a54b9aaa3380e298": "Solve the following math problem step by step...",
  ...
}
```

这是内容寻址去重：同一段长文本（比如 few-shot 示例、系统提示词）如果在很多个 event 里重复出现，只在 `attachments` 字典里存一份，各处用哈希引用它，避免日志文件因为重复文本而膨胀。默认 `inspect log dump` 不会帮你展开这些引用，要看真实内容得加 `--resolve-attachments full`，或者用 Python API 读时传 `resolve_attachments=True`。

## 一句话总结怎么用

- 想看"这次评测整体表现"→ 看 `results`/`stats`
- 想看"某条数据具体输入输出对不对"→ 看 `samples[i].messages`/`output`/`scores`
- 想看"逐步到底发生了什么、模型每一步实际处理了什么上下文"→ 看 `samples[i].events`，尤其是 `event=="model"` 的那些——这正是 `inspect_trace` 三个检测器唯一读取的数据源。

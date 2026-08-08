# vLLM 单次请求背景知识：从"为什么是 HTTP"到响应体每个字段

这篇不是需求 B 的操作文档，是背景知识——用来建立"一次 vLLM 请求到底经历了什么"的概念，方便理解需求 B 一直
在采集的那些指标（TTFT、queue time、KV cache…）具体对应的是请求生命周期里的哪一段。全文围绕
2026-08-08 在真实 NSCC 节点（`vllm-0.26.0-8cfe525c`）上抓到的一个真实响应展开，字段含义逐条核对自 vLLM
官方源码（`vllm/entrypoints/openai/chat_completion/protocol.py`），不是凭印象写的。

用作例子的真实响应：

```json
{
    "id": "chatcmpl-93c860bc0e176e70",
    "object": "chat.completion",
    "created": 1786163366,
    "model": "Qwen/Qwen3-32B",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>\nOkay, the user wants me to say hello in one short sentence. Let me think about",
                "refusal": null, "annotations": null, "audio": null, "function_call": null, "reasoning": null
            },
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null,
            "token_ids": null,
            "routed_experts": null
        }
    ],
    "service_tier": null,
    "system_fingerprint": "vllm-0.26.0-8cfe525c",
    "usage": {"prompt_tokens": 15, "total_tokens": 35, "completion_tokens": 20, "prompt_tokens_details": null},
    "prompt_logprobs": null,
    "prompt_token_ids": null,
    "prompt_text": null,
    "kv_transfer_params": null,
    "ec_transfer_params": null,
    "metrics": {
        "time_to_first_token_ms": 35.1859099464491,
        "generation_time_ms": 158.43987104017287,
        "queue_time_ms": 0.022810068912804127,
        "mean_itl_ms": 8.33894058106173,
        "tokens_per_second": 103.29203011133029
    }
}
```

## 为什么是 URL/HTTP 请求

vLLM 有两种运行形态：**离线批处理**（`vllm.LLM` 这个 Python 类，跑一个脚本、进程内直接调用，跑完退出）和
**在线服务**（`vllm serve`，起一个长期运行的 HTTP 服务进程，暴露 OpenAI 兼容的 `/v1/chat/completions` 等
接口）。这个项目用的是后者——`nscc_model_server/scripts/serve.sh` 启动的就是这样一个持续运行的服务进程，
之后所有的请求（不管是 `curl`、`inspect_ai` 的 benchmark、还是 `run_b5_matrix.sh`）都是对着同一个已经跑
起来的进程发 HTTP 请求，不是每次都重新起一个新进程。

选 HTTP 而不是"每次调用都跑一个新脚本"，核心原因是**模型加载本身极其昂贵，不能每次请求都重来一遍**——这
不是理论推测，是这次在 NSCC 上亲眼看到的真实数字：光是 32B 模型权重加载就花了 113.5 秒，加上 EAGLE-3 草
稿头、torch.compile 编译，从进程启动到能真正处理请求，总共要好几分钟。如果每次请求都要经历这个过程，完
全不可用。HTTP 服务把"加载模型"（只做一次，服务启动时）和"处理一次请求"（每次几十到几百毫秒）彻底解耦：
客户端只需要知道一个 URL（比如 `http://localhost:8000/v1`），发一段文本进去、等 JSON 答案出来，不需要关
心模型是怎么加载、放在哪张卡上的。

用 OpenAI 兼容协议还有一个直接的好处：这是事实上的行业标准，意味着任何已经存在的 OpenAI 客户端代码（包括
`inspect_ai` 自带的 `openai-api` provider、这个项目写的所有 benchmark 脚本、甚至一个裸 `curl`）都能直接
指向这个 URL 使用，不需要为 vLLM 专门写一套客户端逻辑——`local-model-server`/`nscc_model_server` 这两个
项目从头到尾都没有自己写过请求/响应的解析代码，全靠这一点。

## 为什么可以持续运行、还能同时处理多个请求：continuous batching

如果是最朴素的批处理思路——攒一批请求、一起跑完、再处理下一批——批里只要有一个请求特别长，其他短请求也
要陪着等它跑完才能拿到结果。vLLM 的核心设计不是这样：它是**连续批处理（continuous batching）**，不以"整
个请求"为调度单位，而是以"生成一个 token 需要的一步计算"为单位，每一步都重新决定这一步要处理哪些请求。
服务进程里有一个持续运行的调度循环（EngineCore），每一步都在做：

- 看当前有哪些请求在排队（waiting）、哪些正在处理中（running）；
- 根据 KV cache 还剩多少显存空间，决定这一步要不要接纳一个新请求做 prefill（把它整个 prompt 一次性算完，
  产出第一个 token），或者继续推进某些已经在 decode 的请求（每个只往前推进一个 token）。

这就是为什么一个短请求不会被前面一个长请求"卡住"——它们在 token 粒度上交替处理，不是排队等待对方彻底跑
完。这也是为什么需要一个专门的显存管理机制（PagedAttention）：每个正在进行中的请求都占着一块显存存自己
的 KV cache，而且这块显存会随着生成的 token 数不断增长，调度器要能动态地给不同请求分配/释放显存页，而不
是像传统做法那样为每个请求预先留出一整块连续的最坏情况显存。

这次在 NSCC 上启动日志里看到的：

```
Available KV cache memory: 4.71 GiB
GPU KV cache size: 18,976 tokens
Maximum concurrency for 16,384 tokens per request: 1.16x
```

就是这个调度器手里握着的真实"预算"——这几个数字直接决定了需求 B5 要测的并发（concurrency=4/8）到底跑不
跑得动，不是一个抽象概念，是这台服务器当前配置下的硬限制。

## 一次请求的完整生命周期（对照上面的真实数据）

1. 客户端发 HTTP POST 到 `/v1/chat/completions`，带上 `messages`/`max_tokens`/`temperature` 等参数。
2. 请求进服务器后先做 **tokenize**：把 `messages` 用模型自己的聊天模板拼成一段文本，再转成 token id 序
   列——这次的 `usage.prompt_tokens=15` 就是这一步的产出。
3. 请求进入调度队列，等待被调度器选中——这段等待时间就是 `metrics.queue_time_ms`（这次是 0.023ms，几乎
   为零，因为只有这一个请求在跑，没人跟它抢；一旦并发上去，这个数字会有意义得多，这正是需求 B5 要专门测
   的东西）。
4. **Prefill 阶段**：调度器把整个 prompt（15 个 token）一次性喂给模型，算出对应的 KV cache，同时产出第
   一个输出 token——从进队列到拿到这第一个 token 的总耗时就是 `metrics.time_to_first_token_ms`（35ms，
   包含了上面的排队时间）。
5. **Decode 阶段**：从第二个 token 开始，之后每一步只往前推进这一个 token（调度器同一时刻可能在穿插处
   理其他请求），直到碰到停止条件——这次是达到了 `max_tokens=20`，所以在模型还在"思考"（`<think>` 标签
   还没结束）的时候就被截断了，对应 `finish_reason="length"`。这整段耗时是 `metrics.generation_time_ms`
   （158ms），期间每两个连续 token 之间的平均间隔是 `metrics.mean_itl_ms`（inter-token latency，8.3ms）。
6. 请求完成，服务器把结果序列化成一整块 JSON，通过同一条 HTTP 连接返回——这次请求没有开 `stream: true`，
   所以是一次性拿到完整响应，不是像交互式对话界面那样逐字往外吐。
7. `metrics.tokens_per_second`（≈103）是 vLLM 自己算好放进响应里的：输出 token 数 ÷ decode 耗时（20 ÷
   0.15844s），不需要客户端自己再拿 `completion_tokens` 除 `generation_time_ms` 重新算一遍。

## 逐字段详解

来源统一是 vLLM 源码 `vllm/entrypoints/openai/chat_completion/protocol.py`（2026-08-08 读的 `main` 分
支），不是猜的；"vLLM 扩展"指标准 OpenAI 协议里没有、vLLM 自己加的字段。

**顶层字段**

| 字段 | 含义 |
|---|---|
| `id` | 这次请求的唯一标识，`chatcmpl-` 前缀 + 随机 uuid，标准 OpenAI 格式 |
| `object` | 固定值 `"chat.completion"`，标记这条 JSON 是什么类型的响应（跟流式场景下的 chunk 类型区分开） |
| `created` | 响应生成时刻的 Unix 时间戳 |
| `model` | 实际服务这次请求的模型名 |
| `choices` | 生成结果数组，默认长度 1（除非请求里设了 `n>1`，让模型对同一个 prompt 生成多个候选回复），字段详见下表 |
| `service_tier` | OpenAI API 自己的字段（服务等级），vLLM 不使用，恒为 `null` |
| `system_fingerprint` | 标准 OpenAI 字段本意是"标记后端配置是否变化"，vLLM 把它挪用来标注"是哪个版本+commit 处理的这次请求"（这次是 `vllm-0.26.0-8cfe525c`） |
| `usage` | Token 用量：`prompt_tokens`/`completion_tokens`/`total_tokens` 是标准字段；`prompt_tokens_details.cached_tokens` 记录 prompt 里有多少 token 命中了 prefix cache——这次是 `null`，因为这是独立的单次请求，前面没有可复用的历史上下文 |
| `prompt_logprobs` | vLLM 扩展，prompt 每个 token 的对数概率，要请求里显式传 `prompt_logprobs` 才会填，这次没开 |
| `prompt_token_ids` | vLLM 扩展，输入侧的原始 token id 列表，要传 `return_token_ids` 才会填，这次没开 |
| `prompt_text` | vLLM 扩展，聊天模板渲染后的完整 prompt 文本，要传 `return_prompt_text` 才会填，这次没开 |
| `kv_transfer_params` | 给"分离式 prefill/decode 部署"（P/D disaggregation，把 prefill 和 decode 放到不同 GPU/节点上跑）传 KV cache 元数据用的，这次没用这种部署方式，是 `null` |
| `ec_transfer_params` | 跟上面类似，但是给"encoder cache 分离式部署"用的（涉及带独立视觉/编码器组件的模型），同样是 `null` |
| `metrics` | 就是需求 B1 要采集的逐请求指标核心来源，要服务端启动时传 `--enable-per-request-metrics` 才会填（见 `serving_observability_b_howto.md`） |

**`choices[0]` 内部**

| 字段 | 含义 |
|---|---|
| `index` | 这是第几个候选回复（配合 `n>1` 用），这次只有一个，`index=0` |
| `message.role` | 固定 `"assistant"` |
| `message.content` | 真正生成的文本。注意这次里面带了 `<think>...` 标签——Qwen3 是"推理模型"，默认会先输出一段思考过程再给最终答案；这次因为 `max_tokens=20` 太小，还没走出思考阶段就被截断了 |
| `message.refusal`/`annotations`/`audio`/`function_call` | 标准 OpenAI 字段占位，这次都没用到 |
| `message.reasoning` | 有些 provider 会把推理过程单独放这个字段（跟塞进 `content` 里的 `<think>` 标签是两种不同呈现方式），这次没有拆分，是 `null` |
| `logprobs` | 要请求里设 `logprobs: true` 才会填每个 token 的对数概率，这次没开 |
| `finish_reason` | 标准 OpenAI 字段，为什么停止生成——常见取值 `"stop"`（遇到停止条件/eos）、`"length"`（达到 `max_tokens`，这次的情况）、`"tool_calls"`、`"content_filter"` |
| `stop_reason` | vLLM 扩展，`finish_reason="stop"` 时具体是哪个停止 token/字符串触发的——这次是被 `max_tokens` 截断，不是 `stop`，所以是 `null` |
| `token_ids` | vLLM 扩展，输出侧的原始 token id 列表，方便 agent 场景下精确追踪具体 token，要传 `return_token_ids`，这次没开 |
| `routed_experts` | 给 MoE（混合专家）模型用的，记录每个 token 实际路由到了哪些专家（base64 编码的 `.npy` 数据）；Qwen3-32B 是稠密模型不是 MoE，恒为 `null` |

## 跟需求 B 的对应关系

这批字段里，`metrics` 就是需求 B1 逐请求指标（`vllm_per_request_metrics.py`）的真实数据来源；
`usage.prompt_tokens_details.cached_tokens` 是需求 B4"prompt/generated/cached token"里 cached token 
的真实来源（这次是 `null`，等真的有多轮对话/重复上下文时才会有值）；`kv_transfer_params`/
`ec_transfer_params` 这次用不上，但如果以后考虑分离式部署，这就是真实的元数据入口。

## 相关文档

- [`serving_observability_b_howto.md`](./serving_observability_b_howto.md) —— 需求 B 怎么用这些字段、
  跑什么脚本、现场验证进度
- `inspect_trace/src/inspect_trace/vllm_per_request_metrics.py` —— 读取/解析这些字段的实际代码

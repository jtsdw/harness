# inspect_trace

Goal-1 (完整执行轨迹) extension package for [inspect_ai](https://inspect.aisi.org.uk/), built entirely
on the public `Hooks` extension mechanism — no changes to `inspect_ai` source.

inspect_ai's own event log already captures a faithful per-step snapshot of agent execution
(`ModelEvent.input`, tool calls, retries, parallelism, interrupts). What it does not expose are six
derived facts that efficiency research needs, grouped by which of `efficient-harness.md`'s Goal-1
requirements (需求一-四) they satisfy:

**需求一 (complete token-level recording)**

1. **`segment_tokens`** — an approximate split of a model's output tokens across reasoning /
   tool-call / server-tool-use / final-text content, cross-checked against (never overwriting) the
   real billed usage inspect_ai already records.
2. **`token_attribution`** — a stateless join of `prefill_diff` (input side) and `segment_tokens`
   (output side) into the full system-template / tool-schema / conversation / reasoning /
   tool-calling / final-response breakdown, with real vs. estimated fields clearly separated.

**需求二 (new vs. reused context per step) and 需求一's system-template attribution**

3. **`prefill_diff`** — which messages in a step's `ModelEvent.input` already appeared in an earlier
   step (vs. genuinely new context), tagged with `content_category` (`system_template` vs.
   `conversation`) and `tool_call_id`, plus which tools' observations are the ones getting re-sent.
   This also covers `event.tools` (the tool schema definitions themselves): a real multi-turn
   benchmark run showed a ~30-tool, ~23K-character schema list being resent byte-for-byte on every
   call in a 41-step trajectory — a repeated-prefill source invisible to message-only tracking, so
   tool schemas get their own new/reused classification, separate from message counts (see
   `/home/liuyingen/code/efficient-harness/docs/goal1_real_benchmark_findings.md`).

**需求三 (execution topology)**

4. **`execution_topology`** — one record per sample, grouping `ToolEvent`s into stages by their
   parent `ModelEvent`, and reporting whether each stage was genuinely observed to run in parallel
   (overlapping `[timestamp, completed]` windows, not inspect_ai's declared `ToolDef.parallel`),
   `model_waiting_for_tool_seconds`/`tool_waiting_for_model_seconds` (inferred), and
   `tool_semaphore_wait_seconds` (inspect_ai's own real value). Deliberately has no rollback field:
   inspect_ai has no mechanism that reverts `TaskState` and discards a branch (`fork()` runs every
   branch to completion; `BranchEvent`/`AnchorEvent` are timeline-viewer markers; `CheckpointEvent` is
   crash-resume persistence) — the closest real analog is a retry, already in `attempt_group`.

**需求四 (action-parsing and observation write-back tracing)**

5. **`action_parsing`** — one record per `ToolEvent`, capturing tool-call parsing/validation failures
   (`error_type`/`error_message`, real values from inspect_ai's own `ToolCallError`) and
   `tool_call_id` as the join key forward into the next step's `prefill_diff` message.

**Retry grouping (supports 需求三's "重试" and 需求一)**

6. **`attempt_groups`** — which of the several `ModelEvent`s inspect_ai logs for a single logical
   request are retries of each other, how many attempts it took, and how much wait time was spent on
   backoff.

## Goal 2: three-layer cost profiling

Built on top of everything above, not a separate mechanism — see
`/home/liuyingen/code/efficient-harness/docs/goal2_design.md` for the full design and
`goal2_real_validation_findings.md` for real-data validation results.

**Model invocation layer — `vllm_metrics.py`, the one module here with live runtime collection.**
Scrapes vLLM's own `/metrics` Prometheus endpoint once before and once after each model call
(via the same `on_before_model_generate`/`on_sample_event` Hooks everything else uses), and takes
the delta of the `time_to_first_token_seconds`/`time_per_output_token_seconds`/
`e2e_request_latency_seconds` histograms to get real TTFT/ITL/e2e-latency for that specific call —
exact whenever `MAX_CONNECTIONS=1` (the histogram's `_count` delta is exactly 1), and honestly
flagged `attribution_confidence: "ambiguous"` otherwise. Only produces records when the target
model is actually a local vLLM server reachable at `INSPECT_TRACE_VLLM_METRICS_URL` (default
`http://localhost:8000/metrics`); silently produces nothing against a hosted model, by design.

**Token layer and episode layer — `analysis/{token_layer,episode_layer,pricing}.py`, pure offline
analysis, no new collection.** These read the six Goal-1 records above (already on disk) plus the
real `.eval` log and aggregate them per-episode: `token_layer.summarize_run(trace_dir)` for
per-episode token totals broken down by pipeline stage and by new/reused; `episode_layer.
summarize_run(trace_dir)` for end-to-end latency, `total_busy_seconds` vs. naive exclusive-time sum
(the difference being real observed concurrency savings), LLM/tool call counts, retries, success
(via `inspect_ai.scorer.value_to_float()`, not hardcoded to any one scorer), and cost (via
`pricing.py`'s explicit model-name → price table — deliberately has no invented prices for models
never actually run against).

## Install

```bash
pip install -e /home/liuyingen/code/efficient-harness/inspect_trace
```

Installing the package registers its `Hooks` subclass via a standard `inspect_ai` setuptools entry
point — no CLI flag or `eval()` argument is needed to enable it. Any `eval()` run in the same
environment will produce derived-fact JSONL files alongside (not inside) the normal `.eval` log.

## Output

```
${INSPECT_TRACE_DIR:-.inspect_trace}/
  <run_id>/
    <eval_id>/
      sample-<sample_uuid>.jsonl   # one line per prefill_diff / segment_tokens / attempt_group /
                                   # token_attribution / execution_topology / action_parsing /
                                   # vllm_metrics record
      _manifest.jsonl              # eval_id -> original .eval log location
```

Every record links back to the original `.eval` log via `sample_uuid` and `model_event_uuid` — the
original log is never read, copied, or modified.

See `/home/liuyingen/code/efficient-harness/docs/inspect_ai_roadmap.md` for how this fits into the larger four-goal plan,
and `/home/liuyingen/code/efficient-harness/docs/goal1_real_benchmark_findings.md` for what a real multi-turn benchmark
run (BFCL `multi_turn_base`, DeepSeek-chat) showed vs. the toy-example tests, including the tool-schema
gap above.

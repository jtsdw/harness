# tau2_adapter

This directory is the shared, backend-independent tau2 integration. Changes
here are public project code and should be committed and pulled onto NSCC. It
does not own a model backend: local and NSCC runners supply their own
OpenAI-compatible endpoint and deployment lifecycle around this adapter.

Experimental integration between [tau2-bench](https://github.com/sierra-research/tau2-bench)
(a dual-control agent/user-simulator benchmark, not built on inspect_ai) and this project's
`inspect_trace` profiling stack.

Drives a real tau2-bench simulation (`Orchestrator`, `Environment`, `UserSimulator`,
`evaluate_simulation` -- all reused unmodified from the `tau2` package as libraries, not
reimplemented) from inside an inspect_ai `Task`/`Solver`, so that the agent-under-test's model
calls go through `inspect_ai.model.generate()` and `inspect_trace`'s Hooks fire for them.

The adapter resolves environments and task sets through tau2's registry. Its
single Inspect task accepts `domain`, `task_set`, and `task_split` arguments, so
the same bridge runs mock, airline, retail, telecom, and telecom-workflow.

```bash
inspect eval src/tau2_adapter/task.py@tau2 \
  -T domain=airline -T task_split=base \
  --model tau2-agent-vllm/vllm/<served-model>
```

`task_split=auto` selects `base` when the task set provides it and all tasks
otherwise. Use `task_split=all` to explicitly select the complete task file.
Legacy registered sets such as `telecom_small` and `telecom_full` are also
supported through `task_set` without adapter changes.

## Reproducible setup

From `tau2_adapter/`, run:

```bash
./scripts/setup_tau2_bench.sh
```

The setup script fetches tau2-bench into the ignored repository-local path
`../.deps/tau2-bench`, checks out commit
`a1e85084a3960281cb06997594133e8f39ea42a7`, applies the tracked compatibility
patch, and synchronizes the adapter's Python 3.12 environment. Pass
`--with-baseline` only when the native tau2 CLI environment is also needed. It
uses a partial,
sparse checkout containing only the upstream Python source; it does not fetch
or vendor benchmark domain data or result archives.
`TAU2_BENCH_REPO` may point to an internal mirror, but the commit and setup path
are fixed because they are part of the checked-in lock configuration. Run
scripts require `TAU2_DATA_DIR` to point to a separately provisioned data tree
from the same pinned tau2-bench snapshot, for example:

```bash
TAU2_DATA_DIR=/shared/benchmarks/tau2-bench/data \
  NUM_TASKS=1 ./scripts/run_adapter.sh
```

Benchmark data is intentionally outside this repository and must never be
staged or pushed with the adapter code.

The model server is intentionally separate. After an OpenAI-compatible server
is available, configure `VLLM_BASE_URL`, `VLLM_API_KEY`, and `MODEL_NAME`, then
run `scripts/run_adapter.sh` with `TAU2_DATA_DIR`. NSCC should provide
these values and start the backend from its own PBS wrapper. By default, the agent-under-test, user
simulator, and NL-assertion judge use that served model; `USER_MODEL_NAME`,
`JUDGE_MODEL_NAME`, `TAU2_USER_LLM_ARGS`, and `TAU2_JUDGE_LLM_ARGS` can override
the latter two. The script also derives `INSPECT_TRACE_VLLM_METRICS_URL` from
the endpoint so the shared probe captures backend metrics. Native tool calling
is the default; pass `emulate` only for the historical comparison. Set
`TAU2_EMPTY_RESPONSE_RETRIES` to control retries for empty assistant responses.
The former NSCC name `TAU2_AGENT_MAX_EMPTY_RETRIES` remains accepted with a
deprecation warning so existing remote jobs do not silently change behavior.
Each run writes a small `run-manifest.txt` beside its logs so the code, data,
model, and task selection can be recovered without a hand-maintained changelog.

Use `scripts/run_full_core.sh` for the 402-task core selection. It writes a
per-domain status table, validates the newest Inspect log against the expected
sample count, continues after a failed domain, and resumes by skipping domains
that already validate. `../scripts/nscc_tau2_qwen36_job.sh` is the corresponding
parameterized PBS/backend template.

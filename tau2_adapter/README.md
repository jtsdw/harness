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
The optional banking-knowledge domain is also registry-addressable, but still
requires the corresponding tau2 dependencies and `TAU2_ENV_ARGS` retrieval
configuration.

## Reproducible setup

From `tau2_adapter/`, run:

```bash
./scripts/setup_tau2_bench.sh
```

The setup script fetches tau2-bench into the ignored repository-local path
`../.deps/tau2-bench`, checks out commit
`a1e85084a3960281cb06997594133e8f39ea42a7`, applies the tracked compatibility
patch, and synchronizes both Python 3.12 environments. It uses a partial,
sparse checkout containing only the upstream Python source; it does not fetch
or vendor benchmark domain data or result archives.
`TAU2_BENCH_REPO` may point to an internal mirror, but the commit and setup path
are fixed because they are part of the checked-in lock configuration. Run
scripts require `TAU2_DATA_DIR` to point to a separately provisioned data tree
from the same pinned tau2-bench snapshot, for example:

```bash
TAU2_DATA_DIR=/shared/benchmarks/tau2-bench/data \
  ./scripts/run_adapter.sh native
```

Benchmark data is intentionally outside this repository and must never be
staged or pushed with the adapter code.

The model server is intentionally separate. After an OpenAI-compatible server
is available, configure `VLLM_BASE_URL`, `VLLM_API_KEY`, and `MODEL_NAME`, then
run `scripts/run_adapter.sh native` with `TAU2_DATA_DIR`. NSCC should provide
these values and start the backend from its own PBS wrapper. By default, the agent-under-test, user
simulator, and NL-assertion judge use that served model; `USER_MODEL_NAME`,
`JUDGE_MODEL_NAME`, `TAU2_USER_LLM_ARGS`, and `TAU2_JUDGE_LLM_ARGS` can override
the latter two. The script also derives `INSPECT_TRACE_VLLM_METRICS_URL` from
the endpoint so the shared probe captures backend metrics.

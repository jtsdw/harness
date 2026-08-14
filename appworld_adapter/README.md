# AppWorld adapter

Thin Inspect integration for AppWorld Base function-calling tasks. It reuses AppWorld's dataset, prompts, predictor/agent logic, local environment, and official evaluator while routing model and accepted tool calls through Inspect and `inspect_trace`.

## Current status (2026-08-11)

The adapter is **baseline-ready**: the complete 57-sample `dev` split finished successfully with
the official scorer and the normal token/episode analysis path. It is not yet intervention-ready;
the next research step is exposing AppWorld's existing tool-menu policy and budget controls.

The verified run is `runs/appworld-dev-full-20260811T013633Z`:

| Result | Value |
|---|---:|
| Inspect status | `success`, 57/57 completed, 0 sample errors |
| Official task accuracy | 35/57 = 0.6140 (stderr 0.0651) |
| Wall time | 2:17:14 |
| Model usage | 15,889,119 input + 1,202,125 output tokens |
| Model calls | 57 predictor + 1,023 main |
| Tool calls | 2,005 accepted and executed; 0 Inspect transport failures |
| Trace coverage | 57/57 episodes load in both token and episode summaries |

Successful episodes averaged 173,587 billed tokens and 70.1 seconds end-to-end; failed episodes
averaged 500,713 tokens and 262.7 seconds. This is already sufficient for success-conditioned
capability/efficiency analysis rather than treating fast failures as efficient runs.

Known boundaries: tools and samples remain serial by design; 5 of 1,080 model calls lack a vLLM
metrics record (the token and Inspect event data are complete, and missing latency is not written as
zero); the local zero-price model configuration cannot support dollar-cost comparisons. The final
Uvicorn lifespan `CancelledError` is cleanup noise after successful result persistence.

## Setup

From `appworld_adapter/`:

```bash
scripts/setup.sh
```

The script intentionally installs the sibling `inspect_trace` and `tau2_adapter` editable with `--no-deps`. It then overrides OpenAI to 2.53.0: `appworld-agents` metadata pins 1.99.8, while the Inspect 0.3.251 provider requires Responses API types from the newer package.

Set `APPWORLD_ROOT` to the provisioned AppWorld checkout. The local model server uses `VLLM_BASE_URL` and `VLLM_API_KEY`; metrics collection uses `INSPECT_TRACE_VLLM_METRICS_URL`.

The only blessed run entry point is `scripts/run_adapter.sh`. Verify the local backend is healthy and queue-idle, then run:

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy \
  -u HTTPS_PROXY -u https_proxy \
  APPWORLD_ROOT="$(realpath ../.deps/appworld)" \
  VLLM_BASE_URL=http://127.0.0.1:8020/v1 \
  VLLM_API_KEY=local \
  INSPECT_TRACE_VLLM_METRICS_URL=http://127.0.0.1:8020/metrics \
  scripts/run_adapter.sh
```

This defaults to the complete `dev` split, the local `tau2-agent-vllm/vllm/qwen3.6-27b-autoround` model, and unique harness run and AppWorld experiment names. For a smoke run, set a positive `LIMIT` (for example, `LIMIT=5 scripts/run_adapter.sh`). The script always uses `--max-samples 1 --max-connections 1 --display plain`, writes `manifest.txt`, and refuses to start if either selected output directory exists. `RUN_NAME` and `EXPERIMENT_NAME` may be selected explicitly, but must both identify fresh locations.

After failure or interruption, retain the artifacts. Where supported, retry from the Inspect logs using a fresh output location; otherwise invoke the script again to create a fresh unique run. Never directly rerun with the same run directory or AppWorld experiment name.

The P0 measurement contract is now available from the normal artifacts:

- token and episode summaries report `predictor`, `main`, and explicit `unknown` roles;
- scorer metadata contains the per-round call ledger and separates Inspect transport failures from AppWorld environment failures;
- episode timing reports `sample_end_to_end_latency_seconds`, `sample_working_time_seconds`, `model_tool_window_seconds`, interval-union busy time, concurrency savings, and directional transition gaps.

Keep `--max-samples 1 --max-connections 1`; the solver also holds a process-local sample lock, and separate eval processes must not share an AppWorld experiment name.

## Validation

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy \
  -u HTTPS_PROXY -u https_proxy .venv/bin/pytest
tests/test_run_adapter.sh
```

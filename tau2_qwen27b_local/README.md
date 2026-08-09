# Local Qwen3.6-27B profile

This tracked profile binds the shared `tau2_adapter` runner to the workstation's
RTX 5090 Qwen3.6-27B backend. It contains no adapter implementation or Python
environment of its own.

## Setup

Create the machine-local configuration:

```bash
cd tau2_qwen27b_local
cp config.env.example config.env
# Edit data, model-cache, and backend-package paths in config.env.
./scripts/setup.sh
```

`config.env` is ignored; the scripts and example configuration are versioned.
The shared setup provisions the pinned tau2 source under `.deps/` and installs
the `tau2_adapter` environment.

## Run

```bash
./scripts/start_backend.sh
NUM_TASKS=1 ./scripts/run_mock.sh
TAU2_DOMAIN=airline NUM_TASKS=1 ./scripts/run_tau.sh
./scripts/run_full_core.sh
./scripts/stop_backend.sh
```

`run_tau.sh` performs the local backend preflight and delegates the actual
evaluation to `tau2_adapter/scripts/run_adapter.sh`. Useful overrides include
`TAU2_DOMAIN`, `TAU2_TASK_SET`, `TAU2_TASK_SPLIT`, `NUM_TASKS`, `TASK_IDS`,
`RUN_NAME`, `TAU2_MAX_STEPS`, and `TAU2_EMPTY_RESPONSE_RETRIES`.
The full controller is shared with NSCC; it validates expected sample counts,
records per-domain status, and skips already valid domains when resumed with
the same `FULL_RUN_ID`.

Results are written under `runs/`. Each evaluation directory includes Inspect
logs, trace output, and `run-manifest.txt` with the code, data, model, and task
selection metadata.

The backend start/stop scripts are workstation-specific. NSCC should use its
own PBS/container launcher and call the shared adapter runner directly.

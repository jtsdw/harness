# local-model-server

A standalone uv project that serves a local model via vLLM's OpenAI-compatible API, for testing
inspect_ai / `inspect_trace` against a real (non-hosted) model. Lives alongside `inspect_trace/`
in the `efficient-harness/` repo but deliberately keeps its own `uv` environment — vLLM pins a
large, GPU-driver-sensitive dependency tree (torch, CUDA runtime libs) that has nothing to do with
inspect_ai's own dependencies, and mixing the two would make both harder to reason about.

## Quick start

```bash
cd /home/liuyingen/code/efficient-harness/local-model-server
./scripts/setup.sh   # one-time environment setup + GPU sanity check
./scripts/serve.sh    # starts the server, downloads the model on first run, waits until ready
./scripts/stop.sh     # stops it
```

Then point inspect_ai at it (see `scripts/serve.sh`'s own output for the exact command, or
`docs/local_model_deployment.md` for a full worked example running the same BFCL benchmark used to
validate `inspect_trace` against a real hosted model).

## Scripts

`serve.sh` is the one script that actually starts vLLM -- every mode below is the *same* script,
just with different environment variables set before calling it (see `serve.sh`'s own header
comment for the full list of variables: `MODEL`/`PORT`/`GPU_MEMORY_UTILIZATION`/`MAX_MODEL_LEN`
plus `NATIVE_TOOL_CALLING`/`SPECULATIVE_MODE`/`NUM_SPECULATIVE_TOKENS`/
`NGRAM_PROMPT_LOOKUP_MAX`/`NGRAM_PROMPT_LOOKUP_MIN`). Only one server can run at a time --
`serve.sh` refuses to start a second one on top of an already-running `logs/vllm_server.pid`.

The named wrappers below exist so you don't have to remember env var names for the specific
configurations this project has actually used and validated -- each one is a few lines that just
set the right variables and call `serve.sh`. `serve.sh` itself carries all the real logic
(readiness polling, PID/log management, error handling) in exactly one place, so a fix there
applies to every mode without needing to touch N near-duplicate scripts.

| Script | Mode | When to use |
|---|---|---|
| `serve_baseline.sh` | No native tool-calling, no speculative decoding | Default/most battle-tested path. Pair with `-M emulate_tools=true` on the inspect_ai side. |
| `serve_native_tool_calling.sh` | `--enable-auto-tool-choice --tool-call-parser hermes` | Real native tool-calling server-side. inspect_ai's stock `openai-api` provider still can't use it as-is (Bug 3, `docs/tau2_bench_integration_findings.md`) -- either keep `emulate_tools=true`, or use a custom `ModelAPI` that strips the `"strict"` field (see `tau2_adapter/src/tau2_adapter/_registry.py`). |
| `serve_ngram_speculative.sh` | `--speculative-model "[ngram]"`, no draft model needed | Real 1.87x speedup vs its own baseline, but a real 23/100 output-divergence rate -- see `docs/toolspec_vllm_speculative_comparison.md` before trusting outputs for anything correctness-sensitive. |
| `setup.sh` | -- | One-time env setup + GPU/CUDA sanity check. Idempotent. |
| `stop.sh` | -- | Stops whichever server is currently running. |

Want a mode that isn't one of these three (e.g. a different `NUM_SPECULATIVE_TOKENS`, or a future
draft-model-based speculative mode)? Call `serve.sh` directly with the env vars you need -- the
named wrappers only cover the combinations this project has actually run and written findings
about, they're not the only combinations `serve.sh` supports.

## Hardware / driver constraint that shapes every pin in this project

```
$ nvidia-smi
NVIDIA RTX 2000 Ada Generation, 16GB VRAM, Driver 535.230.02, max supported CUDA runtime: 12.2
```

This driver cannot run anything built against a newer CUDA runtime than 12.2. `uv add vllm` with no
version pin resolves to the latest vllm release, whose torch dependency is built against CUDA
12.8+ — that fails at engine startup with `RuntimeError: The NVIDIA driver on your system is too
old`. **Before bumping the `vllm`/`torch` pins in `pyproject.toml`, check `nvidia-smi`'s reported CUDA
version on whatever machine you're deploying to, and confirm the target torch release's default PyPI
wheel doesn't exceed it** (a wheel's bundled CUDA version isn't in its filename — you have to actually
install it and check `torch.version.cuda`, there's no way to tell from PyPI metadata alone).

`vllm==0.6.3.post1` is the newest release found whose declared torch pin (`2.4.0`) still resolves to
a CUDA-12.1 wheel from the default PyPI index — CUDA 12.1 runs fine under a driver that supports up
to 12.2 (older CUDA runtime, newer driver is the compatible direction). The very next vllm release
(`0.6.4.post1`) jumps its torch pin to `2.5.1`, whose default wheel is CUDA 12.4 — one minor version
too new for this driver.

One real capability cost of this pin: `vllm==0.6.3.post1` predates the `--enable-auto-tool-choice` /
`--tool-call-parser` CLI flags (added in a later 0.6.x release), so **this server cannot natively
parse structured tool calls**. Work around it client-side — see "Connecting from inspect_ai" below.

## Five broken/incomplete dependency resolutions this project works around

Discovered by actually trying to run the server and reading the traceback, not by inspection — each
one is a real failure mode worth knowing about if you touch dependencies here.

1. **`transformers`/`torchaudio` drifted to versions from years after `vllm==0.6.3.post1`'s era.**
   `uv add vllm==0.6.3.post1` only pins what vllm's own metadata pins; `transformers` and `torchaudio`
   are pulled in transitively with loose (`>=`) constraints, so uv resolved them to whatever was
   latest at lock time (`transformers==5.14.1`, `torchaudio==2.11.0`). That latest `transformers`
   unconditionally imports `torchaudio` from a code path `vllm.transformers_utils.config` touches at
   import time, and that `torchaudio==2.11.0` build wants `libcudart.so.13` — CUDA 13, which doesn't
   exist anywhere in this environment. Fixed by explicitly pinning both in `pyproject.toml` to
   versions contemporaneous with `vllm==0.6.3.post1` (`transformers==4.46.3`, `torchaudio==2.4.0`,
   matching the `torch==2.4.0` pin).

2. **`pyairports==0.0.1` on PyPI is a metadata-only wheel — it ships no code at all.** `outlines`
   (pulled in by vllm for *any* chat completion request, not just ones using structured/guided
   output — vllm 0.6.3.post1 unconditionally touches
   `vllm.model_executor.guided_decoding.outlines_decoding` on every request) does
   `from pyairports.airports import AIRPORT_LIST`, which fails with `ModuleNotFoundError` against the
   real PyPI package. `pyairports` has had exactly one release, ever, and it's broken — there is no
   newer version to pin to. Fixed with a local shim package at `vendor/pyairports/`, built on top of
   `airportsdata` (which does ship real data), wired in via `[tool.uv.sources]` in `pyproject.toml`
   so it transparently satisfies `outlines`' import.

3. **Some `nvidia-*-cu12` wheels installed from a corrupted/incomplete cache at least once on this
   machine**, leaving empty stub directories (an `__init__.py` and nothing else — no actual `.so`
   files) that only fail once something actually tries to load the library (`cudnn`, `nccl` both hit
   this here). Silent until exercised, which meant it surfaced as an opaque `ImportError:
   libcudnn.so.9: cannot open shared object file` deep inside a request handler, not at install time.
   `scripts/setup.sh` checks every `nvidia/*` directory under `.venv` for at least one `.so` file after
   `uv sync` and force-reinstalls (`--no-cache`) any that come up empty — run it after any dependency
   change, not just once.

4. **On PBS/Slurm-managed multi-GPU clusters (first hit on an NSCC DGX node), `CUDA_VISIBLE_DEVICES`
   can be set to GPU UUIDs instead of plain integer indices** (e.g.
   `GPU-2a95c117-1313-cb8f-323f-ea75d126060d`), as part of cgroup-level device isolation. This vLLM
   version is too old to handle that: `vllm/platforms/cuda.py::device_id_to_physical_device_id()`
   does a bare `int()` on each entry and crashes with `ValueError: invalid literal for int()` before
   the engine even starts, deep in a traceback that gives no hint the actual problem is the
   env var's format. `serve.sh` now detects a non-numeric `CUDA_VISIBLE_DEVICES` and remaps it to
   sequential indices (`0`, `0,1`, ...) before starting vLLM — safe specifically because the
   scheduler's cgroup restriction already means the process can only see its allocated GPU(s); this
   only fixes the label format vLLM's older code expects, it doesn't change which physical GPU(s)
   are used.

5. **HuggingFace caches to `$HOME/.cache/huggingface` by default, which breaks on clusters where
   `$HOME` has a small quota** (also first hit on an NSCC DGX node -- downloading
   `Qwen2.5-32B-Instruct`, ~65GB of weights alone, failed with `OSError: Disk quota exceeded`
   partway through). NSCC's convention is that large files belong under `$HOME/scratch/`, not
   `$HOME` itself (see `docs/remote_compute_workflow.md`). `serve.sh` does **not** default `HF_HOME`
   to a project-relative path itself -- doing so would silently orphan whatever's already cached at
   a machine's real default (this dev machine included, which already has a working
   `~/.cache/huggingface`). It only warns when `$HOME` and the project directory are on different
   filesystems (a `stat -c %d` comparison), since that's the concrete condition this problem needs.
   Set `HF_HOME` yourself before running `serve.sh` when that warning fires, e.g.
   `export HF_HOME=~/scratch/hf_cache`.

## Connecting from inspect_ai

Since this vLLM version has no native tool-call parser, use inspect_ai's `openai-api` provider with
`emulate_tools=true` — inspect_ai prompts for and parses tool calls itself rather than relying on the
server's (nonexistent, in this version) structured tool-calling support:

```bash
VLLM_BASE_URL="http://localhost:8000/v1" VLLM_API_KEY="not-needed" \
  uv run --project /home/liuyingen/code/efficient-harness/inspect_trace inspect eval inspect_evals/bfcl \
  -T "categories=['multi_turn_base']" \
  --model "openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" -M emulate_tools=true \
  --limit 2
```

(`VLLM_BASE_URL`/`VLLM_API_KEY` — the env var names are derived from the `vllm` service-prefix segment
in the model string, `openai-api/vllm/...`; see `inspect_ai`'s own source (upstream reference
clone at `/home/liuyingen/code/inspect_ai/src/inspect_ai/model/_providers/openai_compatible.py`,
not part of this repo) for how that's resolved.)

## Changing the model

16GB VRAM is the binding constraint. `Qwen/Qwen2.5-3B-Instruct` (bf16, ~6GB weights) was chosen to
leave generous headroom for KV cache; a 7B model unquantized (~15GB) leaves almost none and will OOM
on anything but very short sequences. If you need more capability than 3B and are willing to take on
a quantization dependency, `Qwen/Qwen2.5-7B-Instruct-AWQ` (~4.5GB weights) is a reasonable next step —
just re-run `./scripts/setup.sh` first to confirm the environment is still sane, then
`MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ" ./scripts/serve.sh`.

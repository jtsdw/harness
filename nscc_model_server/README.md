# nscc_model_server

A second, separate vLLM-serving project for the NSCC H100 node -- **not** a replacement for
`../local-model-server/`, which stays exactly as-is for this dev machine.

## Why this is a separate project, not a config flag on local-model-server

`local-model-server/pyproject.toml` pins `vllm==0.6.3.post1` (and old `transformers`/`torchaudio`
alongside it) for one specific, narrow reason: this dev machine's GPU driver (535.230.02) only
supports up to CUDA 12.2, and every vLLM release past that pin resolves a torch/CUDA-runtime
combination the driver rejects. See `local-model-server/README.md`'s "driver constraint" section
for the full story.

**The NSCC H100 node doesn't have that constraint.** Confirmed 2026-08-06:

```
Driver Version: 595.71.05, CUDA Version: 13.2
GPU: NVIDIA H100 80GB HBM3
```

That old pin only ever existed to work around this dev machine's specific driver ceiling -- it
was never a requirement of the project itself, and copying it onto a completely different GPU
with a completely different (much newer) driver just because it's what one existing pyproject.toml
happened to say was carrying the constraint somewhere it doesn't apply. This project starts from a
fresh, unconstrained `vllm>=0.9.0` instead of inheriting that pin.

## Why upgrading matters here specifically: EAGLE-3

vLLM 0.6.3.post1 (this dev machine's pin) only supports `[ngram]`/prompt-lookup speculative
decoding -- confirmed for real in `docs/toolspec_vllm_speculative_comparison.md` (1.87x speedup,
23/100 output divergence from baseline on BFCL). vLLM 0.9+ has native EAGLE-3 support, which
published real-world numbers put at **3.0-3.4x decode speedup on H100** (vs 2.4-2.7x for EAGLE-2
on the same hardware) -- a meaningfully stronger comparison point against ToolSpec than what this
dev machine's old vLLM could ever show.

EAGLE-3 needs a draft checkpoint trained specifically for whichever exact target model you serve
-- it isn't a generic "any model" flag like `[ngram]` was. The default here targets
`Qwen/Qwen3-32B` paired with `RedHatAI/Qwen3-32B-speculator.eagle3` (a real, published matching
checkpoint) specifically because a matching EAGLE-3 draft doesn't appear to exist yet for
whatever Qwen2.5 model this team's `tau2_qwen27b_local/` work was targeting -- see
`docs/nscc_h100_speculative_decoding_plan.md` for the full reasoning and the fallback options if
Qwen3-32B isn't the right call for your actual experiments.

## Status: designed from documentation research, NOT yet run on real hardware

Everything in this project (the vLLM version floor, the `--speculative-config` JSON syntax, the
exact checkpoint name) comes from checking vLLM's own docs/blog and HuggingFace, not from an
actual run on the NSCC node -- this session has no access to that machine. Treat every command
here as a first draft to verify for real, not a working known-good recipe (unlike
`local-model-server/`, which every script has actually been run and verified against real
hardware in this repo's history). Run `./scripts/setup.sh` and `./scripts/verify_eagle3.sh` and
report back what actually happens -- especially the real installed vLLM version
(`uv run vllm --version`), since that determines whether the exact `--speculative-config` syntax
below is even right for whatever resolves.

## Quick start

```bash
./scripts/setup.sh              # installs vllm>=0.9.0, reports the exact version resolved
./scripts/serve_eagle3.sh        # starts vLLM with Qwen3-32B + its EAGLE-3 speculator
./scripts/verify_eagle3.sh       # confirms speculative decoding is actually active, prints real numbers
./scripts/stop.sh
```

`./scripts/serve_baseline.sh` (no speculative decoding, same model) is there too, for an
apples-to-apples comparison on this same new vLLM version -- mirrors
`local-model-server/scripts/serve_baseline.sh`.

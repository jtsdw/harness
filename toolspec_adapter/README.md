# toolspec_adapter

Drives [ToolSpec](https://arxiv.org/abs/2604.13519)'s own `baseline_forward()`/`toolspec_forward()`
functions (raw HuggingFace `transformers` generation loop, patched KV-cache modeling code -- not
an OpenAI-compatible service, see `docs/acceleration_methods_survey.md`) from inside a custom
`inspect_ai` `ModelAPI` provider (`toolspec-hf`), so `inspect_trace`'s Hooks fire for real and we
get real token/episode-layer profiling data for both the vanilla-decoding and ToolSpec-decoding
paths, running the same API-Bank samples ToolSpec's own repo ships.

Full narrative, native-repo-vs-adapter comparison, and root-cause investigation of the
non-lossless-in-practice finding: `docs/toolspec_integration_findings.md`.

## Quickstart

```bash
./scripts/run_apibank.sh baseline
./scripts/run_apibank.sh toolspec
```

Requires `TOOLSPEC_REPO_DIR` (default `/home/liuyingen/code/ToolSpec`) to point at a working
ToolSpec checkout with its own venv already set up (`uv venv --python 3.12 .venv && uv pip
install --python .venv/bin/python torch==2.5.1 --index-url
https://download.pytorch.org/whl/cu121 && uv pip install --python .venv/bin/python -r
requirements.txt` -- pinned to a cu121 wheel because the local GPU's driver caps at CUDA 12.2,
same constraint as `local-model-server`).

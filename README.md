# efficient-harness

An agent-efficiency research harness built on top of [inspect_ai](https://inspect.aisi.org.uk/):
full execution-trace capture, three-layer cost profiling (token / model-invocation / episode),
and the visualization built on top of both. See [`docs/efficient-harness.md`](docs/efficient-harness.md)
for the original four-goal spec this project implements.

## Layout

```
efficient-harness/
├── inspect_trace/        Goal 1 + 2 implementation: an inspect_ai Hooks-based package
│                          (execution trace recording, token attribution, execution
│                          topology, action parsing, vLLM metrics, offline analysis).
│                          Own uv project -- cd in and `uv sync --extra dev`.
├── local-model-server/   Standalone vLLM deployment for testing against a real local
│                          model instead of a hosted API. Own uv project, deliberately
│                          isolated (vLLM's dependency stack doesn't mix well with
│                          inspect_trace's) -- cd in and `./scripts/setup.sh`.
├── docs/                 All project documentation. Start at docs/README.md for the
│                          reading-order index.
└── runs/                 Real benchmark run output (.eval logs + inspect_trace JSONL),
                           gitignored -- must be copied by hand when migrating machines.
```

`inspect_trace` and `local-model-server` are independently managed uv projects (separate
venvs, separate lockfiles) that only ever talk to each other over HTTP
(`http://localhost:8000`) -- neither needs the other installed to work.

## Where to start

- Never used this project before: [`docs/README.md`](docs/README.md) is the full reading-order
  index; if you only have ten minutes, read `efficient-harness.md`, `framework-selection.md`,
  and `inspect_ai_roadmap.md` from that list.
- Want to run something: [`docs/environment_checklist.md`](docs/environment_checklist.md) for
  environment setup, [`docs/inspect_ai_quickstart.md`](docs/inspect_ai_quickstart.md) to run
  your first eval.
- Want to see real results without running anything: open
  [`docs/goal1_r3_r4_dashboard.html`](docs/goal1_r3_r4_dashboard.html) directly in a browser
  (self-contained, no server needed).

## History

This repo was consolidated on 2026-08-04 from three previously separate locations
(`inspect_trace` nested inside an upstream `inspect_ai` clone, plus two standalone repos
for `local-model-server` and the docs) ahead of a hardware migration and team rollout.
See [`docs/deployment_migration_guide.md`](docs/deployment_migration_guide.md) for why and
how. Git history was not carried over from the old locations by design.

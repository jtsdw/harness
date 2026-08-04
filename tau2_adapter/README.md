# tau2_adapter

Experimental integration between [tau2-bench](https://github.com/sierra-research/tau2-bench)
(a dual-control agent/user-simulator benchmark, not built on inspect_ai) and this project's
`inspect_trace` profiling stack.

Drives a real tau2-bench simulation (`Orchestrator`, `Environment`, `UserSimulator`,
`evaluate_simulation` -- all reused unmodified from the `tau2` package as libraries, not
reimplemented) from inside an inspect_ai `Task`/`Solver`, so that the agent-under-test's model
calls go through `inspect_ai.model.generate()` and `inspect_trace`'s Hooks fire for them.

See `/home/liuyingen/code/efficient-harness/docs/tau2_bench_integration_findings.md` for why this
needs a sync/async bridge (`tau2`'s `Orchestrator.run()` is synchronous; `inspect_ai`'s Hooks only
fire inside its own Sample execution context) and what the real results were.

Only the `mock` domain is wired up so far.

# nscc_runs/

Real experiment results pulled from the NSCC compute node, kept separate from `../runs/` (which
is for runs executed directly on *this* machine).

Why the split exists: before this, `scripts/pull_runs.sh` rsynced the remote node's `runs/`
straight into this repo's own `runs/`, mixing NSCC-origin results with anything run locally in
the same directory tree -- no way to tell at a glance "did this come off the big GPU box or did I
just run it here for a quick check" (this actually caused a real mix-up during script testing: a
throwaway local smoke-test run landed in the same directory a real NSCC run would have used).

Contents here are gitignored (see `.gitignore`) -- large, regenerable by re-pulling, but not
disposable in the sense that `deployment_migration_guide.md` describes for `../runs/`: if you're
about to wipe this directory, make sure whatever's in it has already been folded into a committed
dashboard/findings doc, or copy it out first.

## How this gets populated

```bash
./scripts/pull_runs.sh preview   # see what would transfer, doesn't touch anything
./scripts/pull_runs.sh pull      # actually pull
./scripts/pull_runs.sh delete    # pull + delete local files no longer present on the remote
```

Full picture (SSH config, per-person remote checkouts, PBS usage) in
[`docs/remote_compute_workflow.md`](../docs/remote_compute_workflow.md).

## Using what's in here

Same tools as `../runs/` work on this directory unmodified -- point any of them at a subdirectory
here instead:

```bash
cd inspect_trace
uv run python scripts/build_eval_report.py ../nscc_runs/<some-run-name>
```

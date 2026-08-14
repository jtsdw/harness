#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
fake_inspect="$tmp/inspect"
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$@" >"%s"\n' "$tmp/args" >"$fake_inspect"
chmod +x "$fake_inspect"

RUNS_ROOT="$tmp/runs" APPWORLD_ROOT="$tmp/appworld" INSPECT_BIN="$fake_inspect" \
  RUN_NAME=first EXPERIMENT_NAME=first-exp "$ROOT/scripts/run_adapter.sh"
[[ -f "$tmp/args" ]]
! grep -Fx -- '--limit' "$tmp/args"
grep -Fx -- '--display' "$tmp/args"
grep -Fx -- 'plain' "$tmp/args"

if RUNS_ROOT="$tmp/runs" APPWORLD_ROOT="$tmp/appworld" INSPECT_BIN="$fake_inspect" \
  RUN_NAME=first EXPERIMENT_NAME=other-exp "$ROOT/scripts/run_adapter.sh" >/dev/null 2>&1; then
  printf 'expected harness run-directory collision refusal\n' >&2
  exit 1
fi
mkdir -p "$tmp/appworld/experiments/outputs/taken-exp"
if RUNS_ROOT="$tmp/runs" APPWORLD_ROOT="$tmp/appworld" INSPECT_BIN="$fake_inspect" \
  RUN_NAME=other EXPERIMENT_NAME=taken-exp "$ROOT/scripts/run_adapter.sh" >/dev/null 2>&1; then
  printf 'expected AppWorld experiment-directory collision refusal\n' >&2
  exit 1
fi

printf 'run_adapter safety verification passed\n'

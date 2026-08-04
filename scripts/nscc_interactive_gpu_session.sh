#!/usr/bin/env bash
# Requests an interactive PBS session with a GPU on the NSCC ASPIRE2 cluster and drops you into
# it once resources are allocated -- this is how we get a long-lived shell to run
# local-model-server/scripts/serve.sh in and keep vLLM up for a real dev/debug session, as
# opposed to a one-shot batch job (see scripts/pbs_vllm_server_job.sh for that alternative).
#
# Run this ON THE LOGIN NODE (after `ssh nscc`), from inside your own checkout under scratch/ --
# NOT from this local machine. It's tracked in git alongside everything else specifically so it
# arrives on the remote side via `git pull`, no separate transfer needed.
#
# Once you're in the session:
#   - It's tied to this SSH connection. Run it inside `tmux`/`screen` on the login node first
#     (i.e. `ssh nscc`, start tmux, THEN run this script) so a dropped connection doesn't kill
#     your GPU allocation and whatever's running in it.
#   - `nvidia-smi` should show your allocated GPU(s) once you're in.
#   - `cd` into your checkout, `cd local-model-server && ./scripts/serve.sh` to start vLLM.
#
# PBS_PROJECT has no sane default -- this script refuses to guess it. Find yours with
# `project -list` on the login node, or ask whoever manages the NSCC allocation.
#
# Usage:
#   PBS_PROJECT=xxxxxxxx ./scripts/nscc_interactive_gpu_session.sh
#   PBS_PROJECT=xxxxxxxx PBS_WALLTIME=24:00:00 PBS_NGPUS=1 ./scripts/nscc_interactive_gpu_session.sh
#
# Env vars (all but PBS_PROJECT have defaults):
#   PBS_PROJECT   required, no default -- your NSCC project/allocation code
#   PBS_QUEUE     default: normal
#   PBS_NGPUS     default: 1
#   PBS_NCPUS     default: 4
#   PBS_MEM       default: 80gb
#   PBS_WALLTIME  default: 48:00:00 -- long enough for a real multi-day dev session; NSCC's
#                 "normal" queue has been seen to accept up to 120:00:00 on this project (see
#                 docs/remote_compute_workflow.md), lower this if your allocation doesn't allow it

set -euo pipefail

if [[ -z "${PBS_PROJECT:-}" ]]; then
  echo "Error: PBS_PROJECT is not set." >&2
  echo "Find your project code with: project -list" >&2
  echo "Then: PBS_PROJECT=xxxxxxxx $0" >&2
  exit 1
fi

: "${PBS_QUEUE:=normal}"
: "${PBS_NGPUS:=1}"
: "${PBS_NCPUS:=4}"
: "${PBS_MEM:=80gb}"
: "${PBS_WALLTIME:=48:00:00}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "Error: qsub not found. This script must run on the NSCC login node (ssh nscc first)." >&2
  exit 1
fi

SELECT_SPEC="1:ngpus=${PBS_NGPUS}:ncpus=${PBS_NCPUS}:mem=${PBS_MEM}"

echo "Requesting interactive PBS session:"
echo "  Project:  $PBS_PROJECT"
echo "  Queue:    $PBS_QUEUE"
echo "  Select:   $SELECT_SPEC"
echo "  Walltime: $PBS_WALLTIME"
echo
echo "Waiting for allocation (this can queue for a while depending on cluster load)..."
echo

exec qsub -I \
  -P "$PBS_PROJECT" \
  -q "$PBS_QUEUE" \
  -l "select=${SELECT_SPEC}" \
  -l "walltime=${PBS_WALLTIME}"

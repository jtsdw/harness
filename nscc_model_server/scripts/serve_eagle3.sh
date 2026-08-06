#!/usr/bin/env bash
# EAGLE-3 speculative decoding. See ../README.md -- untested against real hardware, verify with
# ./scripts/verify_eagle3.sh after this comes up, don't assume it's working just because the
# server started.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SPECULATIVE_MODE=eagle3 exec ./scripts/serve.sh

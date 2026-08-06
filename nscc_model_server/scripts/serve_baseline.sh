#!/usr/bin/env bash
# No speculative decoding -- the apples-to-apples comparison point for serve_eagle3.sh, same
# model, same everything else. See ../README.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec ./scripts/serve.sh

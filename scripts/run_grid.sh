#!/usr/bin/env bash
# Run each eval config RERUNS times with the same seed, rerun-major (every config's first run
# finishes before any second run starts), for the test-retest analysis.
# Usage: scripts/run_grid.sh RERUNS configs/eval/a.yaml [configs/eval/b.yaml ...]
# A run with failed cells exits 1; the loop records that and carries on.
set -euo pipefail
cd "$(dirname "$0")/.."
reruns=${1:?usage: scripts/run_grid.sh RERUNS CONFIG...}
shift
for i in $(seq 1 "$reruns"); do
  for config in "$@"; do
    echo "=== rerun $i/$reruns: $config ($(date -u +%FT%TZ))"
    uv run --no-sync llm-eval run --config "$config" || echo "!!! $config rerun $i exited $?"
  done
done
echo "=== done ($(date -u +%FT%TZ))"

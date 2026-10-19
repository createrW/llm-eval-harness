#!/usr/bin/env bash
# Run one evaluation config.
# Usage: scripts/eval.sh configs/eval/<name>.yaml [extra `llm-eval run` arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
config=${1:?usage: scripts/eval.sh CONFIG_YAML [args...]}
shift
exec uv run --extra local llm-eval run --config "$config" "$@"

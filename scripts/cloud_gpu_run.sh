#!/usr/bin/env bash
# Run the local-model grid on a rented Linux CUDA machine reached over SSH (RunPod, Vast.ai,
# Thunder Compute, ...): batch inference, not training. On the machine, from a clone of the repo:
#     scripts/cloud_gpu_run.sh            # every config 3 times
#     RERUNS=1 scripts/cloud_gpu_run.sh   # one run of each
# Then copy the results back to the laptop:
#     scp <host>:<repo>/runs.tgz . && tar xzf runs.tgz -C data/processed
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv sync --frozen --no-dev --extra local                  # installs the locked CPU wheel on Linux
uv pip install --reinstall-package torch --torch-backend=auto "torch==2.14.1"   # CUDA build
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), torch.__version__; print(torch.cuda.get_device_name(0))"
export UV_NO_SYNC=1 LLM_EVAL_GIT_SHA="$(git rev-parse HEAD)"
scripts/run_grid.sh "${RERUNS:-3}" configs/eval/local_ll.yaml configs/eval/local_gen.yaml
tar czf runs.tgz -C data/processed runs
echo "done: $(pwd)/runs.tgz"

#!/usr/bin/env bash
# Run the local-model grid on a rented Linux CUDA machine reached over SSH (RunPod, Vast.ai,
# Thunder Compute, ...): batch inference, not training. On the machine, from a *git clone* of the
# repo (the commit is recorded in every run), inside tmux so a dropped SSH session cannot kill it:
#     scripts/cloud_gpu_run.sh            # every config 3 times
#     RERUNS=1 scripts/cloud_gpu_run.sh   # one run of each
#     RERUNS=1 CONFIGS=configs/eval/local_gen.yaml scripts/cloud_gpu_run.sh   # only what is missing
#     UV_TORCH_BACKEND=cu126 scripts/cloud_gpu_run.sh   # force one CUDA build (default: auto)
# Then, on the laptop from the repo root (RunPod: the "SSH over exposed TCP" IP and port; the
# ssh.runpod.io proxy does not support scp):
#     scp -P <port> root@<ip>:<repo>/data/processed/runs.tgz data/processed/
#     tar xzf data/processed/runs.tgz -C data/processed
set -euo pipefail
cd "$(dirname "$0")/.."
UV_VERSION=0.12.22  # the uv that wrote uv.lock
export PATH="$HOME/.local/bin:$PATH"
if [ "$(uv --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | UV_NO_MODIFY_PATH=1 sh
fi
sha=$(git rev-parse HEAD)  # stops here when the code is not a git checkout
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "commit tracked changes first: every run records the commit it ran" >&2
  exit 1
fi
uv python install 3.11
uv sync --frozen --no-dev --extra local  # installs the locked CPU wheel on Linux
uv pip install --reinstall-package torch --torch-backend="${UV_TORCH_BACKEND:-auto}" "torch==2.14.1"
uv run --no-sync python -c "
import torch
assert torch.cuda.is_available(), torch.__version__
x = torch.ones(64, 64, device='cuda') @ torch.ones(64, 64, device='cuda')  # a real CUDA kernel
print(torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0), x[0, 0].item())
"
export UV_NO_SYNC=1 LLM_EVAL_GIT_SHA="$sha"  # a plain `uv run` would reinstall the CPU wheel
mkdir -p data/processed
{ nvidia-smi; uv pip freeze; } > data/processed/gpu_env.txt  # GPU, driver, exact packages
read -ra configs <<< "${CONFIGS:-configs/eval/local_ll.yaml configs/eval/local_gen.yaml}"
scripts/run_grid.sh "${RERUNS:-3}" "${configs[@]}" 2>&1 | tee -a data/processed/grid.log
tar czf data/processed/runs.tgz -C data/processed runs gpu_env.txt grid.log
echo "done: $(pwd)/data/processed/runs.tgz"

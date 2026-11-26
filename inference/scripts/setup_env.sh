#!/usr/bin/env bash
# Create the conda env, download the models, run the CPU tests (called by `bash run.sh setup`).
# Versions follow configs/env_jev_fa.lock.txt (python 3.12, torch 2.14.0, transformers 5.17, flash-attn 2.8.3).
# Overrides (normally set in run.env):
#   ENV_NAME      conda env name (default jev_fa)
#   TORCH_CUDA    PyTorch wheel CUDA tag (default cu126)
#   FLASH_ATTN_WHEEL  optional custom flash-attn 2.8.3 wheel path/URL; default: pinned mjun0812 prebuilt wheel
#   CAUSAL_CONV1D_WHEEL  optional causal-conv1d 1.7.0 wheel path/URL; default: compiled from source
#   JEV_CUDA_FORCE_BUILD=1  skip prebuilt wheels and compile both CUDA extensions
#   HF_HOME       where model snapshots go (~35 GB); gated repos need `huggingface-cli login` first
#   JEV_SETUP_CURRENT_ENV=1  use an already-created Python 3.12 env; skip conda activation
#   JEV_SETUP_PACKAGES_ONLY=1  install/import dependencies only (e.g. a CPU container build); no GPU check, models or tests
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_NAME=${ENV_NAME:-jev_fa}
TORCH_CUDA=${TORCH_CUDA:-cu126}
LOCK=configs/env_jev_fa.lock.txt
packages_only=${JEV_SETUP_PACKAGES_ONLY:-0}
case "$packages_only" in 0|1) ;; *) echo 'JEV_SETUP_PACKAGES_ONLY must be 0 or 1' >&2; exit 2 ;; esac

if [ "${JEV_SETUP_CURRENT_ENV:-0}" = 1 ]; then
  python -c 'import sys; assert sys.version_info[:2] == (3, 12), "setup requires Python 3.12"'
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.12
  fi
  conda activate "$ENV_NAME"
fi
if [ "$packages_only" = 1 ]; then
  echo "python: $(which python); installing packages without GPU/model access"
else
  echo "python: $(which python)  driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi

# NVIDIA's Conda toolkit puts CUDA headers/libraries under targets/, whereas
# torch CUDAExtension also invokes the host compiler directly.
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/targets/x86_64-linux/include" ]; then
  export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

# 1. torch first, from the PyTorch index for the chosen CUDA build (it pulls its own nvidia-* / triton wheels)
TV=$(grep -E '^torch==' $LOCK | sed 's/torch==//; s/+.*//')
TVV=$(grep -E '^torchvision==' $LOCK | sed 's/torchvision==//; s/+.*//')
pip install "torch==$TV" "torchvision==$TVV" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

# 2. everything else in the lock file, minus what step 1 / step 3 own (CUDA-build specific)
REQ=$(mktemp)
grep -vE '^(#|torch==|torchvision==|flash-attn==|causal-conv1d==|triton==|nvidia-|cuda-)' $LOCK | grep -v '^$' > "$REQ"
pip install -r "$REQ"
rm -f "$REQ"

# 3. flash-attn (must match this torch + CUDA)
if [ "${JEV_CUDA_FORCE_BUILD:-0}" = 1 ]; then
  MAX_JOBS=${MAX_JOBS:-16} python scripts/install_cuda_extension.py flash-attn 2.8.3
elif [ -n "${FLASH_ATTN_WHEEL:-}" ]; then
  pip install --no-deps --force-reinstall "$FLASH_ATTN_WHEEL"
else
  MAX_JOBS=${MAX_JOBS:-16} python scripts/install_cuda_extension.py flash-attn 2.8.3
fi
# 4. linear-attention kernels of OneJev (Qwen3.5 Gated DeltaNet): fla (Triton, from the lock file) + causal-conv1d (CUDA,
#    must match this torch + CUDA). Without them transformers silently falls back to slow torch loops. Hopper (H100/H200) ok.
if [ "${JEV_CUDA_FORCE_BUILD:-0}" = 1 ]; then
  MAX_JOBS=${MAX_JOBS:-16} python scripts/install_cuda_extension.py causal-conv1d 1.7.0
elif [ -n "${CAUSAL_CONV1D_WHEEL:-}" ]; then
  pip install --no-deps --force-reinstall "$CAUSAL_CONV1D_WHEEL"
elif python -c "import causal_conv1d; assert causal_conv1d.__version__.split('+')[0] == '1.7.0'" 2>/dev/null; then
  echo "causal-conv1d already installed"
else
  MAX_JOBS=${MAX_JOBS:-16} python scripts/install_cuda_extension.py causal-conv1d 1.7.0
fi
python - <<'EOF'
import os
import torch, transformers, flash_attn, fla, causal_conv1d
print(f"torch {torch.__version__} (cuda {torch.version.cuda}, {torch.cuda.device_count()} GPUs visible), "
      f"transformers {transformers.__version__}, flash-attn {flash_attn.__version__}, fla {fla.__version__}, "
      f"causal-conv1d {causal_conv1d.__version__}")
if os.environ.get('JEV_SETUP_PACKAGES_ONLY', '0') != '1':
    assert torch.cuda.is_available(), "torch cannot see a GPU: check TORCH_CUDA against the driver"
EOF

if [ "$packages_only" = 1 ]; then
  echo 'dependency installation done; GPU/model validation runs on the training node'
  exit 0
fi

# The default reader runs outside the training environment.
PY="$(command -v python)" bash scripts/setup_vllm_env.sh

# 5. models: policy OneJev-4B, answerer Qwen3-VL-8B, NanoJev tokenizer (CPU tests only); paths -> configs/model_paths.json
python scripts/download_models.py onejev_4b reader_qwen3vl8b nanojev_tokenizer

# 6. CPU tests (JEV_SETUP_SKIP_TESTS=1 skips them)
if [ "${JEV_SETUP_SKIP_TESTS:-0}" != 1 ]; then python -m pytest tests -q; fi
echo "setup done: env $ENV_NAME"

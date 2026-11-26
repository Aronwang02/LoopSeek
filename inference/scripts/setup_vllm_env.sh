#!/usr/bin/env bash
# Install only the frozen reader's environment. Uses published CUDA wheels; no GPU required.
# VLLM_ENV overrides the destination. PY is used only to create the isolated Python 3.12 venv.
set -euo pipefail
cd "$(dirname "$0")/.."
reader_env=${VLLM_ENV:-$PWD/.venv-vllm}
reader_python=$reader_env/bin/python
req=configs/env_vllm.requirements.txt
signature=$(sha256sum "$req" scripts/setup_vllm_env.sh | sha256sum | cut -d ' ' -f 1)
if [ -x "$reader_python" ] && [ -f "$reader_env/.onejev_vllm_setup" ] && \
   [ "$(cat "$reader_env/.onejev_vllm_setup")" = "$signature" ]; then
  echo "vLLM environment ready: $reader_python"
  exit 0
fi
if [ ! -x "$reader_python" ]; then
  "${PY:-python3}" -m venv "$reader_env"
fi
"$reader_python" -m pip install --upgrade pip
"$reader_python" -m pip install --only-binary=:all: \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 torchcodec==0.16.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
"$reader_python" -m pip install -r "$req"
mkdir -p "$reader_env/share"
cp "$req" "$reader_env/share/"
printf '%s\n' "$signature" > "$reader_env/.onejev_vllm_setup"
echo "vLLM environment ready: $reader_python"

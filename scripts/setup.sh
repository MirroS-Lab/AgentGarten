#!/usr/bin/env bash
# Set up AgentGarten on a CUDA 13.2 host (DGX): venv, CUDA torch, the wm package
# and the released weights. One step per call; nothing runs implicitly.
#
#   bash scripts/setup.sh setup      venv, CUDA 13.2 torch, wm, GPU check
#   bash scripts/setup.sh download   renderer checkpoint + Wan 2.2 VAE, fills .env
#   bash scripts/setup.sh all        setup, then download
#
# Uses Python's built-in venv and pip; uv is not needed.
#
# Overrides (environment): WEIGHTS (default <repo>/weights), PYTHON_BIN (python3;
# 3.11 or newer), CUDA_TAG (cu132 = CUDA 13.2), TORCH_SPEC (the torch and
# torchvision pins; they must exist on that CUDA index).
# Opt-in extras (set to 1): WITH_BASE downloads Cosmos3-Nano too (16B, training
# only), WITH_DATA installs torchcodec, WITH_TE installs transformer-engine
# (compiled against the CUDA toolkit on the host), WITH_WANDB installs wandb.
# A Hugging Face token, if a repo asks for one, is read from HF_TOKEN by
# huggingface_hub and is never printed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

WEIGHTS="${WEIGHTS:-${ROOT}/weights}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_TAG="${CUDA_TAG:-cu132}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.12.0 torchvision==0.27.0}"
WITH_BASE="${WITH_BASE:-0}"
WITH_DATA="${WITH_DATA:-0}"
WITH_TE="${WITH_TE:-0}"
WITH_WANDB="${WITH_WANDB:-0}"

# cu132 -> 13.2, cu130 -> 13.0
cuda_digits="${CUDA_TAG#cu}"
WANT_CUDA="${cuda_digits%?}.${cuda_digits: -1}"

die() {
  echo "error: $*" >&2
  exit 1
}

check_driver() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found; needs an NVIDIA driver"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  local driver_cuda
  driver_cuda="$(nvidia-smi | grep -o 'CUDA Version: *[0-9][0-9.]*' | grep -o '[0-9][0-9.]*$' || true)"
  if [[ -z "${driver_cuda}" ]]; then
    # Some drivers print no CUDA version (or N/A) in the nvidia-smi header; the
    # torch check after the install then decides whether the GPU is usable.
    echo "note: nvidia-smi reports no CUDA version; the torch check below decides"
    return
  fi
  echo "driver supports CUDA ${driver_cuda}, building for CUDA ${WANT_CUDA}"
  local lowest
  lowest="$(printf '%s\n%s\n' "${WANT_CUDA}" "${driver_cuda}" | sort -V | head -n1)"
  [[ "${lowest}" == "${WANT_CUDA}" ]] \
    || die "the driver supports CUDA ${driver_cuda}, below the ${WANT_CUDA} needed"
}

check_torch() {
  .venv/bin/python - "${WANT_CUDA}" <<'PY'
import sys

import torch

want = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see a GPU: " + torch.__version__)
if torch.version.cuda != want:
    raise SystemExit(
        f"torch was built for CUDA {torch.version.cuda}, expected {want}: "
        f"{torch.__version__}"
    )
print(torch.__version__, "CUDA", torch.version.cuda, torch.cuda.get_device_name(0))
PY
}

cmd_setup() {
  command -v "${PYTHON_BIN}" >/dev/null || die "${PYTHON_BIN} not found on PATH"
  "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else f"Python 3.11 or newer is needed, found {sys.version.split()[0]}")' \
    || die "set PYTHON_BIN to a Python 3.11 or newer interpreter"
  check_driver

  # An existing .venv is reused, never recreated or deleted.
  if [[ ! -x .venv/bin/python ]]; then
    [[ ! -e .venv ]] || die ".venv exists but has no python; remove it yourself and rerun"
    "${PYTHON_BIN}" -m venv .venv \
      || die "could not create .venv; on Debian or Ubuntu install the python3-venv package"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  # A venv made by uv, or by a Python without ensurepip, has no pip; ensurepip adds it
  # to the existing environment.
  if ! python -m pip --version >/dev/null 2>&1; then
    echo "--- .venv has no pip; adding it with ensurepip ---"
    python -m ensurepip --upgrade \
      || die ".venv has no pip and ensurepip is unavailable; install python3-venv, or run: curl -sS https://bootstrap.pypa.io/get-pip.py | ${ROOT}/.venv/bin/python"
  fi
  python -m pip install --upgrade pip

  echo "--- CUDA torch from the ${CUDA_TAG} index: ${TORCH_SPEC} ---"
  # TORCH_SPEC is split into separate requirements on purpose.
  # shellcheck disable=SC2086
  python -m pip install ${TORCH_SPEC} --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    || die "the torch install failed, see the pip output above (a 'no matching distribution' error means ${CUDA_TAG} has no ${TORCH_SPEC}: set CUDA_TAG or TORCH_SPEC)"
  check_torch

  local extras=media
  [[ "${WITH_DATA}" == 1 ]] && extras="${extras},data"
  [[ "${WITH_TE}" == 1 ]] && extras="${extras},te"
  [[ "${WITH_WANDB}" == 1 ]] && extras="${extras},wandb"
  echo "--- wm with extras: ${extras} ---"
  python -m pip install huggingface_hub "transformers>=4.57"
  python -m pip install -e ".[${extras}]"
  # The package install must not have replaced the CUDA torch with another build.
  check_torch
  [[ -f .env ]] || cp .env.example .env
}

cmd_download() {
  [[ -x .venv/bin/python ]] || die "no .venv; run: bash scripts/setup.sh setup"
  mkdir -p "${WEIGHTS}"
  echo "weights: ${WEIGHTS}, free: $(df -BG --output=avail "${WEIGHTS}" | tail -1 | tr -d ' ')"
  local base_flag=()
  [[ "${WITH_BASE}" == 1 ]] && base_flag=(--with-base)
  .venv/bin/python scripts/download_weights.py "${WEIGHTS}" \
    --env-file .env --env-example .env.example --python "${ROOT}/.venv/bin/python" \
    "${base_flag[@]}"
}

case "${1:-}" in
  setup) cmd_setup ;;
  download) cmd_download ;;
  all) cmd_setup && cmd_download ;;
  *) die "usage: bash scripts/setup.sh setup|download|all" ;;
esac

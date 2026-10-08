#!/usr/bin/env bash
# Launch one WM training run with torchrun.
#
#   bash scripts/run.sh experiment=<family>/<name> [hydra overrides...]
#
# Single node: nothing else to set; every local GPU is used.
# Multi node:  export NNODES, NODE_RANK and MASTER_ADDR (optionally MASTER_PORT
#              and NPROC_PER_NODE) on each node and run the same command.
#
# Machine-specific settings (interpreter, cache volume, credentials) belong in
# the gitignored .env at the repository root, not here; see .env.example.
# Everything about the experiment itself is a normal config override.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Fill in machine-local settings from .env (KEY=value or export KEY=value, no
# quoting) without overriding anything already exported by the caller.
if [[ -f .env ]]; then
  while IFS='=' read -r key value; do
    key="${key#export }"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!key:-}" ]] || export "${key}=${value}"
  done < .env
fi

if [[ "${NNODES:-1}" == 1 ]]; then
  rendezvous=(--standalone)
else
  rendezvous=(
    --nnodes="${NNODES}"
    --node-rank="${NODE_RANK:?NODE_RANK is required for a multi-node launch}"
    --master-addr="${MASTER_ADDR:?MASTER_ADDR is required for a multi-node launch}"
    --master-port="${MASTER_PORT:-29500}"
  )
fi

# The first compiled forward can write several GiB of Inductor/Triton/CUDA
# caches and temp files; on a small container overlay that surfaces later as an
# NCCL watchdog timeout on a different rank. When WM_RUNTIME_CACHE_ROOT points
# at a large volume, keep those files (and NCCL flight-recorder dumps) under one
# launch-scoped directory there. Explicit per-variable values are never
# overwritten.
if [[ -n "${WM_RUNTIME_CACHE_ROOT:-}" ]]; then
  scope="${WM_RUNTIME_CACHE_SCOPE:-node${NODE_RANK:-0}-${MASTER_PORT:-pid$$}}"
  : "${TORCHINDUCTOR_CACHE_DIR:=${WM_RUNTIME_CACHE_ROOT}/torchinductor/${scope}}"
  : "${TORCH_EXTENSIONS_DIR:=${WM_RUNTIME_CACHE_ROOT}/torch-extensions/${scope}}"
  : "${TRITON_CACHE_DIR:=${WM_RUNTIME_CACHE_ROOT}/triton/${scope}}"
  : "${CUDA_CACHE_PATH:=${WM_RUNTIME_CACHE_ROOT}/cuda/${scope}}"
  : "${TMPDIR:=${WM_RUNTIME_CACHE_ROOT}/tmp/${scope}}"
  : "${TORCH_NCCL_DEBUG_INFO_PIPE_FILE:=${WM_RUNTIME_CACHE_ROOT}/nccl/${scope}/dump-rank-}"
  mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" "${TMPDIR}" "$(dirname "${TORCH_NCCL_DEBUG_INFO_PIPE_FILE}")"
  export TORCHINDUCTOR_CACHE_DIR TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR CUDA_CACHE_PATH \
    TMPDIR TORCH_NCCL_DEBUG_INFO_PIPE_FILE
  echo "[wm.run] runtime caches under ${WM_RUNTIME_CACHE_ROOT} (${scope})"
fi

# Bound Inductor compile-worker fan-out (8 ranks x 32 workers oversubscribe a
# node) and keep NCCL desync diagnostics armed. No-ops for healthy runs;
# explicit values win.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-16}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-32768}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"

exec "${WM_PYTHON:-python}" -m torch.distributed.run \
  --nproc-per-node="${NPROC_PER_NODE:-gpu}" \
  "${rendezvous[@]}" \
  -m wm.train "$@"

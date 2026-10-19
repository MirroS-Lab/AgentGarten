#!/usr/bin/env bash
# Run scripts/smoke_load.py on the DGX: checks free GPU memory, then loads the released renderer checkpoint, prepares serving
# and builds a Cosmos3Stream, reporting load time and GPU memory used. No forward pass (that needs a real depth/normal-annotated
# clip); this only confirms the checkpoint, the venv and the GPU are all working together.
#
# Usage:   bash scripts/run_smoke.sh [--host user@host] [--remote-dir PATH] [checkpoint]
set -euo pipefail

host="kschmid@10.74.11.118"
remote_dir="/localhome/kschmid/AgentGarten"
checkpoint=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) host="${2:?--host needs user@host}"; shift 2 ;;
    --remote-dir) remote_dir="${2:?--remote-dir needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) checkpoint="$1"; shift ;;
  esac
done

min_free="${MIN_FREE_MIB:-20000}"
free_mib="$(ssh -o BatchMode=yes "${host}" 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1' | tr -d ' ')"
echo "GPU free on ${host}: ${free_mib} MiB (needs at least ${min_free})"
if [[ "${free_mib}" -lt "${min_free}" ]]; then
  echo "ERROR: not enough free GPU memory; the GPU is shared. Not starting." >&2
  exit 1
fi

ssh -o BatchMode=yes "${host}" "cd ${remote_dir} && .venv/bin/python scripts/smoke_load.py ${checkpoint}"

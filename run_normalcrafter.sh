#!/usr/bin/env bash
# Run NormalCrafter on a video to produce a normal-map video, on the DGX (checks free GPU memory first).
# Output lands in NormalCrafter/demo_output/<name>_vis.mp4 (confirmed against our take-001 test: same width,
# height, fps and frame count as the input). That file works directly as the "normal" field of an AgentGarten
# manifest row -- ManifestVideoDataset reads it as a video and normalizes it itself, no .npy conversion needed.
#
# NOTE: ViPE (for depth) runs on a separate machine (locally, via a different venv/repo) and is not wrapped
# here; a single script cannot span both environments. Build the manifest row only once both outputs exist.
#
# Usage:   bash scripts/run_normalcrafter.sh VIDEO_PATH [--max-res 864] [--host user@host] [--remote-dir PATH]
set -euo pipefail

host="kschmid@10.74.11.118"
remote_dir="/localhome/kschmid/NormalCrafter"
max_res=864
video=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) host="${2:?--host needs user@host}"; shift 2 ;;
    --remote-dir) remote_dir="${2:?--remote-dir needs a path}"; shift 2 ;;
    --max-res) max_res="${2:?--max-res needs a number}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) video="$1"; shift ;;
  esac
done
[[ -n "${video}" && -f "${video}" ]] || { echo "ERROR: give an existing video path" >&2; exit 2; }

min_free="${MIN_FREE_MIB:-10000}"
free_mib="$(ssh -o BatchMode=yes "${host}" 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1' | tr -d ' ')"
echo "GPU free on ${host}: ${free_mib} MiB (needs at least ${min_free})"
if [[ "${free_mib}" -lt "${min_free}" ]]; then
  echo "ERROR: not enough free GPU memory; the GPU is shared. Not starting." >&2
  exit 1
fi

name="$(basename "${video}")"
name="${name%.*}"
echo "Copying ${video} -> ${host}:${remote_dir}/examples/${name}.mp4"
ssh -o BatchMode=yes "${host}" "mkdir -p ${remote_dir}/examples"
scp -q -o BatchMode=yes "${video}" "${host}:${remote_dir}/examples/${name}.mp4"

echo "Running NormalCrafter (max-res ${max_res})"
ssh -o BatchMode=yes "${host}" "cd ${remote_dir} && .venv/bin/python run.py --video-path examples/${name}.mp4 --max-res ${max_res}"

echo "Output: ${host}:${remote_dir}/demo_output/${name}_vis.mp4"
echo "Copy it locally with: scp ${host}:${remote_dir}/demo_output/${name}_vis.mp4 <local destination>"

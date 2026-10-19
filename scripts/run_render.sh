#!/usr/bin/env bash
# Render a take through the released AgentGarten renderer on the DGX (wraps scripts/run_cosmos_inference.py).
# Checks free GPU memory first, scales an --anchor image to the take's resolution, names the output folder from the
# options when --out is not given, and prints the path of the finished video.
#
# Usage:   bash scripts/run_render.sh [--data DIR] [--anchor IMAGE] [--frames N] [--caption TEXT] [--car-color red|blue|yellow|green] [--out DIR]
#   --data       take folder with normal.mp4, depth.npy, depth_meta.json (and photoreal.mp4 when no --anchor)   [data/matrix4_take001]
#   --anchor     first-frame image of any size; scaled to the take's resolution into <out>/anchor.png
#   --frames     1 + 16*n, at most the take length (399 frames -> 385)                                         [129]
#   --caption    prompt text; one prompt for the whole clip
#   --car-color  recolor the car in the anchor frame and add the color to the caption
#   --out        output folder                                                                                  [outputs/<take>_<frames>[_<color>]]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

data="data/matrix4_take001"
anchor=""
frames=129
caption="A sedan drives down a wet city street between brick buildings, third-person chase camera behind the car, daytime, overcast, realistic."
car_color=""
out=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) data="${2:?--data needs a folder}"; shift 2 ;;
    --anchor) anchor="${2:?--anchor needs an image path}"; shift 2 ;;
    --frames) frames="${2:?--frames needs a number}"; shift 2 ;;
    --caption) caption="${2:?--caption needs text}"; shift 2 ;;
    --car-color) car_color="${2:?--car-color needs red, blue, yellow or green}"; shift 2 ;;
    --out) out="${2:?--out needs a folder}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
  esac
done

[[ -d "${data}" ]] || { echo "ERROR: data folder not found: ${data}" >&2; exit 2; }
for required in normal.mp4 depth.npy depth_meta.json; do
  [[ -f "${data}/${required}" ]] || { echo "ERROR: missing ${data}/${required}" >&2; exit 2; }
done
[[ -n "${anchor}" || -f "${data}/photoreal.mp4" ]] || { echo "ERROR: no --anchor given and ${data}/photoreal.mp4 is missing" >&2; exit 2; }
[[ -z "${anchor}" || -f "${anchor}" ]] || { echo "ERROR: anchor image not found: ${anchor}" >&2; exit 2; }
if (( (frames - 1) % 16 != 0 )); then
  echo "ERROR: --frames must be 1 + 16*n (e.g. 33, 65, 129, 257, 385), got ${frames}" >&2
  exit 2
fi

min_free="${MIN_FREE_MIB:-20000}"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
echo "GPU free: ${free_mib} MiB (needs at least ${min_free})"
if [[ "${free_mib}" -lt "${min_free}" ]]; then
  echo "ERROR: not enough free GPU memory; the GPU is shared. Not starting." >&2
  exit 1
fi

if [[ -z "${out}" ]]; then
  out="outputs/$(basename "${data}")_${frames}${car_color:+_${car_color}}"
fi
mkdir -p "${out}"

args=(--data "${data}" --frames "${frames}" --caption "${caption}" --out "${out}")
if [[ -n "${anchor}" ]]; then
  read -r width height < <(.venv/bin/python -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["width"], m["height"])' "${data}/depth_meta.json")
  ffmpeg -v error -y -i "${anchor}" -vf "scale=${width}:${height}:flags=lanczos" -frames:v 1 "${out}/anchor.png"
  args+=(--anchor "${out}/anchor.png")
fi
if [[ -n "${car_color}" ]]; then
  args+=(--car-color "${car_color}")
fi

.venv/bin/python scripts/run_cosmos_inference.py "${args[@]}"
echo "Output: $(pwd)/${out}/sample.mp4"

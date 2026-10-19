#!/usr/bin/env python3
"""Build one AgentGarten training/inference manifest row for a real (video, depth, normal) triple.

Confirmed real formats, not guessed:
  - video:  any video file ManifestVideoDataset can read (wm/data/video.py's _read_video).
  - normal: a video file works directly -- confirmed against NormalCrafter's own output
            (demo_output/<name>_vis.mp4), which ManifestVideoDataset reads and normalizes
            to [-1, 1] by itself (no .npy conversion needed).
  - depth:  NOT YET WIRED UP. ViPE's dav3-pipeline output format for this clip has not been
            inspected yet, so no depth path is built here -- adding one before confirming the
            real file layout would be a guess, not a fix. Run this again once that is known.

Usage:   python scripts/build_manifest.py --video V --normal N --depth D --caption "..." --depth-scale 1.0 --out manifest.jsonl
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--normal", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path, help=".npy, (T, H, W); not yet produced by this pipeline")
    parser.add_argument("--caption", required=True)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    for path, label in ((args.video, "video"), (args.normal, "normal"), (args.depth, "depth")):
        if not path.is_file():
            raise SystemExit(f"{label} file does not exist: {path}")

    manifest_dir = args.out.parent
    row = {
        "video": str(args.video.resolve().relative_to(manifest_dir.resolve()) if _is_under(args.video, manifest_dir) else args.video.resolve()),
        "depth": str(args.depth.resolve().relative_to(manifest_dir.resolve()) if _is_under(args.depth, manifest_dir) else args.depth.resolve()),
        "normal": str(args.normal.resolve().relative_to(manifest_dir.resolve()) if _is_under(args.normal, manifest_dir) else args.normal.resolve()),
        "caption": args.caption,
        "depth_scale": args.depth_scale,
    }
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"appended one row to {args.out}")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real end-to-end AgentGarten inference on a Matrix-Game-4 engine-rendered take.

Builds anchor + per-block conditions straight from our manifest row (video,
depth.npy, normal.mp4) and streams them through the released renderer
(weights/AgentGarten-renderer) with Cosmos3Stream, exactly as README.md's
"Inference" section shows. Bypasses ManifestVideoDataset's torchcodec reader
(not installed for aarch64 here) and decodes video/normal with ffmpeg instead,
reproducing the same uint8 [T,3,H,W] / normalized-unit-vector conversions
video.py itself does.

Usage: .venv/bin/python scripts/run_matrix4_inference.py \
    --data data/matrix4_take001 --frames 33 --out outputs/matrix4_take001
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch

from wm.codecs import Wan22VAECodec
from wm.inference.cosmos3 import inspect_artifact, load_artifact
from wm.inference.serving import prepare_serving
from wm.models.dmd import RCM_ENDPOINTS
from wm.models.flow import rf_interpolate, rf_x0
from wm.networks.cosmos3.conditioner import Cosmos3Conditioner
from wm.networks.cosmos3.network import Cosmos3Condition
from wm.networks.cosmos3.streaming import Cosmos3Stream, StreamPolicy


def load_dotenv(path: Path) -> None:
    """Fill os.environ from KEY=value / export KEY=value lines, like scripts/run.sh; never overrides the caller's environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.removeprefix("export ").partition("=")
        if sep and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) and not os.environ.get(key):
            os.environ[key] = value


CAR_TINTS = {
    "red": (1.0, 0.10, 0.08),
    "blue": (0.10, 0.25, 1.0),
    "yellow": (1.0, 0.85, 0.10),
    "green": (0.10, 0.80, 0.20),
}


def recolor_car(frame: torch.Tensor, normal_frame: torch.Tensor, tint: tuple[float, float, float]) -> torch.Tensor:
    """Tint the car body in the anchor frame. frame/normal_frame: [1,3,H,W] uint8 (RGB / normal-map video frame 0).

    The car is the box that stands out from the road in the normal map, found in the lower centre of the image;
    only its bright body pixels are tinted (luminance kept, so shading and reflections survive), which leaves
    windows, shadows and the asphalt inside the box alone. The region and thresholds are tuned for the
    behind-the-car chase view of this take.
    """
    rgb = frame[0].float()
    normal = normal_frame[0].float()
    _, h, w = rgb.shape
    road = normal[:, int(h * 0.85):, : int(w * 0.25)].reshape(3, -1).median(dim=1).values
    standout = (normal - road[:, None, None]).norm(dim=0) > 40
    region = torch.zeros_like(standout)
    region[int(h * 0.45): int(h * 0.97), int(w * 0.30): int(w * 0.80)] = True
    mask = region & standout
    columns = (mask.sum(0) > 0.35 * mask.sum(0).max()).nonzero().flatten()
    rows = (mask.sum(1) > 0.35 * mask.sum(1).max()).nonzero().flatten()
    trim = int(0.075 * (columns.max() - columns.min()))
    box = torch.zeros_like(mask)
    box[rows.min(): rows.max() + 1, columns.min() + trim: columns.max() + 1 - trim] = True
    box &= mask
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    body = box & (luminance > 0.55 * torch.quantile(luminance[box], 0.7))
    alpha = torch.nn.functional.avg_pool2d(body[None, None].float(), 5, stride=1, padding=2)[0]
    tinted = (luminance[None] * torch.tensor(tint, device=rgb.device)[:, None, None] * 1.15).clamp(0, 255)
    return (rgb * (1 - alpha) + tinted * alpha).round().to(torch.uint8)[None]


def read_video_ffmpeg(path: Path, width: int, height: int, frames: int) -> torch.Tensor:
    out = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vframes", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True, capture_output=True,
    ).stdout
    array = np.frombuffer(out, dtype=np.uint8).reshape(frames, height, width, 3)
    return torch.from_numpy(array.copy()).permute(0, 3, 1, 2).contiguous()  # [T,3,H,W] uint8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="dir with normal.mp4, depth.npy, depth_meta.json (and photoreal.mp4 unless --anchor is given)")
    parser.add_argument("--anchor", type=Path, default=None, help="first-frame image (or video, first frame used); defaults to <data>/photoreal.mp4")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/AgentGarten-renderer"))
    parser.add_argument("--wan22-vae", type=Path, default=None, help="defaults to $WM_WAN22_VAE")
    parser.add_argument("--caption", default="A car drives through a low-poly city street at dusk, flat-shaded buildings and road markings visible.")
    parser.add_argument("--car-color", choices=sorted(CAR_TINTS), default=None, help="recolor the car in the anchor frame and add the color to the caption")
    parser.add_argument("--frames", type=int, default=33, help="pixel frames; must be 1+4*block_frames*n_blocks")
    parser.add_argument("--block-frames", type=int, default=4)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU is visible")
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    print("loading renderer artifact...")
    spec = inspect_artifact(args.checkpoint)
    artifact = load_artifact(spec, device=device)

    wan22_vae = args.wan22_vae
    if wan22_vae is None:
        if "WM_WAN22_VAE" not in os.environ:
            raise SystemExit("WM_WAN22_VAE is not set: pass --wan22-vae, or run `bash scripts/setup.sh download` to fill .env")
        wan22_vae = Path(os.environ["WM_WAN22_VAE"])
    codec = Wan22VAECodec(pretrained_path=wan22_vae, dtype=torch.bfloat16, device=device)

    conditioner = Cosmos3Conditioner(
        config=spec.config,
        codec=codec,
        tokenizer_path=str(artifact.tokenizer_path),
        negative_prompt_path=None,
        default_fps=float(spec.config.base_fps),
        init_device="meta",
    )
    conditioner.text_encoder = artifact.context_encoder  # real UND weights, not the meta stub

    meta = json.loads((args.data / "depth_meta.json").read_text())
    width, height = meta["width"], meta["height"]
    source_fps = meta["fps"]

    print(f"reading {args.frames} frames of depth/normal at {width}x{height}, {source_fps} fps...")
    # Only the first latent frame (the anchor) comes from the RGB side, and the causal VAE
    # encodes it from pixel frame 0 alone, so one image is enough; repeat it to the clip length
    # so the video latent has the same shape as the depth/normal latents.
    anchor_source = args.anchor if args.anchor is not None else args.data / "photoreal.mp4"
    first_frame = read_video_ffmpeg(anchor_source, width, height, 1)
    normal_u8 = read_video_ffmpeg(args.data / "normal.mp4", width, height, args.frames)
    if args.car_color is not None:
        first_frame = recolor_car(first_frame, normal_u8[:1], CAR_TINTS[args.car_color])
    video = first_frame.expand(args.frames, -1, -1, -1).contiguous()
    normal_rgb = normal_u8.float()
    normal = normal_rgb / 127.5 - 1.0
    normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-6)

    depth_full = np.load(args.data / "depth.npy")[: args.frames]
    depth = torch.from_numpy(depth_full).float()
    if (depth < 0).any():
        raise ValueError("depth must be non-negative")

    batch = {
        "video": video.unsqueeze(0).to(device),
        "depth": depth[None, :, None].to(device),
        "normal": normal.unsqueeze(0).to(device),
        "caption": [args.caption if args.car_color is None else f"{args.caption} The car is bright {args.car_color}."],
        "fps": torch.tensor([source_fps], device=device),
        "depth_scale": torch.tensor([1.0], device=device),
        "sample_id": [0],
    }

    print("encoding video/depth/normal to latents and tokenizing caption...")
    encoded = conditioner.encode(batch, training=False)

    block_frames = args.block_frames
    latent_frames = encoded["latent"].shape[2]
    num_blocks = (latent_frames - 1) // block_frames
    if num_blocks < 1 or 1 + num_blocks * block_frames != latent_frames:
        suggestion = 1 + 4 * block_frames * max(1, num_blocks)
        raise SystemExit(
            f"--frames {args.frames} gives {latent_frames} latent frames, not "
            f"1 + {block_frames}*N; try --frames {suggestion}"
        )
    print(f"{latent_frames} latent frames: 1 anchor + {num_blocks} blocks of {block_frames}")

    # One shared TextContext: Cosmos3Stream identifies "same prompt" by object
    # identity (see streaming.py's validate_chunk), so the text tower runs once
    # here and every block's condition reuses this exact object.
    shared_text = conditioner.text_encoder(
        encoded["text_input_ids"], encoded["text_attention_mask"]
    )
    fps = torch.as_tensor(encoded["fps"], dtype=torch.float32).reshape(-1)

    def slice_condition(start: int, stop: int) -> Cosmos3Condition:
        return Cosmos3Condition(
            text=shared_text,
            depth=encoded["depth_latent"][:, :, start:stop],
            normal=encoded["normal_latent"][:, :, start:stop],
            fps=fps,
        )

    anchor = encoded["latent"][:, :, :1]
    first_condition = slice_condition(0, 1)

    prepare_serving(artifact.student)
    policy = StreamPolicy(block_frames=block_frames)
    stream = Cosmos3Stream(artifact.student, policy)
    stream.start(anchor, first_condition)

    committed = [anchor]
    block_shape = (1, anchor.shape[1], block_frames, *anchor.shape[3:])
    for block_index in range(num_blocks):
        start = 1 + block_index * block_frames
        condition = slice_condition(start, start + block_frames)
        clean = None
        for sigma_value in RCM_ENDPOINTS:
            sigma = torch.tensor([sigma_value], device=device)
            noise = torch.randn(block_shape, device=device, dtype=anchor.dtype)
            noisy = noise if clean is None else rf_interpolate(clean, noise, sigma)
            velocity = stream.denoise(noisy, sigma, condition)
            clean = rf_x0(noisy, sigma, velocity)
        stream.commit(clean, condition)
        committed.append(clean)
        print(f"block {block_index + 1}/{num_blocks} committed")

    print("decoding latents to RGB...")
    full_latent = torch.cat(committed, dim=2)
    video_out = conditioner.decode(full_latent)  # [1,3,T,H,W] in [-1,1]
    frames_u8 = ((video_out.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    frames_u8 = frames_u8[0].permute(1, 2, 3, 0).cpu().numpy()  # [T,H,W,3]

    raw_path = args.out / "sample.raw"
    frames_u8.tofile(raw_path)
    mp4_path = args.out / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{frames_u8.shape[2]}x{frames_u8.shape[1]}", "-r", str(source_fps),
            "-i", str(raw_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
            str(mp4_path),
        ],
        check=True,
    )
    raw_path.unlink()
    print(f"done: {frames_u8.shape[0]} frames -> {mp4_path}")


if __name__ == "__main__":
    main()

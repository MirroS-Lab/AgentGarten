# SPDX-License-Identifier: Apache-2.0

"""Datasets of prepared video clips with aligned depth and normal controls.

Manifest rows (JSONL, paths relative to the manifest), one prepared clip each:
  {"video": "a.mp4", "depth": "a_depth.npy", "normal": "a_normal.mp4",
   "caption": "...", "depth_scale": 1.0}
Every modality holds exactly num_frames frames of height x width, aligned
frame by frame and sampled at `fps`. Clips are read as they are: no temporal
resampling, resizing or cropping. depth is a [T, H, W] metric array; normal
an RGB-encoded video or a [T, H, W, 3] array of unit vectors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ManifestVideoDataset", "SyntheticVideoDataset"]


def _read_video(path: Path) -> torch.Tensor:
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as error:
        raise ImportError(
            "reading manifest videos needs torchcodec: pip install 'wm[data]'"
        ) from error
    decoder = VideoDecoder(str(path))
    return decoder.get_frames_in_range(0, len(decoder)).data


class ManifestVideoDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str,
        *,
        num_frames: int,
        height: int,
        width: int,
        fps: float = 16.0,
    ) -> None:
        self.path = Path(manifest)
        text = self.path.read_text()
        self.rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"manifest {manifest} is empty")
        self.digest = hashlib.sha256(text.encode()).hexdigest()
        self.num_frames = int(num_frames)
        self.height, self.width = int(height), int(width)
        self.fps = float(fps)

    def __len__(self) -> int:
        return len(self.rows)

    def resume_signature(self) -> dict[str, Any]:
        return {
            "manifest": self.digest,
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "fps": self.fps,
        }

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.path.parent / path

    def _check(self, path: Path, value: torch.Tensor, shape: tuple) -> torch.Tensor:
        expected = (self.num_frames, *shape)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"{path}: expected {expected} (frames, ...), got {tuple(value.shape)}"
            )
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{path}: contains non-finite values")
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        size = (self.height, self.width)
        video_path = self._path(row["video"])
        video = self._check(video_path, _read_video(video_path), (3, *size))
        depth_path = self._path(row["depth"])
        depth = torch.from_numpy(np.load(depth_path)).float()
        depth = self._check(depth_path, depth, size)
        if (depth < 0).any():
            raise ValueError(f"{depth_path}: depth must be non-negative")
        normal_path = self._path(row["normal"])
        if normal_path.suffix == ".npy":
            normal = torch.from_numpy(np.load(normal_path)).float()
            normal = self._check(normal_path, normal, (*size, 3)).permute(0, 3, 1, 2)
        else:
            normal = _read_video(normal_path).float() / 127.5 - 1.0
            normal = self._check(normal_path, normal, (3, *size))
        normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "video": video,
            "depth": depth[:, None],
            "normal": normal,
            "caption": str(row.get("caption", "")),
            "fps": self.fps,
            "depth_scale": float(row.get("depth_scale", 1.0)),
            "sample_id": index,
        }


class SyntheticVideoDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        length: int,
        num_frames: int,
        height: int,
        width: int,
        fps: float = 16.0,
    ) -> None:
        self.length, self.num_frames = int(length), int(num_frames)
        self.height, self.width, self.fps = int(height), int(width), float(fps)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(index)
        shape = (self.num_frames, self.height, self.width)
        normal = torch.randn(shape[0], 3, *shape[1:], generator=generator)
        return {
            "video": torch.randint(
                0,
                256,
                (shape[0], 3, *shape[1:]),
                generator=generator,
                dtype=torch.uint8,
            ),
            "depth": 1.0
            + 9.0 * torch.rand(shape[0], 1, *shape[1:], generator=generator),
            "normal": normal / normal.norm(dim=1, keepdim=True),
            "caption": f"synthetic clip {index}",
            "fps": self.fps,
            "depth_scale": 1.0,
            "sample_id": index,
        }

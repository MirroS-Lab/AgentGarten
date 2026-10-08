# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from io import BytesIO
from threading import Thread
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from wm.infra import io as easy_io
from wm.infra import log

_Layout = Literal["BCTHW", "BTCHW"]
_FloatRange = Literal["minus_one_one", "zero_one"]
_ImageFormat = Literal["jpg", "png"]


def _canonical_video_batch(value: Tensor, *, layout: _Layout) -> Tensor:
    input_shape = tuple(value.shape)
    if value.ndim not in (4, 5):
        raise ValueError(
            "media tensor must contain RGB video in BCTHW/CTHW or "
            f"BTCHW/TCHW layout, got {input_shape}"
        )
    if layout == "BCTHW":
        if value.ndim == 4:
            value = value.unsqueeze(0)
        result = value
    elif layout == "BTCHW":
        if value.ndim == 4:
            value = value.unsqueeze(0)
        result = value.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"unsupported media layout: {layout!r}")

    if result.ndim != 5 or result.shape[1] != 3:
        raise ValueError(
            "media tensor must contain RGB video in BCTHW/CTHW or "
            f"BTCHW/TCHW layout, got {input_shape}"
        )
    return result


def _video_to_cpu_uint8(
    video: Tensor,
    *,
    float_range: _FloatRange,
    frame_chunk_size: int,
) -> Tensor:
    channels, frames, height, width = video.shape
    if channels != 3:
        raise ValueError(f"media tensor must have three RGB channels, got {channels}")
    if frame_chunk_size <= 0:
        raise ValueError("frame_chunk_size must be positive")
    if video.dtype != torch.uint8 and not torch.is_floating_point(video):
        raise TypeError(
            f"media tensor must be floating point or uint8, got {video.dtype}"
        )

    result = torch.empty((frames, height, width, channels), dtype=torch.uint8)
    for start in range(0, frames, frame_chunk_size):
        stop = min(start + frame_chunk_size, frames)
        chunk = video[:, start:stop].detach()
        if chunk.dtype != torch.uint8:
            chunk = chunk.float()
            if float_range == "minus_one_one":
                chunk = (chunk + 1.0) * 127.5
            elif float_range == "zero_one":
                chunk = chunk * 255.0
            else:
                raise ValueError(f"unsupported floating-point range: {float_range!r}")
            chunk = chunk.round().clamp_(0, 255).to(torch.uint8)
        result[start:stop].copy_(
            chunk.permute(1, 2, 3, 0).to(device="cpu", non_blocking=False)
        )
    return result


def video_batch_to_cpu_uint8(
    value: Tensor,
    *,
    layout: _Layout = "BCTHW",
    float_range: _FloatRange = "minus_one_one",
    frame_chunk_size: int = 8,
) -> Tensor:
    batch = _canonical_video_batch(value, layout=layout)
    result = torch.empty(tuple(batch.shape), dtype=torch.uint8, device="cpu")
    for batch_index, video in enumerate(batch):
        pixels = _video_to_cpu_uint8(
            video,
            float_range=float_range,
            frame_chunk_size=frame_chunk_size,
        )
        result[batch_index].copy_(pixels.permute(3, 0, 1, 2))
    return result


def _safe_name(value: Any) -> str:
    raw = str(value)
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in raw
    )
    return safe.strip("._") or "sample"


def media_artifact_filename(
    *,
    artifact_name: str,
    iteration: int,
    rank: int,
    sample_batch_index: int,
    extension: str,
    batch_index: int | None = None,
) -> str:
    indices = (int(sample_batch_index),)
    if batch_index is not None:
        indices += (int(batch_index),)
    result = "_".join(f"{index:02d}" for index in indices)
    separator = "@" if any(index >= 100 for index in indices) else ""
    return (
        f"Iter{int(iteration):09d}_Rank{int(rank):04d}_"
        f"{_safe_name(artifact_name)}{separator}{result}.{extension}"
    )


def _grid_options(
    grid_layout: Sequence[Any] | None,
    grid_condition_keys: Sequence[Any] | None,
    grid_reference_key: Any | None,
) -> tuple[tuple[Any, ...] | None, frozenset[Any], Any | None]:
    if grid_layout is None:
        return None, frozenset(), None
    layout = tuple(grid_layout)
    if len(layout) not in (3, 4):
        raise ValueError("video grid layout must contain three or four artifact keys")
    if len(set(layout)) != len(layout):
        raise ValueError("video grid layout keys must be unique")
    default_condition_count = 1 if len(layout) == 3 else 2
    conditions = frozenset(
        layout[:default_condition_count]
        if grid_condition_keys is None
        else grid_condition_keys
    )
    if not conditions.issubset(layout):
        raise ValueError("video grid condition keys must belong to the grid layout")
    reference = layout[-1] if grid_reference_key is None else grid_reference_key
    if reference not in layout:
        raise ValueError("video grid reference key must belong to the grid layout")
    return layout, conditions, reference


def _resize_cpu_uint8_video(
    video: Tensor,
    *,
    size: tuple[int, int],
    frame_chunk_size: int,
) -> Tensor:
    if tuple(video.shape[1:3]) == tuple(size):
        return video
    frames = int(video.shape[0])
    height, width = (int(size[0]), int(size[1]))
    result = torch.empty((frames, height, width, 3), dtype=torch.uint8)
    for start in range(0, frames, frame_chunk_size):
        stop = min(start + frame_chunk_size, frames)
        chunk = video[start:stop].permute(0, 3, 1, 2).float()
        resized = F.interpolate(
            chunk,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        result[start:stop].copy_(
            resized.round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        )
    return result


def _video_grid_batches(
    artifacts: Mapping[Any, Any],
    *,
    grid_layout: tuple[Any, ...],
    grid_condition_keys: frozenset[Any],
    grid_reference_key: Any,
    layout: _Layout,
    float_range: _FloatRange,
    frame_chunk_size: int,
) -> Iterator[tuple[int, Tensor]]:
    selected: dict[Any, Tensor] = {}
    for key in grid_layout:
        value = artifacts[key]
        if not torch.is_tensor(value):
            raise TypeError(f"media artifact {key!r} is not a tensor")
        selected[key] = _canonical_video_batch(value, layout=layout)

    batch_sizes = {int(video.shape[0]) for video in selected.values()}
    if len(batch_sizes) != 1:
        raise ValueError("video grid artifacts must share one batch size")
    common_frames = min(int(video.shape[2]) for video in selected.values())
    if common_frames <= 0:
        raise ValueError("video grid artifacts must contain at least one frame")
    reference = selected[grid_reference_key]
    target_size = (int(reference.shape[-2]), int(reference.shape[-1]))
    for key, video in selected.items():
        if key not in grid_condition_keys and tuple(video.shape[-2:]) != target_size:
            raise ValueError(
                "non-condition video grid artifacts must match the reference "
                f"spatial size; {key!r} has {tuple(video.shape[-2:])}, "
                f"reference {grid_reference_key!r} has {target_size}"
            )

    height, width = target_size
    rows, columns = (1, 3) if len(grid_layout) == 3 else (2, 2)
    for batch_index in range(next(iter(batch_sizes))):
        grid = torch.empty(
            (common_frames, rows * height, columns * width, 3),
            dtype=torch.uint8,
        )
        panels = tuple(
            grid[
                :,
                (index // columns) * height : (index // columns + 1) * height,
                (index % columns) * width : (index % columns + 1) * width,
            ]
            for index in range(len(grid_layout))
        )
        for key, panel in zip(grid_layout, panels, strict=True):
            pixels = _video_to_cpu_uint8(
                selected[key][batch_index, :, :common_frames],
                float_range=float_range,
                frame_chunk_size=frame_chunk_size,
            )
            if key in grid_condition_keys:
                pixels = _resize_cpu_uint8_video(
                    pixels,
                    size=target_size,
                    frame_chunk_size=frame_chunk_size,
                )
            panel.copy_(pixels)
        yield batch_index, grid


def _cpu_media_artifacts(
    value: Any,
    *,
    layout: _Layout,
    float_range: _FloatRange,
    frame_chunk_size: int,
    grid_layout: tuple[Any, Any, Any, Any] | None,
    grid_condition_keys: frozenset[Any],
    grid_reference_key: Any | None,
    grid_name: str,
    save_individual_artifacts: bool,
) -> Iterator[tuple[str, int, int, Tensor]]:
    artifacts = {"sample": value} if torch.is_tensor(value) else value
    if not isinstance(artifacts, Mapping):
        raise TypeError("image/video sample must be a tensor or a mapping of tensors")

    grid_matches = grid_layout is not None and all(
        key in artifacts for key in grid_layout
    )
    used_names: set[str] = set()
    if grid_matches:
        assert grid_layout is not None and grid_reference_key is not None
        artifact_name = _safe_name(grid_name)
        used_names.add(artifact_name)
        batches = _video_grid_batches(
            artifacts,
            grid_layout=grid_layout,
            grid_condition_keys=grid_condition_keys,
            grid_reference_key=grid_reference_key,
            layout=layout,
            float_range=float_range,
            frame_chunk_size=frame_chunk_size,
        )
        batch_size = int(
            _canonical_video_batch(artifacts[grid_reference_key], layout=layout).shape[
                0
            ]
        )
        for batch_index, pixels in batches:
            yield artifact_name, batch_index, batch_size, pixels

    if grid_matches and not save_individual_artifacts:
        return

    for key in sorted(artifacts, key=str):
        tensor = artifacts[key]
        if not torch.is_tensor(tensor):
            raise TypeError(f"media artifact {key!r} is not a tensor")
        base_name = _safe_name(key)
        artifact_name = base_name
        suffix = 2
        while artifact_name in used_names:
            artifact_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(artifact_name)
        batch = _canonical_video_batch(tensor, layout=layout)
        for batch_index, video in enumerate(batch):
            # Encoders already consume THWC. Keep one compact snapshot per
            # video instead of copying through a full BCTHW batch allocation.
            pixels = _video_to_cpu_uint8(
                video,
                float_range=float_range,
                frame_chunk_size=frame_chunk_size,
            )
            yield (
                artifact_name,
                batch_index,
                len(batch),
                pixels,
            )


def _encode_cpu_video(
    pixels: Tensor,
    *,
    fps: int,
    quality: int,
) -> BytesIO:
    return easy_io.encode_video(pixels.numpy(), fps=fps, quality=quality)


def _encode_cpu_image(
    pixels: Tensor,
    *,
    file_format: _ImageFormat,
    quality: int,
) -> BytesIO:
    options = {"quality": int(quality)} if file_format == "jpg" else {}
    return easy_io.encode_image(pixels.numpy(), file_format=file_format, **options)


class ImageVideoArtifactWriter:
    def __init__(
        self,
        *,
        layout: _Layout = "BCTHW",
        float_range: _FloatRange = "minus_one_one",
        fps: int = 16,
        frame_chunk_size: int = 8,
        image_format: _ImageFormat = "jpg",
        image_quality: int = 90,
        video_quality: int = 5,
        grid_layout: Sequence[Any] | None = None,
        grid_condition_keys: Sequence[Any] | None = None,
        grid_reference_key: Any | None = None,
        grid_name: str = "grid",
        save_individual_artifacts: bool = True,
        async_write: bool = False,
    ) -> None:
        self.layout = layout
        self.float_range = float_range
        self.fps = int(fps)
        self.frame_chunk_size = int(frame_chunk_size)
        self.image_format = image_format
        self.image_quality = int(image_quality)
        self.video_quality = int(video_quality)
        (
            self.grid_layout,
            self.grid_condition_keys,
            self.grid_reference_key,
        ) = _grid_options(
            grid_layout,
            grid_condition_keys,
            grid_reference_key,
        )
        self.grid_name = str(grid_name)
        self.save_individual_artifacts = bool(save_individual_artifacts)
        self.async_write = bool(async_write)
        self._write_failed = False
        self._worker: Thread | None = None

    def _join(self, root: str, *parts: str) -> str:
        return easy_io.join_path(root, *parts)

    def _encode_and_publish(self, pixels: Tensor, destination: str) -> None:
        if pixels.shape[0] == 1:
            buffer = _encode_cpu_image(
                pixels[0],
                file_format=self.image_format,
                quality=self.image_quality,
            )
        else:
            buffer = _encode_cpu_video(
                pixels,
                fps=self.fps,
                quality=self.video_quality,
            )
        easy_io.atomic_put(buffer, destination)

    def _publish_artifacts(
        self,
        artifacts: Sequence[tuple[Tensor, str]],
        *,
        directory: str,
    ) -> None:
        del directory
        for pixels, destination in artifacts:
            self._encode_and_publish(pixels, destination)

    def _publish_artifacts_async(
        self,
        artifacts: Sequence[tuple[Tensor, str]],
        *,
        directory: str,
    ) -> None:
        try:
            self._publish_artifacts(artifacts, directory=directory)
        except Exception as error:
            if not self._write_failed:
                self._write_failed = True
                log.warning(
                    "Disabling asynchronous media writes after failure: "
                    f"{type(error).__name__}: {error}"
                )

    def _finish_pending_write(self) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.join()
        self._worker = None

    def write(
        self,
        value: Any,
        *,
        output_dir: str,
        split: str,
        iteration: int,
        rank: int,
        sample_batch_index: int = 0,
    ) -> None:
        del split
        if self._write_failed:
            return
        directory = str(output_dir)
        if self.async_write:
            self._finish_pending_write()
            if self._write_failed:
                return

        artifacts = []
        for artifact_name, batch_index, batch_size, pixels in _cpu_media_artifacts(
            value,
            layout=self.layout,
            float_range=self.float_range,
            frame_chunk_size=self.frame_chunk_size,
            grid_layout=self.grid_layout,
            grid_condition_keys=self.grid_condition_keys,
            grid_reference_key=self.grid_reference_key,
            grid_name=self.grid_name,
            save_individual_artifacts=self.save_individual_artifacts,
        ):
            extension = self.image_format if pixels.shape[0] == 1 else "mp4"
            filename = media_artifact_filename(
                artifact_name=artifact_name,
                iteration=iteration,
                rank=rank,
                sample_batch_index=sample_batch_index,
                batch_index=batch_index if batch_size > 1 else None,
                extension=extension,
            )
            artifacts.append((pixels, self._join(directory, filename)))

        if not artifacts:
            return
        if not self.async_write:
            self._publish_artifacts(artifacts, directory=directory)
            return
        self._worker = Thread(
            target=self._publish_artifacts_async,
            args=(artifacts,),
            kwargs={"directory": directory},
            name="wm-media-writer",
        )
        self._worker.start()

    def close(self) -> None:
        self._finish_pending_write()


__all__ = [
    "ImageVideoArtifactWriter",
    "media_artifact_filename",
    "video_batch_to_cpu_uint8",
]

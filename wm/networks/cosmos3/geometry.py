# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
import torch.nn.functional as F


class GeometryKind(StrEnum):
    DEPTH = "depth"
    NORMAL = "normal"


class ModalitySelection(StrEnum):
    BOTH = "both"
    RANDOM_SINGLE = "random_single"
    DEPTH = "depth"
    NORMAL = "normal"


_GEOMETRY_SLOTS = (GeometryKind.DEPTH, GeometryKind.NORMAL)


@dataclass(frozen=True, slots=True)
class GeometryAugmentationConfig:
    modality_dropout_probability: float = 0.0
    modality_selection: ModalitySelection | str = "both"
    spatial_probability: float = 0.0
    spatial_max_scale: float = 1.0
    spatial_max_shift_fraction: float = 0.0
    spatial_max_shear: float = 0.0
    spatial_max_perspective: float = 0.0
    spatial_anneal_power: float = 1.0
    texture_suppression_probability: float = 0.0
    texture_downsample_min: float = 1.0
    texture_downsample_max: float = 1.0
    texture_gaussian_sigma_scale: float = 0.5
    texture_skip_tsdf: bool = True
    geometry_spatial_downsample: int = 4
    geometry_stretch_divisor: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "modality_selection", ModalitySelection(self.modality_selection)
        )
        for name in (
            "modality_dropout_probability",
            "spatial_probability",
            "texture_suppression_probability",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(self.spatial_max_scale) or self.spatial_max_scale < 1.0:
            raise ValueError("spatial_max_scale must be finite and at least one")
        for name, upper in (
            ("spatial_max_shift_fraction", 0.5),
            ("spatial_max_shear", 0.5),
            ("spatial_max_perspective", 0.25),
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= upper:
                raise ValueError(f"{name} must be finite and in [0, {upper}]")
        if not math.isfinite(self.spatial_anneal_power) or (
            self.spatial_anneal_power <= 0.0
        ):
            raise ValueError("spatial_anneal_power must be finite and positive")
        if (
            not math.isfinite(self.texture_downsample_min)
            or not math.isfinite(self.texture_downsample_max)
            or self.texture_downsample_min < 1.0
            or self.texture_downsample_max < self.texture_downsample_min
        ):
            raise ValueError(
                "texture downsample range must be finite, ordered, and at least one"
            )
        if (
            not math.isfinite(self.texture_gaussian_sigma_scale)
            or self.texture_gaussian_sigma_scale < 0.0
        ):
            raise ValueError(
                "texture_gaussian_sigma_scale must be finite and non-negative"
            )
        if int(self.geometry_spatial_downsample) <= 0:
            raise ValueError("geometry_spatial_downsample must be positive")
        if int(self.geometry_stretch_divisor) <= 0:
            raise ValueError("geometry_stretch_divisor must be positive")


def _rgb_cube_path(t: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    t = torch.where(valid, t.clamp(0.0, 1.0), 0.0)
    scaled = t[:, 0] * 7.0
    red = scaled.clamp(0.0, 1.0)
    red.sub_((scaled - 2.0).clamp(0.0, 1.0))
    red.add_((scaled - 5.0).clamp(0.0, 1.0))
    green = (scaled - 1.0).clamp(0.0, 1.0)
    green.sub_((scaled - 4.0).clamp(0.0, 1.0))
    green.add_((scaled - 6.0).clamp(0.0, 1.0))
    blue = (scaled - 3.0).clamp(0.0, 1.0)
    return torch.stack((red, green, blue), dim=1).masked_fill_(
        ~valid.expand(-1, 3, -1, -1, -1),
        0.0,
    )


def colorize_depth(depth: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(depth) & (depth > 0)
    safe = torch.where(valid, depth.clamp_min(0.0), 0.0)
    scale = scale.to(device=depth.device, dtype=torch.float32).reshape(-1)
    if scale.shape[0] == 1 and depth.shape[0] != 1:
        scale = scale.expand(depth.shape[0])
    t = 1.0 - (1.0 + safe * scale[:, None, None, None, None] / 10.0).pow(-2.0)
    return _rgb_cube_path(t, valid)


def prepare_normal(normal: torch.Tensor) -> torch.Tensor:
    if normal.dtype == torch.uint8:
        normal = normal.to(dtype=torch.float32).div_(127.5).sub_(1.0)
        finite = torch.isfinite(normal).all(dim=1, keepdim=True)
        normal.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        magnitude = torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        valid = finite & (magnitude > torch.finfo(torch.float32).eps)
        normal.div_(magnitude.clamp_min(torch.finfo(torch.float32).eps))
        return normal.masked_fill_(~valid.expand_as(normal), 0.0).contiguous()
    if not normal.is_floating_point():
        raise TypeError(
            f"raw normal must be uint8 or floating point, got {normal.dtype}"
        )
    # Float normals already carry the producer's camera-XYZ normalization.
    # They are kept as they are after the aligned resize: normalizing a second
    # time changes every geometry VAE input slightly and breaks checkpoint
    # parity.
    return normal.to(dtype=torch.float32, copy=True).contiguous()


def downsample_geometry(
    videos: Mapping[str, torch.Tensor],
    *,
    factor: int,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, video in videos.items():
        batch, channels, frames, height, width = video.shape
        rows = (
            torch.nan_to_num(video, nan=0.0, posinf=0.0, neginf=0.0)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch * frames, channels, height, width)
        )
        low = F.avg_pool2d(rows, kernel_size=factor, stride=factor)
        output[name] = (
            low.reshape(batch, frames, channels, *low.shape[-2:])
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
    return output


def _batch_flag(
    data_batch: Mapping[str, Any],
    key: str,
    *,
    batch_size: int,
) -> torch.Tensor:
    value = data_batch.get(key)
    if value is None:
        return torch.zeros(batch_size, dtype=torch.bool)
    flag = torch.as_tensor(value, device="cpu", dtype=torch.bool).reshape(-1)
    if flag.numel() == 1 and batch_size != 1:
        flag = flag.expand(batch_size)
    if flag.numel() != batch_size:
        raise ValueError(
            f"{key} must contain one flag per sample, got {flag.numel()} "
            f"for batch size {batch_size}"
        )
    return flag


def _gaussian_prefilter(
    value: torch.Tensor,
    *,
    sigma: float,
) -> torch.Tensor:
    if sigma <= 0.0:
        return value
    radius = max(1, math.ceil(3.0 * sigma))
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=value.device,
        dtype=value.dtype,
    )
    kernel = torch.exp(coordinates.square().mul(-0.5 / (sigma * sigma)))
    kernel.div_(kernel.sum())
    channels = int(value.shape[1])
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    filtered = F.conv2d(
        F.pad(value, (radius, radius, 0, 0), mode="replicate"),
        horizontal,
        groups=channels,
    )
    return F.conv2d(
        F.pad(filtered, (0, 0, radius, radius), mode="replicate"),
        vertical,
        groups=channels,
    )


def _suppress_geometry_texture(
    videos: Mapping[str, torch.Tensor],
    *,
    data_batch: Mapping[str, Any],
    config: GeometryAugmentationConfig,
) -> dict[str, torch.Tensor]:
    reference = next(iter(videos.values()))
    batch_size, _channels, _frames, height, width = reference.shape
    probability = config.texture_suppression_probability
    if probability <= 0.0:
        return dict(videos)

    applied = torch.rand(batch_size, device="cpu") < probability
    if config.texture_skip_tsdf:
        applied &= ~_batch_flag(
            data_batch,
            "geometry_condition_is_augmented",
            batch_size=batch_size,
        )
    if not bool(applied.any()):
        return dict(videos)

    uniform = torch.rand(batch_size, device="cpu")
    factors = (
        config.texture_downsample_min
        + (config.texture_downsample_max - config.texture_downsample_min) * uniform
    )
    output = {name: value.clone() for name, value in videos.items()}
    names = tuple(videos)
    channel_counts = [int(videos[name].shape[1]) for name in names]
    for batch_index in torch.nonzero(applied, as_tuple=False).flatten().tolist():
        factor = float(factors[batch_index])
        low_height = max(1, round(height / factor))
        low_width = max(1, round(width / factor))
        sample = torch.cat(
            [videos[name][batch_index].permute(1, 0, 2, 3) for name in names],
            dim=1,
        )
        sample = _gaussian_prefilter(
            sample,
            sigma=factor * config.texture_gaussian_sigma_scale,
        )
        low = F.interpolate(sample, size=(low_height, low_width), mode="area")
        smooth = F.interpolate(
            low,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        for name, chunk in zip(
            names,
            smooth.split(channel_counts, dim=1),
            strict=True,
        ):
            output[name][batch_index].copy_(chunk.permute(1, 0, 2, 3))
    return output


def _warp_geometry_from_c0(
    videos: Mapping[str, torch.Tensor],
    *,
    config: GeometryAugmentationConfig,
) -> dict[str, torch.Tensor]:
    reference = next(iter(videos.values()))
    batch_size, _channels, frames, height, width = reference.shape
    applied = torch.zeros(batch_size, dtype=torch.bool)
    if config.spatial_probability > 0.0:
        applied = torch.rand(batch_size, device="cpu") < config.spatial_probability
    if not bool(applied.any()):
        return dict(videos)

    log_scale_bound = math.log(config.spatial_max_scale)
    log_scales = (
        torch.rand((batch_size, 2), device="cpu").mul_(2.0).sub_(1.0) * log_scale_bound
    )
    shifts = (
        torch.rand((batch_size, 2), device="cpu").mul_(2.0).sub_(1.0)
        * config.spatial_max_shift_fraction
    )
    shears = (
        torch.rand((batch_size, 2), device="cpu").mul_(2.0).sub_(1.0)
        * config.spatial_max_shear
    )
    perspective = (
        torch.rand((batch_size, 2), device="cpu").mul_(2.0).sub_(1.0)
        * config.spatial_max_perspective
    )
    active = applied[:, None]
    log_scales = torch.where(active, log_scales, torch.zeros_like(log_scales))
    shifts = torch.where(active, shifts, torch.zeros_like(shifts))
    shears = torch.where(active, shears, torch.zeros_like(shears))
    perspective = torch.where(
        active,
        perspective,
        torch.zeros_like(perspective),
    )

    device = reference.device
    dtype = torch.float32
    y = (
        (torch.arange(height, device=device, dtype=dtype) + 0.5)
        .mul_(2.0 / height)
        .sub_(1.0)
    )
    x = (
        (torch.arange(width, device=device, dtype=dtype) + 0.5)
        .mul_(2.0 / width)
        .sub_(1.0)
    )
    base_y, base_x = torch.meshgrid(y, x, indexing="ij")
    base_x = base_x[None, None]
    base_y = base_y[None, None]
    if frames == 1:
        strength = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    else:
        progress = torch.linspace(0.0, 1.0, frames, device=device, dtype=dtype)
        strength = (1.0 - progress).pow(config.spatial_anneal_power)[
            None, :, None, None
        ]

    def parameter(value: torch.Tensor, index: int) -> torch.Tensor:
        return value[:, index].to(device=device, dtype=dtype)[:, None, None, None]

    log_scales_device = log_scales.to(device=device, dtype=dtype)
    scale_x = (strength * parameter(log_scales_device, 0)).exp()
    scale_y = (strength * parameter(log_scales_device, 1)).exp()
    shift_x = strength * parameter(shifts, 0) * 2.0
    shift_y = strength * parameter(shifts, 1) * 2.0
    shear_x = strength * parameter(shears, 0)
    shear_y = strength * parameter(shears, 1)
    perspective_x = strength * parameter(perspective, 0)
    perspective_y = strength * parameter(perspective, 1)
    denominator = (1.0 + perspective_x * base_x + perspective_y * base_y).clamp_min_(
        0.5
    )
    source_x = (scale_x * base_x + shear_x * base_y + shift_x) / denominator
    source_y = (shear_y * base_x + scale_y * base_y + shift_y) / denominator
    grid = torch.stack((source_x, source_y), dim=-1)

    names = tuple(videos)
    channel_counts = [int(videos[name].shape[1]) for name in names]
    combined = torch.cat([videos[name] for name in names], dim=1)
    rows = combined.permute(0, 2, 1, 3, 4).reshape(
        batch_size * frames,
        sum(channel_counts),
        height,
        width,
    )
    warped = F.grid_sample(
        rows,
        grid.reshape(batch_size * frames, height, width, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    warped = warped.reshape(
        batch_size,
        frames,
        sum(channel_counts),
        height,
        width,
    ).permute(0, 2, 1, 3, 4)
    # Avoid even interpolation-roundoff on the exact identity endpoint.
    if frames > 1:
        warped[:, :, -1].copy_(combined[:, :, -1])
    if not bool(applied.all()):
        active_device = applied.to(device=device)[:, None, None, None, None]
        warped = torch.where(active_device, warped, combined)
    # Channel views: the geometry stretch that follows materializes its own
    # contiguous outputs, so copying here would only raise peak memory.
    return dict(zip(names, warped.split(channel_counts, dim=1), strict=True))


def _drop_geometry_modality(
    videos: Mapping[str, torch.Tensor],
    *,
    config: GeometryAugmentationConfig,
    apply_dropout: bool = True,
    validation_choices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    reference = next(iter(videos.values()))
    batch_size = int(reference.shape[0])
    probability = config.modality_dropout_probability
    selection = config.modality_selection
    mask = torch.zeros((batch_size, 2), dtype=torch.bool)
    names = tuple(name for name in _GEOMETRY_SLOTS if name in videos)
    if not names or (selection is ModalitySelection.BOTH and probability <= 0.0):
        return dict(videos)

    if selection is not ModalitySelection.BOTH:
        if set(names) != set(_GEOMETRY_SLOTS):
            raise ValueError(
                "single-modality selection requires both raw geometry slots"
            )
        choices = (
            (
                torch.randint(2, (batch_size,), device="cpu")
                if validation_choices is None
                else validation_choices
            )
            if selection is ModalitySelection.RANDOM_SINGLE
            else torch.full(
                (batch_size,),
                names.index(
                    GeometryKind.NORMAL
                    if selection is ModalitySelection.DEPTH
                    else GeometryKind.DEPTH
                ),
            )
        )
        mask[torch.arange(batch_size), choices] = True
    if apply_dropout and probability > 0.0:
        drop_all = torch.rand(batch_size, device="cpu") < probability
        for name in names:
            mask[:, _GEOMETRY_SLOTS.index(name)] |= drop_all
    output = dict(videos)
    for name in names:
        fixed_index = 0 if name is GeometryKind.DEPTH else 1
        selected = mask[:, fixed_index]
        if not bool(selected.any()):
            continue
        selected_device = selected.to(device=reference.device)
        value = videos[name]
        if int(value.shape[1]) != 3:
            raise ValueError(
                f"prepared {name} must have three color channels before dropout"
            )
        # Depth is still RGB in [0, 1]; normal is already in [-1, 1].
        # Both become exactly -1 (black) at the VAE input after conversion.
        color = value.new_tensor(0.0 if name is GeometryKind.DEPTH else -1.0)
        output[name] = torch.where(
            selected_device[:, None, None, None, None],
            color,
            value,
        ).contiguous()
    return output


def _validation_modality_choices(
    data_batch: Mapping[str, Any], batch_size: int
) -> torch.Tensor:
    def rows(value: Any) -> list[Any]:
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, (str, int)):
            value = [value] * batch_size
        else:
            value = list(value)
        if len(value) != batch_size:
            raise ValueError("validation modality metadata must match batch size")
        return value

    keys = rows(
        data_batch.get(
            "__key__", data_batch.get("validation_manifest_slot", range(batch_size))
        )
    )
    seeds = rows(data_batch.get("validation_sample_seed", 0))
    choices = []
    for key, seed in zip(keys, seeds, strict=True):
        payload = f"cosmos3-validation-modality-v1:{int(seed)}:{key}".encode()
        choices.append(hashlib.sha256(payload).digest()[0] & 1)
    return torch.tensor(choices, dtype=torch.long)


def augment_geometry(
    videos: Mapping[str, torch.Tensor],
    *,
    data_batch: Mapping[str, Any],
    training: bool,
    config: GeometryAugmentationConfig,
) -> dict[str, torch.Tensor]:
    if not videos:
        return dict(videos)
    if not training:
        if config.modality_selection is ModalitySelection.BOTH:
            return dict(videos)
        return _drop_geometry_modality(
            videos,
            config=config,
            apply_dropout=False,
            validation_choices=(
                _validation_modality_choices(
                    data_batch, int(next(iter(videos.values())).shape[0])
                )
                if config.modality_selection is ModalitySelection.RANDOM_SINGLE
                else None
            ),
        )
    texture = _suppress_geometry_texture(videos, data_batch=data_batch, config=config)
    warped = _warp_geometry_from_c0(texture, config=config)
    return _drop_geometry_modality(warped, config=config)


def stretch_geometry(
    videos: Mapping[str, torch.Tensor],
    *,
    training: bool,
    divisor: int,
) -> dict[str, torch.Tensor]:
    reference = next(iter(videos.values()))
    batch, _channels, _frames, height, width = reference.shape
    target_height = ((height + divisor - 1) // divisor) * divisor
    target_width = ((width + divisor - 1) // divisor) * divisor
    pad_height = target_height - height
    pad_width = target_width - width
    if training:
        top_values = torch.randint(0, pad_height + 1, (batch,), device="cpu")
        left_values = torch.randint(0, pad_width + 1, (batch,), device="cpu")
    else:
        top_values = torch.full((batch,), pad_height // 2, dtype=torch.long)
        left_values = torch.full((batch,), pad_width // 2, dtype=torch.long)

    names = tuple(videos)
    output: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    source_top = height // 2
    source_bottom = height - source_top
    source_left = width // 2
    source_right = width - source_left
    for batch_index in range(batch):
        top = int(top_values[batch_index])
        left = int(left_values[batch_index])
        bottom = pad_height - top
        right = pad_width - left
        sample = torch.cat(
            [videos[name][batch_index].permute(1, 0, 2, 3) for name in names],
            dim=1,
        )
        top_half, bottom_half = sample.split(
            (source_top, source_bottom),
            dim=-2,
        )
        top_half = F.interpolate(
            top_half,
            size=(source_top + top, width),
            mode="bilinear",
            align_corners=False,
        )
        bottom_half = F.interpolate(
            bottom_half,
            size=(source_bottom + bottom, width),
            mode="bilinear",
            align_corners=False,
        )
        stretched = torch.cat((top_half, bottom_half), dim=-2)
        left_half, right_half = stretched.split(
            (source_left, source_right),
            dim=-1,
        )
        left_half = F.interpolate(
            left_half,
            size=(target_height, source_left + left),
            mode="bilinear",
            align_corners=False,
        )
        right_half = F.interpolate(
            right_half,
            size=(target_height, source_right + right),
            mode="bilinear",
            align_corners=False,
        )
        stretched = torch.cat((left_half, right_half), dim=-1)
        chunks = stretched.split(
            [int(videos[name].shape[1]) for name in names],
            dim=1,
        )
        for name, chunk in zip(names, chunks, strict=True):
            output[name].append(chunk.permute(1, 0, 2, 3).unsqueeze(0))
    return {name: torch.cat(rows, dim=0) for name, rows in output.items()}


def depth_scale(
    data_batch: Mapping[str, Any],
    *,
    training: bool,
    batch_size: int | None = None,
    scale_key: str = "depth_scale",
    eligible_key: str = "depth_scale_jitter_eligible",
    bounds_key: str = "depth_scale_jitter_bounds",
) -> torch.Tensor:
    scale = torch.as_tensor(data_batch[scale_key])
    if scale.ndim == 0 or scale.numel() == 0:
        raise ValueError("depth scale must contain one value per batch")
    if scale.is_complex():
        raise TypeError("depth scale must be real-valued")
    if not scale.is_floating_point():
        scale = scale.to(dtype=torch.float32)
    scale = scale.reshape(-1)
    if batch_size is not None and scale.numel() not in (1, int(batch_size)):
        raise ValueError(
            f"depth scale must have batch size 1 or {batch_size}, got {scale.numel()}"
        )
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("depth scale must be finite and positive")
    eligible = data_batch.get(eligible_key)
    bounds = data_batch.get(bounds_key)
    if training and eligible is not None and bounds is not None:
        eligible = torch.as_tensor(eligible, device=scale.device, dtype=torch.bool)
        bounds = torch.as_tensor(bounds, device=scale.device, dtype=torch.float32)
        batch_size = int(scale.shape[0])
        if eligible.numel() not in (1, batch_size):
            raise ValueError(
                "depth scale jitter eligibility must have batch size 1 or "
                f"{batch_size}, got {eligible.numel()}"
            )
        if (
            bounds.ndim != 2
            or bounds.shape[1] != 2
            or bounds.shape[0]
            not in (
                1,
                batch_size,
            )
        ):
            raise ValueError(
                "depth scale jitter bounds must have shape [1, 2] or "
                f"[{batch_size}, 2], got {tuple(bounds.shape)}"
            )
        if bounds.shape[0] == 1 and batch_size != 1:
            bounds = bounds.expand(batch_size, -1)
        if eligible.numel() == 1 and batch_size != 1:
            eligible = eligible.expand(batch_size)
        if not torch.isfinite(bounds).all() or (bounds[:, 1] < bounds[:, 0]).any():
            raise ValueError("depth scale jitter bounds must be finite and ordered")
        uniform = torch.rand((batch_size,), device="cpu").to(device=scale.device)
        sampled = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * uniform
        factor = torch.where(eligible.reshape(-1), sampled, torch.ones_like(sampled))
        scale = scale * factor.to(dtype=scale.dtype)
    return scale


__all__ = [
    "GeometryAugmentationConfig",
    "GeometryKind",
    "ModalitySelection",
    "augment_geometry",
    "colorize_depth",
    "depth_scale",
    "downsample_geometry",
    "prepare_normal",
    "stretch_geometry",
]

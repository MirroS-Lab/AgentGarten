# SPDX-License-Identifier: Apache-2.0

"""Cosmos 3 conditioner: raw batch -> latents and text/geometry conditions.

Raw batch: video [B,T,3,H,W] (uint8 or [-1,1]), depth [B,T,1,H,W] metric,
depth_scale [B], normal [B,T,3,H,W] unit vectors, caption or text_input_ids,
fps [B], optional sample_id [B]. A precomputed latent replaces video and
depth_latent + normal_latent (both or neither) replace depth and normal.
The caption suffix of a cached latent uses its VAE pixel shape, T = 4n+1
frames of 16*h x 16*w.

Optional depth_scale_jitter_eligible [B] and depth_scale_jitter_bounds [B,2]
(or [1,2]) multiply depth_scale by a uniform factor during training.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from wm.codecs import Wan22VAECodec

from .geometry import (
    GeometryAugmentationConfig,
    augment_geometry,
    colorize_depth,
    depth_scale,
    downsample_geometry,
    prepare_normal,
    stretch_geometry,
)
from .layers import Cosmos3Config
from .network import Cosmos3Condition, Cosmos3TextEncoder

__all__ = ["COSMOS3_EMPTY_PROMPT_IDS", "Cosmos3Conditioner"]

# Chat template of an empty user message: the CFG-dropout and default negative prompt.
COSMOS3_EMPTY_PROMPT_IDS = (151644, 872, 198, 151645, 198, 151644, 77091, 198)
# Per-modality seeds (offset by sample_id) keep validation geometry noise
# identical across CFG arms, solver steps, evaluation rounds and batchings.
_VALIDATION_NOISE_SEEDS = {
    "depth": 23_071_996 + 0x1D3E7A11,
    "normal": 23_071_996 + 0x4A6F2C35,
}


class Cosmos3Conditioner(nn.Module):
    def __init__(
        self,
        *,
        config: Cosmos3Config,
        codec: Wan22VAECodec,
        geometry_augmentation: GeometryAugmentationConfig | None = None,
        geometry_noise_sigma: float = 0.4,
        empty_prompt_ids: Sequence[int] = COSMOS3_EMPTY_PROMPT_IDS,
        default_fps: float = 16.0,
        tokenizer_path: str | None = None,
        negative_prompt_path: str | None = None,
        max_text_tokens: int = 512,
        init_device: str = "meta",
    ) -> None:
        super().__init__()
        if not 0.0 <= geometry_noise_sigma <= 1.0:
            raise ValueError(
                f"geometry_noise_sigma must be in [0, 1], got {geometry_noise_sigma}"
            )
        # Built on ``meta`` by default: weights come from the released checkpoint
        # after sharding (see ``wm.infra.parallel``).
        with torch.device(init_device):
            self.text_encoder = Cosmos3TextEncoder(config)
        self.text_encoder.requires_grad_(False)
        # Plain attribute: the codec holds no trainable or checkpointed state.
        self.codec = codec
        self.geometry_augmentation = (
            geometry_augmentation or GeometryAugmentationConfig()
        )
        self.geometry_noise_sigma = float(geometry_noise_sigma)
        self.empty_prompt_ids = tuple(int(token) for token in empty_prompt_ids)
        self.default_fps = float(default_fps)
        self.tokenizer_path = tokenizer_path
        self.negative_prompt_path = negative_prompt_path
        self.max_text_tokens = int(max_text_tokens)
        self._tokenizer: Any = None
        self._negative_prompt: str | None = None

    def train(self, mode: bool = True) -> Cosmos3Conditioner:
        super().train(mode)
        self.text_encoder.eval()  # frozen: never in training mode
        return self

    # ----------------------------------------------------------------- encode
    @torch.no_grad()
    def encode(self, batch: Mapping[str, Any], *, training: bool) -> dict[str, Any]:
        encoded = dict(batch)
        if "latent" in encoded:
            encoded.pop("video", None)
            shape = self._pixel_shape(encoded["latent"])
        else:
            video = _pixels(encoded.pop("video"))
            shape = video.shape
            encoded["latent"] = self._encode_each(video)
        self._tokenize_prompts(encoded, shape, training=training)
        cached = ("depth_latent" in encoded) + ("normal_latent" in encoded)
        if cached == 1:
            raise ValueError("cache both depth_latent and normal_latent, or neither")
        if not cached:
            geometry = self._geometry_pixels(encoded, training=training)
            if not training:
                encoded["depth_video"] = geometry["depth"]
                encoded["normal_video"] = geometry["normal"]
            latents = self._encode_geometry(geometry)
            ids = encoded.get("sample_id")
            ids = (
                list(range(encoded["latent"].shape[0]))
                if ids is None
                else torch.as_tensor(ids).reshape(-1).tolist()
            )
            for name, latent in latents.items():
                encoded[f"{name}_latent"] = self._geometry_noise(
                    latent, name, training=training, sample_ids=ids
                )
        encoded.pop("depth", None)
        encoded.pop("normal", None)
        if "fps" not in encoded:
            encoded["fps"] = torch.full(
                (encoded["latent"].shape[0],),
                self.default_fps,
                device=encoded["latent"].device,
            )
        return encoded

    def _pixel_shape(self, latent: torch.Tensor) -> tuple[int, ...]:
        batch, channels, frames, height, width = latent.shape
        factor = self.codec.spatial_compression_factor
        return (
            batch,
            channels,
            self.codec.pixel_frames(frames),
            height * factor,
            width * factor,
        )

    def _tokenize_prompts(
        self, batch: dict[str, Any], shape: Sequence[int], *, training: bool
    ) -> None:
        if "caption" not in batch or "text_input_ids" in batch:
            return
        _, _, frames, height, width = shape
        fps = torch.as_tensor(batch.get("fps", self.default_fps)).float().reshape(-1)
        fps = fps.expand(shape[0]) if fps.numel() == 1 else fps

        def suffix(index: int) -> str:
            rate = float(fps[index])
            return (
                f" The video is {frames / rate:.1f} seconds long and is of "
                f"{rate:.0f} FPS. This video is of {height}x{width} resolution."
            )

        captions = batch.pop("caption")
        prompts = [caption.strip() + suffix(i) for i, caption in enumerate(captions)]
        batch["text_input_ids"], batch["text_attention_mask"] = self._tokenize(prompts)
        if not training and self.negative_prompt_path is not None:
            if self._negative_prompt is None:
                payload = json.loads(Path(self.negative_prompt_path).read_text())
                self._negative_prompt = json.dumps(payload).rstrip(".") + "."
            negative = [self._negative_prompt + suffix(i) for i in range(len(prompts))]
            ids, mask = self._tokenize(negative)
            batch["negative_text_input_ids"] = ids
            batch["negative_text_attention_mask"] = mask

    def _tokenize(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self._tokenizer is None:
            if self.tokenizer_path is None:
                raise ValueError("captions need Cosmos3Conditioner(tokenizer_path=...)")
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, local_files_only=True
            )
        rows = []
        for prompt in prompts:
            ids = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                add_vision_id=False,
                return_dict=False,
            )
            ids = ids.flatten().tolist() if torch.is_tensor(ids) else list(ids)
            rows.append(torch.tensor(ids[: self.max_text_tokens]))
        width = max(row.numel() for row in rows)
        device = self.text_encoder.embed_tokens.weight.device
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.bool)
        for index, row in enumerate(rows):
            ids[index, : row.numel()] = row
            mask[index, : row.numel()] = True
        return ids.to(device), mask.to(device)

    def _encode_each(self, pixels: torch.Tensor) -> torch.Tensor:
        # The released serving path encodes one sample at a time.
        return torch.cat(
            [
                self.codec.encode(pixels[index : index + 1])
                for index in range(pixels.shape[0])
            ]
        ).float()

    def _geometry_pixels(
        self, batch: dict[str, Any], *, training: bool
    ) -> dict[str, torch.Tensor]:
        config = self.geometry_augmentation
        depth = _canonical(batch["depth"], channels=1).float()
        batch.setdefault("depth_scale", torch.ones(depth.shape[0]))
        scale = depth_scale(batch, training=training, batch_size=depth.shape[0])
        videos = downsample_geometry(
            {
                "depth": colorize_depth(depth, scale),
                "normal": prepare_normal(_canonical(batch["normal"], channels=3)),
            },
            factor=config.geometry_spatial_downsample,
        )
        videos = augment_geometry(
            videos, data_batch=batch, training=training, config=config
        )
        videos = stretch_geometry(
            videos, training=training, divisor=config.geometry_stretch_divisor
        )
        # Colorized depth lives in [0, 1]; the VAE expects [-1, 1].
        videos["depth"] = videos["depth"] * 2.0 - 1.0
        return {
            name: value.to(self.codec.dtype).contiguous()
            for name, value in videos.items()
        }

    def _encode_geometry(
        self, videos: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        latents: dict[str, list[torch.Tensor]] = {"depth": [], "normal": []}
        for index in range(videos["depth"].shape[0]):
            pair = torch.cat(
                (
                    videos["depth"][index : index + 1],
                    videos["normal"][index : index + 1],
                )
            )
            depth, normal = self.codec.encode(pair).float().split(1)
            latents["depth"].append(depth)
            latents["normal"].append(normal)
        return {name: torch.cat(values) for name, values in latents.items()}

    def _geometry_noise(
        self,
        latent: torch.Tensor,
        name: str,
        *,
        training: bool,
        sample_ids: Sequence[int],
    ) -> torch.Tensor:
        sigma = self.geometry_noise_sigma
        if sigma == 0.0:
            return latent
        if training:
            noise = torch.randn(latent.shape, device=latent.device)
        else:
            noise = torch.stack(
                [
                    torch.randn(
                        latent.shape[1:],
                        device=latent.device,
                        generator=torch.Generator(device=latent.device).manual_seed(
                            _VALIDATION_NOISE_SEEDS[name] + int(sample_id)
                        ),
                    )
                    for sample_id in sample_ids
                ]
            )
        return latent * (1.0 - sigma) + noise * sigma

    # -------------------------------------------------------------- condition
    @torch.no_grad()
    def condition(
        self,
        batch: Mapping[str, Any],
        *,
        negative: bool = False,
        drop: torch.Tensor | None = None,
    ) -> Cosmos3Condition:
        ids, mask = self._prompt(batch, negative=negative)
        if drop is not None and not negative:
            empty_ids, empty_mask = self._empty(ids.shape[0], ids.device)
            ids, mask = _select_rows(drop, empty_ids, empty_mask, ids, mask)
        return Cosmos3Condition(
            text=self.text_encoder(ids, mask),
            depth=batch["depth_latent"],
            normal=batch["normal_latent"],
            fps=torch.as_tensor(batch["fps"], dtype=torch.float32).reshape(-1),
        )

    def _prompt(
        self, batch: Mapping[str, Any], *, negative: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = "negative_" if negative else ""
        ids = batch.get(f"{prefix}text_input_ids")
        if ids is None:
            if not negative:
                raise KeyError("batch has neither caption nor text_input_ids")
            return self._empty(batch["latent"].shape[0], batch["latent"].device)
        mask = batch.get(f"{prefix}text_attention_mask")
        return ids, torch.ones_like(
            ids, dtype=torch.bool
        ) if mask is None else mask.bool()

    def _empty(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor(self.empty_prompt_ids, device=device).expand(batch_size, -1)
        return ids, torch.ones_like(ids, dtype=torch.bool)

    # ---------------------------------------------------------------- outputs
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(latent).float()

    def visuals(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {
            name: batch[f"{name}_video"].float()
            for name in ("depth", "normal")
            if f"{name}_video" in batch
        }


def _canonical(value: torch.Tensor, *, channels: int) -> torch.Tensor:
    if value.ndim != 5 or value.shape[2] != channels:
        raise ValueError(f"expected [B, T, {channels}, H, W], got {tuple(value.shape)}")
    return value.permute(0, 2, 1, 3, 4)


def _pixels(video: torch.Tensor) -> torch.Tensor:
    video = _canonical(video, channels=3)
    if video.dtype == torch.uint8:
        return video.float() / 127.5 - 1.0
    return video.float()


def _select_rows(
    select: torch.Tensor,
    ids_a: torch.Tensor,
    mask_a: torch.Tensor,
    ids_b: torch.Tensor,
    mask_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(ids_a.shape[1], ids_b.shape[1])

    def pad(ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        extra = width - ids.shape[1]
        return (
            F.pad(ids, (0, extra)),
            F.pad(mask, (0, extra), value=False),
        )

    (ids_a, mask_a), (ids_b, mask_b) = pad(ids_a, mask_a), pad(ids_b, mask_b)
    row = select.to(device=ids_b.device, dtype=torch.bool)[:, None]
    return torch.where(row, ids_a, ids_b), torch.where(row, mask_a, mask_b)

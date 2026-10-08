# SPDX-License-Identifier: Apache-2.0

"""Cosmos 3 text encoder and geometry-conditioned generation tower.

Every entry point runs the GEN layers over a list of token chunks:
bidirectional = one chunk; teacher/diffusion forcing = [C0, query, pub, ...];
the KV cache = one chunk per call. Per-chunk GEMMs and SDPA are identical in
all three, so a teacher-forcing replay reproduces the cached rollout bitwise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from wm.infra.parallel import ParallelNetwork
from wm.networks.lora import initialize_lora_parameters
from wm.protocols import HistoryWindow

from .attention import (
    AnchorAttention,
    BlockCausalAttention,
    CachedAttention,
    FullAttention,
    KVCache,
    TextContext,
)
from .context_parallel import ContextParallel, ContextParallelAttention
from .layers import (
    Cosmos3Config,
    Cosmos3GenLayer,
    Cosmos3RMSNorm,
    Cosmos3RotaryEmbedding,
    Cosmos3TextLayer,
    TimestepEmbedder,
)

__all__ = [
    "Cosmos3Condition",
    "Cosmos3Network",
    "Cosmos3TextEncoder",
    "GeometryFrames",
    "frame_sigma",
    "pack_text_rows",
    "rgb_tokens",
]

Attention = Callable[
    [int, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
    list[torch.Tensor],
]
Rope = tuple[torch.Tensor, torch.Tensor]


def pack_text_rows(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    special_tokens: dict[str, int],
) -> list[torch.Tensor]:
    rows = (
        list(input_ids.unbind(0))
        if attention_mask is None
        else [
            row[mask.bool()]
            for row, mask in zip(input_ids, attention_mask, strict=True)
        ]
    )
    prefix = (
        [special_tokens["bos_token_id"]] if "bos_token_id" in special_tokens else []
    )
    suffix = [special_tokens["eos_token_id"], special_tokens["start_of_generation"]]
    return [
        torch.cat((row.new_tensor(prefix), row.long(), row.new_tensor(suffix)))
        for row in rows
    ]


class Cosmos3TextEncoder(ParallelNetwork):
    def __init__(self, config: Cosmos3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Cosmos3TextLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.rotary_emb = Cosmos3RotaryEmbedding(config)

    def fsdp_units(self) -> tuple[nn.Module, ...]:
        return (self.embed_tokens,)

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> TextContext:
        rows = pack_text_rows(input_ids, attention_mask, self.config.special_tokens)
        lengths = tuple(int(row.numel()) for row in rows)
        bucket = max(*lengths, int(self.config.text_bucket_tokens or 0))
        device = self.embed_tokens.weight.device
        tokens = torch.zeros(len(rows), bucket, dtype=torch.long, device=device)
        for index, row in enumerate(rows):
            tokens[index, : row.numel()] = row.to(device)
        hidden = self.embed_tokens(tokens)
        positions = torch.arange(bucket, device=device, dtype=torch.float32)
        positions = positions[None, :, None].expand(len(rows), -1, 3)
        rope = self.rotary_emb(positions, hidden.dtype)
        keys, values = [], []
        for layer in self.layers:
            hidden, key, value = layer(hidden, rope)
            keys.append(key)
            values.append(value)
        return TextContext(
            keys=tuple(keys),
            values=tuple(values),
            valid_lengths=torch.tensor(lengths, device=device),
            lengths=lengths,
        )


@dataclass(frozen=True)
class Cosmos3Condition:
    text: TextContext
    depth: torch.Tensor
    normal: torch.Tensor
    fps: torch.Tensor

    def repeat(self, times: int) -> Cosmos3Condition:
        if times == 1:
            return self
        return Cosmos3Condition(
            text=self.text.repeat(times),
            depth=torch.cat((self.depth,) * times),
            normal=torch.cat((self.normal,) * times),
            fps=torch.cat((self.fps,) * times),
        )

    def with_text(self, text: TextContext) -> Cosmos3Condition:
        return replace(self, text=text)


def patchify(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, frames, height, width = latent.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"latent {height}x{width} is not divisible by patch {patch_size}"
        )
    value = latent.reshape(
        batch,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    return torch.einsum("bcthpwq->bthwpqc", value).reshape(
        batch, frames, (height // patch_size) * (width // patch_size), -1
    )


def unpatchify(
    patches: torch.Tensor, *, channels: int, grid: tuple[int, int], patch_size: int
) -> torch.Tensor:
    batch, frames = patches.shape[:2]
    height, width = grid
    value = patches.reshape(
        batch, frames, height, width, patch_size, patch_size, channels
    )
    return torch.einsum("bthwpqc->bcthpwq", value).reshape(
        batch, channels, frames, height * patch_size, width * patch_size
    )


@dataclass(frozen=True)
class GeometryFrames:
    """Embedded geometry tokens of a frame span with the positions of both grids."""

    geometry: torch.Tensor
    geometry_positions: torch.Tensor
    rgb_positions: torch.Tensor

    def chunk(
        self, rgb: torch.Tensor, start: int, stop: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Join RGB tokens with frames ``start:stop``: (tokens, positions)."""
        # Frame-major [R_f, G_f] so that every frame is one contiguous unit
        # (required by frame-routed sparse attention).
        tokens = torch.cat((rgb, self.geometry[:, start:stop]), dim=2).flatten(1, 2)
        positions = torch.cat(
            (self.rgb_positions[:, start:stop], self.geometry_positions[:, start:stop]),
            dim=2,
        ).flatten(1, 2)
        return tokens, positions


def rgb_tokens(
    hidden: torch.Tensor, frames: int, grid: tuple[int, int]
) -> torch.Tensor:
    """Select the RGB tokens of every frame from frame-major hidden states."""
    return hidden.unflatten(1, (frames, -1))[:, :, : grid[0] * grid[1]]


class Cosmos3Network(ParallelNetwork):
    fsdp_entry_points = ("forward_ar", "prefill", "denoise", "commit")
    context_parallel: ContextParallel | None = None

    def __init__(self, config: Cosmos3Config) -> None:
        super().__init__()
        self.config = config
        self.proj_in = nn.Linear(config.patch_dim, config.hidden_size, bias=True)
        self.proj_out = nn.Linear(config.hidden_size, config.patch_dim, bias=True)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.norm_moe_gen = Cosmos3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.geometry_modality_embed = nn.Parameter(torch.zeros(config.hidden_size))
        self.first_frame_reference_embed = nn.Parameter(torch.zeros(config.hidden_size))
        self.layers = nn.ModuleList(
            Cosmos3GenLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.rotary_emb = Cosmos3RotaryEmbedding(config)

    def enable_context_parallel(self, mesh: DeviceMesh) -> None:
        self.context_parallel = ContextParallel(
            mesh, self.config.num_attention_heads, self.config.num_key_value_heads
        )

    def initialize_after_materialization(self) -> None:
        initialize_lora_parameters(self)

        with torch.no_grad():
            self.geometry_modality_embed.zero_()
            self.first_frame_reference_embed.zero_()

    # ------------------------------------------------------------------ embedding
    @property
    def dtype(self) -> torch.dtype:
        return self.proj_in.weight.dtype

    def embed_rgb(
        self,
        latent: torch.Tensor,
        sigma: torch.Tensor | None,
        *,
        reference: bool = False,
    ) -> torch.Tensor:
        """Embed latent frames [B, C, T, H, W] as RGB tokens [B, T, N, D].

        ``sigma`` [B, T] adds the timestep embedding; ``None`` marks clean
        frames. ``reference`` marks a chunk that is exactly the external C0.
        """
        tokens = self.proj_in(patchify(latent.to(self.dtype), self.config.patch_size))
        if reference and self.config.first_frame_reference_embedding:
            tokens = tokens + self.first_frame_reference_embed.to(tokens.dtype)
        if sigma is not None:
            batch, frames = sigma.shape
            timestep = sigma.reshape(-1).float() * 1000.0 * self.config.timestep_scale
            embedding = self.time_embedder(timestep).reshape(batch, frames, 1, -1)
            tokens = tokens + embedding.to(tokens.dtype)
        return tokens

    def geometry_frames(
        self,
        condition: Cosmos3Condition,
        *,
        start: int,
        stop: int,
        rgb_grid: tuple[int, int],
        frame_indices: torch.Tensor | None = None,
    ) -> GeometryFrames:
        """Embed the geometry of condition frames ``start:stop``.

        ``frame_indices`` overrides the temporal positions (streaming inference
        relocates retained history); it defaults to ``start..stop``.
        """
        config = self.config
        depth = patchify(
            condition.depth[:, :, start:stop].to(self.dtype), config.patch_size
        )
        normal = patchify(
            condition.normal[:, :, start:stop].to(self.dtype), config.patch_size
        )
        # proj_in is affine, so projecting the mean equals the mean projection.
        geometry = (
            self.proj_in(torch.lerp(depth, normal, 0.5)) + self.geometry_modality_embed
        )
        geometry_grid = (
            condition.depth.shape[-2] // config.patch_size,
            condition.depth.shape[-1] // config.patch_size,
        )
        device = geometry.device
        temporal = (
            torch.arange(start, stop, device=device, dtype=torch.float32)[None]
            if frame_indices is None
            else frame_indices.to(device=device, dtype=torch.float32)[None]
        )
        if config.enable_fps_modulation:
            temporal = (
                temporal * (config.base_fps / condition.fps.float().to(device))[:, None]
            )
        origin = condition.text.valid_lengths.to(device=device, dtype=torch.float32)
        temporal = (
            temporal + origin[:, None] + config.temporal_modality_margin
        )  # [B, F]
        return GeometryFrames(
            geometry=geometry,
            geometry_positions=_grid_positions(temporal, geometry_grid, rgb_grid),
            rgb_positions=_grid_positions(temporal, rgb_grid, rgb_grid),
        )

    def decode_velocity(
        self, hidden: torch.Tensor, grid: tuple[int, int]
    ) -> torch.Tensor:
        """Project RGB hidden states [B, T, N, D] to a latent velocity."""
        patches = self.proj_out(self.norm_moe_gen(hidden))
        return unpatchify(
            patches.to(torch.promote_types(patches.dtype, torch.float32)),
            channels=self.config.latent_channels,
            grid=grid,
            patch_size=self.config.patch_size,
        )

    # ------------------------------------------------------------------ decoder
    def run_layers(
        self,
        chunks: list[torch.Tensor],
        positions: list[torch.Tensor],
        attention: Attention,
        *,
        feature_layers: Sequence[int] = (),
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """Run the GEN layers over token chunks with one attention policy.

        Returns the final hidden chunks and, for every layer index in
        ``feature_layers``, that layer's output chunks.
        """
        cp = self.context_parallel
        tokens = [chunk.shape[1] for chunk in chunks]
        if cp is not None:
            chunks = [cp.shard(chunk) for chunk in chunks]
            positions = [cp.shard(position) for position in positions]
            attention = ContextParallelAttention(attention, cp, tokens)
        ropes = [
            self.rotary_emb(position, chunk.dtype)
            for chunk, position in zip(chunks, positions, strict=True)
        ]
        features: list[list[torch.Tensor]] = []
        wanted = frozenset(int(index) for index in feature_layers)
        for index, layer in enumerate(self.layers):
            chunks = layer(chunks, ropes, attention, index)
            if index in wanted:
                features.append(chunks)
        if cp is not None:
            chunks = [cp.gather(c, n) for c, n in zip(chunks, tokens, strict=True)]
            features = [
                [cp.gather(c, n) for c, n in zip(group, tokens, strict=True)]
                for group in features
            ]
        return chunks, features

    def _text(self, text: TextContext) -> TextContext:
        return (
            text if self.context_parallel is None else self.context_parallel.text(text)
        )

    # ------------------------------------------------------------ bidirectional
    def forward(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        condition: Cosmos3Condition,
        *,
        known_frames: int = 1,
        feature_layers: Sequence[int] = (),
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        batch, _, frames, height, width = noisy.shape
        grid = (height // self.config.patch_size, width // self.config.patch_size)
        sigma = frame_sigma(sigma, batch, frames)
        known = self.embed_rgb(noisy[:, :, :known_frames], None)
        if known_frames and self.config.first_frame_reference_embedding:
            # Only the true first frame is the external reference (C0).
            known[:, 0] = known[:, 0] + self.first_frame_reference_embed.to(known.dtype)
        rgb = torch.cat(
            (
                known,
                self.embed_rgb(noisy[:, :, known_frames:], sigma[:, known_frames:]),
            ),
            dim=1,
        )
        frame = self.geometry_frames(condition, start=0, stop=frames, rgb_grid=grid)
        tokens, positions = frame.chunk(rgb, 0, frames)
        (hidden,), features = self.run_layers(
            [tokens],
            [positions],
            FullAttention(self._text(condition.text)),
            feature_layers=feature_layers,
        )
        rgb_hidden = rgb_tokens(hidden, frames, grid)
        velocity = self.decode_velocity(rgb_hidden[:, known_frames:], grid)
        velocity = torch.cat(
            (
                velocity.new_zeros(
                    batch, velocity.shape[1], known_frames, height, width
                ),
                velocity,
            ),
            dim=2,
        )
        if feature_layers:
            return velocity, tuple(feature[0] for feature in features)
        return velocity

    # ------------------------------------------------------ block autoregression
    def forward_ar(
        self,
        anchor: torch.Tensor,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        condition: Cosmos3Condition,
        *,
        block_frames: int,
        clean: torch.Tensor | None = None,
        history: HistoryWindow | None = None,
    ) -> torch.Tensor:
        batch, _, frames, height, width = noisy.shape
        if frames % block_frames:
            raise ValueError(
                f"{frames} target frames are not a multiple of {block_frames}"
            )
        grid = (height // self.config.patch_size, width // self.config.patch_size)
        sigma = frame_sigma(sigma, batch, frames)
        chunks, positions = [], []

        def add(rgb: torch.Tensor, start: int) -> None:
            # Geometry is embedded per chunk so every GEMM has the same row
            # count as in the cached rollout (bitwise-identical replay).
            stop = start + rgb.shape[1]
            frame = self.geometry_frames(
                condition, start=start, stop=stop, rgb_grid=grid
            )
            token, position = frame.chunk(rgb, 0, stop - start)
            chunks.append(token)
            positions.append(position)

        add(self.embed_rgb(anchor, None, reference=True), 0)
        for start in range(0, frames, block_frames):
            stop = start + block_frames
            add(
                self.embed_rgb(noisy[:, :, start:stop], sigma[:, start:stop]),
                start + 1,
            )
            if clean is not None:
                add(self.embed_rgb(clean[:, :, start:stop], None), start + 1)
        attention = BlockCausalAttention(
            self._text(condition.text),
            teacher_forcing=clean is not None,
            history=history or HistoryWindow(),
        )
        hidden, _ = self.run_layers(chunks, positions, attention)
        stride = 1 if clean is None else 2
        outputs = [
            self.decode_velocity(rgb_tokens(chunk, block_frames, grid), grid)
            for chunk in hidden[1::stride]
        ]
        return torch.cat(outputs, dim=2)

    # -------------------------------------------------------------- KV cache
    def prefill(
        self,
        anchor: torch.Tensor,
        condition: Cosmos3Condition,
        *,
        history: HistoryWindow | None = None,
    ) -> KVCache:
        patch = self.config.patch_size
        grid = (anchor.shape[-2] // patch, anchor.shape[-1] // patch)
        frame = self.geometry_frames(condition, start=0, stop=1, rgb_grid=grid)
        token, position = frame.chunk(
            self.embed_rgb(anchor, None, reference=True), 0, 1
        )
        text = self._text(condition.text)
        attention = AnchorAttention(text)
        self.run_layers([token], [position], attention)
        return KVCache(
            text=text,
            anchor=tuple(attention.recorded),
            history=history or HistoryWindow(),
            condition=condition,
            grid=grid,
        )

    def denoise(
        self, cache: KVCache, noisy: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        hidden = self._cached_chunk(cache, noisy, sigma, record=None)
        grid = cache.grid
        frames = noisy.shape[2]
        rgb = rgb_tokens(hidden, frames, grid)
        return self.decode_velocity(rgb, grid)

    def commit(
        self, cache: KVCache, clean: torch.Tensor, sigma: torch.Tensor | None = None
    ) -> KVCache:
        recorded: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._cached_chunk(cache, clean, sigma, record=recorded)
        return cache.committed(tuple(recorded), clean.shape[2])

    def _cached_chunk(
        self,
        cache: KVCache,
        latent: torch.Tensor,
        sigma: torch.Tensor | None,
        *,
        record: list[tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> torch.Tensor:
        batch, _, frames = latent.shape[:3]
        start = 1 + cache.frames_committed
        frame = self.geometry_frames(
            cache.condition, start=start, stop=start + frames, rgb_grid=cache.grid
        )
        rgb = self.embed_rgb(
            latent, None if sigma is None else frame_sigma(sigma, batch, frames)
        )
        token, position = frame.chunk(rgb, 0, frames)
        attention = CachedAttention(cache, record=record is not None)
        (hidden,), _ = self.run_layers([token], [position], attention)
        if record is not None:
            record.extend(attention.recorded)
        return hidden


def frame_sigma(sigma: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
    """Broadcast a scalar or per-sample sigma to per-frame [B, T]."""
    sigma = torch.as_tensor(sigma).float()
    if sigma.ndim == 0:
        sigma = sigma.expand(batch)
    if sigma.ndim == 1:
        sigma = sigma[:, None].expand(batch, frames)
    if sigma.shape != (batch, frames):
        raise ValueError(
            f"sigma must be [B] or [B, {frames}], got {tuple(sigma.shape)}"
        )
    return sigma


def _grid_positions(
    temporal: torch.Tensor,
    grid: tuple[int, int],
    rgb_grid: tuple[int, int],
) -> torch.Tensor:
    batch, frames = temporal.shape
    height, width = grid
    device = temporal.device
    rows = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (
        rgb_grid[0] / height
    ) - 0.5
    cols = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (
        rgb_grid[1] / width
    ) - 0.5
    return torch.stack(
        (
            temporal[:, :, None, None].expand(batch, frames, height, width),
            rows[None, None, :, None].expand(batch, frames, height, width),
            cols[None, None, None, :].expand(batch, frames, height, width),
        ),
        dim=-1,
    ).reshape(batch, frames, height * width, 3)

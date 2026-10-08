# SPDX-License-Identifier: Apache-2.0

"""Inference-owned stream state over the same GEN layers used for training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from .attention import TextContext
from .layers import apply_rotary
from .network import Cosmos3Condition, Cosmos3Network, frame_sigma, rgb_tokens
from .streaming_attention import StreamAttentionCache

__all__ = ["Cosmos3Stream", "StreamPolicy", "StreamRoPE"]


class StreamRoPE(StrEnum):
    """Temporal positions of retained history and the next query."""

    ABSOLUTE = "absolute"
    CONTIGUOUS = "contiguous"
    TOP_ALIGNED = "top_aligned"


@dataclass(frozen=True, slots=True)
class StreamPolicy:
    """Frame budgets exclude the current block; a zero sink can evict C0."""

    block_frames: int = 4
    sink_frames: int = 5
    history_frames: int | None = 44
    rope: StreamRoPE = StreamRoPE.TOP_ALIGNED
    train_frames: int = 61

    def __post_init__(self) -> None:
        object.__setattr__(self, "rope", StreamRoPE(self.rope))
        if self.block_frames < 1 or self.sink_frames < 0:
            raise ValueError(
                "block_frames must be positive and sink_frames nonnegative"
            )
        if self.sink_frames and (self.sink_frames - 1) % self.block_frames:
            raise ValueError("sink_frames must be C0 plus complete generated blocks")
        if self.history_frames is None:
            if self.rope is not StreamRoPE.ABSOLUTE:
                raise ValueError("unbounded history requires absolute RoPE")
        elif self.history_frames < 0 or self.capacity < 1:
            raise ValueError("history_frames must be nonnegative and capacity positive")
        if self.rope is StreamRoPE.TOP_ALIGNED:
            if self.train_frames < self.capacity + self.block_frames:
                raise ValueError(
                    "train_frames must contain history and the current block"
                )
            if (self.train_frames - 1) % self.block_frames:
                raise ValueError(
                    "train_frames must follow the C0 + N * block_frames layout"
                )

    @property
    def capacity(self) -> int:
        """Maximum retained frames; only defined for a bounded stream."""
        if self.history_frames is None:
            raise ValueError("unbounded streams have no fixed capacity")
        return self.sink_frames + self.history_frames

    def positions(self, frames: tuple[int, ...], next_frame: int) -> tuple[int, ...]:
        """Map absolute frame IDs without changing text or spatial positions."""
        match self.rope:
            case StreamRoPE.ABSOLUTE:
                return frames
            case StreamRoPE.CONTIGUOUS:
                # Both the retained history and the following query use this
                # one shift. The permanent sink retains its original positions.
                shift = max(0, next_frame - self.capacity)
            case StreamRoPE.TOP_ALIGNED:
                shift = max(0, next_frame + self.block_frames - self.train_frames)
        return tuple(f if f < self.sink_frames else f - shift for f in frames)


class Cosmos3Stream:
    """One sequential latent stream; reset reuses the model and compiled layers.

    Conditions contain only the current chunk, already VAE-encoded and noised.
    No generated output is committed until the caller explicitly publishes its
    clean latent. Text changes preserve visual history and require equal lengths.
    """

    def __init__(
        self,
        network: Cosmos3Network,
        policy: StreamPolicy,
        *,
        initial_capacity_frames: int = 256,
    ) -> None:
        if initial_capacity_frames < 1:
            raise ValueError("initial_capacity_frames must be positive")
        self.network = network
        self.policy = policy
        self.initial_capacity_frames = initial_capacity_frames
        self.reset()

    def reset(self) -> None:
        """Release semantic history without touching model or compiler state."""
        self.text: TextContext | None = None
        self.layers: list[StreamAttentionCache] = []
        self.frame_ids: tuple[int, ...] = ()
        self.next_frame = 0
        self.positions: torch.Tensor | None = None
        self.grid = (0, 0)

    @property
    def keys(self) -> list[torch.Tensor]:
        return [layer.visual_key for layer in self.layers]

    @property
    def values(self) -> list[torch.Tensor]:
        return [layer.visual_value for layer in self.layers]

    @torch.inference_mode()
    def start(self, anchor: torch.Tensor, condition: Cosmos3Condition) -> None:
        """Prefill the separate clean C0 once, with its matching condition."""
        if self.next_frame:
            raise RuntimeError("reset the stream before starting a new anchor")
        if anchor.shape[2] != 1:
            raise ValueError("anchor must contain exactly one latent frame")
        if anchor.shape[0] != 1:
            raise ValueError("a serving stream owns exactly one sample")
        if (
            condition.depth.shape[2] != 1
            or condition.normal.shape != condition.depth.shape
        ):
            raise ValueError("anchor condition must contain exactly one latent frame")
        if condition.text.batch_size != 1:
            raise ValueError("a serving stream needs one text context")
        self.text = condition.text
        patch = self.network.config.patch_size
        self.grid = (anchor.shape[-2] // patch, anchor.shape[-1] // patch)
        self._run(anchor, None, condition, record=True, reference=True)
        self.next_frame = 1
        self.frame_ids = (0,)

    @torch.inference_mode()
    def denoise(
        self, noisy: torch.Tensor, sigma: torch.Tensor, condition: Cosmos3Condition
    ) -> torch.Tensor:
        """Predict velocity without changing the completed history."""
        self.validate_chunk(noisy, condition)
        hidden = self._run(noisy, sigma, condition, record=False)
        return self.network.decode_velocity(
            rgb_tokens(hidden, noisy.shape[2], self.grid), self.grid
        )

    @torch.inference_mode()
    def commit(self, clean: torch.Tensor, condition: Cosmos3Condition) -> None:
        """Publish one clean block, then evict and relocate history for the next."""
        self.validate_chunk(clean, condition)
        self._run(clean, None, condition, record=True)
        self.frame_ids += tuple(
            range(self.next_frame, self.next_frame + clean.shape[2])
        )
        self.next_frame += clean.shape[2]
        self._trim(condition)

    def validate_chunk(self, value: torch.Tensor, condition: Cosmos3Condition) -> None:
        """Check that ``value`` and its condition are one block of this stream."""
        if self.text is None:
            raise RuntimeError("start the stream before denoising or committing")
        if value.shape[2] != self.policy.block_frames:
            raise ValueError("current latent must contain one complete block")
        if (
            condition.depth.shape[2] != value.shape[2]
            or condition.normal.shape != condition.depth.shape
        ):
            raise ValueError("condition must contain exactly the current block")
        if condition.text is not self.text:
            raise ValueError("update_text explicitly before changing stream text")

    @torch.inference_mode()
    def update_text(self, text: TextContext) -> None:
        """Switch a prepared prompt while keeping the visual cache and RoPE origin."""
        if self.text is None:
            raise RuntimeError("start the stream before updating text")
        if text.lengths != self.text.lengths or len(text.keys) != len(self.text.keys):
            raise ValueError(
                "prompt changes must preserve text lengths and layer count"
            )
        pairs = tuple(
            zip(
                (*self.text.keys, *self.text.values),
                (*text.keys, *text.values),
                strict=True,
            )
        )
        if any(
            a.shape != b.shape or a.dtype != b.dtype or a.device != b.device
            for a, b in pairs
        ):
            raise ValueError(
                "prompt changes must preserve the complete text cache layout"
            )
        for destination, source in pairs:
            destination.copy_(source)
        for layer, key, value in zip(self.layers, text.keys, text.values, strict=True):
            layer.update_text(key[:, : text.lengths[0]], value[:, : text.lengths[0]])

    def embed_chunk(
        self,
        latent: torch.Tensor,
        sigma: torch.Tensor | None,
        condition: Cosmos3Condition,
        *,
        reference: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed the next chunk at its stream positions: (tokens, positions)."""
        count = latent.shape[2]
        indices = self.policy.positions(
            tuple(range(self.next_frame, self.next_frame + count)), self.next_frame
        )
        frame = self.network.geometry_frames(
            condition,
            start=0,
            stop=count,
            rgb_grid=self.grid,
            frame_indices=torch.arange(
                indices[0], indices[0] + count, device=latent.device
            ),
        )
        rgb = self.network.embed_rgb(
            latent,
            None if sigma is None else frame_sigma(sigma, latent.shape[0], count),
            reference=reference,
        )
        return frame.chunk(rgb, 0, count)

    def _run(
        self,
        latent: torch.Tensor,
        sigma: torch.Tensor | None,
        condition: Cosmos3Condition,
        *,
        record: bool,
        reference: bool = False,
    ) -> torch.Tensor:
        tokens, positions = self.embed_chunk(
            latent, sigma, condition, reference=reference
        )

        def attend(
            index: int,
            queries: list[torch.Tensor],
            keys: list[torch.Tensor],
            values: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            query, key, value = queries[0], keys[0], values[0]
            length = self.text.lengths[0]
            if index == len(self.layers):
                self.layers.append(
                    StreamAttentionCache(
                        self.text.keys[index][:, :length],
                        self.text.values[index][:, :length],
                        key.shape[1],
                        self.policy.capacity
                        if self.policy.history_frames is not None
                        else self.initial_capacity_frames,
                        self.policy.block_frames,
                    )
                )
            layer = self.layers[index]
            if layer.frames + latent.shape[2] > layer.capacity + layer.block_frames:
                replacement = StreamAttentionCache(
                    layer.key[:, :length],
                    layer.value[:, :length],
                    layer.frame_tokens,
                    max(layer.capacity * 2, layer.frames + latent.shape[2]),
                    layer.block_frames,
                )
                replacement.append(layer.visual_key, layer.visual_value)
                layer = self.layers[index] = replacement
            output = layer.attend(query, key, value)
            if record:
                layer.append(key, value)
            return [output]

        (hidden,), _ = self.network.run_layers([tokens], [positions], attend)
        if record:
            self.positions = (
                positions
                if self.positions is None
                else torch.cat((self.positions, positions), 1)
            )
        return hidden

    def _trim(self, condition: Cosmos3Condition) -> None:
        if self.policy.history_frames is None:
            return
        count, capacity = len(self.frame_ids), self.policy.capacity
        if count <= capacity:
            return
        sink = self.policy.sink_frames
        kept = tuple(range(sink)) + tuple(range(count - (capacity - sink), count))
        frame_tokens = self.keys[0].shape[1] // count
        device = self.keys[0].device
        indices = torch.cat(
            (
                torch.arange(sink, device=device),
                torch.arange(count - (capacity - sink), count, device=device),
            )
        )
        self.frame_ids = tuple(self.frame_ids[i] for i in kept)

        def retain(tokens: torch.Tensor) -> torch.Tensor:
            # Keep the tokens of the retained frames of a frame-major sequence.
            return (
                tokens.unflatten(1, (count, frame_tokens))
                .index_select(1, indices)
                .flatten(1, 2)
            )

        old = retain(self.positions)
        if self.policy.rope is StreamRoPE.ABSOLUTE:
            for layer in self.layers:
                layer.replace_visual(
                    retain(layer.visual_key), retain(layer.visual_value)
                )
            self.positions = old
            return
        new = old.clone().unflatten(1, (capacity, frame_tokens))
        target = self.policy.positions(self.frame_ids, self.next_frame)
        times = (
            torch.cat(
                (
                    torch.arange(sink, device=device),
                    torch.arange(
                        target[sink], target[sink] + capacity - sink, device=device
                    ),
                )
            ).float()
            if sink < capacity
            else torch.arange(sink, device=device).float()
        )
        if self.network.config.enable_fps_modulation:
            times = times * (self.network.config.base_fps / condition.fps.float())
        new[..., 0] = (
            times + self.text.lengths[0] + self.network.config.temporal_modality_margin
        )[None, :, None]
        new = new.flatten(1, 2)
        dtype = self.keys[0].dtype
        old_c, old_s = (x.float() for x in self.network.rotary_emb(old, dtype))
        new_c, new_s = (x.float() for x in self.network.rotary_emb(new, dtype))
        # Invert the *rounded native* phases; cos(new-old) loses accuracy at
        # the large Cosmos modality offset. Values and sink keys do not rotate.
        norm = old_c.square() + old_s.square()
        cosine = (new_c * old_c + new_s * old_s) / norm
        sine = (new_s * old_c - new_c * old_s) / norm
        # On the GPU one kernel per tensor evicts and rotates; the eager path
        # below computes the same and serves every other device and dtype.
        fused = self.keys[0].is_cuda and dtype == torch.bfloat16
        for layer in self.layers:
            if fused:
                layer.relocate(sink, count - capacity, (cosine, sine))
                continue
            key = retain(layer.visual_key)
            rotated = apply_rotary(key.float(), cosine, sine).to(dtype)
            rotated[:, : sink * frame_tokens] = key[:, : sink * frame_tokens]
            layer.replace_visual(rotated, retain(layer.visual_value))
        self.positions = new

# SPDX-License-Identifier: Apache-2.0

"""Offline training-free clean-image conditioning of a cached AR query.

This borrows native Nano image editing's clean GEN tokens and bidirectional
source/query attention. It is an AR adaptation, not an exact reproduction of
native whole-sequence image editing: completed history remains frozen, and the
usual clean commit is unchanged. References are never C0 or video history.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .attention import sdpa
from .network import Cosmos3Condition, rgb_tokens
from .streaming import Cosmos3Stream

__all__ = ["denoise_with_references"]


@torch.inference_mode()
def denoise_with_references(
    stream: Cosmos3Stream,
    noisy: torch.Tensor,
    sigma: torch.Tensor,
    condition: Cosmos3Condition,
    references: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Recompute clean reference states alongside each noisy current block.

    Reference latents have shape [1,C,1,H,W], with native (unscaled) patch-grid
    spatial coordinates. Their time slots precede C0: origin-R,...,origin-1.
    This preserves the already-prefilled video/text RoPE and every normal
    token's alignment. It differs from moving all video times to make room for
    source images at the beginning of a newly packed native editing sequence.

    All queries (reference and current) read text, completed history and the
    whole current/reference group. No reference states/KV are persisted between
    denoise calls. Clean pixels/latents are fixed, NOT their contextual states.
    Empty references delegate to the original stream without numerical changes.
    """
    if not references:
        return stream.denoise(noisy, sigma, condition)
    stream.validate_chunk(noisy, condition)
    network = stream.network
    if network.context_parallel is not None:
        raise ValueError("reference queries do not support context parallelism")
    if len(stream.layers) != len(network.layers):
        raise RuntimeError("Prefill C0 before using reference images")

    tokens, positions = stream.embed_chunk(noisy, sigma, condition)
    chunks, coordinates = [], []
    origin = condition.text.valid_lengths.to(tokens.device, torch.float32)
    origin = origin + network.config.temporal_modality_margin
    patch = network.config.patch_size
    for index, reference in enumerate(references):
        if (
            reference.ndim != 5
            or reference.shape[:3] != (1, network.config.latent_channels, 1)
            or reference.device != noisy.device
        ):
            raise ValueError(
                "Reference must be a single clean image latent on the query device"
            )
        # None is essential: clean images do NOT receive time_embedder(0).
        # reference=False avoids the trained external-C0-only embedding.
        embedded = network.embed_rgb(reference, None, reference=False).flatten(1, 2)
        h, w = reference.shape[-2] // patch, reference.shape[-1] // patch
        yy, xx = torch.meshgrid(
            torch.arange(h, device=tokens.device, dtype=torch.float32),
            torch.arange(w, device=tokens.device, dtype=torch.float32),
            indexing="ij",
        )
        time = (origin + index - len(references))[:, None].expand(-1, h * w)
        position = torch.stack((time, yy.flatten()[None], xx.flatten()[None]), -1)
        chunks.append(embedded)
        coordinates.append(position)
    count = sum(chunk.shape[1] for chunk in chunks)

    def attend(
        index: int,
        queries: list[torch.Tensor],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        cache = stream.layers[index]
        end = cache.text_tokens + cache.frames * cache.frame_tokens
        # One GEMM group for [reference,current] also avoids aliasing the output
        # storage of project/finish CUDA graphs across multiple invocations.
        key = torch.cat((cache.key[:, :end], keys[0]), dim=1)
        value = torch.cat((cache.value[:, :end], values[0]), dim=1)
        return [sdpa(queries[0], key, value, None)]

    (hidden,), _ = network.run_layers(
        [torch.cat((*chunks, tokens), dim=1)],
        [torch.cat((*coordinates, positions), dim=1)],
        attend,
    )
    return network.decode_velocity(
        rgb_tokens(hidden[:, count:], noisy.shape[2], stream.grid), stream.grid
    )

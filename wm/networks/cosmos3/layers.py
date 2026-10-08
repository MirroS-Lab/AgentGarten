# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "Cosmos3Config",
    "Cosmos3GenLayer",
    "Cosmos3MLP",
    "Cosmos3RMSNorm",
    "Cosmos3RotaryEmbedding",
    "Cosmos3TextLayer",
    "TimestepEmbedder",
    "apply_rotary",
]


@dataclass
class Cosmos3Config:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    latent_channels: int = 48
    patch_size: int = 2
    base_fps: float = 24.0
    enable_fps_modulation: bool = True
    temporal_modality_margin: int = 15000
    timestep_scale: float = 0.001
    # Zero-initialized type embedding added to the RGB patches of the external
    # first frame (C0). It adds no token and changes no position.
    first_frame_reference_embedding: bool = True
    # Stable text width shared by every rank; avoids shape churn when captions
    # have different token counts. Longer rows widen the bucket locally.
    text_bucket_tokens: int | None = 514
    special_tokens: dict[str, int] = field(
        default_factory=lambda: {
            "eos_token_id": 151_645,
            "start_of_generation": 151_652,
        }
    )

    def __post_init__(self) -> None:
        # Configs read back from YAML carry lists.
        self.mrope_section = tuple(int(value) for value in self.mrope_section)

    @property
    def patch_dim(self) -> int:
        return self.latent_channels * self.patch_size**2


class Cosmos3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        # At least FP32 statistics (and never a downcast of FP64 inputs).
        value = hidden_states.to(torch.promote_types(dtype, torch.float32))
        value = value * torch.rsqrt(
            value.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * value.to(dtype)


class Cosmos3MLP(nn.Module):
    def __init__(self, config: Cosmos3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        arguments = timestep[:, None].float() * frequencies[None]
        frequency = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
        # Features are rounded to the parameter dtype, then the MLP runs in FP32
        # with upcast weights. This is the autocast-FP32 path the production
        # AR/DMD checkpoints were trained with.
        frequency = frequency.to(self.linear_1.weight.dtype).float()
        with torch.autocast(timestep.device.type, enabled=False):
            weight_1 = self.linear_1.weight.float()
            bias_1 = self.linear_1.bias.float()
            hidden = self.act(F.linear(frequency, weight_1, bias_1))
            return F.linear(
                hidden, self.linear_2.weight.float(), self.linear_2.bias.float()
            )


class Cosmos3RotaryEmbedding(nn.Module):
    def __init__(self, config: Cosmos3Config) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.mrope_section = config.mrope_section

    @torch.no_grad()
    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inverse = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0, self.head_dim, 2, dtype=torch.float32, device=positions.device
                )
                / self.head_dim
            )
        )
        # [3, B, N, D/2]
        frequencies = positions.float().permute(2, 0, 1)[..., None] * inverse
        mixed = frequencies[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            stop = self.mrope_section[axis] * 3
            mixed[..., offset:stop:3] = frequencies[axis, ..., offset:stop:3]
        embedding = torch.cat((mixed, mixed), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    cosine = cosine.unsqueeze(2)
    sine = sine.unsqueeze(2)
    return value * cosine + _rotate_half(value) * sine


class _TextAttention(nn.Module):
    """Projections of a text layer, named as in the released checkpoint."""

    def __init__(self, config: Cosmos3Config) -> None:
        super().__init__()
        hidden, dim, bias = config.hidden_size, config.head_dim, config.attention_bias
        heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.to_q = nn.Linear(hidden, heads * dim, bias=bias)
        self.to_k = nn.Linear(hidden, kv_heads * dim, bias=bias)
        self.to_v = nn.Linear(hidden, kv_heads * dim, bias=bias)
        self.to_out = nn.Linear(heads * dim, hidden, bias=bias)
        self.norm_q = Cosmos3RMSNorm(dim, eps=config.rms_norm_eps)
        self.norm_k = Cosmos3RMSNorm(dim, eps=config.rms_norm_eps)


class _GenAttention(nn.Module):
    """Projections of a generation layer, named as in the released checkpoint."""

    def __init__(self, config: Cosmos3Config) -> None:
        super().__init__()
        hidden, dim, bias = config.hidden_size, config.head_dim, config.attention_bias
        heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.add_q_proj = nn.Linear(hidden, heads * dim, bias=bias)
        self.add_k_proj = nn.Linear(hidden, kv_heads * dim, bias=bias)
        self.add_v_proj = nn.Linear(hidden, kv_heads * dim, bias=bias)
        self.to_add_out = nn.Linear(heads * dim, hidden, bias=bias)
        self.norm_added_q = Cosmos3RMSNorm(dim, eps=config.rms_norm_eps)
        self.norm_added_k = Cosmos3RMSNorm(dim, eps=config.rms_norm_eps)


class Cosmos3TextLayer(nn.Module):
    def __init__(self, config: Cosmos3Config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.self_attn = _TextAttention(config)
        self.mlp = Cosmos3MLP(config)
        self.input_layernorm = Cosmos3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Cosmos3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self, hidden: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, _ = hidden.shape
        attn = self.self_attn
        normalized = self.input_layernorm(hidden)
        query = attn.norm_q(
            attn.to_q(normalized).view(
                batch, tokens, self.num_attention_heads, self.head_dim
            )
        )
        key = attn.norm_k(
            attn.to_k(normalized).view(
                batch, tokens, self.num_key_value_heads, self.head_dim
            )
        )
        value = attn.to_v(normalized).view(
            batch, tokens, self.num_key_value_heads, self.head_dim
        )
        query = apply_rotary(query, *rope)
        key = apply_rotary(key, *rope)
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        )
        residual = hidden + attn.to_out(output.transpose(1, 2).flatten(-2))
        output = residual + self.mlp(self.post_attention_layernorm(residual))
        return output, key, value


class Cosmos3GenLayer(nn.Module):
    # Compiled individually by ``wm.infra.parallel``; attention stays eager.
    compile_methods = ("project", "finish")

    def __init__(self, config: Cosmos3Config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.self_attn = _GenAttention(config)
        self.mlp_moe_gen = Cosmos3MLP(config)
        self.input_layernorm_moe_gen = Cosmos3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_moe_gen = Cosmos3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def project(
        self, hidden: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, _ = hidden.shape
        attn = self.self_attn
        normalized = self.input_layernorm_moe_gen(hidden)
        query = attn.norm_added_q(
            attn.add_q_proj(normalized).view(
                batch, tokens, self.num_attention_heads, self.head_dim
            )
        )
        key = attn.norm_added_k(
            attn.add_k_proj(normalized).view(
                batch, tokens, self.num_key_value_heads, self.head_dim
            )
        )
        value = attn.add_v_proj(normalized).view(
            batch, tokens, self.num_key_value_heads, self.head_dim
        )
        return apply_rotary(query, cosine, sine), apply_rotary(key, cosine, sine), value

    def forward(
        self,
        chunks: list[torch.Tensor],
        ropes: list[tuple[torch.Tensor, torch.Tensor]],
        attention: Callable[..., list[torch.Tensor]],
        layer_index: int,
    ) -> list[torch.Tensor]:
        projected = [
            self.project(hidden, *rope)
            for hidden, rope in zip(chunks, ropes, strict=True)
        ]
        queries, keys, values = (list(items) for items in zip(*projected, strict=True))
        outputs = attention(layer_index, queries, keys, values)
        return [
            self.finish(hidden, output)
            for hidden, output in zip(chunks, outputs, strict=True)
        ]

    def finish(self, hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        residual = hidden + self.self_attn.to_add_out(attention.flatten(-2))
        return residual + self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(residual)
        )

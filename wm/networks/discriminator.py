# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from wm.infra.parallel import ActivationCheckpointing, ParallelConfig, init_meshes

__all__ = ["DiscriminatorHead"]


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, value: Tensor) -> Tensor:
        value_float = value.to(torch.promote_types(value.dtype, torch.float32))
        normalized = value_float * torch.rsqrt(
            value_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(value) * self.weight


class _CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("discriminator hidden size must be divisible by heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query_token = nn.Parameter(torch.empty(1, 1, 1, hidden_size))
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pre_norm_kv = _RMSNorm(hidden_size)
        self.post_norm_q = _RMSNorm(hidden_size)
        self.post_norm_k = _RMSNorm(hidden_size)
        self.to_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, features: Tensor) -> Tensor:
        # Promote backbone features to the head dtype before normalizing, so
        # the small differences the paired/R3 objectives rely on survive.
        features = features.to(self.to_k.weight.dtype)
        batch, tokens = features.shape[:2]
        residual = self.query_token[:, 0].expand(batch, -1, -1)
        query = self.post_norm_q(self.to_q(residual)).view(batch, 1, self.num_heads, -1)
        key_value = self.pre_norm_kv(features)
        key = self.post_norm_k(self.to_k(key_value)).view(
            batch, tokens, self.num_heads, -1
        )
        value = self.to_v(key_value).view(batch, tokens, self.num_heads, -1)
        # One query per sample: explicit attention is cheap and, unlike fused
        # SDPA, supports double backward (needed by exact R1/R2).
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * self.head_dim**-0.5
        attended = torch.einsum("bhqk,bkhd->bhqd", scores.softmax(-1), value)
        return (self.to_out(attended.transpose(1, 2).reshape(batch, 1, -1)) + residual)[
            :, 0
        ]


class _MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm = _RMSNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.fc2(self.activation(self.fc1(self.norm(value))))


class _Branch(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, num_heads: int) -> None:
        super().__init__()
        self.cross_attention = _CrossAttention(hidden_size, num_heads)
        self.mlp = _MLP(hidden_size, mlp_ratio)

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(self.cross_attention(features))


class DiscriminatorHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 4096,
        feature_layers: Sequence[int] = (11, 23, 35),
        mlp_ratio: float = 4.0,
        num_heads: int = 32,
        initial_logit_sign: float = 1.0,
        seed: int = 0xD2D20000,
        init_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.feature_layers = tuple(sorted(int(layer) for layer in feature_layers))
        if not self.feature_layers or len(set(self.feature_layers)) != len(
            self.feature_layers
        ):
            raise ValueError(
                f"feature_layers must be unique and non-empty: {feature_layers}"
            )
        if initial_logit_sign not in (-1.0, 1.0):
            raise ValueError("initial_logit_sign must be -1 or 1")
        self.initial_logit_sign = float(initial_logit_sign)
        self.seed = int(seed)
        with torch.device(init_device):
            self.branches = nn.ModuleList(
                _Branch(hidden_size, mlp_ratio, num_heads) for _ in self.feature_layers
            )
            joined = hidden_size * len(self.feature_layers)
            self.final_norm = nn.LayerNorm(joined, eps=1.0e-6)
            self.final_linear = nn.Linear(joined, 1)
        if not any(parameter.is_meta for parameter in self.parameters()):
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, shard_index: int = 0) -> None:
        def local(parameter: Tensor) -> Tensor:
            return parameter.to_local() if isinstance(parameter, DTensor) else parameter

        generator = torch.Generator(device=self.final_linear.weight.device)
        generator.manual_seed(self.seed + int(shard_index))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = (6.0 / float(fan_in + fan_out)) ** 0.5
                local(module.weight).uniform_(-bound, bound, generator=generator)
                if module.bias is not None:
                    local(module.bias).zero_()
            elif isinstance(module, (_RMSNorm, nn.LayerNorm)):
                local(module.weight).fill_(1.0)
                if getattr(module, "bias", None) is not None:
                    local(module.bias).zero_()
            elif isinstance(module, _CrossAttention):
                local(module.query_token).normal_(0.0, 0.02, generator=generator)
        local(self.final_linear.weight).mul_(self.initial_logit_sign)

    def parallelize(self, config: ParallelConfig, device: torch.device | str) -> None:
        device = torch.device(device)
        if config.activation_checkpointing is not ActivationCheckpointing.NONE:
            for index, branch in enumerate(self.branches):
                self.branches[index] = checkpoint_wrapper(branch)
        if config.compile:
            for branch in self.branches:
                branch.compile(dynamic=config.compile_dynamic, fullgraph=True)
        dtype = getattr(torch, config.parameter_dtype)
        on_meta = any(parameter.is_meta for parameter in self.parameters())
        if on_meta:
            self.to(dtype=dtype)
        else:
            self.to(device=device, dtype=dtype)
        mesh, _ = init_meshes(config, device.type)
        if mesh is not None:
            policy = MixedPrecisionPolicy(
                param_dtype=dtype,
                reduce_dtype=getattr(torch, config.reduce_dtype),
                cast_forward_inputs=False,
            )
            fully_shard(self, mesh=mesh, mp_policy=policy)
        elif not on_meta and dist.is_available() and dist.is_initialized():
            for parameter in self.parameters():
                dist.broadcast(parameter.detach(), src=0)
        if on_meta:
            self.to_empty(device=device)
            coordinate = None if mesh is None else mesh.get_coordinate()
            self.reset_parameters(0 if coordinate is None else int(coordinate[-1]))

    def _group_logits(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != len(self.branches):
            raise ValueError("feature count does not match discriminator branches")
        joined = torch.cat(
            [branch(f) for branch, f in zip(self.branches, features, strict=True)], -1
        )
        return self.final_linear(self.final_norm(joined))

    def forward(self, feature_groups: Sequence[Sequence[Tensor]]) -> Tensor:
        return (
            torch.cat([self._group_logits(group) for group in feature_groups])
            .float()
            .reshape(-1)
        )

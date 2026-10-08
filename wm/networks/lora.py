# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

LORA_PARAMETER_PATTERN = r"(?:.+\.)?lora_[AB]\.weight"


class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self._configure_lora(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

    def _configure_lora(
        self,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Linear(
            self.in_features,
            self.rank,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.lora_B = nn.Linear(
            self.rank,
            self.out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.reset_lora_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> LoRALinear:
        # Do not allocate a throwaway full-size base projection when attaching
        # adapters to an already materialized model.
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features = linear.in_features
        result.out_features = linear.out_features
        result.weight = linear.weight
        if linear.bias is None:
            result.register_parameter("bias", None)
        else:
            result.bias = linear.bias
        result._configure_lora(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        result.train(linear.training)
        return result

    @torch.no_grad()
    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base = super().forward(input)
        update = self.lora_B(self.lora_A(self.lora_dropout(input)))
        return base + update * self.scaling


def _lora_options(config: Mapping[str, Any]) -> tuple[int, float, float, Sequence[str]]:
    rank = int(config.get("rank", 16))
    alpha = float(config.get("alpha", rank))
    dropout = float(config.get("dropout", 0.0))
    patterns = config.get("module_patterns", ())
    if isinstance(patterns, str):
        patterns = (patterns,)
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")
    return rank, alpha, dropout, tuple(patterns)


def inject_lora(
    module: nn.Module,
    config: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if config is None:
        return ()
    rank, alpha, dropout, raw_patterns = _lora_options(config)
    patterns = tuple(re.compile(pattern) for pattern in raw_patterns)
    if not patterns:
        return ()

    aliases_by_identity: dict[int, list[str]] = {}
    linear_by_identity: dict[int, nn.Linear] = {}
    for name, child in module.named_modules(remove_duplicate=False):
        if not name or type(child) is not nn.Linear:
            continue
        identity = id(child)
        aliases_by_identity.setdefault(identity, []).append(name)
        linear_by_identity[identity] = child

    # Preserve ``named_modules`` traversal order. Iterating a set of object
    # identities makes RNG consumption depend on process-specific addresses,
    # so identical seeds can initialize different adapters on different ranks.
    selected = tuple(
        identity
        for identity, aliases in aliases_by_identity.items()
        if any(pattern.fullmatch(alias) for alias in aliases for pattern in patterns)
    )
    replacements = {
        identity: LoRALinear.from_linear(
            linear_by_identity[identity],
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        for identity in selected
    }
    replaced_names: list[str] = []
    for identity in selected:
        replacement = replacements[identity]
        for name in aliases_by_identity[identity]:
            parent_name, _, child_name = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            parent._modules[child_name] = replacement
            replaced_names.append(name)
    return tuple(sorted(replaced_names))


@torch.no_grad()
def initialize_lora_parameters(module: nn.Module, seed: int = 0) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for child in module.modules():
            if isinstance(child, LoRALinear):
                child.reset_lora_parameters()


__all__ = [
    "LORA_PARAMETER_PATTERN",
    "LoRALinear",
    "initialize_lora_parameters",
    "inject_lora",
]

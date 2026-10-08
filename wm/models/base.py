# SPDX-License-Identifier: Apache-2.0

"""Lifecycle shared by every objective that trains against a frozen conditioner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import torch
from torch import nn

from wm.infra.model import Model
from wm.infra.parallel import ParallelConfig, parallelize_network
from wm.protocols import Conditioner

__all__ = ["ConditionedModel"]


class ConditionedModel(Model):
    """Objective with a frozen conditioner and a fixed set of sampling options.

    The conditioner (codec and text encoder) is never trained and never written
    to a checkpoint. Subclasses that hold further frozen roles extend
    ``checkpoint_excluded_prefixes``.
    """

    checkpoint_excluded_prefixes: ClassVar[tuple[str, ...]] = ("conditioner.",)

    def __init__(
        self,
        *,
        conditioner: nn.Module,
        sampling_defaults: Mapping[str, Any],
        sampling: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.conditioner: Conditioner = conditioner
        self.conditioner.requires_grad_(False)
        self.sampling: dict[str, Any] = {**sampling_defaults, **(sampling or {})}

    def parallelize_conditioner(
        self, config: ParallelConfig, device: torch.device | str
    ) -> None:
        parallelize_network(self.conditioner.text_encoder, config, device)

    def prepare_data_batch(self, data_batch: dict[str, Any]) -> dict[str, Any]:
        return self.conditioner.encode(data_batch, training=self.training)

    def checkpoint_model_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        excluded = self.checkpoint_excluded_prefixes
        return {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(excluded)
        }

    def sampling_options(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        """Merge per-call overrides into the configured sampling options."""
        unknown = set(overrides) - set(self.sampling)
        if unknown:
            raise TypeError(f"unknown sampling options {sorted(unknown)}")
        return {**self.sampling, **overrides}

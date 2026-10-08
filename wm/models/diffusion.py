# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from wm.infra.model import StepOutput
from wm.infra.parallel import ParallelConfig, parallelize_network
from wm.models.base import ConditionedModel
from wm.models.flow import (
    FlowUniPC,
    RectifiedFlowSchedule,
    invariant_randn,
    rf_interpolate,
    sampling_seeds,
)
from wm.protocols import DenoisingNetwork

__all__ = ["DiffusionModel"]


class DiffusionModel(ConditionedModel):
    def __init__(
        self,
        *,
        net: nn.Module,
        conditioner: nn.Module,
        schedule: RectifiedFlowSchedule | None = None,
        known_frames: int = 1,
        text_dropout: float = 0.1,
        trainable_patterns: Sequence[str] | None = None,
        parallel: ParallelConfig | None = None,
        conditioner_parallel: ParallelConfig | None = None,
        sampling: Mapping[str, Any] | None = None,
        validation_seed: int = 0,
    ) -> None:
        if not 0.0 <= text_dropout <= 1.0:
            raise ValueError(f"text_dropout must be in [0, 1], got {text_dropout}")
        super().__init__(
            conditioner=conditioner,
            sampling_defaults={"num_steps": 35, "guidance": 6.0, "seed": 0},
            sampling=sampling,
        )
        self.net: DenoisingNetwork = net
        self.schedule = schedule or RectifiedFlowSchedule(
            shift=5.0, train_distribution="waver"
        )
        self.known_frames = int(known_frames)
        self.text_dropout = float(text_dropout)
        self.parallel = parallel or ParallelConfig(fully_shard=False, compile=False)
        self.conditioner_parallel = conditioner_parallel or self.parallel
        self.validation_seed = int(validation_seed)
        self.configure_trainable_parameters(trainable_patterns, module=self.net)

    # ------------------------------------------------------------- lifecycle
    def parallelize(self, device: torch.device | str) -> None:
        mesh = parallelize_network(self.net, self.parallel, device)
        self.parallelize_conditioner(self.conditioner_parallel, device)
        self.register_context_parallel_mesh(mesh)

    # --------------------------------------------------------------- training
    def _loss(
        self,
        batch: Mapping[str, Any],
        *,
        generator: torch.Generator | None,
        drop_text: bool,
    ) -> StepOutput:
        x0 = batch["latent"].float()
        size, device = x0.shape[0], x0.device
        sigma = self.schedule.sample(size, device=device, generator=generator)
        noise = torch.randn(x0.shape, device=device, generator=generator)
        noisy = rf_interpolate(x0, noise, sigma)
        known = self.known_frames
        noisy[:, :, :known] = x0[:, :, :known]
        drop = (
            torch.rand(size, device=device, generator=generator) < self.text_dropout
            if drop_text
            else None
        )
        condition = self.conditioner.condition(batch, drop=drop)
        velocity = self.net(noisy, sigma, condition, known_frames=known).float()
        squared = (velocity - (noise - x0)).square()[:, :, known:]
        loss_per_sample = squared.flatten(1).sum(1) / x0[0].numel()
        loss = loss_per_sample.mean()
        return {
            "loss_per_sample": loss_per_sample.detach(),
            "sigma": sigma.detach(),
        }, loss

    def training_step(self, data_batch: dict[str, Any], iteration: int) -> StepOutput:
        del iteration
        return self._loss(data_batch, generator=None, drop_text=True)

    @torch.no_grad()
    def validation_step(self, data_batch: dict[str, Any], iteration: int) -> StepOutput:
        del iteration
        generator = torch.Generator(device=data_batch["latent"].device)
        generator.manual_seed(self.validation_seed)
        return self._loss(data_batch, generator=generator, drop_text=False)

    # --------------------------------------------------------------- sampling
    @torch.inference_mode()
    def sample(
        self, data_batch: dict[str, Any], **options: Any
    ) -> dict[str, torch.Tensor]:
        options = self.sampling_options(options)
        x0 = data_batch["latent"].float()
        known = self.known_frames
        condition = self.conditioner.condition(data_batch)
        guidance = float(options["guidance"])
        negative = (
            self.conditioner.condition(data_batch, negative=True)
            if guidance != 1.0
            else None
        )
        latent = invariant_randn(x0, sampling_seeds(data_batch, int(options["seed"])))
        latent[:, :, :known] = x0[:, :, :known]

        def velocity(current: torch.Tensor, sigma: float) -> torch.Tensor:
            sigma_batch = torch.full((x0.shape[0],), sigma, device=x0.device)
            result = self.net(
                current, sigma_batch, condition, known_frames=known
            ).float()
            if negative is not None:
                unconditional = self.net(
                    current, sigma_batch, negative, known_frames=known
                )
                result = unconditional.float() + guidance * (
                    result - unconditional.float()
                )
            return result

        def keep_known(current: torch.Tensor) -> torch.Tensor:
            current[:, :, :known] = x0[:, :, :known]
            return current

        solver = FlowUniPC(int(options["num_steps"]), shift=self.schedule.shift)
        latent = solver.run(latent, velocity, post_step=keep_known)
        return {
            **self.conditioner.visuals(data_batch),
            "ground_truth": self.conditioner.decode(x0),
            "sample": self.conditioner.decode(latent),
        }

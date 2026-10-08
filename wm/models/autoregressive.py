# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
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
from wm.protocols import BlockARNetwork, HistoryWindow

__all__ = ["ARMode", "AutoregressiveModel", "corrupt_history"]


class ARMode(StrEnum):
    TEACHER_FORCING = "teacher_forcing"
    DIFFUSION_FORCING = "diffusion_forcing"


# Frame-aware corruption of clean history (exposure, noise, rescale or no-op),
# which makes a teacher-forced model tolerant to its own imperfect history.
_EXPOSURE_RANGE = (0.3, 1.7)
_NOISE_RANGE = (0.0, 1.0 / 3.0)
_SCALE_RANGE = (0.9, 1.0)


@torch.no_grad()
def corrupt_history(
    frames: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    batch, _, count, height, width = frames.shape
    route = torch.randint(4, (batch, count), device=frames.device, generator=generator)
    draw = torch.rand(batch, count, device=frames.device, generator=generator)

    def span(bounds: tuple[float, float]) -> torch.Tensor:
        return bounds[0] + draw * (bounds[1] - bounds[0])

    value = torch.where(
        route == 0,
        span(_EXPOSURE_RANGE),
        torch.where(route == 1, span(_NOISE_RANGE), span(_SCALE_RANGE)),
    )[:, None, :, None, None]
    route = route[:, None, :, None, None]
    mean = frames.mean(dim=1, keepdim=True)
    noise = torch.randn(frames.shape, device=frames.device, generator=generator)
    output = torch.where(route == 0, (frames - mean) * value + mean, frames)
    output = torch.where(route == 1, value * noise + (1.0 - value) * frames, output)
    for b, t in (route[:, 0, :, 0, 0] == 2).nonzero().tolist():
        scale = float(value[b, 0, t])
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        frame = frames[b : b + 1, :, t]
        resized = F.interpolate(frame, size=size, mode="bilinear", antialias=True)
        output[b : b + 1, :, t] = F.interpolate(
            resized, size=(height, width), mode="bilinear", antialias=True
        )
    return output


class AutoregressiveModel(ConditionedModel):
    def __init__(
        self,
        *,
        net: nn.Module,
        conditioner: nn.Module,
        mode: ARMode | str = ARMode.TEACHER_FORCING,
        block_frames: int = 4,
        schedule: RectifiedFlowSchedule | None = None,
        text_dropout: float = 0.1,
        corrupt_clean_history: bool = True,
        diffusion_forcing_clean_sigma: float = 0.001,
        trainable_patterns: Sequence[str] | None = None,
        parallel: ParallelConfig | None = None,
        conditioner_parallel: ParallelConfig | None = None,
        sampling: Mapping[str, Any] | None = None,
        validation_seed: int = 0,
    ) -> None:
        # ``sink_blocks``/``recent_blocks``: bounded KV cache; ``None`` keeps all.
        super().__init__(
            conditioner=conditioner,
            sampling_defaults={
                "num_steps": 16,
                "guidance": 6.0,
                "seed": 0,
                "sink_blocks": 0,
                "recent_blocks": None,
            },
            sampling=sampling,
        )
        self.net: BlockARNetwork = net
        self.mode = ARMode(mode)
        self.block_frames = int(block_frames)
        self.schedule = schedule or RectifiedFlowSchedule(
            shift=5.0, train_distribution="waver"
        )
        self.text_dropout = float(text_dropout)
        self.corrupt_clean_history = bool(corrupt_clean_history)
        self.diffusion_forcing_clean_sigma = float(diffusion_forcing_clean_sigma)
        self.parallel = parallel or ParallelConfig(fully_shard=False, compile=False)
        self.conditioner_parallel = conditioner_parallel or self.parallel
        self.validation_seed = int(validation_seed)
        self.configure_trainable_parameters(trainable_patterns, module=self.net)

    def parallelize(self, device: torch.device | str) -> None:
        mesh = parallelize_network(self.net, self.parallel, device)
        self.parallelize_conditioner(self.conditioner_parallel, device)
        self.register_context_parallel_mesh(mesh)

    # --------------------------------------------------------------- training
    def _block_sigma(
        self,
        batch: int,
        frames: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        blocks = frames // self.block_frames
        sigma = self.schedule.sample(
            (batch, blocks), device=device, generator=generator
        )
        return sigma.repeat_interleave(self.block_frames, dim=1)

    def _loss(
        self,
        batch: Mapping[str, Any],
        *,
        generator: torch.Generator | None,
        training: bool,
    ) -> StepOutput:
        latent = batch["latent"].float()
        anchor, target = latent[:, :, :1], latent[:, :, 1:]
        size, _, frames = target.shape[:3]
        if frames % self.block_frames:
            raise ValueError(
                f"{frames} target frames are not a multiple of {self.block_frames}"
            )
        device = latent.device
        sigma = self._block_sigma(size, frames, device, generator)
        noise = torch.randn(target.shape, device=device, generator=generator)
        noisy = rf_interpolate(target, noise, sigma)
        clean = None
        if self.mode is ARMode.TEACHER_FORCING:
            clean = target
            if training and self.corrupt_clean_history:
                clean = corrupt_history(target, generator)
        drop = (
            torch.rand(size, device=device, generator=generator) < self.text_dropout
            if training
            else None
        )
        condition = self.conditioner.condition(batch, drop=drop)
        velocity = self.net.forward_ar(
            anchor, noisy, sigma, condition, block_frames=self.block_frames, clean=clean
        ).float()
        loss_per_sample = (velocity - (noise - target)).square().flatten(1).mean(1)
        return {"loss_per_sample": loss_per_sample.detach(), "sigma": sigma.detach()}, (
            loss_per_sample.mean()
        )

    def training_step(self, data_batch: dict[str, Any], iteration: int) -> StepOutput:
        del iteration
        return self._loss(data_batch, generator=None, training=True)

    @torch.no_grad()
    def validation_step(self, data_batch: dict[str, Any], iteration: int) -> StepOutput:
        del iteration
        generator = torch.Generator(device=data_batch["latent"].device)
        generator.manual_seed(self.validation_seed)
        return self._loss(data_batch, generator=generator, training=False)

    # --------------------------------------------------------------- sampling
    def _guided_velocity(
        self,
        caches: Sequence[Any],
        guidance: float,
        current: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        # ``caches``: the conditional cache, then (with CFG) the negative one.
        sigma_batch = torch.full((current.shape[0],), sigma, device=current.device)
        outputs = [
            self.net.denoise(cache, current, sigma_batch).float() for cache in caches
        ]
        if len(outputs) == 1:
            return outputs[0]
        return outputs[1] + guidance * (outputs[0] - outputs[1])

    @torch.inference_mode()
    def sample(
        self, data_batch: dict[str, Any], **options: Any
    ) -> dict[str, torch.Tensor]:
        options = self.sampling_options(options)
        latent = data_batch["latent"].float()
        history = HistoryWindow(int(options["sink_blocks"]), options["recent_blocks"])
        guidance = float(options["guidance"])
        conditions = [self.conditioner.condition(data_batch)]
        if guidance != 1.0:
            conditions.append(self.conditioner.condition(data_batch, negative=True))
        anchor = latent[:, :, :1]
        caches = [
            self.net.prefill(anchor, condition, history=history)
            for condition in conditions
        ]
        noise = invariant_randn(
            latent[:, :, 1:], sampling_seeds(data_batch, int(options["seed"]))
        )
        clean_sigma = (
            None
            if self.mode is ARMode.TEACHER_FORCING
            else torch.full(
                (latent.shape[0],),
                self.diffusion_forcing_clean_sigma,
                device=latent.device,
            )
        )
        solver = FlowUniPC(int(options["num_steps"]), shift=self.schedule.shift)
        blocks = []
        for start in range(0, noise.shape[2], self.block_frames):
            block = solver.run(
                noise[:, :, start : start + self.block_frames].clone(),
                partial(self._guided_velocity, caches, guidance),
            )
            caches = [self.net.commit(cache, block, clean_sigma) for cache in caches]
            blocks.append(block)
        generated = torch.cat((anchor, *blocks), dim=2)
        return {
            **self.conditioner.visuals(data_batch),
            "ground_truth": self.conditioner.decode(latent),
            "sample": self.conditioner.decode(generated),
        }

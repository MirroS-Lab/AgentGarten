# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# FlowUniPC is reduced from Hugging Face diffusers v0.31.0 UniPC and the
# flow-matching adaptation released by the Alibaba Wan team (Apache-2.0).

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any

import numpy as np
import torch

__all__ = [
    "FlowUniPC",
    "RectifiedFlowSchedule",
    "TrainDistribution",
    "UniformShift",
    "VelocityFunction",
    "invariant_randn",
    "rf_interpolate",
    "rf_x0",
    "sampling_seeds",
]

# ``velocity(sample, sigma)``: the network prediction at one solver step.
VelocityFunction = Callable[[torch.Tensor, float], torch.Tensor]


class TrainDistribution(StrEnum):
    UNIFORM = "uniform"
    LOGITNORMAL = "logitnormal"
    WAVER = "waver"


def shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _view(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value.view(-1, *([1] * (like.ndim - 1)))
    return value[:, None, :, None, None]


def rf_interpolate(
    x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    sigma = _view(sigma.to(x0.dtype), x0)
    return (1.0 - sigma) * x0 + sigma * noise


def rf_x0(
    noisy: torch.Tensor, sigma: torch.Tensor, velocity: torch.Tensor
) -> torch.Tensor:
    return noisy - _view(sigma.to(noisy.dtype), noisy) * velocity


class RectifiedFlowSchedule:
    _WAVER_MODE_S = 1.29

    def __init__(
        self,
        *,
        shift: float = 1.0,
        train_distribution: TrainDistribution | str = TrainDistribution.UNIFORM,
    ) -> None:
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self.shift = float(shift)
        self.train_distribution = TrainDistribution(train_distribution)

    def sample(
        self,
        shape: int | Sequence[int],
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        uniform = torch.rand(shape, device=device, generator=generator)
        match self.train_distribution:
            case TrainDistribution.UNIFORM:
                data_time = uniform
            case TrainDistribution.LOGITNORMAL:
                data_time = torch.sigmoid(
                    torch.randn(shape, device=device, generator=generator)
                )
            case TrainDistribution.WAVER:
                data_time = (
                    1.0
                    - uniform
                    - self._WAVER_MODE_S
                    * (torch.cos(torch.pi * 0.5 * uniform).square() - 1.0 + uniform)
                )
        return shift_sigma(1.0 - data_time, self.shift)


class UniformShift:
    def __init__(self, shift: float = 5.0) -> None:
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self.shift = float(shift)

    def sample(
        self,
        shape: int | Sequence[int],
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return shift_sigma(
            torch.rand(shape, device=device, generator=generator), self.shift
        )


# Per-sample noise that depends only on (seed, stream), never on batching.
# Stream 0 is the plain seeded stream; others give independent draws.
def invariant_randn(
    reference: torch.Tensor,
    seeds: Sequence[int],
    dtype: torch.dtype = torch.float32,
    *,
    stream: int = 0,
) -> torch.Tensor:
    if len(seeds) != reference.shape[0]:
        raise ValueError(f"need {reference.shape[0]} seeds, got {len(seeds)}")
    samples = [
        np.random.RandomState(
            int(seed) if stream == 0 else (int(seed), int(stream))
        ).standard_normal(tuple(reference.shape[1:]))
        for seed in seeds
    ]
    return torch.from_numpy(np.stack(samples).astype(np.float32)).to(
        device=reference.device, dtype=dtype
    )


def sampling_seeds(batch: Mapping[str, Any], seed: int) -> list[int]:
    """Per-sample seeds: ``seed + sample_id``, or the batch index without ids."""
    ids = batch.get("sample_id")
    size = int(batch["latent"].shape[0])
    if ids is None:
        return [seed + index for index in range(size)]
    return [seed + int(value) for value in torch.as_tensor(ids).reshape(-1).tolist()]


class FlowUniPC:
    def __init__(
        self, num_steps: int, *, shift: float, num_train_timesteps: int = 1000
    ) -> None:
        sigma_max = float(np.float32(1.0 - 1.0 / num_train_timesteps))
        values = np.linspace(sigma_max, 0.0, int(num_steps) + 1)[:-1]
        values = shift * values / (1.0 + (shift - 1.0) * values)
        self.num_train_timesteps = int(num_train_timesteps)
        self.sigmas = torch.from_numpy(
            np.concatenate((values, [0.0])).astype(np.float32)
        )
        # The released solver deliberately truncates fractional timesteps.
        self.timesteps = torch.from_numpy(values * num_train_timesteps).to(torch.int64)
        self.order = 2

    def network_sigmas(self) -> torch.Tensor:
        return self.timesteps.float() / self.num_train_timesteps

    def run(
        self,
        sample: torch.Tensor,
        velocity: VelocityFunction,
        post_step: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        sigmas = self.sigmas.to(sample.device)
        network_sigmas = self.network_sigmas()
        outputs: list[torch.Tensor] = []
        last_sample: torch.Tensor | None = None
        lower_order = 0
        this_order = 1
        steps = len(self.timesteps)
        for step in range(steps):
            prediction = velocity(sample, float(network_sigmas[step]))
            converted = sample - sigmas[step] * prediction.to(sample.dtype)
            if step > 0 and last_sample is not None:
                sample = self._corrector(
                    converted, outputs, last_sample, sigmas, step, this_order
                )
            outputs = [*outputs[-(self.order - 1) :], converted]
            order = min(self.order, steps - step)
            this_order = min(order, lower_order + 1)
            last_sample = sample
            sample = self._predictor(sample, outputs, sigmas, step, this_order)
            lower_order = min(self.order, lower_order + 1)
            if post_step is not None:
                sample = post_step(sample)
        return sample

    @staticmethod
    def _lambda(sigma: torch.Tensor) -> torch.Tensor:
        return torch.log(1.0 - sigma) - torch.log(sigma)

    def _predictor(
        self,
        sample: torch.Tensor,
        outputs: list[torch.Tensor],
        sigmas: torch.Tensor,
        step: int,
        order: int,
    ) -> torch.Tensor:
        m0 = outputs[-1]
        sigma_t, sigma_s0 = sigmas[step + 1], sigmas[step]
        h = self._lambda(sigma_t) - self._lambda(sigma_s0)
        phi_1 = torch.expm1(-h)
        predicted = sigma_t / sigma_s0 * sample - (1.0 - sigma_t) * phi_1 * m0
        if order == 2:
            ratio = (self._lambda(sigmas[step - 1]) - self._lambda(sigma_s0)) / h
            difference = (outputs[-2] - m0) / ratio
            predicted = predicted - (1.0 - sigma_t) * phi_1 * 0.5 * difference
        return predicted.to(sample.dtype)

    def _corrector(
        self,
        model_output: torch.Tensor,
        outputs: list[torch.Tensor],
        last_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step: int,
        order: int,
    ) -> torch.Tensor:
        m0 = outputs[-1]
        sigma_t, sigma_s0 = sigmas[step], sigmas[step - 1]
        h = self._lambda(sigma_t) - self._lambda(sigma_s0)
        hh = -h
        phi_1 = torch.expm1(hh)
        base = phi_1
        phi_k = phi_1 / hh - 1.0
        ratios: list[torch.Tensor | float] = []
        differences = []
        if order == 2:
            ratio = (self._lambda(sigmas[step - 2]) - self._lambda(sigma_s0)) / h
            ratios.append(ratio)
            differences.append((outputs[-2] - m0) / ratio)
        ratios.append(1.0)
        if order == 1:
            rho = torch.tensor(
                [0.5], dtype=last_sample.dtype, device=last_sample.device
            )
        else:
            r = torch.tensor(
                [float(value) for value in ratios], device=last_sample.device
            )
            rows, rhs, factorial = [], [], 1
            for index in range(1, order + 1):
                rows.append(torch.pow(r, index - 1))
                rhs.append(phi_k * factorial / base)
                factorial *= index + 1
                phi_k = phi_k / hh - 1.0 / factorial
            rho = torch.linalg.solve(
                torch.stack(rows), torch.tensor(rhs, device=last_sample.device)
            ).to(last_sample.dtype)
        corrected = sigma_t / sigma_s0 * last_sample - (1.0 - sigma_t) * phi_1 * m0
        residual = rho[0] * differences[0] if differences else 0
        corrected = corrected - (1.0 - sigma_t) * base * (
            residual + rho[-1] * (model_output - m0)
        )
        return corrected.to(last_sample.dtype)

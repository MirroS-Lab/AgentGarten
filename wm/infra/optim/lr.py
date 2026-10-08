# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math


class LinearWarmup:
    def __init__(
        self,
        *,
        warmup_steps: int = 100,
        start_factor: float = 1.0e-6,
    ) -> None:
        self.warmup_steps = int(warmup_steps)
        self.start_factor = float(start_factor)

    def __call__(self, step: int) -> float:
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return 1.0
        progress = max(float(step), 0.0) / float(self.warmup_steps)
        return self.start_factor + (1.0 - self.start_factor) * progress


class CosineDecay:
    def __init__(
        self,
        *,
        decay_steps: int,
        min_factor: float = 0.0,
    ) -> None:
        self.decay_steps = int(decay_steps)
        self.min_factor = float(min_factor)
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive")
        if not math.isfinite(self.min_factor) or not 0.0 <= self.min_factor <= 1.0:
            raise ValueError("min_factor must be finite and in [0, 1]")

    def __call__(self, step: int) -> float:
        progress = min(max(float(step), 0.0), float(self.decay_steps))
        progress /= float(self.decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_factor + (1.0 - self.min_factor) * cosine


class LinearWarmupCosineDecay:
    def __init__(
        self,
        *,
        warmup_steps: int,
        decay_steps: int,
        start_factor: float = 0.0,
        min_factor: float = 0.0,
    ) -> None:
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = int(decay_steps)
        self.start_factor = float(start_factor)
        self.min_factor = float(min_factor)
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.decay_steps <= self.warmup_steps:
            raise ValueError("decay_steps must be greater than warmup_steps")
        if not 0.0 <= self.start_factor <= 1.0:
            raise ValueError("start_factor must be in [0, 1]")
        if not 0.0 <= self.min_factor <= 1.0:
            raise ValueError("min_factor must be in [0, 1]")

    def __call__(self, step: int) -> float:
        bounded_step = min(max(float(step), 0.0), float(self.decay_steps))
        if self.warmup_steps and bounded_step < self.warmup_steps:
            progress = bounded_step / float(self.warmup_steps)
            return self.start_factor + (1.0 - self.start_factor) * progress
        progress = (bounded_step - self.warmup_steps) / (
            self.decay_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_factor + (1.0 - self.min_factor) * cosine

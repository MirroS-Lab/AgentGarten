# SPDX-License-Identifier: Apache-2.0

"""Distribution matching distillation (DMD) for a block-autoregressive student.

Roles: student, frozen real score (teacher, CFG), fake score trained on student
rollouts, optional DMD2 discriminator head.

Student update (Self Gradient Forcing): Pass 1 rolls out every block with
causal rCM up to one exit step (shared by blocks and ranks) without grad;
Pass 2 is one teacher-forcing forward over [C0, (x_t, x0)...] that reproduces
Pass 1 bitwise, so gradients reach the denoiser and the clean-context K/V.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wm.infra import log
from wm.infra.config import LazyDict, instantiate
from wm.infra.model import StepOutput, TrainingClosureSpec
from wm.infra.parallel import ParallelConfig, parallelize_network
from wm.infra.rng_state import isolated_rng
from wm.infra.utils import frozen_parameters
from wm.models.base import ConditionedModel
from wm.models.exact_regularization import (
    check_exact_regularization,
    exact_penalties,
)
from wm.models.flow import (
    UniformShift,
    invariant_randn,
    rf_interpolate,
    rf_x0,
    sampling_seeds,
)
from wm.protocols import (
    BlockARNetwork,
    Condition,
    DenoisingNetwork,
    Discriminator,
    HistoryWindow,
)

__all__ = ["RCM_ENDPOINTS", "DMDModel", "GANConfig", "PhaseClock", "Regularization"]

# Causal-rCM 4-step endpoint plan; an ``n``-step rollout uses its prefix.
RCM_ENDPOINTS = (1600.0 / 1601.0, 15.0 / 16.0, 5.0 / 6.0, 5.0 / 8.0)


class Regularization(StrEnum):
    FINITE_DIFFERENCE = "finite_difference"
    EXACT = "exact"


@dataclass(frozen=True)
class GANConfig:
    generator_weight: float = 0.01
    discriminator_weight: float = 0.01
    relativistic: bool = True
    regularization: Regularization = Regularization.FINITE_DIFFERENCE
    # Finite-difference R3: weight * E[(D(x + sigma * eps) - D(x))^2].
    regularization_weight: float = 30.0
    regularization_sigma: float = 0.05
    # Exact R1/R2: weight * E[||grad_x D(x)||^2]; 0.075 = 30 * 0.05^2.
    exact_regularization_weight: float = 0.075
    warmup_updates: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "regularization", Regularization(self.regularization))
        if self.warmup_updates < 0:
            raise ValueError("warmup_updates must be non-negative")


@dataclass(frozen=True)
class PhaseClock:
    fake_score_updates_per_student: int = 5
    warmup_updates: int = 0

    def __post_init__(self) -> None:
        if self.fake_score_updates_per_student < 1 or self.warmup_updates < 0:
            raise ValueError(f"invalid phase clock {self}")

    @property
    def cycle(self) -> int:
        return self.fake_score_updates_per_student + 1

    def is_warmup(self, iteration: int) -> bool:
        return iteration < self.warmup_updates

    def is_student(self, iteration: int) -> bool:
        return (
            not self.is_warmup(iteration)
            and (iteration - self.warmup_updates) % self.cycle == 0
        )

    def role_updates(self, iteration: int) -> dict[str, int]:
        normal = max(0, iteration - self.warmup_updates)
        student = (normal + self.cycle - 1) // self.cycle
        return {"student": student, "fake_score": normal - student}

    def is_cadence_due(self, iteration: int, interval: int | None) -> bool:
        if interval is None or interval <= 0 or iteration <= 0:
            return False
        counts = self.role_updates(iteration)
        return (
            counts["student"] > 0
            and counts["student"] % interval == 0
            and counts["fake_score"] % (interval * self.fake_score_updates_per_student)
            == 0
            and counts["fake_score"] > 0
        )


@dataclass
class _Rollout:
    noisy: Tensor  # [B, C, T, H, W] query at the exit step
    sigma: Tensor  # [B, T]
    clean: Tensor  # [B, C, T, H, W] committed x0 (= generated video)


def _pair_loss(
    real_logits: Tensor, fake_logits: Tensor, *, relativistic: bool
) -> Tensor:
    if relativistic:
        return F.softplus(-(real_logits - fake_logits)).mean()
    return (F.softplus(-real_logits) + F.softplus(fake_logits)).mean()


def _pair_metrics(
    pair: Tensor, real_logits: Tensor, fake_logits: Tensor
) -> dict[str, Tensor]:
    return {
        "metrics/discriminator/pair_loss": pair.detach(),
        "metrics/discriminator/accuracy": (real_logits > fake_logits)
        .float()
        .mean()
        .detach(),
    }


class DMDModel(ConditionedModel):
    # The real score is frozen and restored from its own initialization.
    checkpoint_excluded_prefixes = ("real_score.", "conditioner.")

    def __init__(
        self,
        *,
        student: nn.Module,
        real_score: nn.Module,
        fake_score: nn.Module,
        conditioner: nn.Module,
        discriminator: Discriminator | None = None,
        gan: GANConfig | None = None,
        block_frames: int = 4,
        rollout_endpoints: Sequence[float] = RCM_ENDPOINTS,
        score_shift: float = 5.0,
        real_score_guidance: float = 6.0,
        fake_score_updates_per_student: int = 5,
        dmd_loss_scale: float = 0.5,
        history: HistoryWindow | None = None,
        student_trainable_patterns: Sequence[str] | None = None,
        fake_score_trainable_patterns: Sequence[str] | None = None,
        student_parallel: ParallelConfig | None = None,
        score_parallel: ParallelConfig | None = None,
        real_score_parallel: ParallelConfig | None = None,
        discriminator_parallel: ParallelConfig | None = None,
        conditioner_parallel: ParallelConfig | None = None,
        sampling: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        if (discriminator is None) != (gan is None):
            raise ValueError("pass both discriminator and gan, or neither")
        endpoints = tuple(float(value) for value in rollout_endpoints)
        if (
            not endpoints
            or any(not 0 < value <= 1 for value in endpoints)
            or any(a <= b for a, b in itertools.pairwise(endpoints))
        ):
            raise ValueError(
                f"rollout endpoints must be strictly descending in (0, 1]: {endpoints}"
            )
        if block_frames < 1 or int(block_frames) != block_frames:
            raise ValueError(
                f"block_frames must be a positive integer, got {block_frames}"
            )
        super().__init__(
            conditioner=conditioner,
            sampling_defaults={"num_steps": len(endpoints), "seed": 0},
            sampling=sampling,
        )
        self.student: BlockARNetwork = student
        self.real_score: DenoisingNetwork = real_score
        self.fake_score: DenoisingNetwork = fake_score
        self.discriminator = discriminator
        self.gan = gan
        self.block_frames = int(block_frames)
        self.rollout_endpoints = endpoints
        self.score_time = UniformShift(score_shift)
        self.real_score_guidance = float(real_score_guidance)
        self.clock = PhaseClock(
            fake_score_updates_per_student, gan.warmup_updates if gan else 0
        )
        self.dmd_loss_scale = float(dmd_loss_scale)
        self.history = history or HistoryWindow()
        default = ParallelConfig(fully_shard=False, compile=False)
        self.student_parallel = student_parallel or default
        self.score_parallel = score_parallel or default
        self.real_score_parallel = real_score_parallel or self.score_parallel
        self.discriminator_parallel = discriminator_parallel or default
        self.conditioner_parallel = conditioner_parallel or self.score_parallel
        self.seed = int(seed)
        self.real_score.requires_grad_(False)
        self.configure_trainable_parameters(
            student_trainable_patterns, module=self.student
        )
        self.configure_trainable_parameters(
            fake_score_trainable_patterns, module=self.fake_score
        )
        if self.student_parallel.compile:
            log.warning(
                "DMD student is compiled: compiled training graphs round differently "
                "from the no-grad rollout, so SGF Pass 2 only approximately "
                "reproduces Pass 1 (see metrics/student/sgf_recovery_max_abs)."
            )
        if gan is not None and gan.regularization is Regularization.EXACT:
            if self.real_score_parallel.compile:
                raise ValueError(
                    "exact R1/R2 runs the real score in forward mode; set "
                    "real_score_parallel.compile=False"
                )
            if self.real_score_parallel.context_parallel_size > 1:
                raise ValueError(
                    "exact R1/R2 does not support context parallelism: its "
                    "forward-mode pass cannot cross the CP all-to-all. Use "
                    "context_parallel_size=1 or finite_difference regularization"
                )
            check_exact_regularization(self.real_score, discriminator)

    # ------------------------------------------------------------- lifecycle
    def parallelize(self, device: torch.device | str) -> None:
        # The student keeps parameters gathered from rollout to replay.
        student_parallel = replace(self.student_parallel, reshard_after_forward=False)
        mesh = parallelize_network(self.student, student_parallel, device)
        meshes = [
            parallelize_network(self.real_score, self.real_score_parallel, device),
            parallelize_network(self.fake_score, self.score_parallel, device),
        ]
        self.parallelize_conditioner(self.conditioner_parallel, device)
        if self.discriminator is not None:
            self.discriminator.parallelize(self.discriminator_parallel, device)
        self.real_score.eval()
        self.register_context_parallel_mesh(mesh, *meshes)

    def train(self, mode: bool = True) -> DMDModel:
        super().train(mode)
        self.real_score.eval()
        return self

    def is_cadence_due(self, iteration: int, interval: int | None) -> bool:
        return self.clock.is_cadence_due(int(iteration), interval)

    def get_optimizer_names(self, iteration: int) -> tuple[str, ...]:
        if self.clock.is_warmup(iteration):
            return ("discriminator",)
        if self.clock.is_student(iteration):
            return ("student",)
        return ("fake_score", "discriminator") if self.gan else ("fake_score",)

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ):
        roles = {"student": self.student, "fake_score": self.fake_score}
        if self.discriminator is not None:
            roles["discriminator"] = self.discriminator
        missing = set(roles) - set(optimizer_config)
        if missing:
            raise KeyError(f"optimizer config lacks roles {sorted(missing)}")
        self.optimizer_dict, self.scheduler_dict = {}, {}
        for name, module in roles.items():
            optimizer = instantiate(
                optimizer_config[name],
                params=self._group_module_trainable_parameters(module),
            )
            self.optimizer_dict[name] = optimizer
            self.scheduler_dict[name] = instantiate(
                scheduler_config, optimizer=optimizer
            )
        return self.optimizer_dict, self.scheduler_dict

    # --------------------------------------------------------------- rollout
    def _exit_step(self, iteration: int) -> int:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + iteration)
        return int(
            torch.randint(len(self.rollout_endpoints), (1,), generator=generator)
        )

    @torch.no_grad()
    def _rollout(
        self,
        anchor: Tensor,
        condition: Condition,
        frames: int,
        steps: int,
        *,
        noise: Tensor | None = None,
    ) -> _Rollout:
        # ``noise``: [steps, B, C, frames, H, W], the draw of every rCM step;
        # ``None`` draws from the global generator.
        self._validate_rollout(frames, steps)
        endpoints = self.rollout_endpoints[:steps]
        batch = anchor.shape[0]
        shape = (batch, anchor.shape[1], self.block_frames, *anchor.shape[3:])
        cache = self.student.prefill(anchor, condition, history=self.history)
        noisy_blocks, clean_blocks = [], []
        clean = anchor

        def draw(step: int, start: int) -> Tensor:
            if noise is None:
                return torch.randn(shape, device=anchor.device)
            return noise[step, :, :, start : start + self.block_frames]

        for block, start in enumerate(range(0, frames, self.block_frames)):
            current = draw(0, start)
            for step, sigma in enumerate(endpoints):
                if step:
                    current = rf_interpolate(
                        clean,
                        draw(step, start),
                        torch.full((batch,), sigma, device=anchor.device),
                    )
                sigma_batch = torch.full((batch,), sigma, device=anchor.device)
                velocity = self.student.denoise(cache, current, sigma_batch).float()
                clean = rf_x0(current, sigma_batch, velocity)
            noisy_blocks.append(current)
            clean_blocks.append(clean)
            if block < frames // self.block_frames - 1:
                cache = self.student.commit(cache, clean)
        exit_sigma = torch.full((batch, frames), endpoints[-1], device=anchor.device)
        return _Rollout(
            torch.cat(noisy_blocks, 2), exit_sigma, torch.cat(clean_blocks, 2)
        )

    def _validate_rollout(self, frames: int, steps: int) -> None:
        if frames < 1 or frames % self.block_frames:
            raise ValueError(
                f"target latent frames must be a positive multiple of "
                f"block_frames={self.block_frames}, got {frames}"
            )
        if int(steps) != steps or not 1 <= steps <= len(self.rollout_endpoints):
            raise ValueError(
                f"num_steps must be an integer in [1, {len(self.rollout_endpoints)}], "
                f"got {steps}"
            )

    def _score(
        self,
        network: DenoisingNetwork,
        anchor: Tensor,
        noisy: Tensor,
        sigma: Tensor,
        condition: Condition,
    ) -> Tensor:
        velocity = network(
            torch.cat((anchor, noisy), dim=2), sigma, condition, known_frames=1
        )
        return velocity[:, :, 1:].float()

    def _gan_roles(self) -> tuple[GANConfig, Discriminator]:
        if self.gan is None or self.discriminator is None:
            raise RuntimeError("this objective was built without a discriminator")
        return self.gan, self.discriminator

    def _real_features(
        self, anchor: Tensor, noisy: Tensor, sigma: Tensor, condition: Condition
    ) -> tuple[Tensor, ...]:
        # Hidden states of the frozen real score at the discriminator's layers.
        _, discriminator = self._gan_roles()
        _, features = self.real_score(
            torch.cat((anchor, noisy), dim=2),
            sigma,
            condition,
            known_frames=1,
            feature_layers=discriminator.feature_layers,
        )
        return features

    def _real_velocity(
        self,
        anchor: Tensor,
        noisy: Tensor,
        sigma: Tensor,
        conditions: Sequence[Condition],
    ) -> Tensor:
        velocity = self._score(self.real_score, anchor, noisy, sigma, conditions[0])
        if len(conditions) == 1:
            return velocity
        unconditional = self._score(
            self.real_score, anchor, noisy, sigma, conditions[1]
        )
        return unconditional + self.real_score_guidance * (velocity - unconditional)

    # -------------------------------------------------------------- closures
    def training_step_closures(
        self, data_batch: dict[str, Any], iteration: int
    ) -> Iterator[TrainingClosureSpec]:
        latent = data_batch["latent"].float()
        anchor, real = latent[:, :, :1], latent[:, :, 1:]
        condition = self.conditioner.condition(data_batch)
        steps = self._exit_step(iteration) + 1
        rollout = self._rollout(anchor, condition, real.shape[2], steps)
        if self.clock.is_student(iteration):
            conditions = [condition]
            if self.real_score_guidance != 1.0:
                conditions.append(self.conditioner.condition(data_batch, negative=True))
            yield (
                "student",
                partial(self._student_loss, anchor, real, rollout, conditions),
                True,
            )
        else:
            warmup = self.clock.is_warmup(iteration)
            yield (
                "fake_score",
                partial(
                    self._fake_score_loss,
                    anchor,
                    real,
                    rollout.clean,
                    condition,
                    warmup=warmup,
                ),
                True,
            )

    def training_step(self, data_batch: dict[str, Any], iteration: int) -> StepOutput:
        outputs, losses = {}, []
        for _name, closure, _last in self.training_step_closures(data_batch, iteration):
            output, loss = closure()
            outputs.update(output)
            losses.append(loss)
        return outputs, torch.stack(losses).sum()

    def _student_loss(
        self,
        anchor: Tensor,
        real: Tensor,
        rollout: _Rollout,
        conditions: Sequence[Condition],
    ) -> StepOutput:
        # Pass 2: teacher forcing over the detached Pass-1 queries and context.
        velocity = self.student.forward_ar(
            anchor,
            rollout.noisy,
            rollout.sigma,
            conditions[0],
            block_frames=self.block_frames,
            clean=rollout.clean,
            history=self.history,
        ).float()
        x0 = rf_x0(rollout.noisy, rollout.sigma, velocity)
        target, gan_gradient, metrics = self._student_targets(
            anchor, real, x0.detach(), conditions
        )
        error = (x0.double() - target.double()).square().sum()
        dmd_loss = self.dmd_loss_scale * error / x0.numel()
        loss = dmd_loss.float()
        if gan_gradient is not None:
            loss = loss + (x0 * gan_gradient).sum()
        with torch.no_grad():
            recovery = (x0 - rollout.clean).abs().max()
        return {
            **metrics,
            "metrics/student/dmd_loss": dmd_loss.detach(),
            "metrics/student/sgf_recovery_max_abs": recovery,
            "metrics/student/exit_sigma": rollout.sigma[0, 0],
        }, loss

    def _student_targets(
        self, anchor: Tensor, real: Tensor, x0: Tensor, conditions: Sequence[Condition]
    ) -> tuple[Tensor, Tensor | None, dict[str, Tensor]]:
        batch = x0.shape[0]
        sigma = self.score_time.sample(batch, device=x0.device)
        noise = torch.randn_like(x0)
        gan_gradient = None
        metrics: dict[str, Tensor] = {}
        with torch.no_grad():
            noisy = rf_interpolate(x0, noise, sigma)
            fake_velocity = self._score(
                self.fake_score, anchor, noisy, sigma, conditions[0]
            )
            real_velocity = self._real_velocity(anchor, noisy, sigma, conditions)
            x0_fake = rf_x0(noisy, sigma, fake_velocity).double()
            x0_real = rf_x0(noisy, sigma, real_velocity).double()
            normalizer = (
                (x0.double() - x0_real).abs().flatten(1).mean(1).clamp_min(1e-5)
            )
            gradient = (x0_fake - x0_real) / normalizer.view(-1, 1, 1, 1, 1)
            target = x0.double() - gradient
        if self.gan is not None:
            gan_gradient, metrics = self._generator_gradient(
                anchor, real, x0, noise, sigma, conditions[0]
            )
        metrics["metrics/student/dmd_gradient_rms"] = (
            gradient.float().square().mean().sqrt()
        )
        return target, gan_gradient, metrics

    def _generator_gradient(
        self,
        anchor: Tensor,
        real: Tensor,
        x0: Tensor,
        noise: Tensor,
        sigma: Tensor,
        condition: Condition,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        gan, discriminator = self._gan_roles()
        leaf = x0.detach().requires_grad_(True)
        # Backward (and any activation recompute) must see the same frozen head
        # as the forward did, so the head stays frozen for the whole block.
        with torch.enable_grad(), frozen_parameters(discriminator):
            fake_features = self._real_features(
                anchor, rf_interpolate(leaf, noise, sigma), sigma, condition
            )
            fake_logits = discriminator([fake_features])
            with torch.no_grad():
                real_features = self._real_features(
                    anchor, rf_interpolate(real, noise, sigma), sigma, condition
                )
                real_logits = discriminator([real_features])
            margin = fake_logits - real_logits if gan.relativistic else fake_logits
            loss = gan.generator_weight * F.softplus(-margin).mean()
            (gradient,) = torch.autograd.grad(loss, leaf)
        return torch.nan_to_num(gradient.detach()), {
            "metrics/student/gan_loss": loss.detach(),
            "metrics/student/gan_fake_minus_real_logit": margin.detach().mean(),
        }

    def _fake_score_loss(
        self,
        anchor: Tensor,
        real: Tensor,
        generated: Tensor,
        condition: Condition,
        *,
        warmup: bool,
    ) -> StepOutput:
        batch = generated.shape[0]
        sigma = self.score_time.sample(batch, device=generated.device)
        noise = torch.randn_like(generated)
        noisy = rf_interpolate(generated, noise, sigma)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not warmup):
            velocity = self._score(self.fake_score, anchor, noisy, sigma, condition)
        fake_loss = (velocity - (noise - generated)).square().mean()
        outputs = {"metrics/fake_score/loss": fake_loss.detach()}
        loss = fake_loss if not warmup else fake_loss.detach() * 0
        if self.gan is not None:
            gan_loss, gan_metrics = self._discriminator_loss(
                anchor, real, noisy, noise, sigma, condition
            )
            loss = loss + gan_loss
            outputs.update(gan_metrics)
        return outputs, loss

    def _discriminator_loss(
        self,
        anchor: Tensor,
        real: Tensor,
        fake_noisy: Tensor,
        noise: Tensor,
        sigma: Tensor,
        condition: Condition,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        gan, discriminator = self._gan_roles()
        real_noisy = rf_interpolate(real, noise, sigma)
        if gan.regularization is Regularization.EXACT:
            return self._exact_discriminator_loss(
                anchor, real_noisy, fake_noisy, sigma, condition
            )
        rows = [fake_noisy, real_noisy]
        if gan.regularization_weight > 0:
            rows += [
                real_noisy + gan.regularization_sigma * torch.randn_like(real_noisy),
                fake_noisy + gan.regularization_sigma * torch.randn_like(fake_noisy),
            ]
        count = len(rows)
        with torch.no_grad():
            features = self._real_features(
                torch.cat((anchor,) * count),
                torch.cat(rows),
                sigma.repeat(count),
                condition.repeat(count),
            )
        logits = discriminator([features]).split(fake_noisy.shape[0])
        fake_logits, real_logits = logits[0], logits[1]
        pair = _pair_loss(real_logits, fake_logits, relativistic=gan.relativistic)
        loss = gan.discriminator_weight * pair
        metrics = _pair_metrics(pair, real_logits, fake_logits)
        if gan.regularization_weight > 0:
            r1 = (logits[2] - real_logits).square().mean()
            r2 = (logits[3] - fake_logits).square().mean()
            loss = loss + gan.discriminator_weight * 0.5 * gan.regularization_weight * (
                r1 + r2
            )
            metrics.update(
                {
                    "metrics/discriminator/r1": r1.detach(),
                    "metrics/discriminator/r2": r2.detach(),
                }
            )
        return loss, metrics

    def _exact_discriminator_loss(
        self,
        anchor: Tensor,
        real_noisy: Tensor,
        fake_noisy: Tensor,
        sigma: Tensor,
        condition: Condition,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        gan, discriminator = self._gan_roles()
        batch = fake_noisy.shape[0]
        features = partial(
            self._real_features,
            torch.cat((anchor, anchor)),
            sigma=sigma.repeat(2),
            condition=condition.repeat(2),
        )
        result = exact_penalties(
            features, discriminator, torch.cat((fake_noisy, real_noisy))
        )
        fake_logits, real_logits = result.logits.split(batch)
        pair = _pair_loss(real_logits, fake_logits, relativistic=gan.relativistic)
        r2, r1 = result.penalty.split(batch)
        # Value: weighted (R1 + R2); gradient: exact dR/dtheta.
        scale = gan.discriminator_weight * 0.5 * gan.exact_regularization_weight
        regularization = scale * (
            (r1.mean() + r2.mean()) + result.surrogate.view(2, batch).mean(1).sum()
        )
        loss = gan.discriminator_weight * pair + regularization
        return loss, {
            **_pair_metrics(pair, real_logits, fake_logits),
            "metrics/discriminator/r1": r1.mean(),
            "metrics/discriminator/r2": r2.mean(),
        }

    # ------------------------------------------------------------- evaluation
    @torch.no_grad()
    def validation_step(self, data_batch: dict[str, Any], iteration: int) -> StepOutput:
        del iteration
        latent = data_batch["latent"].float()
        anchor, real = latent[:, :, :1], latent[:, :, 1:]
        condition = self.conditioner.condition(data_batch)
        # A fixed seed for every validation batch, without moving the
        # caller's random streams.
        with isolated_rng():
            torch.manual_seed(self.seed)
            rollout = self._rollout(
                anchor, condition, real.shape[2], len(self.rollout_endpoints)
            )
            return self._fake_score_loss(
                anchor, real, rollout.clean, condition, warmup=False
            )

    @torch.inference_mode()
    def sample(self, data_batch: dict[str, Any], **options: Any) -> dict[str, Tensor]:
        options = self.sampling_options(options)
        latent = data_batch["latent"].float()
        anchor, target = latent[:, :, :1], latent[:, :, 1:]
        self._validate_rollout(target.shape[2], options["num_steps"])
        steps = int(options["num_steps"])
        seeds = sampling_seeds(data_batch, int(options["seed"]))
        noise = torch.stack(
            [invariant_randn(target, seeds, stream=step) for step in range(steps)]
        )
        rollout = self._rollout(
            anchor,
            self.conditioner.condition(data_batch),
            target.shape[2],
            steps,
            noise=noise,
        )
        return {
            **self.conditioner.visuals(data_batch),
            "ground_truth": self.conditioner.decode(latent),
            "sample": self.conditioner.decode(
                torch.cat((anchor, rollout.clean), dim=2)
            ),
        }

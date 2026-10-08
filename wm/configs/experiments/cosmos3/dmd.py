# SPDX-License-Identifier: Apache-2.0

from wm.configs.experiments.cosmos3 import (
    conditioner,
    cosmos3_base,
    gen_initialization,
    network,
    optimizer,
    parallel,
)
from wm.infra.config import LazyCall, LazyDict
from wm.models.dmd import DMDModel

SAMPLING = {"num_steps": 4, "seed": 2026}


def build_config() -> LazyDict:
    config = cosmos3_base(name="dmd", pixel_frames=81, sampling=SAMPLING)
    config.model = LazyCall(DMDModel)(
        student=network(),
        real_score=network(),
        fake_score=network(),
        conditioner=conditioner(),
        block_frames=4,
        score_shift=5.0,
        real_score_guidance=6.0,
        fake_score_updates_per_student=5,
        dmd_loss_scale=0.5,
        # Eager student: SGF Pass 2 then reproduces the rollout bitwise.
        student_parallel=parallel(activation_checkpointing="full", compile=False),
        score_parallel=parallel(activation_checkpointing="full"),
        conditioner_parallel=parallel(activation_checkpointing="none", compile=False),
        sampling=SAMPLING,
    )
    config.optimizer = {
        "student": optimizer(2e-6, betas=(0.0, 0.999)),
        "fake_score": optimizer(4e-7, betas=(0.0, 0.999)),
    }
    inits = config.train.model_initializations
    inits.student = gen_initialization("student")
    inits.real_score = gen_initialization("real_score", frozen=True)
    inits.fake_score = gen_initialization("fake_score")
    # ``checkpoint_interval``/``validation_interval`` count student updates.
    config.train.checkpoint_interval = 10
    config.train.validation_interval = 10
    config.trainer.grad_accum_steps = 2
    return config

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
from wm.models.diffusion import DiffusionModel
from wm.models.flow import RectifiedFlowSchedule


def build_config() -> LazyDict:
    config = cosmos3_base(
        name="bidirectional",
        pixel_frames=81,
        sampling={"num_steps": 35, "guidance": 6.0, "seed": 2026},
    )
    config.model = LazyCall(DiffusionModel)(
        net=network(),
        conditioner=conditioner(),
        schedule=LazyCall(RectifiedFlowSchedule)(shift=5.0, train_distribution="waver"),
        text_dropout=0.1,
        parallel=parallel(),
        conditioner_parallel=parallel(activation_checkpointing="none", compile=False),
        sampling={"num_steps": 35, "guidance": 6.0, "seed": 2026},
    )
    config.optimizer = optimizer(3e-5)
    config.train.model_initializations.net = gen_initialization("net")
    config.trainer.grad_accum_steps = 4
    return config

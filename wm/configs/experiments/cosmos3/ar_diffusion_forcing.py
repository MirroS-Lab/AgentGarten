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
from wm.models.autoregressive import AutoregressiveModel
from wm.models.flow import RectifiedFlowSchedule

SAMPLING = {
    "num_steps": 16,
    "guidance": 6.0,
    "seed": 2026,
    "sink_blocks": 0,
    "recent_blocks": None,
}


def build_config() -> LazyDict:
    config = cosmos3_base(
        name="ar_diffusion_forcing", pixel_frames=81, sampling=SAMPLING
    )
    config.model = LazyCall(AutoregressiveModel)(
        net=network(),
        conditioner=conditioner(),
        mode="diffusion_forcing",
        block_frames=4,
        schedule=LazyCall(RectifiedFlowSchedule)(shift=5.0, train_distribution="waver"),
        text_dropout=0.1,
        corrupt_clean_history=False,
        parallel=parallel(),
        conditioner_parallel=parallel(activation_checkpointing="none", compile=False),
        sampling=SAMPLING,
    )
    config.optimizer = optimizer(3e-5)
    config.train.model_initializations.net = gen_initialization("net")
    config.trainer.grad_accum_steps = 2
    return config

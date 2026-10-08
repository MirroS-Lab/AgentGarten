# SPDX-License-Identifier: Apache-2.0

from wm.configs.experiments.cosmos3 import LORA_TRAINABLE, lora, network, optimizer
from wm.configs.experiments.cosmos3.dmd_gan import build_config as dmd_gan_config
from wm.infra.config import LazyDict


def build_config() -> LazyDict:
    config = dmd_gan_config()
    config.job.output_dir = "${paths.output_root}/cosmos3/dmd_gan_lora"
    config.model.student = network(lora=lora(64))
    config.model.student_trainable_patterns = list(LORA_TRAINABLE)
    config.optimizer.student = optimizer(2e-5, betas=(0.0, 0.999))
    return config

# SPDX-License-Identifier: Apache-2.0

from wm.configs.experiments.cosmos3 import parallel
from wm.configs.experiments.cosmos3.dmd_gan import build_config as dmd_gan_config
from wm.infra.config import LazyDict


def build_config() -> LazyDict:
    config = dmd_gan_config()
    config.job.output_dir = "${paths.output_root}/cosmos3/dmd_gan_exact"
    config.model.gan.regularization = "exact"
    # 0.075 = 30 * 0.05^2 matches the local scale of the finite-difference recipe.
    config.model.gan.exact_regularization_weight = 0.075
    config.model.real_score_parallel = parallel(
        activation_checkpointing="full", compile=False
    )
    return config

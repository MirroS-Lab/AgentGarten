# SPDX-License-Identifier: Apache-2.0

from wm.configs.experiments.cosmos3 import optimizer, parallel
from wm.configs.experiments.cosmos3.dmd import build_config as dmd_config
from wm.infra.config import LazyCall, LazyDict
from wm.models.dmd import GANConfig
from wm.networks.discriminator import DiscriminatorHead


def build_config() -> LazyDict:
    config = dmd_config()
    config.job.output_dir = "${paths.output_root}/cosmos3/dmd_gan"
    config.model.discriminator = LazyCall(DiscriminatorHead)(
        hidden_size=4096,
        feature_layers=(11, 23, 35),
        mlp_ratio=4.0,
        num_heads=32,
        initial_logit_sign=-1.0,
        init_device="meta",
    )
    config.model.gan = LazyCall(GANConfig)(
        generator_weight=0.01,
        discriminator_weight=0.01,
        relativistic=True,
        regularization="finite_difference",
        regularization_weight=30.0,
        regularization_sigma=0.05,
        warmup_updates=10,
    )
    config.model.discriminator_parallel = parallel(
        parameter_dtype="float32", activation_checkpointing="full", compile=False
    )
    config.optimizer.discriminator = optimizer(2e-7, betas=(0.0, 0.999))
    return config

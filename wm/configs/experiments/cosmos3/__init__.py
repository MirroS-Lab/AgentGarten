# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from wm.codecs import Wan22VAECodec
from wm.configs.base import make_base_config
from wm.data.loader import StatefulDataLoader
from wm.data.video import ManifestVideoDataset
from wm.infra.callbacks import ImageVideoArtifactWriter, PeriodicSampleCallback
from wm.infra.config import LazyCall, LazyDict
from wm.infra.optim import FusedAdam, LinearWarmup
from wm.infra.parallel import ParallelConfig
from wm.networks.cosmos3 import (
    GEN_KEYS,
    LORA_ATTENTION_MLP_MODULES,
    LORA_ATTENTION_MODULES,
    TEXT_ENCODER_KEYS,
    Cosmos3Conditioner,
    Cosmos3Config,
    build_network,
)
from wm.networks.cosmos3.geometry import GeometryAugmentationConfig
from wm.networks.lora import LORA_PARAMETER_PATTERN

__all__ = [
    "GEOMETRY_AUGMENTATION",
    "LORA_TRAINABLE",
    "RELEASED_TRANSFORMER",
    "conditioner",
    "cosmos3_base",
    "gen_initialization",
    "loader",
    "lora",
    "network",
    "optimizer",
    "parallel",
    "text_initialization",
]

# Robustness policy for imperfect depth/normal controls (training only).
GEOMETRY_AUGMENTATION = {
    "modality_dropout_probability": 0.10,
    "modality_selection": "both",
    "spatial_probability": 0.50,
    "spatial_max_scale": 1.15,
    "spatial_max_shift_fraction": 0.06,
    "spatial_max_shear": 0.03,
    "spatial_max_perspective": 0.02,
    "spatial_anneal_power": 1.5,
    "texture_suppression_probability": 0.50,
    "texture_downsample_min": 1.5,
    "texture_downsample_max": 3.0,
    "texture_gaussian_sigma_scale": 0.5,
    "texture_skip_tsdf": True,
}

RELEASED_TRANSFORMER = "${paths.cosmos3_nano}/transformer"


def parallel(**overrides: object) -> LazyDict:
    return LazyCall(ParallelConfig)(
        **{
            "parameter_dtype": "bfloat16",
            "fully_shard": True,
            "context_parallel_size": "${model_parallel.context_parallel_size}",
            "fsdp_mesh_shape": "${model_parallel.fsdp_mesh_shape}",
            "activation_checkpointing": "selective",
            "compile": True,
            **overrides,
        }
    )


def network(lora: dict | None = None) -> LazyDict:
    return LazyCall(build_network)(config="${cosmos3}", init_device="meta", lora=lora)


def lora(rank: int = 64, *, mlp: bool = True) -> dict:
    modules = LORA_ATTENTION_MLP_MODULES if mlp else LORA_ATTENTION_MODULES
    return {
        "rank": rank,
        "alpha": float(rank),
        "dropout": 0.0,
        "module_patterns": list(modules),
    }


# With LoRA only the adapters and the (small) control embeddings train.
LORA_TRAINABLE = (
    LORA_PARAMETER_PATTERN,
    r"geometry_modality_embed",
    r"first_frame_reference_embed",
)


def conditioner() -> LazyDict:
    return LazyCall(Cosmos3Conditioner)(
        config="${cosmos3}",
        codec=LazyCall(Wan22VAECodec)(
            pretrained_path="${paths.wan22_vae}", dtype="bfloat16", decoder=None
        ),
        geometry_augmentation=LazyCall(GeometryAugmentationConfig)(
            **GEOMETRY_AUGMENTATION
        ),
        geometry_noise_sigma=0.4,
        tokenizer_path="${paths.cosmos3_nano}/text_tokenizer",
        negative_prompt_path="${paths.cosmos3_nano}/assets/negative_prompt.json",
        default_fps=16.0,
    )


def gen_initialization(
    *destinations: str, path: str = RELEASED_TRANSFORMER, frozen: bool = False
) -> dict:
    return {
        "path": path,
        "model_key_prefixes": list(destinations),
        "source_model_key_prefix": "",
        "source_model_key_patterns": list(GEN_KEYS),
        "resume_model_key_prefixes": list(destinations) if frozen else None,
    }


def text_initialization() -> dict:
    return {
        "path": RELEASED_TRANSFORMER,
        "model_key_prefix": "conditioner.text_encoder",
        "source_model_key_prefix": "",
        "source_model_key_patterns": list(TEXT_ENCODER_KEYS),
        "resume_model_key_prefixes": ["conditioner.text_encoder"],
    }


def optimizer(lr: float, betas: tuple[float, float] = (0.9, 0.99)) -> LazyDict:
    return LazyCall(FusedAdam)(
        lr=lr,
        betas=betas,
        eps=1e-8,
        weight_decay=0.0,
        adam_w_mode=True,
        use_te_capturable_kernel=True,
        master_weights=True,
    )


def loader(manifest: str, *, frames: int, shuffle: bool, workers: int) -> LazyDict:
    return LazyCall(StatefulDataLoader)(
        dataset=LazyCall(ManifestVideoDataset)(
            manifest=manifest,
            num_frames=frames,
            height=480,
            width=832,
            fps=16.0,
        ),
        batch_size=1,
        shuffle=shuffle,
        num_workers=workers,
    )


def cosmos3_base(*, name: str, pixel_frames: int, sampling: dict) -> LazyDict:
    config = make_base_config()
    config.paths = {
        "cosmos3_nano": "${oc.env:WM_COSMOS3_NANO,/path/to/Cosmos3-Nano}",
        "wan22_vae": "${oc.env:WM_WAN22_VAE,/path/to/Wan2.2_VAE.pth}",
        "train_manifest": "${oc.env:WM_TRAIN_MANIFEST,/path/to/train.jsonl}",
        "val_manifest": "${oc.env:WM_VAL_MANIFEST,/path/to/val.jsonl}",
        "output_root": "${oc.env:WM_OUTPUT_ROOT,outputs}",
    }
    config.job.output_dir = f"${{paths.output_root}}/cosmos3/{name}"
    config.cosmos3 = LazyCall(Cosmos3Config)()
    config.model_parallel = {"fsdp_mesh_shape": None, "context_parallel_size": 1}
    config.dataloader_train = loader(
        "${paths.train_manifest}", frames=pixel_frames, shuffle=True, workers=6
    )
    config.dataloader_val = loader(
        "${paths.val_manifest}", frames=pixel_frames, shuffle=False, workers=0
    )
    config.scheduler = LazyCall(torch.optim.lr_scheduler.LambdaLR)(
        lr_lambda=LazyCall(LinearWarmup)(warmup_steps=100, start_factor=0.1)
    )
    config.train.model_initializations = {"text": text_initialization()}
    config.trainer.callbacks.samples = LazyCall(PeriodicSampleCallback)(
        every_n="${train.validation_interval}",
        output_dir="${job.output_dir}/samples",
        writer=LazyCall(ImageVideoArtifactWriter)(
            fps=16,
            async_write=True,
            grid_layout=("depth", "normal", "ground_truth", "sample"),
            grid_reference_key="sample",
            grid_name="comparison",
            save_individual_artifacts=False,
        ),
        source="validation",
        sampling_options=sampling,
        seed_batch_key=None,
    )
    return config

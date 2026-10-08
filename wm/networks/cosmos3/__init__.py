# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from wm.networks.lora import inject_lora

from .conditioner import COSMOS3_EMPTY_PROMPT_IDS, Cosmos3Conditioner
from .layers import Cosmos3Config
from .network import Cosmos3Condition, Cosmos3Network, Cosmos3TextEncoder

__all__ = [
    "COSMOS3_EMPTY_PROMPT_IDS",
    "GEN_KEYS",
    "LORA_ATTENTION_MLP_MODULES",
    "LORA_ATTENTION_MODULES",
    "TEXT_ENCODER_KEYS",
    "Cosmos3Condition",
    "Cosmos3Conditioner",
    "Cosmos3Config",
    "Cosmos3Network",
    "Cosmos3TextEncoder",
    "build_network",
]

TEXT_ENCODER_KEYS = (
    r"embed_tokens\.weight",
    r"layers\.\d+\.self_attn\.(?:to_q|to_k|to_v|to_out|norm_q|norm_k)\.weight",
    r"layers\.\d+\.(?:input_layernorm|post_attention_layernorm)\.weight",
    r"layers\.\d+\.mlp\.(?:gate|up|down)_proj\.weight",
)

GEN_KEYS = (
    r"layers\.\d+\.self_attn\.(?:add_[qkv]_proj|to_add_out|norm_added_[qk])\.weight",
    r"layers\.\d+\.(?:input_layernorm_moe_gen|post_attention_layernorm_moe_gen)\.weight",
    r"layers\.\d+\.mlp_moe_gen\.(?:gate|up|down)_proj\.weight",
    r"norm_moe_gen\.weight",
    r"proj_(?:in|out)\.(?:weight|bias)",
    r"time_embedder\.linear_[12]\.(?:weight|bias)",
    # Control-v1 additions: present in trained checkpoints, absent (zero) in
    # the released one.
    r"geometry_modality_embed",
    r"first_frame_reference_embed",
)


# LoRA targets of the GEN tower: attention projections, optionally the MLP.
LORA_ATTENTION_MODULES = (r"layers\.\d+\.self_attn\.(?:add_[qkv]_proj|to_add_out)",)
LORA_ATTENTION_MLP_MODULES = (
    *LORA_ATTENTION_MODULES,
    r"layers\.\d+\.mlp_moe_gen\.(?:gate|up|down)_proj",
)


def build_network(
    config: Cosmos3Config,
    *,
    init_device: str = "meta",
    lora: Mapping[str, Any] | None = None,
) -> Cosmos3Network:
    with torch.device(init_device):
        network = Cosmos3Network(config)
        inject_lora(network, lora)
    return network

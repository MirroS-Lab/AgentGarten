# SPDX-License-Identifier: Apache-2.0

"""Sharding, compilation and activation checkpointing of a network.

Order: CP -> activation checkpointing -> compile -> FSDP2 layers -> root.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import prod
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    DefaultDeviceType,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from wm.infra.distributed import get_local_world_size

__all__ = [
    "ATTENTION_SAVE_OPS",
    "ActivationCheckpointing",
    "ParallelConfig",
    "ParallelNetwork",
    "init_meshes",
    "parallelize_network",
]

# Save attention outputs, recompute everything else (projections, MLPs).
ATTENTION_SAVE_OPS = (r"scaled_dot_product", r"flash_attn", r"fmha")


class ActivationCheckpointing(StrEnum):
    NONE = "none"
    SELECTIVE = "selective"
    FULL = "full"


@dataclass(frozen=True)
class ParallelConfig:
    parameter_dtype: str = "bfloat16"
    fully_shard: bool = True
    # ``None``: 1D FSDP on one node, (nodes, local) HSDP on several nodes.
    fsdp_mesh_shape: tuple[int, ...] | None = None
    context_parallel_size: int = 1
    activation_checkpointing: ActivationCheckpointing = (
        ActivationCheckpointing.SELECTIVE
    )
    selective_save_ops: tuple[str, ...] = ATTENTION_SAVE_OPS
    compile: bool = True
    compile_dynamic: bool = False
    compile_options: Mapping[str, Any] = field(default_factory=dict)
    reduce_dtype: str = "float32"
    # Keep parameters gathered between forward and backward (e.g. an AR student
    # that replays after a long rollout).
    reshard_after_forward: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "activation_checkpointing",
            ActivationCheckpointing(self.activation_checkpointing),
        )
        for name in ("parameter_dtype", "reduce_dtype"):
            if not isinstance(getattr(torch, getattr(self, name), None), torch.dtype):
                raise ValueError(
                    f"{name} must name a torch dtype, got {getattr(self, name)!r}"
                )
        if self.context_parallel_size < 1:
            raise ValueError("context_parallel_size must be positive")
        if self.fsdp_mesh_shape is not None:
            object.__setattr__(
                self, "fsdp_mesh_shape", tuple(int(v) for v in self.fsdp_mesh_shape)
            )
        object.__setattr__(self, "selective_save_ops", tuple(self.selective_save_ops))
        object.__setattr__(self, "compile_options", dict(self.compile_options))


class ParallelNetwork(nn.Module):
    """A layered network that `parallelize_network` shards and compiles.

    Each entry of ``layers`` becomes one FSDP unit and one activation
    checkpoint. A layer may name the methods to compile individually in a
    ``compile_methods`` tuple; otherwise the whole layer is compiled.
    """

    layers: nn.ModuleList
    # Root methods besides ``forward`` that run the layers (FSDP2 root hooks).
    fsdp_entry_points: ClassVar[tuple[str, ...]] = ()

    def fsdp_units(self) -> tuple[nn.Module, ...]:
        # Modules outside ``layers`` that are sharded as units of their own.
        return ()

    def enable_context_parallel(self, mesh: DeviceMesh) -> None:
        # Networks that do not split tokens across ranks ignore the mesh.
        del mesh

    def initialize_after_materialization(self) -> None:
        # Initialize parameters that no checkpoint provides, after ``to_empty``.
        return


_MESHES: dict[tuple, tuple[DeviceMesh | None, DeviceMesh | None]] = {}


def init_meshes(
    config: ParallelConfig, device_type: str
) -> tuple[DeviceMesh | None, DeviceMesh | None]:
    if not (dist.is_available() and dist.is_initialized()):
        return None, None
    world = dist.get_world_size()
    local = get_local_world_size()
    shape = config.fsdp_mesh_shape
    if (
        shape is None
        and config.fully_shard
        and 0 < local < world
        and world % local == 0
    ):
        shape = (world // local, local)
    key = (device_type, world, config.fully_shard, config.context_parallel_size, shape)
    if key in _MESHES:
        return _MESHES[key]
    fsdp_mesh = None
    if config.fully_shard:
        shape = shape or (world,)
        if prod(shape) != world:
            raise ValueError(f"fsdp_mesh_shape {shape} does not cover {world} ranks")
        names = ("shard",) if len(shape) == 1 else ("replicate", "shard")
        fsdp_mesh = init_device_mesh(device_type, shape, mesh_dim_names=names)
    cp_mesh = None
    cp = config.context_parallel_size
    if cp > 1:
        if world % cp:
            raise ValueError(
                f"context_parallel_size {cp} does not divide {world} ranks"
            )
        cp_mesh = init_device_mesh(
            device_type, (world // cp, cp), mesh_dim_names=("rest", "cp")
        )["cp"]
    _MESHES[key] = (fsdp_mesh, cp_mesh)
    return fsdp_mesh, cp_mesh


def _selective_context(save_ops: Sequence[str]) -> Callable[[], Any]:
    patterns = tuple(re.compile(pattern) for pattern in save_ops)

    def policy(
        _context: Any, function: Any, *args: Any, **kwargs: Any
    ) -> CheckpointPolicy:
        name = getattr(function, "__name__", str(function))
        if any(pattern.search(name) for pattern in patterns):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return lambda: create_selective_checkpoint_contexts(policy)


def _checkpoint_when_training(
    function: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    options = kwargs.pop("_options")
    if not torch.is_grad_enabled():
        return function(*args, **kwargs)
    return checkpoint(function, *args, use_reentrant=False, **options, **kwargs)


def _inner(layer: nn.Module) -> nn.Module:
    return getattr(layer, "_checkpoint_wrapped_module", layer)


def parallelize_network(
    network: ParallelNetwork, config: ParallelConfig, device: torch.device | str
) -> DeviceMesh | None:
    device = torch.device(device)
    dtype = getattr(torch, config.parameter_dtype)
    on_meta = any(parameter.is_meta for parameter in network.parameters())
    if on_meta:
        network.to(dtype=dtype)
    else:
        network.to(device=device, dtype=dtype)
    fsdp_mesh, cp_mesh = init_meshes(config, device.type)

    if cp_mesh is not None:
        network.enable_context_parallel(cp_mesh)

    if config.activation_checkpointing is not ActivationCheckpointing.NONE:
        DefaultDeviceType.set_device_type(device.type)
        options: dict[str, Any] = {"preserve_rng_state": True}
        if config.activation_checkpointing is ActivationCheckpointing.SELECTIVE:
            options["context_fn"] = _selective_context(config.selective_save_ops)
        for index, layer in enumerate(network.layers):
            network.layers[index] = checkpoint_wrapper(
                layer, checkpoint_fn=_checkpoint_when_training, _options=options
            )

    if config.compile:
        compile_options = {"emulate_precision_casts": True, **config.compile_options}
        kwargs = {
            "dynamic": config.compile_dynamic,
            "fullgraph": True,
            "options": compile_options,
        }
        for layer in network.layers:
            inner = _inner(layer)
            methods = getattr(inner, "compile_methods", ())
            if methods:
                for name in methods:
                    setattr(inner, name, torch.compile(getattr(inner, name), **kwargs))
            else:
                layer.compile(**kwargs)

    if fsdp_mesh is not None:
        policy = MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=getattr(torch, config.reduce_dtype),
            cast_forward_inputs=False,
        )
        shard = dict(
            mesh=fsdp_mesh,
            mp_policy=policy,
            reshard_after_forward=config.reshard_after_forward,
        )
        for layer in network.layers:
            fully_shard(layer, **shard)
        for unit in network.fsdp_units():
            fully_shard(unit, **shard)
        fully_shard(network, **shard)
        for name in network.fsdp_entry_points:
            register_fsdp_forward_method(network, name)

    if on_meta:
        network.to_empty(device=device)
        network.initialize_after_materialization()
    return cp_mesh

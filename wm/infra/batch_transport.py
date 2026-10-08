# SPDX-License-Identifier: Apache-2.0

"""Fetch one batch per context-parallel group and share it with the group.

Without context parallelism every rank reads its own batch. With it, one owner
rank per group reads and prepares the batch, then broadcasts the tensors and
its RNG state so the whole group computes on identical data and noise. A
failure on any rank is raised on every rank instead of leaving peers blocked in
a collective.
"""

from __future__ import annotations

import pickle
import random
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from wm.infra.model import ContextParallelContext

__all__ = ["fetch_batch", "prepare_batch"]

Batch = dict[str, Any]

_ProcessRNGState = tuple[
    object,
    tuple[Any, ...],
    torch.Tensor,
    torch.Tensor | None,
]

_FETCH_OK = 0
_FETCH_STOP = 1
_FETCH_ERROR = 2
_TREE_TENSOR = "tensor"
_TREE_MAPPING = "mapping"
_TREE_LIST = "list"
_TREE_TUPLE = "tuple"
_TREE_NAMEDTUPLE = "namedtuple"
_TREE_TYPED_MAPPING = "typed_mapping"
_TREE_LEAF = "leaf"

_TensorSpec = tuple[tuple[int, ...], torch.dtype]


def _compact_fetch_error(error: BaseException) -> str:
    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message[:400]}"


def _collective_status_device() -> torch.device:
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _encode_tensor_tree(
    value: Any,
    *,
    device: torch.device,
) -> tuple[Any, list[torch.Tensor], list[_TensorSpec]]:
    tensors: list[torch.Tensor] = []
    specs: list[_TensorSpec] = []
    tensor_indices: dict[int, int] = {}

    def encode(item: Any) -> Any:
        if torch.is_tensor(item):
            identity = id(item)
            existing = tensor_indices.get(identity)
            if existing is not None:
                return (_TREE_TENSOR, existing)
            tensor = item.detach().to(device=device).contiguous()
            index = len(tensors)
            tensor_indices[identity] = index
            tensors.append(tensor)
            specs.append((tuple(int(size) for size in tensor.shape), tensor.dtype))
            return (_TREE_TENSOR, index)
        if isinstance(item, Mapping):
            children = [(key, encode(child)) for key, child in item.items()]
            if isinstance(item, (defaultdict, OrderedDict)):
                factory = (
                    item.default_factory if isinstance(item, defaultdict) else None
                )
                return (_TREE_TYPED_MAPPING, (type(item), factory, children))
            return (
                _TREE_MAPPING,
                children,
            )
        if isinstance(item, list):
            return (_TREE_LIST, [encode(child) for child in item])
        if isinstance(item, tuple):
            if hasattr(item, "_fields"):
                return (
                    _TREE_NAMEDTUPLE,
                    (type(item), [encode(child) for child in item]),
                )
            return (_TREE_TUPLE, [encode(child) for child in item])
        return (_TREE_LEAF, item)

    schema = encode(value)
    # Validate before the owner enters broadcast_objects: a local class or
    # unpicklable metadata must use the existing collective preparation-error
    # path, rather than leaving peers blocked in an object broadcast.
    pickle.dumps(schema)
    return schema, tensors, specs


def _decode_tensor_tree(schema: Any, tensors: list[torch.Tensor]) -> Any:
    tag, payload = schema
    if tag == _TREE_TENSOR:
        return tensors[int(payload)]
    if tag == _TREE_MAPPING:
        return {key: _decode_tensor_tree(child, tensors) for key, child in payload}
    if tag == _TREE_TYPED_MAPPING:
        container_type, factory, children = payload
        values = {key: _decode_tensor_tree(child, tensors) for key, child in children}
        if issubclass(container_type, defaultdict):
            return container_type(factory, values)
        return container_type(values)
    if tag == _TREE_NAMEDTUPLE:
        container_type, children = payload
        return container_type(
            *(_decode_tensor_tree(child, tensors) for child in children)
        )
    if tag == _TREE_LIST:
        return [_decode_tensor_tree(child, tensors) for child in payload]
    if tag == _TREE_TUPLE:
        return tuple(_decode_tensor_tree(child, tensors) for child in payload)
    return payload


def _capture_process_rng_state() -> _ProcessRNGState:
    cuda_state = (
        torch.cuda.get_rng_state(torch.cuda.current_device()).cpu()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        cuda_state,
    )


def _restore_process_rng_state(state: _ProcessRNGState) -> None:
    python_state, numpy_state, cpu_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, torch.cuda.current_device())


def _synchronize_status(
    status: int, error_rank: int | None, error_message: str | None
) -> tuple[int, int | None, str | None]:
    if not (dist.is_available() and dist.is_initialized()):
        return status, error_rank, error_message
    world_size = dist.get_world_size()
    state = torch.tensor(
        [status, world_size - int(error_rank) if status == _FETCH_ERROR else 0],
        dtype=torch.int64,
        device=_collective_status_device(),
    )
    dist.all_reduce(state, op=dist.ReduceOp.MAX)
    status, source = state.tolist()
    if status == _FETCH_ERROR:
        error_rank = world_size - source
        message = [error_message if dist.get_rank() == error_rank else None]
        dist.broadcast_object_list(message, src=error_rank)
        error_message = message[0]
    return status, error_rank, error_message


def _raise_on_error(
    *,
    status: int,
    error_rank: int | None,
    error_message: str | None,
    local_error: BaseException | None,
    stage: str,
) -> None:
    if int(status) != _FETCH_ERROR:
        return
    details = (
        "" if error_message is None else f": rank {int(error_rank)}: {error_message}"
    )
    raise RuntimeError(
        f"Distributed dataloader {stage} failed{details}"
    ) from local_error


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def fetch_batch(
    context: ContextParallelContext,
    iterator: Iterator[Batch],
    *,
    sequence_index: int,
) -> tuple[Batch | None, bool, int | None]:
    """Read the next batch: (batch on the reading rank, exhausted, owner rank)."""
    if not context.enabled and not _distributed():
        try:
            return next(iterator), False, None
        except StopIteration:
            return None, True, None

    data_batch = None
    status = _FETCH_OK
    error_rank = None
    error_message = None
    local_error: BaseException | None = None
    owner_rank = int(sequence_index) % context.size if context.enabled else None
    if not context.enabled or context.rank == owner_rank:
        try:
            data_batch = next(iterator)
        except StopIteration:
            status = _FETCH_STOP
        except BaseException as error:
            local_error = error
            status = _FETCH_ERROR
            error_rank = dist.get_rank()
            error_message = _compact_fetch_error(error)

    if context.enabled:
        payload = [status, error_rank, error_message]
        context.broadcast_objects(payload, source_rank=owner_rank)
        status, error_rank, error_message = payload
    status, error_rank, error_message = _synchronize_status(
        status, error_rank, error_message
    )
    _raise_on_error(
        status=int(status),
        error_rank=error_rank,
        error_message=error_message,
        local_error=local_error,
        stage="fetch",
    )
    if int(status) == _FETCH_STOP:
        return None, True, owner_rank
    return data_batch, False, owner_rank


def prepare_batch(
    context: ContextParallelContext,
    data_batch: Batch | None,
    *,
    owner_rank: int | None,
    prepare: Callable[[Batch], Batch],
) -> Batch:
    """Run ``prepare`` where the batch was read and share the result."""
    if not context.enabled and not _distributed():
        return prepare(data_batch)

    status = _FETCH_OK
    error_rank = None
    error_message = None
    local_error: BaseException | None = None
    prepared: Batch | None = None
    rng_state = None
    schema = None
    tensor_specs = None
    tensors: list[torch.Tensor] = []
    transport_device = _collective_status_device()

    if not context.enabled or context.rank == owner_rank:
        try:
            prepared = prepare(data_batch)
            if context.enabled:
                schema, tensors, tensor_specs = _encode_tensor_tree(
                    prepared, device=transport_device
                )
                rng_state = _capture_process_rng_state()
        except BaseException as error:
            local_error = error
            status = _FETCH_ERROR
            error_rank = dist.get_rank()
            error_message = _compact_fetch_error(error)

    if context.enabled:
        payload = [status, error_rank, error_message, rng_state, schema, tensor_specs]
        context.broadcast_objects(payload, source_rank=int(owner_rank))
        status, error_rank, error_message, rng_state, schema, tensor_specs = payload
    status, error_rank, error_message = _synchronize_status(
        status, error_rank, error_message
    )
    _raise_on_error(
        status=int(status),
        error_rank=error_rank,
        error_message=error_message,
        local_error=local_error,
        stage="prepare",
    )
    if not context.enabled:
        return prepared

    if context.rank != owner_rank:
        tensors = [
            torch.empty(shape, dtype=dtype, device=transport_device)
            for shape, dtype in tensor_specs
        ]
    source_rank = context.global_rank(int(owner_rank))
    for tensor in tensors:
        if tensor.numel() > 0:
            dist.broadcast(tensor, src=source_rank, group=context.group)

    _restore_process_rng_state(rng_state)
    return _decode_tensor_tree(schema, tensors)

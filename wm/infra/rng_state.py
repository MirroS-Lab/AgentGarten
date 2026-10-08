# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import pickle
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

RNG_CHECKPOINT_FORMAT = "wm-trainer-rng-v1"
RNG_CHECKPOINT_SCHEMA_VERSION = 1


@contextmanager
def isolated_rng(*, cuda_devices: Sequence[int] | None = None) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = (
        list(cuda_devices)
        if cuda_devices is not None
        else list(range(torch.cuda.device_count()))
        if torch.cuda.is_initialized()
        else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _rank_key(rank: int) -> str:
    return f"rank_{rank:08d}"


def _bytes_tensor(payload: bytes) -> torch.Tensor:
    # Keep the serialized blob variable-length.  The payload is already a
    # checkpoint tensor, so padding it to an arbitrary fixed capacity only
    # wastes storage and turns harmless future state growth into a save-time
    # failure.
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()


def _tensor_bytes(value: torch.Tensor) -> bytes:
    if value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
        raise RuntimeError(
            "RNG serialized state must be a one-dimensional CPU uint8 tensor, "
            f"got device={value.device}, dtype={value.dtype}, shape={tuple(value.shape)}"
        )
    return value.contiguous().numpy().tobytes()


# Globals that a pickled Python or NumPy RNG state references (old and new
# NumPy module layouts). Anything else in a checkpoint blob is rejected rather
# than imported, so resuming from an untrusted checkpoint cannot execute code
# through this path.
_RNG_STATE_GLOBALS = frozenset(
    {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
    }
)


class _RNGStateUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _RNG_STATE_GLOBALS:
            raise pickle.UnpicklingError(f"RNG state may not reference {module}.{name}")
        return super().find_class(module, name)


def _decode_blob(value: torch.Tensor, length: int, *, name: str) -> Any:
    try:
        payload = io.BytesIO(_tensor_bytes(value)[:length])
        return _RNGStateUnpickler(payload).load()
    except (pickle.UnpicklingError, EOFError) as error:
        raise RuntimeError(
            f"RNG checkpoint {name} serialized state is invalid"
        ) from error


def _capture_local(rank: int) -> dict[str, Any]:
    python_payload = pickle.dumps(random.getstate(), protocol=pickle.HIGHEST_PROTOCOL)
    numpy_payload = pickle.dumps(
        np.random.get_state(), protocol=pickle.HIGHEST_PROTOCOL
    )
    cpu_state = torch.get_rng_state().cpu().clone()

    cuda_initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    if cuda_initialized:
        cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        current_cuda_device = torch.cuda.current_device()
    else:
        cuda_states = []
        current_cuda_device = -1
    return {
        "global_rank": int(rank),
        "python_blob": _bytes_tensor(python_payload),
        "python_length": len(python_payload),
        "numpy_blob": _bytes_tensor(numpy_payload),
        "numpy_length": len(numpy_payload),
        "torch_cpu": cpu_state,
        "cuda_initialized": cuda_initialized,
        "cuda_device_count": len(cuda_states),
        "current_cuda_device": current_cuda_device,
        "cuda_states": {
            f"device_{index:04d}": state for index, state in enumerate(cuda_states)
        },
    }


def capture_distributed_rng_state() -> dict[str, Any]:
    _rank, world_size = _rank_world()
    local = _capture_local(_rank)
    if world_size > 1:
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local)
        rank_states = [state for state in gathered if state is not None]
    else:
        rank_states = [local]
    return {
        "format": RNG_CHECKPOINT_FORMAT,
        "schema_version": RNG_CHECKPOINT_SCHEMA_VERSION,
        "world_size": world_size,
        "ranks": {_rank_key(index): state for index, state in enumerate(rank_states)},
    }


def _decode(
    state: Mapping[str, Any],
) -> tuple[object, tuple[Any, ...], torch.Tensor, list[torch.Tensor]]:
    if (
        state.get("format") != RNG_CHECKPOINT_FORMAT
        or state.get("schema_version") != RNG_CHECKPOINT_SCHEMA_VERSION
    ):
        raise RuntimeError("Unsupported WM trainer RNG checkpoint format")
    rank, world_size = _rank_world()
    if int(state.get("world_size", -1)) != world_size:
        raise RuntimeError(
            f"RNG checkpoint world-size mismatch: checkpoint={state.get('world_size')}, current={world_size}"
        )
    ranks = state.get("ranks")
    if not isinstance(ranks, Mapping) or set(ranks) != {
        _rank_key(i) for i in range(world_size)
    }:
        raise RuntimeError(
            "RNG checkpoint rank topology does not match the current process group"
        )
    item = ranks[_rank_key(rank)]
    if not isinstance(item, Mapping):
        raise RuntimeError("RNG checkpoint rank entry is not a mapping")
    if item.get("global_rank") != rank:
        raise RuntimeError("RNG checkpoint rank entry belongs to a different rank")
    python_length = int(item["python_length"])
    numpy_length = int(item["numpy_length"])
    python_blob = item["python_blob"]
    numpy_blob = item["numpy_blob"]
    cpu_state = item["torch_cpu"]
    if (
        not torch.is_tensor(python_blob)
        or not torch.is_tensor(numpy_blob)
        or not torch.is_tensor(cpu_state)
    ):
        raise RuntimeError("RNG checkpoint contains a non-tensor serialized state")
    if (
        not 0 < python_length <= python_blob.numel()
        or not 0 < numpy_length <= numpy_blob.numel()
    ):
        raise RuntimeError("RNG checkpoint serialized state length is invalid")
    python_state = _decode_blob(python_blob, python_length, name="Python")
    numpy_state = _decode_blob(numpy_blob, numpy_length, name="NumPy")
    # Validate every payload using private generators before touching any
    # process-global stream. Successful deserialization alone is insufficient.
    random.Random(0).setstate(python_state)
    np.random.RandomState(0).set_state(numpy_state)
    torch.Generator(device="cpu").set_state(cpu_state)
    cuda_states = item["cuda_states"]
    if not isinstance(cuda_states, Mapping):
        raise RuntimeError("RNG checkpoint CUDA state is not a mapping")
    saved_cuda = bool(item["cuda_initialized"])
    if saved_cuda and not torch.cuda.is_available():
        raise RuntimeError("RNG checkpoint contains CUDA state but CUDA is unavailable")
    if saved_cuda:
        if int(item["cuda_device_count"]) != torch.cuda.device_count():
            raise RuntimeError("RNG checkpoint CUDA device count differs")
        if set(cuda_states) != {
            f"device_{index:04d}" for index in range(torch.cuda.device_count())
        }:
            raise RuntimeError("RNG checkpoint CUDA device entries differ")
        ordered = [
            cuda_states[f"device_{i:04d}"] for i in range(torch.cuda.device_count())
        ]
        for index, cuda_state in enumerate(ordered):
            torch.Generator(device=torch.device("cuda", index)).set_state(cuda_state)
    else:
        # A CPU-only checkpoint may be inspected by a process that initialized
        # CUDA for unrelated reasons before loading. There is no CUDA stream
        # to restore in that checkpoint, so leave the current CUDA generators
        # untouched.
        ordered = []
    return python_state, numpy_state, cpu_state, ordered


def validate_distributed_rng_state(state: Mapping[str, Any]) -> None:
    _decode(state)


def restore_distributed_rng_state(state: Mapping[str, Any]) -> None:
    python_state, numpy_state, cpu_state, cuda_states = _decode(state)
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    torch.set_rng_state(cpu_state)
    np.random.set_state(numpy_state)
    random.setstate(python_state)


__all__ = [
    "capture_distributed_rng_state",
    "restore_distributed_rng_state",
    "validate_distributed_rng_state",
]

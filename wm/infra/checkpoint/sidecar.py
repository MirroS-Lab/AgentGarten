# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from wm.infra import io as easy_io
from wm.infra.checkpoint.common import warn_rank0
from wm.infra.distributed import is_rank0

_SIDECAR_VERSION = 1


@dataclass(frozen=True, slots=True)
class DataloaderSnapshot:
    state: Any
    metadata: dict[str, Any]


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _all_gather_object(value: Any) -> list[Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _compact_error(error: BaseException | None) -> str | None:
    if error is None:
        return None
    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message[:240]}"


def _raise_collective_errors(stage: str, local_error: BaseException | None) -> None:
    failures = [
        f"rank {rank}: {message}"
        for rank, message in enumerate(_all_gather_object(_compact_error(local_error)))
        if message is not None
    ]
    if failures:
        summary = "; ".join(failures[:4])
        if len(failures) > 4:
            summary += f"; ... ({len(failures)} ranks failed)"
        raise RuntimeError(
            f"Dataloader checkpoint {stage} failed: {summary}"
        ) from local_error


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _runtime(runtime_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return _json_value(dict(runtime_metadata or {}))


def _worker_signature(dataloader: Any) -> dict[str, Any]:
    signature = {
        "num_workers": getattr(dataloader, "num_workers", None),
        "persistent_workers": getattr(dataloader, "persistent_workers", None),
        "prefetch_factor": getattr(dataloader, "prefetch_factor", None),
        "in_order": getattr(dataloader, "in_order", None),
    }
    resume_signature = getattr(dataloader, "resume_signature", None)
    if callable(resume_signature):
        resume_signature = resume_signature()
    if resume_signature is not None:
        signature["loader"] = _json_value(resume_signature)
    return signature


def _supports_state(dataloader: Any | None) -> bool:
    return (
        dataloader is not None
        and callable(getattr(dataloader, "state_dict", None))
        and callable(getattr(dataloader, "load_state_dict", None))
    )


def _gather_loader_capabilities(dataloader: Any | None) -> list[dict[str, Any]]:
    capability = None
    local_error = None
    try:
        capability = {
            "supported": _supports_state(dataloader),
            "worker": _json_value(_worker_signature(dataloader))
            if dataloader is not None
            else None,
        }
    except BaseException as error:
        local_error = error
    gathered = _all_gather_object((capability, _compact_error(local_error)))
    failures = [
        f"rank {rank}: {message}"
        for rank, (_, message) in enumerate(gathered)
        if message is not None
    ]
    if failures:
        raise RuntimeError(
            "Dataloader checkpoint metadata capture failed: " + "; ".join(failures[:4])
        ) from local_error
    return [item for item, _ in gathered]


def capture_dataloader_snapshot(
    dataloader: Any | None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> DataloaderSnapshot | None:
    capabilities = _gather_loader_capabilities(dataloader)
    unsupported = [
        rank for rank, item in enumerate(capabilities) if not item["supported"]
    ]
    if unsupported:
        warn_rank0(
            "Training dataloader does not provide state_dict/load_state_dict on "
            f"{len(unsupported)} rank(s); skipping exact dataloader checkpointing."
        )
        return None

    workers = [item["worker"] for item in capabilities]
    if any(
        worker["num_workers"] not in (None, 0) and worker["in_order"] is False
        for worker in workers
    ):
        warn_rank0(
            "Training dataloader uses num_workers > 0 with in_order=False; "
            "skipping exact dataloader checkpointing."
        )
        return None

    local_error: BaseException | None = None
    state: Any = None
    try:
        state = dataloader.state_dict()
    except BaseException as error:
        local_error = error
    _raise_collective_errors("state capture", local_error)

    runtime = _runtime(runtime_metadata)
    return DataloaderSnapshot(
        state=state,
        metadata={
            "version": _SIDECAR_VERSION,
            "world_size": _world_size(),
            "mesh": runtime.get("mesh"),
            "workers": workers,
        },
    )


def save_dataloader_snapshot(
    checkpoint_path: str | Path,
    snapshot: DataloaderSnapshot,
) -> None:
    sidecar_path = Path(checkpoint_path) / "dataloader"
    rank_path = sidecar_path / f"rank_{_rank():05d}.pt"
    local_error: BaseException | None = None
    try:
        payload = BytesIO()
        torch.save(snapshot.state, payload)
        easy_io.atomic_put(payload, rank_path)
    except BaseException as error:
        local_error = error
    _raise_collective_errors("rank-state write", local_error)

    metadata_error: BaseException | None = None
    if is_rank0():
        try:
            payload = json.dumps(snapshot.metadata, indent=2, sort_keys=True) + "\n"
            easy_io.atomic_put_text(payload, sidecar_path / "metadata.json")
        except BaseException as error:
            metadata_error = error
    _raise_collective_errors("metadata write", metadata_error)


def save_checkpoint_metadata(
    checkpoint_path: str | Path,
    *,
    iteration: int,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> None:
    local_error: BaseException | None = None
    if is_rank0():
        try:
            metadata = {
                "version": _SIDECAR_VERSION,
                "iteration": int(iteration),
                "world_size": _world_size(),
                "runtime": _runtime(runtime_metadata),
            }
            payload = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            easy_io.atomic_put_text(payload, Path(checkpoint_path) / "metadata.json")
        except BaseException as error:
            local_error = error
    _raise_collective_errors("runtime metadata write", local_error)


def warn_runtime_metadata_changes(
    checkpoint_path: str | Path,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> None:
    if not is_rank0():
        return
    metadata_path = Path(checkpoint_path) / "metadata.json"
    try:
        if not easy_io.exists(metadata_path):
            return
        saved = json.loads(easy_io.get_text(metadata_path))
        if not isinstance(saved, Mapping):
            raise TypeError("metadata root is not a mapping")

        changes = []
        saved_world_size = saved.get("world_size")
        if saved_world_size != _world_size():
            changes.append(f"world_size {saved_world_size} -> {_world_size()}")
        current_runtime = _runtime(runtime_metadata)
        saved_runtime = saved.get("runtime", {})
        if not isinstance(saved_runtime, Mapping):
            raise TypeError("runtime metadata is not a mapping")
        for key in sorted(set(saved_runtime).intersection(current_runtime)):
            if saved_runtime[key] != current_runtime[key]:
                changes.append(f"{key} changed")
    except Exception as error:
        warn_rank0(f"Checkpoint runtime metadata is unreadable; ignoring it ({error}).")
        return

    if changes:
        warn_rank0(
            "Checkpoint runtime differs from the current run ("
            + ", ".join(changes)
            + "); restoring selected tensor state with the current configuration."
        )


def _load_decision(
    checkpoint_path: Path,
    workers: list[dict[str, Any]],
    runtime_metadata: Mapping[str, Any] | None,
) -> tuple[bool, str | None]:
    decision: list[Any] = [None]
    if is_rank0():
        metadata_path = checkpoint_path / "dataloader" / "metadata.json"
        try:
            if not easy_io.exists(metadata_path):
                decision[0] = (
                    False,
                    "checkpoint has no dataloader sidecar metadata",
                )
            else:
                metadata = json.loads(easy_io.get_text(metadata_path))
                reason = None
                if metadata.get("version") != _SIDECAR_VERSION:
                    reason = (
                        "dataloader sidecar version is unsupported "
                        f"({metadata.get('version')})"
                    )
                elif metadata.get("world_size") != _world_size():
                    reason = (
                        "world size differs "
                        f"({metadata.get('world_size')} -> {_world_size()})"
                    )
                elif metadata.get("mesh") != _runtime(runtime_metadata).get("mesh"):
                    reason = "parallel mesh differs"
                elif metadata.get("workers") != workers:
                    reason = "dataloader settings differ"
                decision[0] = (reason is None, reason)
        except Exception as error:
            decision[0] = (
                False,
                f"dataloader sidecar metadata is unreadable ({error})",
            )
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(decision, src=0)
    return decision[0]


def restore_dataloader_snapshot(
    checkpoint_path: str | Path,
    dataloader: Any | None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> bool:
    capabilities = _gather_loader_capabilities(dataloader)
    if not all(item["supported"] for item in capabilities):
        warn_rank0(
            "Training dataloader does not provide state_dict/load_state_dict on "
            "every rank; starting it from the beginning."
        )
        return False
    workers = [item["worker"] for item in capabilities]
    compatible, reason = _load_decision(
        Path(checkpoint_path), workers, runtime_metadata
    )
    if not compatible:
        warn_rank0(f"Skipping exact dataloader restore: {reason}.")
        return False

    rank_path = Path(checkpoint_path) / "dataloader" / f"rank_{_rank():05d}.pt"
    local_error: BaseException | None = None
    state: Any = None
    try:
        if not easy_io.exists(rank_path):
            raise FileNotFoundError(rank_path)
        state = torch.load(
            BytesIO(easy_io.get(rank_path)),
            map_location="cpu",
            weights_only=False,
        )
    except BaseException as error:
        local_error = error
    read_failures = _all_gather_object(_compact_error(local_error))
    if any(message is not None for message in read_failures):
        failed = [str(rank) for rank, message in enumerate(read_failures) if message]
        warn_rank0(
            "Skipping exact dataloader restore because rank-local state is "
            f"missing or unreadable on rank(s) {', '.join(failed)}."
        )
        return False

    load_error: BaseException | None = None
    try:
        dataloader.load_state_dict(state)
    except BaseException as error:
        load_error = error
    _raise_collective_errors("state restore", load_error)
    return True

# SPDX-License-Identifier: Apache-2.0

"""Load trained Cosmos towers without constructing a trainer or score networks."""

from __future__ import annotations

import json
import os
import re
import struct
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file
from torch import nn
from torch.distributed.checkpoint import FileSystemReader

from wm.infra.checkpoint import DistributedCheckpointer
from wm.networks.cosmos3 import (
    Cosmos3Config,
    Cosmos3Network,
    Cosmos3TextEncoder,
    build_network,
)

__all__ = [
    "Cosmos3Artifact",
    "Cosmos3ArtifactSpec",
    "WeightSource",
    "inspect_artifact",
    "load_artifact",
    "prompt_pixel_frames",
    "save_artifact",
]

MANIFEST_SCHEMA = "v2v.raw-dmd-safetensors"
MANIFEST_VERSION = 1
_TEXT_PREFIX = "conditioner.text_encoder"
_READ_THREADS = 16
_READ_CHUNK = 64 << 20
_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
}


@dataclass(frozen=True, slots=True)
class WeightSource:
    """A checkpoint overlay with explicit parameter ownership."""

    path: Path
    prefix: str
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Cosmos3ArtifactSpec:
    """Resolved inference inputs; configuration targets are never executed."""

    checkpoint: Path
    config: Cosmos3Config
    student: tuple[WeightSource, ...]
    text: tuple[WeightSource, ...]
    tokenizer_path: Path
    prompt_pixel_frames: int
    iteration: int
    # Keys are owned by wm.networks.lora.inject_lora.
    lora: Mapping[str, Any] | None = None


@dataclass
class Cosmos3Artifact:
    """Resident GEN/UND weights; stream state is owned by individual sessions."""

    student: Cosmos3Network
    context_encoder: Cosmos3TextEncoder
    spec: Cosmos3ArtifactSpec

    @property
    def tokenizer_path(self) -> Path:
        return self.spec.tokenizer_path

    @property
    def prompt_pixel_frames(self) -> int:
        return self.spec.prompt_pixel_frames

    @property
    def special_tokens(self) -> dict[str, int]:
        return self.spec.config.special_tokens

    @property
    def report(self) -> dict[str, object]:
        """Small JSON-compatible provenance for application status panels."""
        config = self.spec.config
        return {
            "implementation": "wm.networks.cosmos3",
            "checkpoint_path": str(self.spec.checkpoint),
            "iteration": self.spec.iteration,
            "und_checkpoint_path": str(self.spec.text[-1].path),
            "lora_loaded": self.spec.lora is not None,
            "first_frame_reference_embedding": config.first_frame_reference_embedding,
        }


def prompt_pixel_frames(config: Mapping[str, Any]) -> int:
    """Read the caption horizon (pixel frames) a checkpoint was trained with."""
    count = config["dataloader_val"]["dataset"]["num_frames"]
    if type(count) is not int or count <= 1 or (count - 1) % 4:
        raise ValueError(f"caption horizon must be 1 + 4N pixel frames, got {count!r}")
    return count


def _resolve(path: str, root: Path) -> Path:
    location = Path(path).expanduser()
    if not location.is_absolute():
        location = root / location
    return location.resolve()


def _recipe(config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], Cosmos3Config]:
    # Diffusion teachers own ``net``; distilled recipes own ``student``.
    # This selects weight ownership only, never a sampler or training target.
    model = config["model"]
    role = "student" if "student" in model else "net"
    network = model[role]
    raw = network["config"]
    if not isinstance(raw, Mapping):
        raise ValueError(f"unresolved network config: {raw!r}")
    allowed = {field.name for field in fields(Cosmos3Config)}
    topology = Cosmos3Config(**{k: v for k, v in raw.items() if k in allowed})
    return role, network, topology


def inspect_artifact(checkpoint: str | Path) -> Cosmos3ArtifactSpec:
    """Read a run's metadata without loading tensors or importing its recipe."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    manifest_root = checkpoint.parent if checkpoint.is_file() else checkpoint
    if (manifest_root / "manifest.json").is_file():
        return _inspect_export(manifest_root)
    if not (checkpoint / "tensor" / ".metadata").is_file():
        raise FileNotFoundError(
            f"expected a DCP checkpoint with tensor/.metadata: {checkpoint}"
        )
    candidates = (
        checkpoint / "config.yaml",
        checkpoint.parent.parent / "config.yaml",
        checkpoint.parent / "config.yaml",
    )
    config_path = next((p for p in candidates if p.is_file()), None)
    if config_path is None:
        raise FileNotFoundError(f"no resolved config.yaml beside {checkpoint}")
    config = yaml.safe_load(config_path.read_text())
    role, network, topology = _recipe(config)

    # Frozen or initial weights come from the run's initializations; the
    # checkpoint itself overlays every tensor it owns.
    student_sources: list[WeightSource] = []
    text_sources: list[WeightSource] = []
    for entry in config["train"].get("model_initializations", {}).values():
        if entry.get("path") is None:
            continue
        destinations = entry.get("model_key_prefixes", (entry.get("model_key_prefix"),))
        source = WeightSource(
            _resolve(entry["path"], config_path.parent),
            (entry.get("source_model_key_prefix") or "").removeprefix("model."),
            tuple(entry.get("source_model_key_patterns", ())),
        )
        if role in destinations:
            student_sources.append(source)
        if _TEXT_PREFIX in destinations:
            text_sources.append(source)
    keys = FileSystemReader(checkpoint / "tensor").read_metadata().state_dict_metadata
    if f"model.{_TEXT_PREFIX}.embed_tokens.weight" in keys:
        text_sources.append(WeightSource(checkpoint, _TEXT_PREFIX))
    if not text_sources:
        raise ValueError("checkpoint does not identify its frozen UND weight source")
    student_sources.append(WeightSource(checkpoint, role))

    match = re.fullmatch(r"iter_(\d+)", checkpoint.name)
    return Cosmos3ArtifactSpec(
        checkpoint=checkpoint,
        config=topology,
        student=tuple(student_sources),
        text=tuple(text_sources),
        tokenizer_path=_resolve(
            config["model"]["conditioner"]["tokenizer_path"], config_path.parent
        ),
        prompt_pixel_frames=prompt_pixel_frames(config),
        iteration=int(match[1]) if match else 0,
        lora=network.get("lora"),
    )


def _inspect_export(root: Path) -> Cosmos3ArtifactSpec:
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("version") != MANIFEST_VERSION
    ):
        raise ValueError("unsupported serving manifest schema/version")

    def local_file(name: str) -> Path:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                f"serving manifest must name an existing local file: {name!r}"
            )
        return path

    config = yaml.safe_load(
        local_file(manifest.get("config", "config.yaml")).read_text()
    )
    _, network, topology = _recipe(config)
    # Exported tensors use local module keys, so the sources have no prefix.
    student, text = (
        WeightSource(local_file(manifest[role]["file"]), "")
        for role in ("student", "und")
    )
    iteration = manifest["iteration"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError("manifest iteration must be a nonnegative integer")
    return Cosmos3ArtifactSpec(
        checkpoint=root,
        config=topology,
        student=(student,),
        text=(text,),
        tokenizer_path=_resolve(config["model"]["conditioner"]["tokenizer_path"], root),
        prompt_pixel_frames=prompt_pixel_frames(config),
        iteration=iteration,
        lora=network.get("lora"),
    )


def _tensor_layout(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    """A safetensors file's data offset and its per-tensor header entries."""
    with path.open("rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
        entries = json.loads(file.read(length))
    entries.pop("__metadata__", None)
    return 8 + length, entries


def _load_tensor_file(module: nn.Module, source: WeightSource) -> set[str]:
    prefix = source.prefix.rstrip(".") + "." if source.prefix else ""
    parameters = dict(module.named_parameters())
    data_start, entries = _tensor_layout(source.path)
    selected = {name: prefix + name for name in parameters if prefix + name in entries}
    for name, key in selected.items():
        if tuple(entries[key]["shape"]) != tuple(parameters[name].shape):
            raise ValueError(f"shape mismatch for {key!r} in {source.path}")
    _reject_unused_lora(
        key for key in entries.keys() - set(selected.values()) if key.startswith(prefix)
    )

    # Positional reads from several threads: a network filesystem serves them
    # many times faster than one stream of page faults through a mapping.
    descriptor = os.open(source.path, os.O_RDONLY)

    def read(name: str) -> None:
        entry = entries[selected[name]]
        start, stop = entry["data_offsets"]
        staging = torch.empty(
            stop - start, dtype=torch.uint8, pin_memory=parameters[name].is_cuda
        )
        view = memoryview(staging.numpy())
        done = 0
        while done < len(view):
            count = os.preadv(
                descriptor,
                [view[done : done + _READ_CHUNK]],
                data_start + start + done,
            )
            if count <= 0:
                raise OSError(f"truncated tensor {selected[name]!r} in {source.path}")
            done += count
        # Gradient mode is per thread; the caller's no_grad does not reach here.
        with torch.no_grad():
            parameters[name].copy_(
                staging.view(_SAFETENSORS_DTYPES[entry["dtype"]]).view(entry["shape"])
            )

    try:
        # Largest first, so no thread is left alone with one big tensor at the end.
        order = sorted(
            selected,
            key=lambda name: (
                -(
                    entries[selected[name]]["data_offsets"][1]
                    - entries[selected[name]]["data_offsets"][0]
                )
            ),
        )
        with ThreadPoolExecutor(_READ_THREADS) as pool:
            list(pool.map(read, order))
    finally:
        os.close(descriptor)
    return set(selected)


def _reject_unused_lora(unused_keys: Iterable[str]) -> None:
    unused = sorted(key for key in unused_keys if ".lora_" in key)
    if unused:
        raise ValueError(f"checkpoint has undeclared LoRA weights: {unused[:4]}")


def _load_tower(module: nn.Module, sources: tuple[WeightSource, ...]) -> None:
    # Reuse training's DCP/HF readers, prefix mapping and shape validation.
    # The union of overlays must cover every live parameter: uninitialized
    # meta storage and accidentally omitted adapters are never allowed.
    checkpointer = DistributedCheckpointer(sources[-1].path.parent)
    loaded: set[str] = set()
    for source in sources:
        if source.path.suffix == ".safetensors":
            loaded.update(_load_tensor_file(module, source))
            continue
        result = checkpointer.load(
            module,
            path=source.path,
            components=("model",),
            source_model_key_prefix=source.prefix,
            model_key_prefix="",
            source_model_key_patterns=source.patterns,
        )
        _reject_unused_lora(result.unexpected_model_keys)
        loaded.update(result.loaded_model_keys)
    missing = set(module.state_dict()) - loaded
    if missing:
        raise ValueError(f"inference weights are incomplete: {sorted(missing)[:12]}")


def load_artifact(
    spec: Cosmos3ArtifactSpec,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    progress: Callable[[str, int, int], None] | None = None,
) -> Cosmos3Artifact:
    """Materialize only the student and text tower, including declared LoRA."""
    with torch.device("meta"):
        student = build_network(spec.config, lora=spec.lora)
        text = Cosmos3TextEncoder(spec.config)
    for label, tower, sources in (
        ("und", text, spec.text),
        ("student", student, spec.student),
    ):
        if progress is not None:
            progress(label, 0, 0)
        tower.to(dtype=dtype).to_empty(device=device)
        _load_tower(tower, sources)
        tower.eval().requires_grad_(False)
        if progress is not None:
            progress(label, len(tower.state_dict()), len(tower.state_dict()))
    return Cosmos3Artifact(student, text, spec)


def save_artifact(artifact: Cosmos3Artifact, directory: str | Path) -> Path:
    """Export both towers as safetensors with a manifest `inspect_artifact` reads."""
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = artifact.spec
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "iteration": spec.iteration,
        "config": "config.yaml",
    }
    for role, tower in (
        ("student", artifact.student),
        ("und", artifact.context_encoder),
    ):
        name = f"{role}.safetensors"
        save_file(
            {key: value.detach().cpu() for key, value in tower.state_dict().items()},
            str(root / name),
        )
        manifest[role] = {"file": name}
    lora = None if spec.lora is None else _plain(spec.lora)
    config = {
        "model": {
            "student": {"config": _plain(asdict(spec.config)), "lora": lora},
            "conditioner": {"tokenizer_path": str(spec.tokenizer_path)},
        },
        "dataloader_val": {"dataset": {"num_frames": spec.prompt_pixel_frames}},
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def _plain(value: Any) -> Any:
    # YAML-safe containers: mappings and sequences of plain scalars.
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from torch.distributed.checkpoint import (
    FileSystemReader,
    HuggingFaceStorageReader,
    StorageReader,
)
from torch.distributed.checkpoint.metadata import Metadata

TENSOR_COMPONENTS = ("model", "optimizers", "schedulers", "trainer")


ALL_COMPONENTS = (*TENSOR_COMPONENTS, "dataloader")


def _state_keys(metadata: Metadata) -> set[str]:
    return {str(key) for key in metadata.state_dict_metadata}


def _load_reader(checkpoint_path: Path) -> tuple[StorageReader, bool]:
    tensor_path = checkpoint_path / "tensor"
    if (tensor_path / ".metadata").is_file():
        return FileSystemReader(str(tensor_path)), False
    if (checkpoint_path / ".metadata").is_file():
        return FileSystemReader(str(checkpoint_path)), False
    model_path = checkpoint_path / "model"
    if (model_path / ".metadata").is_file():
        return FileSystemReader(str(model_path)), False
    return HuggingFaceStorageReader(str(checkpoint_path)), True


def component_names(keys: Iterable[str], component: str) -> set[str]:
    prefix = f"{component}."
    return {
        suffix.split(".", maxsplit=1)[0]
        for key in keys
        if (suffix := key.removeprefix(prefix)) != key and suffix
    }


def component_fields(
    keys: Iterable[str], component: str, name: str | None = None
) -> set[str]:
    prefix = f"{component}." if name is None else f"{component}.{name}."
    return {
        suffix for key in keys if (suffix := key.removeprefix(prefix)) != key and suffix
    }


def normalise_components(
    components: Iterable[str] | None,
    *,
    include_dataloader: bool = False,
) -> tuple[str, ...]:
    if components is None:
        return ALL_COMPONENTS if include_dataloader else TENSOR_COMPONENTS
    requested = set(components)
    unknown = requested.difference(ALL_COMPONENTS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown checkpoint components: {names}")
    return tuple(component for component in ALL_COMPONENTS if component in requested)


@dataclass(frozen=True, slots=True)
class LoadSource:
    reader: StorageReader
    metadata: Metadata
    state_keys: set[str]
    is_huggingface: bool
    # A HuggingFace directory or a raw (non-nested) DCP model overlay stores
    # model tensors at the top level instead of under ``model.``.
    flat_model: bool
    saved_components: frozenset[str]

    @classmethod
    def open(
        cls, checkpoint_path: Path, source_model_key_prefix: str | None
    ) -> LoadSource:
        reader, is_huggingface = _load_reader(checkpoint_path)
        metadata = reader.read_metadata()
        state_keys = _state_keys(metadata)
        nested_dcp_model = any(key.startswith("model.") for key in state_keys)
        raw_dcp_model_overlay = (
            not is_huggingface
            and source_model_key_prefix is not None
            and not nested_dcp_model
            and bool(state_keys)
        )
        flat_model = is_huggingface or raw_dcp_model_overlay
        saved_components = frozenset(
            {"model"}
            if flat_model and state_keys
            else {key.split(".", maxsplit=1)[0] for key in state_keys}
        )
        return cls(
            reader=reader,
            metadata=metadata,
            state_keys=state_keys,
            is_huggingface=is_huggingface,
            flat_model=flat_model,
            saved_components=saved_components,
        )

    def has(self, component: str, selected: Iterable[str]) -> bool:
        return component in selected and component in self.saved_components

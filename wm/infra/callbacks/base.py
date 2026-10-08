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

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch
from omegaconf import ListConfig

from wm.infra.config import instantiate

__all__ = ["Callback", "CallbackGroup"]


class Callback:
    def __init__(self) -> None:
        self.config: Any | None = None
        self.trainer: Any | None = None

    def on_train_start(self, model: Any, iteration: int = 0) -> None: ...

    def on_before_dataloading(self, iteration: int = 0) -> None: ...

    def on_training_step_start(
        self, model: Any, data: dict[str, Any], iteration: int = 0
    ) -> None: ...

    def on_before_optimizer_step(
        self,
        model_ddp: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None: ...

    def on_after_optimizer_step(
        self,
        model_ddp: torch.nn.Module,
        name: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None: ...

    def on_optimizer_step_skipped(
        self,
        model_ddp: torch.nn.Module,
        optimizer_names: tuple[str, ...],
        iteration: int = 0,
    ) -> None: ...

    def on_before_zero_grad(
        self,
        model_ddp: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None: ...

    # Once per microbatch (also on accumulation steps that do not update).
    def on_training_step_batch_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None: ...

    # Once per successful optimizer update; ``iteration`` counts updates.
    def on_training_step_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None: ...

    def on_validation_start(
        self, model: Any, dataloader_val: Any, iteration: int = 0
    ) -> None: ...

    def on_validation_step_end(
        self,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None: ...

    def on_validation_end(self, model: Any, iteration: int = 0) -> None: ...

    def on_train_end(self, model: Any, iteration: int = 0) -> None: ...

    def on_app_end(self) -> None: ...


class CallbackGroup(Callback):
    """Runs every lifecycle hook on its callbacks, in configuration order."""

    def __init__(self, config: Any | None, trainer: Any | None) -> None:
        super().__init__()
        self._callbacks: list[Callback] = []
        callback_configs = self._read_callback_configs(config)
        if callback_configs is None:
            return
        for name, callback_config in self._iter_callback_configs(callback_configs):
            # A null entry is the smallest useful CLI switch for an optional
            # callback: ``trainer.callbacks.samples=null``.  Keep the mapping
            # key/order stable without asking every experiment to rebuild the
            # whole callback collection.
            if callback_config is None:
                continue
            callback = instantiate(callback_config)
            if not isinstance(callback, Callback):
                raise TypeError(
                    f"Callback {name!r} resolved to {type(callback).__name__}; "
                    "expected an instance of wm.infra.callbacks.Callback"
                )
            callback.config = config
            callback.trainer = trainer
            self._callbacks.append(callback)

    @staticmethod
    def _read_callback_configs(config: Any | None) -> Any | None:
        if config is None:
            return None
        if isinstance(config, Mapping):
            trainer_config = config.get("trainer")
        else:
            trainer_config = getattr(config, "trainer", None)
        if trainer_config is None:
            return None
        if isinstance(trainer_config, Mapping):
            return trainer_config.get("callbacks")
        return getattr(trainer_config, "callbacks", None)

    @staticmethod
    def _iter_callback_configs(
        callback_configs: Any,
    ) -> Iterator[tuple[Any, Any]]:
        if isinstance(callback_configs, Mapping):
            yield from callback_configs.items()
            return
        if isinstance(callback_configs, (Sequence, ListConfig)) and not isinstance(
            callback_configs, (str, bytes)
        ):
            for index, callback_config in enumerate(callback_configs):
                yield f"callback_{index}", callback_config
            return
        raise TypeError(
            "config.trainer.callbacks must be an ordered mapping or sequence, "
            f"got {type(callback_configs).__name__}"
        )

    @property
    def callbacks(self) -> tuple[Callback, ...]:
        return tuple(self._callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    def __iter__(self) -> Iterator[Callback]:
        return iter(self._callbacks)

    def on_app_end(self) -> None:
        # Cleanup reaches every callback even when an earlier one fails.
        errors: list[Exception] = []
        for callback in self._callbacks:
            try:
                callback.on_app_end()
            except Exception as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Callback cleanup failed", errors)


def _fan_out(name: str) -> Callable[..., None]:
    def hook(group: CallbackGroup, *args: Any, **kwargs: Any) -> None:
        for callback in group:
            getattr(callback, name)(*args, **kwargs)

    hook.__name__ = name
    hook.__qualname__ = f"{CallbackGroup.__name__}.{name}"
    return hook


for _name in vars(Callback):
    if _name.startswith("on_") and _name not in vars(CallbackGroup):
        setattr(CallbackGroup, _name, _fan_out(_name))

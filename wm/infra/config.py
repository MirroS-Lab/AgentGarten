# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy object configuration.

LazyCall records a keyword call as an OmegaConf node ({_target_: dotted
path, **kwargs}); instantiate() builds it recursively. Adapted from
Imaginaire; targets must be importable, so resolved configs round-trip YAML.
"""

from __future__ import annotations

import collections.abc as abc
import dataclasses
import importlib
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, cast

import attrs
from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["LazyCall", "LazyDict", "instantiate"]


def locate(name: str) -> Any:
    if not isinstance(name, str) or "." not in name or "<locals>" in name:
        raise ImportError(f"Target must be an importable dotted path, got {name!r}")

    parts = name.split(".")
    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            obj: Any = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            # Trying progressively shorter prefixes is expected. If importing
            # a real module failed because one of its dependencies is missing,
            # preserve that useful underlying error instead.
            missing_name = error.name or ""
            if module_name != missing_name and not module_name.startswith(
                f"{missing_name}."
            ):
                raise
            continue

        try:
            for attribute in parts[split_at:]:
                obj = getattr(obj, attribute)
        except AttributeError as error:
            raise ImportError(f"Cannot locate target {name!r}") from error
        return obj

    raise ImportError(f"Cannot import target {name!r}")


def convert_target_to_string(target: Any) -> str:
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise TypeError(f"LazyCall target is not an importable callable: {target!r}")
    if module in {"__main__", "__mp_main__"}:
        raise ValueError(
            f"LazyCall target is not importable from a module: {module}.{qualname}"
        )
    if "<locals>" in qualname or "<lambda>" in qualname:
        raise ValueError(f"LazyCall target is not importable: {module}.{qualname}")

    # Match the reference implementation's preference for a stable exported
    # API path over a deeper implementation path when both name the same object.
    module_parts = module.split(".")
    candidates = [
        f"{'.'.join(module_parts[:end])}.{qualname}"
        for end in range(1, len(module_parts) + 1)
    ]
    for candidate in candidates:
        try:
            if locate(candidate) is target:
                return candidate
        except ImportError:
            continue

    full_name = f"{module}.{qualname}"
    raise ValueError(f"LazyCall target {full_name!r} cannot be imported by dotted path")


T = TypeVar("T")

if TYPE_CHECKING:
    # Static type checkers treat LazyDict[T] like T, while the runtime object is
    # OmegaConf's mutable DictConfig.
    LazyDict: TypeAlias = T
else:
    LazyDict = DictConfig


def _default_keyword_arguments(target: Callable[..., Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        # A few importable C-extension/builtin callables do not expose a
        # signature. They remain valid lazy targets; they simply contribute no
        # discoverable defaults.
        return {}
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


class LazyCall(Generic[T]):
    def __init__(self, target: Callable[..., T] | str):
        if isinstance(target, str):
            resolved_target = locate(target)
            if not callable(resolved_target):
                raise TypeError(f"LazyCall target {target!r} is not callable")
            self._target = target
            self._callable = cast(Callable[..., T], resolved_target)
        elif callable(target):
            self._target = convert_target_to_string(target)
            self._callable = target
        else:
            raise TypeError(
                f"LazyCall target must be an importable callable, got {target!r}"
            )

    def __call__(self, **kwargs: Any) -> LazyDict[T]:
        parameters = _default_keyword_arguments(self._callable)
        parameters.update(kwargs)
        parameters["_target_"] = self._target
        return cast(LazyDict[T], DictConfig(parameters, flags={"allow_objects": True}))


def _is_dataclass_or_attrs(target: Any) -> bool:
    return dataclasses.is_dataclass(target) or attrs.has(target)


def instantiate(cfg: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(cfg, ListConfig):
        return ListConfig(
            [instantiate(value) for value in cfg], flags={"allow_objects": True}
        )
    if isinstance(cfg, list):
        return [instantiate(value) for value in cfg]

    # Preserve OmegaConf's structured-config behavior from the reference
    # implementation.
    if isinstance(cfg, DictConfig) and _is_dataclass_or_attrs(
        cfg._metadata.object_type
    ):
        return OmegaConf.to_object(cfg)

    if isinstance(cfg, abc.Mapping) and "_target_" in cfg:
        # Resolve the outer factory before constructing potentially expensive
        # child objects. An invalid parent must not instantiate its children.
        target_name = cfg["_target_"]
        if not isinstance(target_name, str):
            raise TypeError(
                f"_target_ must be an importable dotted string, got {target_name!r}"
            )
        target = locate(target_name)
        if not callable(target):
            raise TypeError(f"_target_ {target_name!r} does not name a callable")

        recursive = cfg.get("_recursive_", True)
        values = {}
        # DictConfig.items() eagerly resolves missing values/interpolations,
        # including ones that a runtime injection completely replaces.
        for key in cfg:
            if key in {"_target_", "_recursive_"}:
                continue
            if key in kwargs:
                values[key] = kwargs[key]
            else:
                value = cfg[key]
                values[key] = instantiate(value) if recursive else value

        values.update(
            {key: value for key, value in kwargs.items() if key not in values}
        )
        return target(*args, **values)

    if isinstance(cfg, DictConfig):
        return DictConfig(
            {key: instantiate(value) for key, value in cfg.items()},
            flags={"allow_objects": True},
        )
    if isinstance(cfg, abc.Mapping):
        return {key: instantiate(value) for key, value in cfg.items()}

    return cfg

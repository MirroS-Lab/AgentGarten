# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import cast

from hydra import compose as hydra_compose
from hydra import initialize
from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf

from wm.infra.config import LazyDict

EXPERIMENT_MODULE_ROOT = "wm.configs.experiments"
_PRIVATE_CONFIG_NAME = "_wm_selected_experiment"


def _experiment_selector(overrides: Sequence[str]) -> str:
    parsed = OverridesParser.create().parse_overrides(list(overrides))
    selected = [item for item in parsed if item.key_or_group == "experiment"]
    if len(selected) != 1:
        raise ValueError("Exactly one plain experiment=... selector is required")

    override = selected[0]
    value = override.value()
    if (
        override.is_add()
        or override.is_force_add()
        or override.is_delete()
        or override.is_sweep_override()
        or not isinstance(value, str)
    ):
        raise ValueError("experiment must be a single plain scalar selector")
    return value


def _experiment_module(selector: str, experiment_module_root: str) -> str:
    parts = selector.split("/")
    if len(parts) < 2 or any(not part.isidentifier() for part in parts):
        raise ValueError(
            "experiment must be an importable path such as family/name or "
            "family/objective/version"
        )
    return ".".join((experiment_module_root, *parts))


def compose_config(
    overrides: Sequence[str], *, experiment_module_root: str = EXPERIMENT_MODULE_ROOT
) -> LazyDict:
    """Build the selected experiment's config and apply Hydra overrides.

    ``overrides`` must contain one ``experiment=family/name`` selector, which
    names a module under ``experiment_module_root`` exposing ``build_config()``.
    """
    selector = _experiment_selector(overrides)
    module = importlib.import_module(
        _experiment_module(selector, experiment_module_root)
    )
    builder = getattr(module, "build_config", None)
    if not callable(builder):
        raise TypeError(f"{module.__name__}.build_config is not callable")

    config = builder()
    if not isinstance(config, DictConfig):
        raise TypeError(f"{module.__name__}.build_config() must return LazyDict")

    OmegaConf.update(config, "experiment", selector, force_add=True)
    ConfigStore.instance().store(name=_PRIVATE_CONFIG_NAME, node=config)

    with initialize(version_base=None, config_path=None):
        resolved = hydra_compose(
            config_name=_PRIVATE_CONFIG_NAME,
            overrides=list(overrides),
        )
        OmegaConf.resolve(resolved)
        OmegaConf.to_container(resolved, resolve=True, throw_on_missing=True)

    return cast(LazyDict, resolved)


__all__ = ["compose_config"]

# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from re import Pattern
from typing import Any

import torch


def model_prefix(value: str | None) -> str:
    normalized = "" if value is None else str(value).strip(".")
    return f"{normalized}." if normalized else ""


@dataclass(frozen=True, slots=True)
class ModelLoadEntry:
    source_key: str
    targets: tuple[str, ...]

    @property
    def template_key(self) -> str:
        return self.targets[0]


@dataclass(frozen=True, slots=True)
class ModelRestorePlan:
    entries: tuple[ModelLoadEntry, ...]
    checkpoint_model_keys: frozenset[str]
    missing_model_keys: tuple[str, ...]
    unexpected_model_keys: tuple[str, ...]


def build_model_restore_plan(
    *,
    live_keys: Iterable[str],
    saved_keys: Iterable[str],
    overlay: bool,
    source_prefix: str | None = None,
    destination_prefixes: Sequence[str | None] = (None,),
    source_patterns: Sequence[Pattern[str]] = (),
) -> ModelRestorePlan:
    ordered_live_keys = tuple(live_keys)
    live = set(ordered_live_keys)
    saved = set(saved_keys)
    if not overlay:
        return ModelRestorePlan(
            entries=tuple(
                ModelLoadEntry(key, (key,)) for key in ordered_live_keys if key in saved
            ),
            checkpoint_model_keys=frozenset(saved),
            missing_model_keys=tuple(sorted(live - saved)),
            unexpected_model_keys=tuple(sorted(saved - live)),
        )

    source = model_prefix(source_prefix)
    destinations = tuple(dict.fromkeys(model_prefix(p) for p in destination_prefixes))
    if not destinations:
        raise ValueError("checkpoint overlay requires at least one destination")
    selected = sorted(
        key
        for key in saved
        if key.startswith(source)
        and (
            not source_patterns
            or any(
                pattern.fullmatch(key.removeprefix(source))
                for pattern in source_patterns
            )
        )
    )
    scoped_live = {key for key in live if key.startswith(destinations)}
    mapped_keys: set[str] = set()
    destination_sources: dict[str, str] = {}
    entries = []
    for key in selected:
        targets = tuple(prefix + key.removeprefix(source) for prefix in destinations)
        mapped_keys.update(targets)
        live_targets = tuple(target for target in targets if target in live)
        for target in live_targets:
            previous = destination_sources.get(target)
            if previous is not None and previous != key:
                raise ValueError(
                    f"Ambiguous checkpoint overlay for {target!r}: "
                    f"both {previous!r} and {key!r} map to it"
                )
            destination_sources[target] = key
        if live_targets:
            entries.append(ModelLoadEntry(key, live_targets))
    return ModelRestorePlan(
        entries=tuple(entries),
        checkpoint_model_keys=frozenset(mapped_keys),
        missing_model_keys=tuple(sorted(scoped_live - mapped_keys)),
        unexpected_model_keys=tuple(sorted(mapped_keys - scoped_live)),
    )


def validate_overlay_target_shapes(
    plan: ModelRestorePlan, live_state: Mapping[str, Any]
) -> None:
    for entry in plan.entries:
        template = live_state[entry.template_key]
        if not torch.is_tensor(template):
            continue
        for target in entry.targets[1:]:
            value = live_state[target]
            if torch.is_tensor(value) and value.shape != template.shape:
                raise ValueError(
                    f"Checkpoint overlay target shape mismatch for {entry.source_key!r}: "
                    f"{entry.template_key}={tuple(template.shape)}, {target}={tuple(value.shape)}"
                )


def prepare_model_restore_plan(
    *,
    live_state: Mapping[str, Any],
    saved_keys: Iterable[str],
    overlay: bool,
    source_prefix: str | None,
    destination_prefixes: Sequence[str | None],
    source_patterns: Sequence[Pattern[str]],
) -> ModelRestorePlan:
    plan = None
    local_error: Exception | None = None
    try:
        plan = build_model_restore_plan(
            live_keys=live_state,
            saved_keys=saved_keys,
            overlay=overlay,
            source_prefix=source_prefix,
            destination_prefixes=destination_prefixes,
            source_patterns=source_patterns,
        )
        validate_overlay_target_shapes(plan, live_state)
    except Exception as error:
        local_error = error
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        errors: list[str | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(
            errors,
            None
            if local_error is None
            else f"{type(local_error).__name__}: {local_error}",
        )
        failures = [
            f"rank {rank}: {error}" for rank, error in enumerate(errors) if error
        ]
        if failures:
            raise RuntimeError(
                "Checkpoint model restore planning failed: " + "; ".join(failures)
            ) from local_error
    elif local_error is not None:
        raise local_error
    assert plan is not None
    return plan


@dataclass(frozen=True, slots=True)
class ModelInitialization:
    path: str
    destinations: tuple[str | None, ...]
    plural_destinations: bool
    source_model_key_prefix: str | None = None
    source_model_key_patterns: tuple[str, ...] = ()

    @property
    def coalescible(self) -> bool:
        return bool(self.destinations) and all(
            destination is not None for destination in self.destinations
        )

    def reads_same_source(self, other: "ModelInitialization") -> bool:
        return (
            self.path == other.path
            and self.source_model_key_prefix == other.source_model_key_prefix
            and self.source_model_key_patterns == other.source_model_key_patterns
        )

    def load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "path": self.path,
            "source_model_key_prefix": self.source_model_key_prefix,
        }
        if self.source_model_key_patterns:
            kwargs["source_model_key_patterns"] = self.source_model_key_patterns
        if self.plural_destinations:
            kwargs["model_key_prefixes"] = self.destinations
        else:
            kwargs["model_key_prefix"] = self.destinations[0]
        return kwargs


def _initialization_destinations(
    entry: Mapping[str, Any], *, resume: bool
) -> tuple[tuple[str | None, ...], bool] | None:
    if resume:
        configured = entry.get("resume_model_key_prefixes")
        if configured is None or not tuple(configured):
            return None
        return tuple(configured), True
    configured = entry.get("model_key_prefixes")
    if configured is None:
        return (entry.get("model_key_prefix"),), False
    return tuple(configured), True


def plan_model_initializations(
    entries: Mapping[str, Mapping[str, Any]], *, resume: bool
) -> tuple[ModelInitialization, ...]:
    planned: list[ModelInitialization] = []
    last_index: int | None = None
    for index, (label, entry) in enumerate(entries.items()):
        selected = _initialization_destinations(entry, resume=resume)
        if selected is None:
            continue
        destinations, plural = selected
        path = entry.get("path")
        if path is None:
            warnings.warn(
                f"Skipping model initialization {label!r}: path is null.",
                stacklevel=2,
            )
            continue
        current = ModelInitialization(
            path=path,
            destinations=destinations,
            plural_destinations=plural,
            source_model_key_prefix=entry.get("source_model_key_prefix"),
            source_model_key_patterns=tuple(entry.get("source_model_key_patterns", ())),
        )
        previous = planned[-1] if planned else None
        if (
            previous is not None
            and last_index == index - 1
            and previous.reads_same_source(current)
            and previous.coalescible
            and current.coalescible
        ):
            planned[-1] = replace(
                previous,
                destinations=(*previous.destinations, *current.destinations),
                plural_destinations=True,
            )
        else:
            planned.append(current)
        last_index = index
    return tuple(planned)

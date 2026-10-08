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

import os
import sys
from typing import Any

import torch.distributed as dist
from loguru import logger as _loguru_logger

__all__ = [
    "add_sink",
    "complete",
    "configure",
    "exception",
    "info",
    "logger",
    "warning",
]

# Messages are printed by rank 0 only unless logged with ``rank0_only=False``,
# which also prefixes them with the rank.
_FORMAT = (
    "[<green>{time:MM-DD HH:mm:ss}</green>|"
    "<level>{level}</level>|"
    "<cyan>{extra[relative_path]}:{line}:{function}</cyan>] "
    "{extra[rank_prefix]}{message}\n{exception}"
)


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _add_log_context(record: dict[str, Any]) -> None:
    extra = record["extra"]
    try:
        extra["relative_path"] = os.path.relpath(record["file"].path, os.getcwd())
    except OSError:
        extra["relative_path"] = f"<cwd-unavailable>:{record['file'].path}"
    rank = _get_rank()
    extra["rank"] = rank
    extra["rank_prefix"] = "" if extra.get("rank0_only", True) else f"[RANK {rank}] "


logger = _loguru_logger.patch(_add_log_context)


def _format(record: dict[str, Any]) -> str:
    # Records from other loguru users reach the same sinks without wm's context.
    extra = record["extra"]
    extra.setdefault("relative_path", record["file"].name)
    extra.setdefault("rank_prefix", "")
    return _FORMAT


def _rank_filter(record: dict[str, Any]) -> bool:
    return not record["extra"].get("rank0_only", True) or _get_rank() == 0


def add_sink(sink: Any, *, level: str = "INFO", **options: Any) -> int:
    """Add a rank-aware sink (a stream or a file path); returns its handler id."""
    return logger.add(
        sink, level=level, format=_format, filter=_rank_filter, catch=False, **options
    )


def configure(level: str = "INFO") -> int:
    """Route all loguru output to stdout. Call from an application entry point."""
    logger.remove()
    return add_sink(sys.stdout, level=level)


def complete() -> None:
    logger.complete()


def info(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).info(message)


def warning(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).warning(message)


def exception(message: str, rank0_only: bool = True) -> None:
    logger.opt(depth=1).bind(rank0_only=rank0_only).exception(message)

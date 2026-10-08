# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from itertools import islice
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
)

__all__ = ["StatefulDataLoader"]


class _Skip(Sampler[list[int]]):
    def __init__(self, batches: BatchSampler, offset: int) -> None:
        self.batches, self.offset = batches, offset

    def __iter__(self) -> Iterator[list[int]]:
        yield from islice(self.batches, self.offset, None)

    def __len__(self) -> int:
        return max(len(self.batches) - self.offset, 0)


class StatefulDataLoader:
    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        batch_size: int,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        num_workers: int = 0,
        prefetch_factor: int | None = 2,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.collate_fn = collate_fn
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.prefetch_factor = prefetch_factor if self.num_workers else None
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        distributed = dist.is_available() and dist.is_initialized()
        self.num_replicas = dist.get_world_size() if distributed else 1
        self.rank = dist.get_rank() if distributed else 0
        self.epoch = 0
        self.batch_offset = 0
        self._generation = 0

    def _batches(self, epoch: int) -> BatchSampler:
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.num_replicas,
            rank=self.rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        sampler.set_epoch(epoch)
        return BatchSampler(
            sampler, batch_size=self.batch_size, drop_last=self.drop_last
        )

    def __len__(self) -> int:
        batches = self._batches(self.epoch)
        return len(batches)

    def __iter__(self) -> Iterator[Any]:
        epoch, start = self.epoch, self.batch_offset
        self._generation += 1
        loader = DataLoader(
            self.dataset,
            batch_sampler=_Skip(self._batches(epoch), start),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(self.seed + epoch),
        )
        # Start workers now: the trainer creates this iterator before
        # callbacks spawn background threads (fork-after-threads is unsafe).
        return self._iterate(iter(loader), epoch, start, self._generation)

    def _iterate(
        self, batches: Iterator[Any], epoch: int, start: int, generation: int
    ) -> Iterator[Any]:
        total = len(self._batches(epoch))
        for index, batch in enumerate(batches, start=start):
            if generation != self._generation:
                raise RuntimeError(
                    "loader iterator superseded; call iter(loader) again"
                )
            # Publish the next cursor before yielding: a checkpoint may be
            # written before this generator is resumed.
            if index + 1 == total:
                self.epoch, self.batch_offset = epoch + 1, 0
            else:
                self.batch_offset = index + 1
            yield batch
        if self.epoch == epoch:
            self.epoch, self.batch_offset = epoch + 1, 0

    # Compared by the checkpoint sidecar: a cursor only resumes the same stream.
    def resume_signature(self) -> dict[str, Any]:
        dataset = getattr(self.dataset, "resume_signature", None)
        return {
            "dataset_size": len(self.dataset),
            "dataset": dataset() if callable(dataset) else None,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
        }

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_offset": self.batch_offset}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        epoch, offset = int(state["epoch"]), int(state["batch_offset"])
        if epoch < 0 or offset < 0:
            raise ValueError(f"invalid loader cursor {dict(state)}")
        total = len(self._batches(epoch))
        if total and offset >= total:
            epoch, offset = epoch + offset // total, offset % total
        self.epoch, self.batch_offset = epoch, offset
        self._generation += 1

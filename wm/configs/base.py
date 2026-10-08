# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from omegaconf import DictConfig

from wm.infra.callbacks import GradClipMonitor, LoguruMetricSink, ScalarMetricCallback
from wm.infra.checkpoint import DistributedCheckpointer
from wm.infra.config import LazyCall, LazyDict
from wm.infra.trainer import Trainer

__all__ = ["make_base_config"]


def make_base_config() -> LazyDict:
    return DictConfig(
        {
            "job": {"output_dir": "???"},
            "runtime": {
                "device": "cuda",
                "backend": "nccl",
                "process_group_timeout_seconds": 300.0,
                "seed": 0,
                "seed_by_rank": True,
                "torch": {
                    "benchmark": True,
                    "allow_tf32": True,
                    "float32_matmul_precision": "high",
                    "compile": {"recompile_limit": 32, "use_duck_shape": False},
                },
            },
            "model": "???",
            "optimizer": "???",
            "scheduler": "???",
            "trainer": LazyCall(Trainer)(
                grad_accum_steps=1,
                callbacks={
                    # Order matters: the monitor publishes gradient norms into
                    # the step outputs before the metric callback reads them.
                    "grad_monitor": LazyCall(GradClipMonitor)(
                        max_norm=1.0,
                        spike_threshold=10.0,
                        spike_ratio=5.0,
                        warmup_updates=10,
                        output_dir="${job.output_dir}/diagnostics/grad_spikes",
                    ),
                    "metrics": LazyCall(ScalarMetricCallback)(
                        every_n=10,
                        report_update_speed=True,
                        sinks=[LazyCall(LoguruMetricSink)()],
                    ),
                },
            ),
            "checkpoint": LazyCall(DistributedCheckpointer)(
                save_dir="${job.output_dir}/checkpoints",
                async_mode="thread",
                async_timeout_seconds=3600.0,
            ),
            "grad_scaler": LazyCall(torch.amp.GradScaler)(device="cuda", enabled=False),
            "dataloader_train": "???",
            "dataloader_val": None,
            "model_wrapper": None,
            "train": {
                "max_iterations": 10_000,
                "model_initializations": {},
                "checkpoint_path": None,
                "checkpoint_components": None,
                "checkpoint_optimizer_names": None,
                "checkpoint_scheduler_names": None,
                "checkpoint_interval": 500,
                "validation_interval": 500,
                "max_val_steps": 8,
                "validate_at_start": True,
                "save_final_checkpoint": True,
                "runtime_metadata": None,
            },
        },
        flags={"allow_objects": True},
    )

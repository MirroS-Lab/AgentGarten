# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from wm.configs.compose import compose_config
from wm.infra import distributed, log
from wm.infra import io as easy_io
from wm.infra.callbacks import CallbackGroup
from wm.infra.checkpoint import DistributedCheckpointer
from wm.infra.config import instantiate
from wm.infra.model import Model
from wm.infra.trainer import Trainer
from wm.infra.utils import set_random_seed


def _compact_error(error: BaseException) -> str:
    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message[:400]}"


def _publish_resolved_config(config: DictConfig) -> None:
    local_error: BaseException | None = None
    if distributed.is_rank0():
        try:
            payload = OmegaConf.to_yaml(config, resolve=True, sort_keys=False)
            output_path = easy_io.join_path(config.job.output_dir, "config.yaml")
            easy_io.atomic_put_text(payload, output_path)
        except BaseException as error:
            local_error = error

    error_message = None if local_error is None else _compact_error(local_error)
    if dist.is_available() and dist.is_initialized():
        publication_status = [error_message]
        dist.broadcast_object_list(publication_status, src=0)
        error_message = publication_status[0]

    if error_message is not None:
        raise RuntimeError(
            f"Resolved config publication failed on rank 0: {error_message}"
        ) from local_error


def _configure_torch(settings: Mapping[str, Any] | None) -> None:
    if settings is None:
        return
    if "deterministic" in settings:
        torch.backends.cudnn.deterministic = bool(settings["deterministic"])
    if "benchmark" in settings:
        torch.backends.cudnn.benchmark = bool(settings["benchmark"])
    if "allow_tf32" in settings:
        allow_tf32 = bool(settings["allow_tf32"])
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if "float32_matmul_precision" in settings:
        torch.set_float32_matmul_precision(settings["float32_matmul_precision"])
    # PyTorch exposes these compiler knobs only under private module paths.
    compile_settings = settings.get("compile") or {}
    recompile_limit = compile_settings.get("recompile_limit")
    if recompile_limit is not None:
        torch._dynamo.config.recompile_limit = int(recompile_limit)
    use_duck_shape = compile_settings.get("use_duck_shape")
    if use_duck_shape is not None:
        torch.fx.experimental._config.use_duck_shape = bool(use_duck_shape)


def _delayed_model_wrapper(
    wrapper_config: Mapping[str, Any] | None,
    device: torch.device,
) -> Callable[[Model], torch.nn.Module] | None:
    if wrapper_config is None:
        return None

    def wrap(model: Model) -> torch.nn.Module:
        return instantiate(wrapper_config, model=model, device=device)

    return wrap


def _seed_runtime_rng(runtime: Mapping[str, Any]) -> None:
    seed = runtime.get("seed")
    if seed is not None:
        set_random_seed(int(seed), by_rank=bool(runtime.get("seed_by_rank", False)))


def _run(config: DictConfig, device: torch.device) -> int:
    model = cast(Model, instantiate(config.model))
    model.parallelize(device)

    callbacks = CallbackGroup(config, trainer=None)
    trainer = cast(
        Trainer, instantiate(config.trainer, callbacks=callbacks, device=device)
    )
    checkpointer = cast(DistributedCheckpointer, instantiate(config.checkpoint))
    train_kwargs = dict(config.get("train", {}))
    components = train_kwargs.get("checkpoint_components")
    checkpointer.initialize_model(
        model,
        train_kwargs.pop("model_initializations", {}),
        resume=train_kwargs.get("checkpoint_path") is not None
        and (components is None or "model" in components),
    )

    model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = instantiate(config.grad_scaler)

    # CUDA DTensor initialization can synchronize the default RNG to rank 0,
    # overriding the per-rank seed set before model construction. Reestablish
    # independent training streams after all parameter/optimizer initialization.
    # CP still synchronizes its own group at the batch boundary. Exact resume
    # restores checkpoint RNG later, in Trainer.train(), and must take priority.
    _seed_runtime_rng(config.get("runtime", {}))

    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val_config = config.get("dataloader_val")
    dataloader_val = (
        None if dataloader_val_config is None else instantiate(dataloader_val_config)
    )
    model_wrapper = _delayed_model_wrapper(config.get("model_wrapper"), device)

    return trainer.train(
        model,
        dataloader_train,
        dataloader_val,
        grad_scaler=grad_scaler,
        checkpointer=checkpointer,
        model_wrapper=model_wrapper,
        **train_kwargs,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    experiment_module_root: str = "wm.configs.experiments",
) -> int:
    # Keep credentials and other machine-local settings out of experiment
    # configs. Explicitly exported variables win over the repository .env.
    load_dotenv(Path.cwd() / ".env", override=False)
    log.configure()
    overrides = list(sys.argv[1:] if argv is None else argv)
    config = compose_config(overrides, experiment_module_root=experiment_module_root)
    runtime = config.get("runtime", {})
    process_group_existed = dist.is_available() and dist.is_initialized()
    process_group_owned = False
    distributed_initializing = True

    try:
        process_group_timeout_seconds = float(
            runtime.get("process_group_timeout_seconds", 300.0)
        )
        if process_group_timeout_seconds <= 0:
            raise ValueError("runtime.process_group_timeout_seconds must be positive")
        device = distributed.init(
            device=runtime.get("device"),
            backend=runtime.get("backend"),
            timeout=timedelta(seconds=process_group_timeout_seconds),
        )
        process_group_owned = (
            not process_group_existed and dist.is_available() and dist.is_initialized()
        )
        distributed_initializing = False
        _publish_resolved_config(config)
        _configure_torch(runtime.get("torch"))
        _seed_runtime_rng(runtime)
        return _run(config, device)
    finally:
        process_group_is_active = dist.is_available() and dist.is_initialized()
        owns_failed_initialization = (
            distributed_initializing and not process_group_existed
        )
        if process_group_is_active and (
            process_group_owned or owns_failed_initialization
        ):
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()

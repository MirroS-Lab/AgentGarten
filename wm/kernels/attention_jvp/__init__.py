# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import torch

__all__ = ["attention_jvp", "backend_for"]


@functools.cache
def _has_cutlass() -> bool:
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


def backend_for(device: torch.device) -> str:
    if torch.cuda.get_device_capability(device) == (9, 0) and _has_cutlass():
        return "cute"
    return "triton"


@torch.no_grad()
def attention_jvp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    *,
    key_mask: torch.Tensor | None = None,
    scale: float | None = None,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    args = [x.contiguous() for x in (q, k, v, dq, dk, dv)]
    key_mask = None if key_mask is None else key_mask.contiguous()
    match backend or backend_for(q.device):
        case "cute":
            from .cute_backend import attention_jvp_cute

            output, tangent, _ = attention_jvp_cute(
                *args, key_mask=key_mask, scale=scale
            )
        case "triton":
            from .triton import attention_jvp as attention_jvp_triton

            output, tangent, _ = attention_jvp_triton(
                *args, key_mask=key_mask, scale=scale
            )
        case other:
            raise ValueError(f"unknown attention JVP backend {other!r}")
    return output, tangent

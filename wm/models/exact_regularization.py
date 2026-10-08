# SPDX-License-Identifier: Apache-2.0

"""Exact R1/R2 gradient penalties of a head on a frozen feature backbone.

Exact R = ||grad_x D||^2 with D = h_theta(f(x)) and f frozen:

    u = grad_x D                      (reverse mode)
    t = J_f(x) u                      (forward mode; attention via JVP kernel)
    dR/dtheta = 2 d<grad_phi h_theta, t>/dtheta   (double backward, head only)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor, nn

from wm.infra.utils import frozen_parameters

__all__ = ["ExactPenalty", "check_exact_regularization", "exact_penalties"]

FeatureFunction = Callable[[Tensor], tuple[Tensor, ...]]


@dataclass
class ExactPenalty:
    logits: Tensor  # [N], differentiable w.r.t. the head
    penalty: Tensor  # [N] = ||grad_x D||^2 per row, detached
    surrogate: Tensor  # [N] with value 0 and gradient dR/dtheta


def exact_penalties(
    features: FeatureFunction, head: nn.Module, rows: Tensor
) -> ExactPenalty:
    # 1. Input gradient through the frozen backbone and a frozen head.
    leaf = rows.detach().requires_grad_(True)
    with frozen_parameters(head), torch.enable_grad():
        (direction,) = torch.autograd.grad(head([features(leaf)]).sum(), leaf)
    direction = direction.detach()

    # 2. Feature tangents along that gradient (forward mode, no graph).
    with torch.no_grad(), fwAD.dual_level():
        dual = features(fwAD.make_dual(rows.detach(), direction.to(rows.dtype)))
        primal = tuple(fwAD.unpack_dual(value).primal.detach() for value in dual)
        tangent = tuple(fwAD.unpack_dual(value).tangent.detach() for value in dual)

    # 3. Head directional derivative <grad_phi h, t>, differentiable w.r.t. the
    #    head: double backward through the (small) head only.
    with torch.enable_grad():
        inputs = [p.requires_grad_(True) for p in primal]
        logits = head([inputs])
        gradients = torch.autograd.grad(logits.sum(), inputs, create_graph=True)
        wide = torch.promote_types(rows.dtype, torch.float32)
        directional = sum(
            (g.to(wide) * t.to(wide)).flatten(1).sum(1)
            for g, t in zip(gradients, tangent, strict=True)
        )
    penalty = direction.to(wide).square().flatten(1).sum(1)
    surrogate = 2.0 * directional
    return ExactPenalty(
        logits=logits,
        penalty=penalty,
        surrogate=surrogate - surrogate.detach(),
    )


def check_exact_regularization(real_score: nn.Module, discriminator: nn.Module) -> None:
    if any(parameter.requires_grad for parameter in real_score.parameters()):
        raise ValueError("exact R1/R2 requires a frozen real score")
    if not isinstance(getattr(discriminator, "feature_layers", None), Sequence):
        raise TypeError("exact R1/R2 needs a discriminator exposing feature_layers")

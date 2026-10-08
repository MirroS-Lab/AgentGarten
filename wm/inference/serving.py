# SPDX-License-Identifier: Apache-2.0

"""Prepare serving kernels and graphs without capturing mutable attention caches."""

from collections.abc import Callable
from functools import partial

import torch

from wm.networks.cosmos3 import Cosmos3Network

__all__ = ["prepare_serving", "release_serving"]

_Output = torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class _GraphCall:
    """One bound projection owns its graphs and their output lifetimes."""

    def __init__(self, function: Callable[..., _Output], pool: tuple[int, int]) -> None:
        self.function = function
        self.pool = pool
        self.graphs = {}

    def __call__(self, *arguments: torch.Tensor) -> _Output:
        if torch.is_grad_enabled():
            raise RuntimeError("serving CUDA graphs require disabled gradients")
        signature = tuple((x.shape, x.stride(), x.dtype, x.device) for x in arguments)
        if signature not in self.graphs:
            inputs = tuple(x.clone() for x in arguments)
            stream = torch.cuda.Stream(device=arguments[0].device)
            current = torch.cuda.current_stream(arguments[0].device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.function(*inputs)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                outputs = self.function(*inputs)
            self.graphs[signature] = (graph, inputs, outputs)
        graph, inputs, outputs = self.graphs[signature]
        for destination, source in zip(inputs, arguments, strict=True):
            destination.copy_(source)
        graph.replay()
        # Consumers must finish using these tensors before the next invocation
        # of this same bound method. Other layers have independent graph pools.
        return outputs


def prepare_serving(
    network: Cosmos3Network,
    *,
    dynamic: bool = False,
    cuda_graphs: bool = True,
    inductor: bool = False,
) -> None:
    """Prepare shared GEN project/finish methods, preserving parameter names.

    By default the methods use BF16 CUDA kernels from ``wm.kernels.gen_layer``
    and CUDA graphs, without calling ``torch.compile``. Triton kernel JIT and
    graph warmup still happen on first use. ``inductor=True`` explicitly opts
    into the compiler path for comparisons. Training policies are independent.
    """
    if not inductor:
        from wm.kernels import gen_layer
    for layer in network.layers:
        pool = torch.cuda.graph_pool_handle() if cuda_graphs else None
        for name in layer.compile_methods:
            if inductor:
                function = torch.compile(
                    getattr(layer, name),
                    dynamic=dynamic,
                    fullgraph=True,
                    options={
                        "emulate_precision_casts": True,
                        "triton.cudagraphs": False,
                    },
                )
            else:
                function = partial(getattr(gen_layer, name), layer)
            setattr(
                layer, name, _GraphCall(function, pool) if cuda_graphs else function
            )


def release_serving(network: Cosmos3Network) -> None:
    """Release inference graph ownership after all stream work has completed.

    Removing instance overrides also breaks the bound-method cycle,
    allowing a closed worker to release weights without waiting for GC.
    """
    for layer in network.layers:
        for name in layer.compile_methods:
            if name in vars(layer):
                delattr(layer, name)

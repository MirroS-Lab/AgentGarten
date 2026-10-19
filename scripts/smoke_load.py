"""Smoke test: load the released renderer on the GPU and build a stream.

Needs no input data. It confirms that the checkpoint loads, that serving can be
prepared on this GPU, and reports the memory used. It does not run a forward
pass: that needs real anchor and depth/normal condition latents, and the Triton
kernel JIT and CUDA graph capture happen on the first such pass.

Run from the repository root (any CUDA GPU with room for the student and the
text tower):

    python scripts/smoke_load.py [weights/AgentGarten-renderer]
"""

import argparse
import time
from pathlib import Path

import torch

from wm.inference.cosmos3 import inspect_artifact, load_artifact
from wm.inference.serving import prepare_serving
from wm.networks.cosmos3.streaming import Cosmos3Stream, StreamPolicy


def gib(nbytes: int) -> str:
    return f"{nbytes / 2**30:.1f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=Path("weights/AgentGarten-renderer"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU is visible")
    print(
        f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
        f"{torch.cuda.get_device_name(0)}"
    )
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory free before loading: {gib(free)} of {gib(total)}")

    spec = inspect_artifact(args.checkpoint)
    print(
        f"checkpoint: {spec.checkpoint}, iteration {spec.iteration}, "
        f"prompt pixel frames {spec.prompt_pixel_frames}"
    )

    start = time.perf_counter()
    artifact = load_artifact(spec)
    torch.cuda.synchronize()
    print(
        f"loaded in {time.perf_counter() - start:.1f} s, "
        f"allocated {gib(torch.cuda.memory_allocated())}"
    )

    prepare_serving(artifact.student)
    policy = StreamPolicy()
    Cosmos3Stream(artifact.student, policy)
    print(f"serving prepared, stream built with {policy}")
    print(f"peak allocated: {gib(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    main()

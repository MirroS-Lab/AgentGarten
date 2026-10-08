# SPDX-License-Identifier: Apache-2.0

"""Optional Super decoding of normalized Wan 2.2 48-channel latents."""

from pathlib import Path

import torch

from wm.codecs._taehv.taehv import TAEHV, StreamingTAEHV

__all__ = ["TAEW22SuperDecoder", "TAEW22SuperStream"]


def _validate(latent: torch.Tensor) -> None:
    if not latent.is_floating_point():
        raise TypeError("Super decoder requires floating-point latents")
    if latent.ndim != 5 or latent.shape[1] != 48 or min(latent.shape) < 1:
        raise ValueError("Super decoder expects nonempty N,48,T,H,W latents")


class TAEW22SuperDecoder:
    """Decode diffusion-normalized latents; encoding remains native Wan 2.2.

    FP16 matches the upstream inference recipe. Decode frames sequentially to
    bound intermediate activation memory independently of the clip length.
    Each full-clip call starts with empty recurrence and trims startup once.
    """

    def __init__(self, pretrained_path: str | Path) -> None:
        self.pretrained_path = str(pretrained_path)
        self._model: TAEHV | None = None

    def _get_model(self, device: torch.device) -> TAEHV:
        if self._model is None:
            self._model = (
                TAEHV(self.pretrained_path, arch_name="taew2_2_super")
                .eval()
                .requires_grad_(False)
            )
        # FP32 CPU supports portable tests; CUDA uses the upstream FP16 path.
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        return self._model.to(device=device, dtype=dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Return N,3,4T-3,16H,16W RGB in [-1,1], preserving input dtype."""
        _validate(latent)
        model = self._get_model(latent.device)
        with torch.autocast(device_type=latent.device.type, enabled=False):
            source = latent.transpose(1, 2).to(next(model.parameters()).dtype)
            # TAEHV already consumes normalized latents: no Wan mean/std here.
            video = model.decode_video(source, parallel=False, show_progress_bar=False)
            return (video.transpose(1, 2) * 2 - 1).to(latent.dtype)

    def stream(self, device: str | torch.device) -> "TAEW22SuperStream":
        """Create independent recurrence on shared frozen decoder weights."""
        return TAEW22SuperStream(self._get_model(torch.device(device)))


class TAEW22SuperStream:
    """Decode contiguous chunks, trimming three startup frames only once."""

    def __init__(self, model: TAEHV) -> None:
        self._stream = StreamingTAEHV(model)
        self._shape: tuple[int, int, int] | None = None

    def reset(self) -> None:
        """Begin an independent video with empty decoder memory."""
        self._stream.reset()
        self._shape = None

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a C0/chunk into RGB; continuation chunks emit 4T frames."""
        _validate(latent)
        weight = next(self._stream.taehv.parameters())
        if latent.device != weight.device:
            raise ValueError("stream latent device must match decoder device")
        shape = (latent.shape[0], latent.shape[3], latent.shape[4])
        if self._shape is not None and shape != self._shape:
            raise ValueError("reset the stream before changing batch or spatial shape")
        self._shape = shape
        with torch.autocast(device_type=latent.device.type, enabled=False):
            source = latent.transpose(1, 2).to(weight.dtype)
            first = self._stream.decode(source)
            frames = [first, *self._stream.flush_decoder()]
            video = torch.cat(frames, dim=1).transpose(1, 2)
            return (video * 2 - 1).to(latent.dtype)

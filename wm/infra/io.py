# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "atomic_put",
    "atomic_put_text",
    "encode_image",
    "encode_video",
    "exists",
    "get",
    "get_text",
    "join_path",
]

PathLike = str | os.PathLike[str]


def join_path(root: PathLike, *parts: PathLike) -> str:
    return str(Path(root).joinpath(*parts))


def exists(path: PathLike) -> bool:
    return Path(path).exists()


def get(path: PathLike) -> bytes:
    return Path(path).read_bytes()


def get_text(path: PathLike, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def atomic_put(payload: bytes | BytesIO, path: PathLike) -> None:
    data = payload.getvalue() if isinstance(payload, BytesIO) else bytes(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_put_text(text: str, path: PathLike, encoding: str = "utf-8") -> None:
    atomic_put(text.encode(encoding), path)


def encode_video(pixels: np.ndarray, *, fps: int, quality: int) -> BytesIO:
    import imageio

    height, width = pixels.shape[1:3]
    buffer = BytesIO()
    imageio.mimsave(
        buffer,
        pixels,
        "mp4",
        fps=int(fps),
        quality=int(quality),
        macro_block_size=1,
        ffmpeg_params=["-s", f"{width}x{height}"],
        output_params=["-f", "mp4"],
    )
    buffer.seek(0)
    return buffer


def encode_image(pixels: np.ndarray, *, file_format: str, **options: Any) -> BytesIO:
    from PIL import Image

    buffer = BytesIO()
    pil_format = "JPEG" if file_format in {"jpg", "jpeg"} else file_format.upper()
    Image.fromarray(pixels).save(buffer, format=pil_format, **options)
    buffer.seek(0)
    return buffer

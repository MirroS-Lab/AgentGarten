# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.2 VAE (the Cosmos 3 codec): 16x spatial / 4x temporal compression,
48 latent channels, causal chunked encoding.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import assert_never

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from wm.codecs.taew22 import TAEW22SuperDecoder
from wm.infra import log
from wm.infra.utils import prefetch

__all__ = ["Wan22VAECodec"]


CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )


class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        # compute query, key, value
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanResampleMode(StrEnum):
    NONE = "none"
    UPSAMPLE2D = "upsample2d"
    UPSAMPLE3D = "upsample3d"
    DOWNSAMPLE2D = "downsample2d"
    DOWNSAMPLE3D = "downsample3d"


# Released Wan2.2/Cosmos3 48-channel latent normalization.
WAN22_LATENTS_MEAN = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)
WAN22_LATENTS_STD = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        mode = WanResampleMode(mode)
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        match mode:
            case WanResampleMode.UPSAMPLE2D:
                self.resample = nn.Sequential(
                    Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                    nn.Conv2d(dim, dim, 3, padding=1),
                )
            case WanResampleMode.UPSAMPLE3D:
                self.resample = nn.Sequential(
                    Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                    nn.Conv2d(dim, dim, 3, padding=1),
                )
                self.time_conv = CausalConv3d(
                    dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
                )
            case WanResampleMode.DOWNSAMPLE2D:
                self.resample = nn.Sequential(
                    nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
                )
            case WanResampleMode.DOWNSAMPLE3D:
                self.resample = nn.Sequential(
                    nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
                )
                self.time_conv = CausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )
            case WanResampleMode.NONE:
                self.resample = nn.Identity()
            case _:
                assert_never(mode)

    def forward(self, x, feat_cache=None, feat_idx=None):
        b, c, t, h, w = x.size()
        if self.mode is WanResampleMode.UPSAMPLE3D:
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] != "Rep"
                    ):
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] == "Rep"
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode is WanResampleMode.DOWNSAMPLE3D:
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        b, c, t, h, w = x.shape
        x = x.view(
            b,
            c,
            t // self.factor_t,
            self.factor_t,
            h // self.factor_s,
            self.factor_s,
            w // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            b,
            c * self.factor,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )
        x = x.view(
            b,
            self.out_channels,
            self.group_size,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class DownResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        mult,
        temperal_downsample=False,
        down_flag=False,
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=None):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class UpResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        mult,
        temperal_upsample=False,
        up_flag=False,
    ):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(True, True, False),
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in (1, *dim_mult)]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i] if i < len(temperal_downsample) else False
            )
            downsamples.append(
                DownResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # output blocks
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_upsample=(False, True, True),
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in (dim_mult[-1], *dim_mult[::-1])]
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                UpResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class _Wan22VAE(nn.Module):
    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(True, True, False),
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

    def encode(self, x, scale, *, temporal_window: int):
        if (
            isinstance(temporal_window, bool)
            or not isinstance(temporal_window, int)
            or temporal_window <= 0
        ):
            raise ValueError(
                f"temporal_window must be a positive integer, got {temporal_window!r}"
            )
        if temporal_window % 4 != 0:
            raise ValueError(
                f"temporal_window must be divisible by 4, got {temporal_window}"
            )
        if x.ndim != 5:
            raise ValueError(
                f"Wan2.2 VAE input must be [B,C,T,H,W], got {tuple(x.shape)}"
            )
        original_frames = int(x.shape[2])
        if original_frames != 1 and (original_frames - 1) % 4 != 0:
            raise ValueError(
                f"Wan2.2 VAE temporal length must be 1 or 4n+1, got {original_frames}"
            )
        latent_frames = 1 + (original_frames - 1) // 4
        padded_frames = (
            1
            + ((original_frames - 1 + temporal_window - 1) // temporal_window)
            * temporal_window
        )
        if padded_frames != original_frames:
            x = F.pad(x, (0, 0, 0, 0, 0, padded_frames - original_frames))

        self.clear_cache()
        x = patchify(x, patch_size=2)
        outputs = []
        for start in (0, *range(1, padded_frames, temporal_window)):
            self._enc_conv_idx = [0]
            chunk = (
                x[:, :, :1] if start == 0 else x[:, :, start : start + temporal_window]
            )
            encoded = self.encoder(
                chunk,
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            mu, _log_var = self.conv1(encoded).chunk(2, dim=1)
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1
                )
            else:
                mu = (mu - scale[0]) * scale[1]
            outputs.append(mu)

        result = torch.cat(outputs, dim=2)[:, :, :latent_frames]
        self.clear_cache()
        return result

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        outputs = []
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                decoded = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=True,
                )
            else:
                decoded = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
            outputs.append(decoded)
        out = torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
        out = unpatchify(out, patch_size=2)
        self.clear_cache()
        return out

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(
    pretrained_path=None,
    z_dim=48,
    dim=160,
    device="cpu",
    **kwargs,
):
    # params
    cfg = dict(
        dim=dim,
        z_dim=z_dim,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True),
        dropout=0.0,
    )
    cfg.update(**kwargs)

    # init model
    with torch.device("meta"):
        model = _Wan22VAE(**cfg)

    if pretrained_path is None:
        model.to_empty(device=device)
    else:
        prefetch(pretrained_path)
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        # The codec is lazily materialized by the current context-parallel
        # batch owner.  Loading is consequently rank-local: a WORLD
        # collective here would wait for ranks that are receiving the
        # prepared batch instead of entering the codec.
        log.info(f"loading {pretrained_path}")
        model.load_state_dict(ckpt, assign=True)

    return model


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_ENCODE_CHUNK_FRAMES = {
    "256": 68,
    "480": 24,
    "720": 12,
}


class Wan22VAECodec:
    def __init__(
        self,
        pretrained_path: str | Path,
        *,
        dtype: torch.dtype | str = "bfloat16",
        device: torch.device | str | None = None,
        temporal_compression_factor: int = 4,
        decoder: TAEW22SuperDecoder | None = None,
    ) -> None:
        self.pretrained_path = str(pretrained_path)
        self.decoder = decoder
        self.dtype = _DTYPES[dtype] if isinstance(dtype, str) else dtype
        self.preferred_device = None if device is None else torch.device(device)
        self.temporal_compression_factor = int(temporal_compression_factor)
        self._vae: nn.Module | None = None
        self._scale: tuple[torch.Tensor, torch.Tensor] | None = None

    def _get_vae(self, device: torch.device | str | None = None) -> nn.Module:
        target_device = (
            self.preferred_device if device is None else torch.device(device)
        )
        if self._vae is None:
            self._vae = _video_vae(
                pretrained_path=self.pretrained_path,
                device=str(target_device or "cpu"),
            )
            self._vae = self._vae.eval().requires_grad_(False).to(dtype=self.dtype)
        return self._vae

    def _normalization(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._scale is None or self._scale[0].device != device:
            mean = torch.tensor(WAN22_LATENTS_MEAN, dtype=self.dtype, device=device)
            standard_deviation = torch.tensor(
                WAN22_LATENTS_STD,
                dtype=self.dtype,
                device=device,
            )
            self._scale = mean, 1.0 / standard_deviation
        return self._scale

    def stream_components(
        self, device: torch.device | str
    ) -> tuple[nn.Module, tuple[torch.Tensor, torch.Tensor]]:
        """Share the native Wan VAE and normalization with an online codec.

        This always returns Wan for encoding. An optional Super decoder owns
        separate recurrence; use ``decoder.stream(device)`` for RGB output.

        Streaming callers own separate encoder/decoder recurrence. They must
        serialize use of the shared modules with full-clip encode/decode calls.
        """
        device = torch.device(device)
        return self._get_vae(device), self._normalization(device)

    @staticmethod
    def _encode_window(spatial_shape: tuple[int, int]) -> int:
        height, width = (int(value) for value in spatial_shape)
        min_dimension = min(height, width)
        resolution = (
            "256" if min_dimension <= 256 else "480" if min_dimension <= 640 else "720"
        )
        return _ENCODE_CHUNK_FRAMES[resolution]

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        if not video.is_floating_point():
            raise TypeError("Wan 2.2 VAE input must be floating point")
        input_dtype = video.dtype
        vae = self._get_vae(video.device)
        with torch.autocast(device_type=video.device.type, enabled=False):
            source = video.to(self.dtype)
            scale = self._normalization(video.device)
            temporal_window = self._encode_window(video.shape[-2:])
            latent = vae.encode(
                source,
                scale,
                temporal_window=temporal_window,
            )
        return latent.to(input_dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if not latent.is_floating_point():
            raise TypeError("Wan 2.2 VAE latent must be floating point")
        if self.decoder is not None:
            return self.decoder.decode(latent)
        input_dtype = latent.dtype
        vae = self._get_vae(latent.device)
        with torch.autocast(device_type=latent.device.type, enabled=False):
            video = vae.decode(
                latent.to(self.dtype),
                self._normalization(latent.device),
            )
        return video.to(input_dtype)

    @property
    def spatial_compression_factor(self) -> int:
        return 16

    @property
    def latent_channels(self) -> int:
        return 48

    def latent_frames(self, pixel_frames: int) -> int:
        return (int(pixel_frames) - 1) // self.temporal_compression_factor + 1

    def pixel_frames(self, latent_frames: int) -> int:
        return (int(latent_frames) - 1) * self.temporal_compression_factor + 1

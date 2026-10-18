"""Normalization helpers used by lat_dit / pcd_dit / pcd_vae."""

from __future__ import annotations

import numpy as np
import torch


def resample(x, f=(1, 3, 3, 1), mode="keep"):
    if mode == "keep":
        return x

    f = torch.tensor(f, dtype=x.dtype, device=x.device)
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    filt = f[None, None, :]

    dims = x.ndim - 2
    for _ in range(dims - 1):
        filt = torch.matmul(filt[..., None], f[None, :])

    c = x.shape[1]
    filt = filt.repeat_interleave(c, dim=0)

    if mode == "down":
        if dims == 2:
            x = torch.nn.functional.conv2d(x, filt, groups=c, stride=2, padding=(pad,))
        if dims == 3:
            x = torch.nn.functional.conv3d(x, filt, groups=c, stride=2, padding=(pad,))
    if mode == "up":
        if dims == 2:
            x = torch.nn.functional.conv_transpose2d(
                x, filt * 2**dims, groups=c, stride=2, padding=(pad,)
            )
        if dims == 3:
            x = torch.nn.functional.conv_transpose3d(
                x, filt * 2**dims, groups=c, stride=2, padding=(pad,)
            )
    return x


def layer_norm(x, eps=1e-4):
    xcenter = x - x.mean(dim=-1, keepdim=True, dtype=torch.float32).type_as(x)
    xnrm = xcenter / (
        torch.linalg.vector_norm(xcenter, dim=-1, dtype=torch.float32, keepdim=True)
        / np.sqrt(x.shape[-1])
        + eps
    ).type_as(x)
    return xnrm


def rms_norm(x, eps=1e-4):
    return x / (
        torch.linalg.vector_norm(x, dim=-1, dtype=torch.float32, keepdim=True)
        / np.sqrt(x.shape[-1])
        + eps
    ).type_as(x)


class LayerNorm(torch.nn.Module):

    def __init__(self, hidden_dim, learnable=True):
        super(LayerNorm, self).__init__()
        self.learnable = learnable
        if self.learnable:
            self.weight = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
        else:
            self.weight = self.bias = None

    def forward(self, x):
        if not self.learnable:
            return layer_norm(x)
        return layer_norm(x) * (1 + self.weight.type_as(x)) + self.bias.type_as(x)


class ScaleNorm(torch.nn.Module):

    def __init__(self, hidden_dim, learnable=True):
        super(ScaleNorm, self).__init__()
        self.learnable = learnable
        if self.learnable:
            self.weight = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, x):
        if not self.learnable:
            return rms_norm(x)
        return rms_norm(x) * (1 + self.weight.type_as(x))


class PixelNorm(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(PixelNorm, self).__init__()
        self.shape = (hidden_dim,)
        self.weight = torch.nn.Parameter(torch.zeros(self.shape, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(self.shape, dtype=torch.float32))

    def forward(self, x):
        return (
            layer_norm(x.movedim(1, -1)) * (1 + self.weight.type_as(x))
            + self.bias.type_as(x)
        ).movedim(-1, 1)


class QKNorm(torch.nn.Module):
    def __init__(self, dim):
        super(QKNorm, self).__init__()
        self.shape = (dim,)
        self.q_scale = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.k_scale = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, q, k):
        q = rms_norm(q) * (1 + self.q_scale.type_as(q))
        k = rms_norm(k) * (1 + self.k_scale.type_as(k))
        return q, k

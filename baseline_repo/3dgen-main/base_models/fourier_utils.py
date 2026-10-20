"""Positional / time Fourier encodings used by lat_dit, lat_vae, and pcd_*."""

from __future__ import annotations

import numpy as np
import torch


def aslist(obj):
    return obj if isinstance(obj, list) or obj is None else [obj]


@torch.amp.autocast("cuda", enabled=False)
def get_positional_encoding(x, features, scale=16, include_pi=False, logspace=True):
    dim = x.shape[-1]
    coord_dim = features // dim // 2
    powers = torch.arange(0, coord_dim, device=x.device, dtype=torch.float64) / coord_dim
    if logspace:
        omega = scale ** powers
    else:
        omega = scale * powers
    h = x[..., None] * omega
    if include_pi:
        h = h * torch.pi
    h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)
    h = h.view(*h.shape[:-2], h.shape[-2] * h.shape[-1])  # dim, p -> dim * p
    if h.shape[-1] < features:
        h = torch.nn.functional.pad(h, pad=(0, features - h.shape[-1]), value=0)
    return h.float()


@torch.compile()
@torch.amp.autocast("cuda", enabled=False)
def get_time_encoding(t, dim, max_period=10000):
    omega = torch.exp(
        -np.log(max_period)
        * torch.arange(0, dim, 2, dtype=torch.float64, device=t.device)
        / dim
    )
    freqs = t[..., None] * omega
    out = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    if out.shape[-1] < dim:
        out = torch.nn.functional.pad(out, pad=(0, dim - out.shape[-1]), value=0)
    return out.float()

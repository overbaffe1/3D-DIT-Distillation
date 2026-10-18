"""High-performance fused Triton/CUDA kernels and precomputed lookup tables for LatDiT."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

_FLASH_ATTN = None
_FLASH_ATTN_3 = None
_TRITON_LN = None
_TRITON_RMS = None
_SWIGLU_FN = None
_LIBSPARSE = None
_CUROPE = None

# Cache of dummy ones tensors keyed by (device, dtype, dim) to avoid per-call allocations
_ONES_CACHE: Dict[Tuple[torch.device, torch.dtype, int], torch.Tensor] = {}


def get_cached_ones(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype, dim)
    t = _ONES_CACHE.get(key)
    if t is None:
        t = torch.ones(dim, device=device, dtype=dtype)
        _ONES_CACHE[key] = t
    return t


def preload_extensions(verbose: bool = False) -> None:
    """Pre-load CUDA/Triton extensions at model load time so zero JIT happens during generate()."""
    global _FLASH_ATTN, _FLASH_ATTN_3, _TRITON_LN, _TRITON_RMS, _SWIGLU_FN, _LIBSPARSE, _CUROPE

    if not torch.cuda.is_available():
        return

    try:
        import flash_attn_interface

        _FLASH_ATTN_3 = flash_attn_interface
    except ImportError:
        pass

    try:
        import flash_attn

        _FLASH_ATTN = flash_attn
        from flash_attn.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn
        from flash_attn.ops.activations import swiglu

        _TRITON_LN = layer_norm_fn
        _TRITON_RMS = rms_norm_fn
        _SWIGLU_FN = swiglu
    except ImportError:
        pass

    try:
        import ops.sparse as ops_sparse

        ops_sparse._init(verbose=verbose)
        _LIBSPARSE = ops_sparse._libsparse
    except Exception:
        pass

    try:
        import ops.rope as ops_rope

        ops_rope._init()
        _CUROPE = ops_rope._curope
    except Exception:
        pass


def fast_layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """LayerNorm without per-call torch.ones allocation; fuses scale & shift directly into Triton."""
    if x.is_cuda and _TRITON_LN is not None:
        w = weight if weight is not None else get_cached_ones(x.shape[-1], x.device, x.dtype)
        return _TRITON_LN(x, w, bias, eps=eps)
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    out = (xf - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def fast_modulate(
    x: torch.Tensor,
    sizes: List[int],
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused LayerNorm + affine modulation (1 + scale) * LN(x) + shift."""
    if len(sizes) == 1:
        w = 1.0 + scale.view(-1)
        b = shift.view(-1)
        return fast_layer_norm(x, w, b, eps=eps)
    return torch.cat(
        [
            fast_layer_norm(hh, 1.0 + scale[idx].view(-1), shift[idx].view(-1), eps=eps)
            for idx, hh in enumerate(x.split(sizes))
        ],
        dim=0,
    )


def fast_modulate_cond(
    cond: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused LayerNorm + affine modulation for [B, L, D] conditioning tensor."""
    if cond.shape[0] == 1:
        return fast_layer_norm(cond, 1.0 + scale.view(-1), shift.view(-1), eps=eps)
    return fast_layer_norm(cond, eps=eps) * (1.0 + scale[:, None]) + shift[:, None]


def fast_gate(x: torch.Tensor, sizes: List[int], scale: torch.Tensor) -> torch.Tensor:
    if len(sizes) == 1:
        return x * scale.view(1, -1).type_as(x)
    return torch.cat([hh * scale[idx].type_as(hh) for idx, hh in enumerate(x.split(sizes))], dim=0)


def fast_rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if x.is_cuda and _TRITON_RMS is not None:
        w = weight if weight is not None else get_cached_ones(x.shape[-1], x.device, x.dtype)
        return _TRITON_RMS(x, w, bias=None, eps=eps)
    xf = x.float()
    rrms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = (xf * rrms).to(x.dtype)
    if weight is not None:
        out = out * weight.to(x.dtype)
    return out


def fast_swiglu(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and _SWIGLU_FN is not None:
        return _SWIGLU_FN(gate, x)
    return (F.silu(gate.float()) * x.float()).to(x.dtype)


def precompute_rope_3d_table(
    res: int = 64,
    base: float = 128.0,
    head_dim: int = 96,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute 1D coordinate lookup table [res, Q] for 3D RoPE (DIM=3, Q=head_dim//6=16).
    Matches ops/curope/rope.cu FP64 frequency calculation.
    """
    Q = head_dim // 6
    coords_1d = (torch.arange(res, dtype=torch.float64, device=device) + 0.5) / float(res) * 2.0 - 1.0
    inv_freq = float(base) ** (-torch.arange(Q, dtype=torch.float64, device=device) / float(Q))
    freqs = coords_1d.unsqueeze(-1) * inv_freq.unsqueeze(0)  # [res, Q]
    cos_table = torch.cos(freqs).to(torch.float32).contiguous()
    sin_table = torch.sin(freqs).to(torch.float32).contiguous()
    return cos_table, sin_table


def lookup_rope_3d(
    voxel_xyz: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given integer voxel coordinates [N, 3] in [0, res-1], returns (cos_vox, sin_vox)
    of shape [N, 1, 3, Q] (ready for broadcasting over H heads).
    """
    idx = voxel_xyz.long().clamp(0, cos_table.shape[0] - 1)
    cos_vox = cos_table[idx].unsqueeze(1).contiguous()  # [N, 1, 3, Q]
    sin_vox = sin_table[idx].unsqueeze(1).contiguous()  # [N, 1, 3, Q]
    return cos_vox, sin_vox


def precompute_rope_2d_cond(
    cond_size: int = 64,
    base: float = 64.0,
    head_dim: int = 96,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute 2D RoPE (cos, sin) of shape [1, cond_size*cond_size, 1, 2, Q] (Q=24)
    for the DINOv3 spatial grid. Matches ops/curope/rope.cu FP64 math.
    """
    Q = head_dim // 4
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(cond_size, device=device, dtype=torch.long),
            torch.arange(cond_size, device=device, dtype=torch.long),
            indexing="ij",
        ),
        dim=-1,
    ).view(1, -1, 2)
    cond_pos = (grid.to(torch.float64) + 0.5) / float(cond_size) * 2.0 - 1.0
    inv_freq = float(base) ** (-torch.arange(Q, dtype=torch.float64, device=device) / float(Q))
    freq = cond_pos.unsqueeze(2).unsqueeze(-1) * inv_freq.view(1, 1, 1, 1, Q)
    cos_cond = torch.cos(freq).to(torch.float32).contiguous()  # [1, 4096, 1, 2, 24]
    sin_cond = torch.sin(freq).to(torch.float32).contiguous()  # [1, 4096, 1, 2, 24]
    return cos_cond, sin_cond


def apply_rope_cached_3d(
    tokens: torch.Tensor,
    cos_vox: torch.Tensor,
    sin_vox: torch.Tensor,
) -> torch.Tensor:
    """Apply 3D RoPE using precomputed (cos_vox, sin_vox) [N, 1, 3, 16] to tokens [N, H, 96]."""
    N, H, D = tokens.shape
    Q = cos_vox.shape[-1]
    tok = tokens.float().view(N, H, 3, 2, Q)
    u = tok[..., 0, :]
    v = tok[..., 1, :]
    out_u = u * cos_vox - v * sin_vox
    out_v = v * cos_vox + u * sin_vox
    return torch.stack([out_u, out_v], dim=-2).reshape(N, H, D).to(tokens.dtype)


def apply_rope_cached_2d(
    tokens: torch.Tensor,
    cos_cond: torch.Tensor,
    sin_cond: torch.Tensor,
) -> torch.Tensor:
    """Apply 2D RoPE using precomputed (cos_cond, sin_cond) [1, L_spatial, 1, 2, 24] to tokens [B, L_spatial, H, 96]."""
    B, L, H, D = tokens.shape
    Q = cos_cond.shape[-1]
    tok = tokens.float().view(B, L, H, 2, 2, Q)
    u = tok[..., 0, :]
    v = tok[..., 1, :]
    out_u = u * cos_cond - v * sin_cond
    out_v = v * cos_cond + u * sin_cond
    return torch.stack([out_u, out_v], dim=-2).reshape(B, L, H, D).to(tokens.dtype)


if HAS_TRITON:

    @triton.jit
    def _fused_qknorm_rope3d_gather_kernel(
        qkv_ptr,
        q_scale_ptr,
        k_scale_ptr,
        cos_ptr,
        sin_ptr,
        fwd_idx_ptr,
        out_qkv_ptr,
        N,
        H: tl.constexpr,
        Q: tl.constexpr,
        eps: tl.constexpr,
    ):
        """
        Fuses for 3D voxels (D = 6 * Q = 96, H = 16):
          1. Gather from `src_n = fwd_idx[dst_n]`
          2. RMSNorm(q) * q_scale, RMSNorm(k) * k_scale
          3. 3D RoPE rotation on q and k using precomputed cos/sin [N, 3, Q]
          4. Copy v and write packed [N, 3, H, D] directly in Hilbert-sorted order!
        """
        dst_n = tl.program_id(0)
        h = tl.program_id(1)
        src_n = tl.load(fwd_idx_ptr + dst_n)

        offs = tl.arange(0, Q)
        D: tl.constexpr = 6 * Q
        stride_n = 3 * H * D
        stride_3 = H * D
        stride_h = D

        # Load cos/sin for src_n (3 axes * Q)
        cos_base = cos_ptr + src_n * (3 * Q)
        sin_base = sin_ptr + src_n * (3 * Q)
        c0 = tl.load(cos_base + 0 * Q + offs).to(tl.float32)
        c1 = tl.load(cos_base + 1 * Q + offs).to(tl.float32)
        c2 = tl.load(cos_base + 2 * Q + offs).to(tl.float32)
        s0 = tl.load(sin_base + 0 * Q + offs).to(tl.float32)
        s1 = tl.load(sin_base + 1 * Q + offs).to(tl.float32)
        s2 = tl.load(sin_base + 2 * Q + offs).to(tl.float32)

        # --- Process Q (idx 0) ---
        q_base = qkv_ptr + src_n * stride_n + 0 * stride_3 + h * stride_h
        qs_base = q_scale_ptr + h * D
        qu0 = tl.load(q_base + 0 * Q + offs)
        qv0 = tl.load(q_base + 1 * Q + offs)
        qu1 = tl.load(q_base + 2 * Q + offs)
        qv1 = tl.load(q_base + 3 * Q + offs)
        qu2 = tl.load(q_base + 4 * Q + offs)
        qv2 = tl.load(q_base + 5 * Q + offs)

        qu0_f, qv0_f = qu0.to(tl.float32), qv0.to(tl.float32)
        qu1_f, qv1_f = qu1.to(tl.float32), qv1.to(tl.float32)
        qu2_f, qv2_f = qu2.to(tl.float32), qv2.to(tl.float32)

        q_sq = tl.sum(
            qu0_f * qu0_f + qv0_f * qv0_f + qu1_f * qu1_f + qv1_f * qv1_f + qu2_f * qu2_f + qv2_f * qv2_f,
            axis=0,
        )
        q_rrms = tl.rsqrt(q_sq * (1.0 / 96.0) + eps)

        qsu0 = tl.load(qs_base + 0 * Q + offs).to(qu0.dtype).to(tl.float32)
        qsv0 = tl.load(qs_base + 1 * Q + offs).to(qu0.dtype).to(tl.float32)
        qsu1 = tl.load(qs_base + 2 * Q + offs).to(qu0.dtype).to(tl.float32)
        qsv1 = tl.load(qs_base + 3 * Q + offs).to(qu0.dtype).to(tl.float32)
        qsu2 = tl.load(qs_base + 4 * Q + offs).to(qu0.dtype).to(tl.float32)
        qsv2 = tl.load(qs_base + 5 * Q + offs).to(qu0.dtype).to(tl.float32)

        qu0_n = (qu0_f * q_rrms).to(qu0.dtype).to(tl.float32) * qsu0
        qv0_n = (qv0_f * q_rrms).to(qu0.dtype).to(tl.float32) * qsv0
        qu1_n = (qu1_f * q_rrms).to(qu0.dtype).to(tl.float32) * qsu1
        qv1_n = (qv1_f * q_rrms).to(qu0.dtype).to(tl.float32) * qsv1
        qu2_n = (qu2_f * q_rrms).to(qu0.dtype).to(tl.float32) * qsu2
        qv2_n = (qv2_f * q_rrms).to(qu0.dtype).to(tl.float32) * qsv2

        out_q_base = out_qkv_ptr + dst_n * stride_n + 0 * stride_3 + h * stride_h
        tl.store(out_q_base + 0 * Q + offs, (qu0_n * c0 - qv0_n * s0).to(qu0.dtype))
        tl.store(out_q_base + 1 * Q + offs, (qv0_n * c0 + qu0_n * s0).to(qu0.dtype))
        tl.store(out_q_base + 2 * Q + offs, (qu1_n * c1 - qv1_n * s1).to(qu0.dtype))
        tl.store(out_q_base + 3 * Q + offs, (qv1_n * c1 + qu1_n * s1).to(qu0.dtype))
        tl.store(out_q_base + 4 * Q + offs, (qu2_n * c2 - qv2_n * s2).to(qu0.dtype))
        tl.store(out_q_base + 5 * Q + offs, (qv2_n * c2 + qu2_n * s2).to(qu0.dtype))

        # --- Process K (idx 1) ---
        k_base = qkv_ptr + src_n * stride_n + 1 * stride_3 + h * stride_h
        ks_base = k_scale_ptr + h * D
        ku0 = tl.load(k_base + 0 * Q + offs)
        kv0 = tl.load(k_base + 1 * Q + offs)
        ku1 = tl.load(k_base + 2 * Q + offs)
        kv1 = tl.load(k_base + 3 * Q + offs)
        ku2 = tl.load(k_base + 4 * Q + offs)
        kv2 = tl.load(k_base + 5 * Q + offs)

        ku0_f, kv0_f = ku0.to(tl.float32), kv0.to(tl.float32)
        ku1_f, kv1_f = ku1.to(tl.float32), kv1.to(tl.float32)
        ku2_f, kv2_f = ku2.to(tl.float32), kv2.to(tl.float32)

        k_sq = tl.sum(
            ku0_f * ku0_f + kv0_f * kv0_f + ku1_f * ku1_f + kv1_f * kv1_f + ku2_f * ku2_f + kv2_f * kv2_f,
            axis=0,
        )
        k_rrms = tl.rsqrt(k_sq * (1.0 / 96.0) + eps)

        ksu0 = tl.load(ks_base + 0 * Q + offs).to(ku0.dtype).to(tl.float32)
        ksv0 = tl.load(ks_base + 1 * Q + offs).to(ku0.dtype).to(tl.float32)
        ksu1 = tl.load(ks_base + 2 * Q + offs).to(ku0.dtype).to(tl.float32)
        ksv1 = tl.load(ks_base + 3 * Q + offs).to(ku0.dtype).to(tl.float32)
        ksu2 = tl.load(ks_base + 4 * Q + offs).to(ku0.dtype).to(tl.float32)
        ksv2 = tl.load(ks_base + 5 * Q + offs).to(ku0.dtype).to(tl.float32)

        ku0_n = (ku0_f * k_rrms).to(ku0.dtype).to(tl.float32) * ksu0
        kv0_n = (kv0_f * k_rrms).to(ku0.dtype).to(tl.float32) * ksv0
        ku1_n = (ku1_f * k_rrms).to(ku0.dtype).to(tl.float32) * ksu1
        kv1_n = (kv1_f * k_rrms).to(ku0.dtype).to(tl.float32) * ksv1
        ku2_n = (ku2_f * k_rrms).to(ku0.dtype).to(tl.float32) * ksu2
        kv2_n = (kv2_f * k_rrms).to(ku0.dtype).to(tl.float32) * ksv2

        out_k_base = out_qkv_ptr + dst_n * stride_n + 1 * stride_3 + h * stride_h
        tl.store(out_k_base + 0 * Q + offs, (ku0_n * c0 - kv0_n * s0).to(ku0.dtype))
        tl.store(out_k_base + 1 * Q + offs, (kv0_n * c0 + ku0_n * s0).to(ku0.dtype))
        tl.store(out_k_base + 2 * Q + offs, (ku1_n * c1 - kv1_n * s1).to(ku0.dtype))
        tl.store(out_k_base + 3 * Q + offs, (kv1_n * c1 + ku1_n * s1).to(ku0.dtype))
        tl.store(out_k_base + 4 * Q + offs, (ku2_n * c2 - kv2_n * s2).to(ku0.dtype))
        tl.store(out_k_base + 5 * Q + offs, (kv2_n * c2 + ku2_n * s2).to(ku0.dtype))

        # --- Process V (idx 2): direct gather ---
        v_base = qkv_ptr + src_n * stride_n + 2 * stride_3 + h * stride_h
        out_v_base = out_qkv_ptr + dst_n * stride_n + 2 * stride_3 + h * stride_h
        tl.store(out_v_base + 0 * Q + offs, tl.load(v_base + 0 * Q + offs))
        tl.store(out_v_base + 1 * Q + offs, tl.load(v_base + 1 * Q + offs))
        tl.store(out_v_base + 2 * Q + offs, tl.load(v_base + 2 * Q + offs))
        tl.store(out_v_base + 3 * Q + offs, tl.load(v_base + 3 * Q + offs))
        tl.store(out_v_base + 4 * Q + offs, tl.load(v_base + 4 * Q + offs))
        tl.store(out_v_base + 5 * Q + offs, tl.load(v_base + 5 * Q + offs))


def fused_qknorm_rope3d_gather(
    qkv: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cos_vox: torch.Tensor,
    sin_vox: torch.Tensor,
    fwd_indices: torch.Tensor,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Input:
      qkv: [N, 3, H, D] (H=16, D=96)
      q_scale, k_scale: [H, D]
      cos_vox, sin_vox: [N, 1, 3, 16]
      fwd_indices: [N]
    Output:
      qkv_sorted: [N, 3, H, D] gathered in fwd_indices order with QKNorm + 3D RoPE applied.
    """
    N, _, H, D = qkv.shape
    if use_triton and HAS_TRITON and qkv.is_cuda and D == 96:
        out_qkv = torch.empty_like(qkv)
        _fused_qknorm_rope3d_gather_kernel[(N, H)](
            qkv,
            q_scale,
            k_scale,
            cos_vox,
            sin_vox,
            fwd_indices,
            out_qkv,
            N=N,
            H=H,
            Q=16,
            eps=1e-6,
            num_warps=1,
        )
        return out_qkv

    q, k, v = qkv.unbind(dim=-3)
    q = fast_rms_norm(q) * q_scale.to(q.dtype)
    k = fast_rms_norm(k) * k_scale.to(k.dtype)
    q = apply_rope_cached_3d(q, cos_vox, sin_vox)
    k = apply_rope_cached_3d(k, cos_vox, sin_vox)
    return torch.stack([q, k, v], dim=-3)[fwd_indices]


def flash_varlen_qkvpacked_already_sorted(
    qkv_sorted: torch.Tensor,
    bwd_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Run FlashAttention varlen on pre-sorted qkv_sorted [N, 3, H, D] and unpermute by bwd_indices."""
    if qkv_sorted.is_cuda:
        if _FLASH_ATTN_3 is not None:
            q, k, v = qkv_sorted.unbind(dim=-3)
            out = _FLASH_ATTN_3.flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
            )
            return out[bwd_indices]
        if _FLASH_ATTN is not None:
            out = _FLASH_ATTN.flash_attn_varlen_qkvpacked_func(
                qkv_sorted, cu_seqlens, max_seqlen
            )
            return out[bwd_indices]

    # CPU / fallback path using PyTorch SDPA per window chunk
    q_s, k_s, v_s = qkv_sorted.unbind(dim=-3)
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    outs = []
    start = 0
    for sl in seqlens:
        end = start + sl
        out_c = F.scaled_dot_product_attention(
            q_s[start:end].unsqueeze(0).transpose(1, 2),
            k_s[start:end].unsqueeze(0).transpose(1, 2),
            v_s[start:end].unsqueeze(0).transpose(1, 2),
        ).transpose(1, 2).squeeze(0)
        outs.append(out_c)
        start = end
    return torch.cat(outs, dim=0)[bwd_indices]


def flash_cond_self_attn(qkv_cond: torch.Tensor) -> torch.Tensor:
    """Run FlashAttention on [B, L, 3, H, D]."""
    if qkv_cond.is_cuda:
        if _FLASH_ATTN_3 is not None:
            return _FLASH_ATTN_3.flash_attn_qkvpacked_func(qkv_cond)
        if _FLASH_ATTN is not None:
            return _FLASH_ATTN.flash_attn_qkvpacked_func(qkv_cond)
    q, k, v = qkv_cond.unbind(dim=-3)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


def flash_cross_attn(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    kv: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_kv: int,
) -> torch.Tensor:
    """Run FlashAttention cross-attention between q [T_q, H, D] and kv [T_kv, 2, H, D]."""
    if q.is_cuda:
        if _FLASH_ATTN_3 is not None:
            k, v = kv.unbind(dim=-3)
            return _FLASH_ATTN_3.flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
            )
        if _FLASH_ATTN is not None:
            return _FLASH_ATTN.flash_attn_varlen_kvpacked_func(
                q, kv, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
            )
    k, v = kv.unbind(dim=-3)
    return F.scaled_dot_product_attention(
        q.unsqueeze(0).transpose(1, 2),
        k.unsqueeze(0).transpose(1, 2),
        v.unsqueeze(0).transpose(1, 2),
    ).transpose(1, 2).squeeze(0)


def build_conv_indices_fast(coords: torch.Tensor) -> List[torch.Tensor]:
    """Compute 3D submanifold 3x3x3 neighbor indices once per sample."""
    if coords.is_cuda:
        if _LIBSPARSE is None:
            import ops.sparse as ops_sparse

            ops_sparse._init(verbose=False)
            return ops_sparse.get_conv_indices(coords)
        return _LIBSPARSE.conv_indices(coords.int().contiguous())

    # CPU fallback
    N = coords.shape[0]
    b = coords[:, 0].to(torch.int64)
    x = coords[:, 1].to(torch.int64)
    y = coords[:, 2].to(torch.int64)
    z = coords[:, 3].to(torch.int64)
    keys = (b << 36) | (x << 24) | (y << 12) | z
    order = torch.argsort(keys)
    sorted_keys = keys[order]

    out = torch.full((N, 27), -1, dtype=torch.int32, device=coords.device)
    d = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                xx = x + dx
                yy = y + dy
                zz = z + dz
                valid = (xx >= 0) & (yy >= 0) & (zz >= 0)
                qkeys = (b << 36) | (xx.clamp(min=0) << 24) | (yy.clamp(min=0) << 12) | zz.clamp(min=0)
                idx = torch.searchsorted(sorted_keys, qkeys)
                idx_clamped = idx.clamp(max=N - 1)
                found = valid & (idx < N) & (sorted_keys[idx_clamped] == qkeys)
                out[found, d] = order[idx_clamped[found]].to(torch.int32)
                d += 1
    return [out]


def fast_sparse_conv3d(
    feat: torch.Tensor,
    weight_fp32: torch.Tensor,
    bias_fp32: torch.Tensor,
    conv_indices: List[torch.Tensor],
) -> torch.Tensor:
    """Direct inference call to _libsparse.conv_fwd without autograd overhead or redundant index build."""
    if feat.is_cuda:
        if _LIBSPARSE is not None:
            return _LIBSPARSE.conv_fwd(feat.contiguous(), weight_fp32, bias_fp32, conv_indices)
        import ops.sparse as ops_sparse

        ops_sparse._init(verbose=False)
        return ops_sparse._libsparse.conv_fwd(feat.contiguous(), weight_fp32, bias_fp32, conv_indices)

    # CPU fallback
    fwd_indices = conv_indices[0]
    N, C = feat.shape
    acc = bias_fp32.unsqueeze(0).expand(N, C).clone()
    w_flat = weight_fp32.view(27, C)
    feat_f = feat.float()
    for d in range(27):
        idx_d = fwd_indices[:, d]
        mask_d = idx_d >= 0
        if mask_d.any():
            acc[mask_d] += feat_f[idx_d[mask_d]] * w_flat[d]
    return acc.to(feat.dtype)

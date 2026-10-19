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


def install_cpu_fallbacks() -> None:
    """
    When running on a CPU-only machine (e.g., CI testing without CUDA/flash_attn),
    provide exact PyTorch CPU fallbacks for `base_models.order`, `ops.sparse`,
    and `flash_attn` so both baseline `LatDitModel` and `Solution` can execute on CPU.
    Does NOT modify anything when CUDA is available.
    """
    if torch.cuda.is_available():
        return

    import importlib.machinery
    import importlib.util
    import sys
    import types

    class _DummyModule(types.ModuleType):
        def __getattr__(self, name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            val = type(name, (), {})
            setattr(self, name, val)
            return val

    for mod_name in (
        "spconv",
        "spconv.pytorch",
        "nvdiffrast",
        "nvdiffrast.torch",
        "torch_scatter",
        "torch_cluster",
        "igl",
        "mapbox_earcut",
        "sklearn",
        "sklearn.neighbors",
        "torchvision",
        "torchvision.transforms",
    ):
        if mod_name not in sys.modules:
            top = mod_name.split(".")[0]
            try:
                spec = importlib.util.find_spec(top) if top not in sys.modules else getattr(sys.modules[top], "__spec__", None)
            except Exception:
                spec = None
            if spec is None or isinstance(sys.modules.get(top), _DummyModule):
                m = _DummyModule(mod_name)
                m.__path__ = []
                m.__spec__ = importlib.machinery.ModuleSpec(mod_name, None, is_package=True)
                sys.modules[mod_name] = m

    class _StubSparseConvTensor:
        def __init__(
            self,
            features,
            indices,
            spatial_shape=None,
            batch_size=None,
            grid=None,
            voxel_num=None,
            indice_dict=None,
            benchmark=False,
            **kwargs,
        ):
            self._features = features
            self.indices = indices
            self.spatial_shape = spatial_shape
            self.batch_size = batch_size
            self.grid = grid
            self.voxel_num = voxel_num
            self.indice_dict = indice_dict if indice_dict is not None else {}
            self.benchmark = benchmark
            self.benchmark_record = {}
            self.thrust_allocator = None
            self._timer = None
            self.force_algo = None
            self.int8_scale = None

        @property
        def features(self):
            return self._features

        @features.setter
        def features(self, val):
            self._features = val

    if "spconv.pytorch" in sys.modules and isinstance(sys.modules["spconv.pytorch"], _DummyModule):
        sys.modules["spconv.pytorch"].SparseConvTensor = _StubSparseConvTensor

    if "flash_attn" not in sys.modules:
        def _fa_varlen_qkvpacked_cpu(qkv, cu_seqlens, max_seqlen, *args, **kwargs):
            N, _, H, HD = qkv.shape
            out = torch.empty((N, H, HD), device=qkv.device, dtype=qkv.dtype)
            seqlens = cu_seqlens.tolist()
            for i in range(len(seqlens) - 1):
                s, e = seqlens[i], seqlens[i + 1]
                if e > s:
                    q_i = qkv[s:e, 0].transpose(0, 1).unsqueeze(0)
                    k_i = qkv[s:e, 1].transpose(0, 1).unsqueeze(0)
                    v_i = qkv[s:e, 2].transpose(0, 1).unsqueeze(0)
                    o_i = F.scaled_dot_product_attention(q_i, k_i, v_i)
                    out[s:e] = o_i.squeeze(0).transpose(0, 1)
            return out

        def _fa_qkvpacked_cpu(qkv, *args, **kwargs):
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous()

        def _fa_varlen_kvpacked_cpu(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs):
            return flash_cross_attn(q, cu_seqlens_q, max_seqlen_q, kv, cu_seqlens_k, max_seqlen_k)

        def _fa_swiglu(gate, y):
            return F.silu(gate) * y

        def _fa_layer_norm(x, weight, bias, eps=1e-6):
            return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

        def _fa_rms_norm(x, weight, bias=None, eps=1e-6):
            return F.rms_norm(x, (x.shape[-1],), weight, eps)

        for mname in (
            "flash_attn",
            "flash_attn.ops",
            "flash_attn.ops.activations",
            "flash_attn.ops.layer_norm",
            "flash_attn.ops.rms_norm",
            "flash_attn.ops.triton",
            "flash_attn.ops.triton.layer_norm",
        ):
            mod = types.ModuleType(mname)
            mod.__path__ = []
            mod.__spec__ = importlib.machinery.ModuleSpec(mname, None, is_package=True)
            sys.modules[mname] = mod

        sys.modules["flash_attn"].flash_attn_varlen_qkvpacked_func = _fa_varlen_qkvpacked_cpu
        sys.modules["flash_attn"].flash_attn_qkvpacked_func = _fa_qkvpacked_cpu
        sys.modules["flash_attn"].flash_attn_varlen_kvpacked_func = _fa_varlen_kvpacked_cpu
        sys.modules["flash_attn.ops.activations"].swiglu = _fa_swiglu
        sys.modules["flash_attn.ops.layer_norm"].layer_norm = _fa_layer_norm
        sys.modules["flash_attn.ops.rms_norm"].rms_norm = _fa_rms_norm
        sys.modules["flash_attn.ops.triton.layer_norm"].layer_norm_fn = _fa_layer_norm
        sys.modules["flash_attn.ops.triton.layer_norm"].rms_norm_fn = _fa_rms_norm

    try:
        import base_models.order as order_mod

        _orig_z = order_mod.encode_z
        _orig_h = order_mod.encode_hilbert
        _orig_x = order_mod.encode_xor
        _orig_b = order_mod.assign_batch_index

        def _cpu_split_by_3(a: torch.Tensor) -> torch.Tensor:
            w = a.to(torch.int64) & 0x1FFFFF
            w = (w | (w << 32)) & 0x001F00000000FFFF
            w = (w | (w << 16)) & 0x001F0000FF0000FF
            w = (w | (w << 8)) & 0x010F00F00F00F00F
            w = (w | (w << 4)) & 0x10C30C30C30C30C3
            w = (w | (w << 2)) & 0x1249249249249249
            return w

        def _encode_z_cpu(xyz: torch.Tensor, convention: str = "xyz", assign_batch_index: bool = True) -> torch.Tensor:
            if xyz.is_cuda:
                return _orig_z(xyz, convention=convention, assign_batch_index=assign_batch_index)
            convention = convention.lower()
            x_off, y_off, z_off = convention.find("x"), convention.find("y"), convention.find("z")
            bx = xyz[:, 0].to(torch.int64)
            x = xyz[:, 1 + x_off].to(torch.int64)
            y = xyz[:, 1 + y_off].to(torch.int64)
            z = xyz[:, 1 + z_off].to(torch.int64)
            ret = (_cpu_split_by_3(x) << 2) | (_cpu_split_by_3(y) << 1) | _cpu_split_by_3(z)
            if assign_batch_index:
                ret = ret | (bx << 48)
            return ret

        def _encode_hilbert_cpu(xyz: torch.Tensor, convention: str = "xyz", assign_batch_index: bool = True) -> torch.Tensor:
            if xyz.is_cuda:
                return _orig_h(xyz, convention=convention, assign_batch_index=assign_batch_index)
            convention = convention.lower()
            x_off, y_off, z_off = convention.find("x"), convention.find("y"), convention.find("z")
            bx = xyz[:, 0].to(torch.int64)
            x = xyz[:, 1 + x_off].to(torch.int64).clone()
            y = xyz[:, 1 + y_off].to(torch.int64).clone()
            z = xyz[:, 1 + z_off].to(torch.int64).clone()
            for i in range(15, 0, -1):
                q = 1 << i
                p = q - 1
                x = torch.where((x & q) > 0, x ^ p, x)
                cond = (y & q) > 0
                t = (x ^ y) & p
                x = x ^ torch.where(cond, torch.full_like(t, p), t)
                y = y ^ torch.where(cond, torch.zeros_like(t), t)
                cond = (z & q) > 0
                t = (x ^ z) & p
                x = x ^ torch.where(cond, torch.full_like(t, p), t)
                z = z ^ torch.where(cond, torch.zeros_like(t), t)
            y ^= x
            z ^= y
            t = torch.zeros_like(z)
            for i in range(15, 0, -1):
                q = 1 << i
                t = t ^ torch.where((z & q) > 0, torch.full_like(z, q - 1), torch.zeros_like(z))
            x ^= t
            y ^= t
            z ^= t
            ret = (_cpu_split_by_3(x) << 2) | (_cpu_split_by_3(y) << 1) | _cpu_split_by_3(z)
            if assign_batch_index:
                ret = ret | (bx << 48)
            return ret

        def _encode_xor_cpu(xyz: torch.Tensor, assign_batch_index: bool = True) -> torch.Tensor:
            if xyz.is_cuda:
                return _orig_x(xyz, assign_batch_index=assign_batch_index)
            bx = xyz[:, 0].to(torch.int64)
            ret = xyz[:, 1].to(torch.int64) ^ xyz[:, 2].to(torch.int64) ^ xyz[:, 3].to(torch.int64)
            if assign_batch_index:
                ret = ret | (bx << 48)
            return ret

        def _assign_batch_index_cpu(code: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
            if code.is_cuda:
                return _orig_b(code, batch_idx)
            return code | (batch_idx.to(torch.int64) << 48)

        order_mod.encode_z = _encode_z_cpu
        order_mod.encode_hilbert = _encode_hilbert_cpu
        order_mod.encode_xor = _encode_xor_cpu
        order_mod.assign_batch_index = _assign_batch_index_cpu
    except Exception:
        pass

    try:
        import ops.sparse as ops_sparse

        _orig_get_idx = ops_sparse.get_conv_indices
        _orig_conv = ops_sparse.conv

        def _get_conv_indices_fallback(pos: torch.Tensor) -> torch.Tensor:
            if pos.is_cuda and _LIBSPARSE is not None:
                return _orig_get_idx(pos)
            return build_conv_indices_fast(pos)

        def _conv_fallback(
            feat: torch.Tensor,
            pos: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            indices: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if feat.is_cuda and _LIBSPARSE is not None:
                return _orig_conv(feat, pos, weight, bias, indices=indices)
            if indices is None:
                indices = build_conv_indices_fast(pos)
            w_fp32 = weight.float().contiguous()
            b_fp32 = bias.float().contiguous() if bias is not None else torch.zeros(weight.shape[-1], device=feat.device)
            return fast_sparse_conv3d(feat, w_fp32, b_fp32, indices)

        ops_sparse.get_conv_indices = _get_conv_indices_fallback
        ops_sparse.conv = _conv_fallback
    except Exception:
        pass

    try:
        import ops.rope as ops_rope
        import models.generators.lat_dit as gen_lat_dit

        def _rope_nd_cpu(x: torch.Tensor, pos: torch.Tensor, freq: float = 100.0) -> torch.Tensor:
            B, N, H, K = x.shape
            D = pos.shape[-1]
            M = K // (2 * D)
            inv_freq = float(freq) ** (-torch.arange(M, dtype=torch.float64, device=x.device) / float(M))
            out_chunks = []
            for d in range(D):
                angles = pos[:, :, d : d + 1].to(torch.float64) * inv_freq.view(1, 1, M)
                c = torch.cos(angles).to(x.dtype).unsqueeze(2)
                s = torch.sin(angles).to(x.dtype).unsqueeze(2)
                xd = x[:, :, :, d * 2 * M : (d + 1) * 2 * M]
                u = xd[..., :M]
                v = xd[..., M:]
                out_chunks.append(u * c - v * s)
                out_chunks.append(u * s + v * c)
            if 2 * D * M < K:
                out_chunks.append(x[..., 2 * D * M :])
            return torch.cat(out_chunks, dim=-1)

        ops_rope.rope_nd = _rope_nd_cpu
        gen_lat_dit.rope_nd = _rope_nd_cpu
    except Exception:
        pass

    try:
        import base_models.sparse as bm_sparse
        import models.generators.lat_dit as gen_lat_dit

        if getattr(gen_lat_dit, "flash_attn_varlen_qkvpacked_func", None) is None:

            def _fa_varlen_qkvpacked_cpu(qkv, cu_seqlens, max_seqlen, *args, **kwargs):
                N, _, H, HD = qkv.shape
                out = torch.empty((N, H, HD), device=qkv.device, dtype=qkv.dtype)
                seqlens = cu_seqlens.tolist()
                for i in range(len(seqlens) - 1):
                    s, e = seqlens[i], seqlens[i + 1]
                    if e > s:
                        q_i = qkv[s:e, 0].transpose(0, 1).unsqueeze(0)
                        k_i = qkv[s:e, 1].transpose(0, 1).unsqueeze(0)
                        v_i = qkv[s:e, 2].transpose(0, 1).unsqueeze(0)
                        o_i = F.scaled_dot_product_attention(q_i, k_i, v_i)
                        out[s:e] = o_i.squeeze(0).transpose(0, 1)
                return out

            def _fa_qkvpacked_cpu(qkv, *args, **kwargs):
                q = qkv[:, :, 0].transpose(1, 2)
                k = qkv[:, :, 1].transpose(1, 2)
                v = qkv[:, :, 2].transpose(1, 2)
                return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous()

            def _fa_varlen_kvpacked_cpu(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs):
                return flash_cross_attn(q, cu_seqlens_q, max_seqlen_q, kv, cu_seqlens_k, max_seqlen_k)

            bm_sparse.flash_attn_varlen_qkvpacked_func = _fa_varlen_qkvpacked_cpu
            gen_lat_dit.flash_attn_varlen_qkvpacked_func = _fa_varlen_qkvpacked_cpu
            gen_lat_dit.flash_attn_qkvpacked_func = _fa_qkvpacked_cpu
            gen_lat_dit.flash_attn_varlen_kvpacked_func = _fa_varlen_kvpacked_cpu
    except Exception:
        pass


def preload_extensions(verbose: bool = False) -> None:
    """Pre-load CUDA/Triton extensions at model load time so zero JIT happens during generate()."""
    global _FLASH_ATTN, _FLASH_ATTN_3, _TRITON_LN, _TRITON_RMS, _SWIGLU_FN, _LIBSPARSE, _CUROPE

    install_cpu_fallbacks()
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
        w = getattr(scale, "_w_1ps", None)
        if w is None:
            w = 1.0 + scale.view(-1)
        b = getattr(shift, "_b_1d", None)
        if b is None:
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
        w = getattr(scale, "_w_1ps", None)
        if w is None:
            w = 1.0 + scale.view(-1)
        b = getattr(shift, "_b_1d", None)
        if b is None:
            b = shift.view(-1)
        return fast_layer_norm(cond, w, b, eps=eps)
    return fast_layer_norm(cond, eps=eps) * (1.0 + scale[:, None]) + shift[:, None]


def fast_gate(x: torch.Tensor, sizes: List[int], scale: torch.Tensor) -> torch.Tensor:
    if len(sizes) == 1:
        g = getattr(scale, "_g_2d", None)
        if g is not None and g.dtype == x.dtype:
            return x * g
        return x * scale.view(1, -1).type_as(x)
    return torch.cat([hh * scale[idx].type_as(hh) for idx, hh in enumerate(x.split(sizes))], dim=0)


def fast_gate_add(
    residual: torch.Tensor,
    h: torch.Tensor,
    sizes: List[int],
    scale: torch.Tensor,
) -> torch.Tensor:
    """Fused residual + h * gate using single-kernel torch.addcmul when batch_size == 1."""
    if len(sizes) == 1:
        g = getattr(scale, "_g_2d", None)
        if g is None or g.dtype != h.dtype:
            g = scale.view(1, -1).to(dtype=h.dtype)
        return torch.addcmul(residual, h, g)
    return residual + fast_gate(h, sizes, scale)


def fast_gate_add_cond(
    cond: torch.Tensor,
    h: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Fused cond + h * cg[:, None] using single-kernel torch.addcmul for [B, L, D] conditioning."""
    if cond.shape[0] == 1:
        g = getattr(scale, "_g_3d", None)
        if g is None or g.dtype != h.dtype:
            g = scale.view(1, 1, -1).to(dtype=h.dtype)
        return torch.addcmul(cond, h, g)
    return torch.addcmul(cond, h, scale[:, None].to(dtype=h.dtype))


def fast_modulate_preln_cond(
    ln_cond: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Fused affine modulation `ln_cond * (1 + scale) + shift` on pre-normalized `ln_cond_after_7`."""
    if ln_cond.shape[0] == 1:
        w = getattr(scale, "_w_3d", None)
        b = getattr(shift, "_b_3d", None)
        if w is not None and b is not None and w.dtype == ln_cond.dtype:
            return torch.addcmul(b, ln_cond, w)
    return torch.addcmul(shift[:, None], ln_cond, 1.0 + scale[:, None])


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


if HAS_TRITON:

    @triton.jit
    def _fused_gate_add_ln_modulate_kernel(
        X_ptr,
        H_ptr,
        G_ptr,
        W_ptr,
        B_ptr,
        X_OUT_ptr,
        MOD_OUT_ptr,
        M: int,
        H_STRIDE_ROW: int,
        D: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= M:
            return
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        base = row * D + cols

        x = tl.load(X_ptr + base, mask=mask, other=0.0).to(tl.float32)
        h = tl.load(H_ptr + row * H_STRIDE_ROW + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        x_next = (x + h * g).to(X_OUT_ptr.dtype.element_ty)
        tl.store(X_OUT_ptr + base, x_next, mask=mask)

        xf = x_next.to(tl.float32)
        mean = tl.sum(xf, axis=0) / D
        diff = tl.where(mask, xf - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / D
        rstd = tl.rsqrt(var + eps)

        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mod_out = (diff * rstd * w + b).to(MOD_OUT_ptr.dtype.element_ty)
        tl.store(MOD_OUT_ptr + base, mod_out, mask=mask)

    @triton.jit
    def _fused_out_modulate_unpatchify_kernel(
        X_ptr,
        W_MOD_ptr,
        B_MOD_ptr,
        W_UNPATCH_ptr,
        OUT_ptr,
        M: int,
        D: tl.constexpr,
        C_OUT: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= M:
            return
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        base = row * D + cols

        xf = tl.load(X_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(xf, axis=0) / D
        diff = tl.where(mask, xf - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / D
        rstd = tl.rsqrt(var + eps)

        w_mod = tl.load(W_MOD_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b_mod = tl.load(B_MOD_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mod_x = ((diff * rstd * w_mod + b_mod).to(X_ptr.dtype.element_ty)).to(tl.float32)
        mod_x = tl.where(mask, mod_x, 0.0)

        out_base = row * C_OUT
        for c in tl.static_range(C_OUT):
            w_u = tl.load(W_UNPATCH_ptr + c * D + cols, mask=mask, other=0.0).to(tl.float32)
            acc = tl.sum(mod_x * w_u, axis=0)
            tl.store(OUT_ptr + out_base + c, acc)

    @triton.jit
    def _fused_packed_swiglu_kernel(
        IN_ptr,
        OUT_ptr,
        M: int,
        D_half: int,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_blk = tl.program_id(1)
        if row >= M:
            return
        cols = col_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = cols < D_half
        in_base = row * (2 * D_half)
        h1 = tl.load(IN_ptr + in_base + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(IN_ptr + in_base + D_half + cols, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(gate)
        out = (gate * sig * h1).to(OUT_ptr.dtype.element_ty)
        tl.store(OUT_ptr + row * D_half + cols, out, mask=mask)


def fast_gate_add_and_modulate(
    residual: torch.Tensor,
    h: torch.Tensor,
    sizes: List[int],
    gate_scale: torch.Tensor,
    next_scale: torch.Tensor,
    next_shift: torch.Tensor,
    eps: float = 1e-6,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused `x_next = residual + h * gate_scale` AND `h_mod = LayerNorm(x_next) * (1 + next_scale) + next_shift`
    in a single Triton kernel pass (eliminating a full HBM round-trip of `x_next`).
    """
    if (
        use_triton
        and HAS_TRITON
        and residual.is_cuda
        and len(sizes) == 1
        and residual.is_contiguous()
        and h.is_contiguous()
        and residual.shape[-1] <= 2048
    ):
        D = residual.shape[-1]
        M = residual.shape[0]
        h_stride_row = 0 if h.shape[0] == 1 else D
        g = getattr(gate_scale, "_g_2d", None)
        if g is None:
            g = gate_scale.view(-1)
        w = getattr(next_scale, "_w_1ps", None)
        if w is None:
            w = 1.0 + next_scale.view(-1)
        b = getattr(next_shift, "_b_1d", None)
        if b is None:
            b = next_shift.view(-1)
        x_out = torch.empty_like(residual)
        mod_out = torch.empty_like(residual)
        block_d = triton.next_power_of_2(D)
        _fused_gate_add_ln_modulate_kernel[(M,)](
            residual,
            h,
            g,
            w,
            b,
            x_out,
            mod_out,
            M=M,
            H_STRIDE_ROW=h_stride_row,
            D=D,
            eps=eps,
            BLOCK_D=block_d,
            num_warps=8 if block_d >= 2048 else 4,
        )
        return x_out, mod_out
    x_next = fast_gate_add(residual, h, sizes, gate_scale)
    h_mod = fast_modulate(x_next, sizes, next_scale, next_shift, eps=eps)
    return x_next, h_mod


def fast_gate_add_and_modulate_cond(
    cond: torch.Tensor,
    h: torch.Tensor,
    gate_scale: torch.Tensor,
    next_scale: torch.Tensor,
    next_shift: torch.Tensor,
    eps: float = 1e-6,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused `cond_next = cond + h * gate_scale` AND `h_cond = LayerNorm(cond_next) * (1 + next_scale) + next_shift`
    for `[1, L, D]` conditioning tensors in a single Triton kernel pass.
    """
    if (
        use_triton
        and HAS_TRITON
        and cond.is_cuda
        and cond.shape[0] == 1
        and cond.is_contiguous()
        and h.is_contiguous()
        and cond.shape[-1] <= 2048
    ):
        D = cond.shape[-1]
        M = cond.shape[1]
        h_stride_row = 0 if h.shape[1] == 1 else D
        g = getattr(gate_scale, "_g_2d", None)
        if g is None:
            g = gate_scale.view(-1)
        w = getattr(next_scale, "_w_1ps", None)
        if w is None:
            w = 1.0 + next_scale.view(-1)
        b = getattr(next_shift, "_b_1d", None)
        if b is None:
            b = next_shift.view(-1)
        c_out = torch.empty_like(cond)
        mod_out = torch.empty_like(cond)
        block_d = triton.next_power_of_2(D)
        _fused_gate_add_ln_modulate_kernel[(M,)](
            cond,
            h,
            g,
            w,
            b,
            c_out,
            mod_out,
            M=M,
            H_STRIDE_ROW=h_stride_row,
            D=D,
            eps=eps,
            BLOCK_D=block_d,
            num_warps=8 if block_d >= 2048 else 4,
        )
        return c_out, mod_out
    c_next = fast_gate_add_cond(cond, h, gate_scale)
    h_mod = fast_modulate_cond(c_next, next_scale, next_shift, eps=eps)
    return c_next, h_mod


def fast_out_modulate_unpatchify(
    x: torch.Tensor,
    sizes: List[int],
    out_scale: torch.Tensor,
    out_shift: torch.Tensor,
    unpatchify_w_fp32: torch.Tensor,
    eps: float = 1e-6,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Fused Output LayerNorm + AdaLN modulation + FP32 Unpatchify projection (`[N, D] -> [N, 16]`):
    Eliminates writing `[N, 1536]` BF16 and `[N, 1536]` FP32 intermediate buffers to HBM!
    """
    if (
        use_triton
        and HAS_TRITON
        and x.is_cuda
        and len(sizes) == 1
        and x.is_contiguous()
        and unpatchify_w_fp32.is_contiguous()
        and x.shape[-1] <= 2048
        and unpatchify_w_fp32.shape[0] <= 32
    ):
        M, D = x.shape
        c_out = unpatchify_w_fp32.shape[0]
        w_mod = getattr(out_scale, "_w_1ps", None)
        if w_mod is None:
            w_mod = 1.0 + out_scale.view(-1)
        b_mod = getattr(out_shift, "_b_1d", None)
        if b_mod is None:
            b_mod = out_shift.view(-1)
        out = torch.empty((M, c_out), device=x.device, dtype=torch.float32)
        block_d = triton.next_power_of_2(D)
        _fused_out_modulate_unpatchify_kernel[(M,)](
            x,
            w_mod,
            b_mod,
            unpatchify_w_fp32,
            out,
            M=M,
            D=D,
            C_OUT=c_out,
            eps=eps,
            BLOCK_D=block_d,
            num_warps=8 if block_d >= 2048 else 4,
        )
        return out
    x_mod = fast_modulate(x, sizes, out_scale, out_shift, eps=eps)
    return F.linear(x_mod.float(), unpatchify_w_fp32)


def fast_swiglu(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and _SWIGLU_FN is not None:
        return _SWIGLU_FN(gate, x)
    return (F.silu(gate.float()) * x.float()).to(x.dtype)


def fast_swiglu_from_packed(h_packed: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    """
    Fused SwiGLU directly from packed `[..., 2 * D_half]` tensor (`h1, h_gate = h_packed.chunk(2, dim=-1)`),
    eliminating `.chunk(2, dim=-1)` and the two hidden `.contiguous()` copies inside `flash_attn`'s `swiglu`.
    """
    D2 = h_packed.shape[-1]
    D_half = D2 // 2
    if use_triton and HAS_TRITON and h_packed.is_cuda and h_packed.is_contiguous():
        M = h_packed.numel() // D2
        out = torch.empty((*h_packed.shape[:-1], D_half), device=h_packed.device, dtype=h_packed.dtype)
        block_d = 1024 if D_half >= 1024 else triton.next_power_of_2(max(32, D_half))
        grid = (M, triton.cdiv(D_half, block_d))
        _fused_packed_swiglu_kernel[grid](
            h_packed,
            out,
            M=M,
            D_half=D_half,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return out
    h1, h_gate = h_packed.chunk(2, dim=-1)
    return fast_swiglu(h_gate, h1)


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
    def _triton_hilbert_dist_3d(fx, fy, fz):
        x = fx.to(tl.int64)
        y = fy.to(tl.int64)
        z = fz.to(tl.int64)
        for i in tl.static_range(15, 0, -1):
            q = 1 << i
            p = q - 1
            x ^= tl.where((x & q) > 0, p, 0)
            cond = (y & q) > 0
            t = (x ^ y) & p
            x ^= tl.where(cond, p, t)
            y ^= tl.where(cond, 0, t)
            cond = (z & q) > 0
            t = (x ^ z) & p
            x ^= tl.where(cond, p, t)
            z ^= tl.where(cond, 0, t)
        y ^= x
        z ^= y
        t = tl.zeros_like(z)
        for i in tl.static_range(15, 0, -1):
            q = 1 << i
            t ^= tl.where((z & q) > 0, q - 1, 0)
        x ^= t
        y ^= t
        z ^= t
        ret = tl.zeros_like(z)
        for i in tl.static_range(0, 16):
            q = 1 << i
            ret |= (x & q) << (2 * i + 2)
            ret |= (y & q) << (2 * i + 1)
            ret |= (z & q) << (2 * i + 0)
        return ret

    @triton.jit
    def _encode_hilbert_4_kernel(
        xyz_ptr,
        codes_ptr,
        N,
        str_xyz_N,
        str_xyz_C,
        BLK: tl.constexpr,
    ):
        """Fused single-pass 4-mode Hilbert curve encoder for ('h', 'ht1', 'h1', 'ht')."""
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLK + tl.arange(0, BLK)
        mask_n = offs_n < N

        xyz_ptrs = xyz_ptr + offs_n * str_xyz_N
        bx = tl.load(xyz_ptrs, mask=mask_n, other=0).to(tl.int64) << 48
        fx = tl.load(xyz_ptrs + 1 * str_xyz_C, mask=mask_n, other=0)
        fy = tl.load(xyz_ptrs + 2 * str_xyz_C, mask=mask_n, other=0)
        fz = tl.load(xyz_ptrs + 3 * str_xyz_C, mask=mask_n, other=0)

        c_h = _triton_hilbert_dist_3d(fx, fy, fz) | bx
        c_ht1 = _triton_hilbert_dist_3d(fy + 1, fx + 1, fz + 1) | bx
        c_h1 = _triton_hilbert_dist_3d(fx + 1, fy + 1, fz + 1) | bx
        c_ht = _triton_hilbert_dist_3d(fy, fx, fz) | bx

        tl.store(codes_ptr + 0 * N + offs_n, c_h, mask=mask_n)
        tl.store(codes_ptr + 1 * N + offs_n, c_ht1, mask=mask_n)
        tl.store(codes_ptr + 2 * N + offs_n, c_h1, mask=mask_n)
        tl.store(codes_ptr + 3 * N + offs_n, c_ht, mask=mask_n)

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

    @triton.jit
    def _fused_qknorm_rope2d_cond_inplace_kernel(
        qkv_ptr,
        q_scale_ptr,
        k_scale_ptr,
        cos2d_ptr,
        sin2d_ptr,
        L,
        H: tl.constexpr,
        prefix_tokens: tl.constexpr,
        eps: tl.constexpr,
    ):
        """
        In-place QKNorm + 2D RoPE on `qkv_cond` [1, L, 3, H, 96]:
          - For all l in [0, L): applies RMSNorm(q) * q_scale and RMSNorm(k) * k_scale
          - For l >= prefix_tokens (5): applies 2D RoPE rotation (Q=24) in registers
          - Leaves V (idx 2) untouched in-place (zero memory allocations!)
        """
        l_idx = tl.program_id(0)
        h = tl.program_id(1)

        offs = tl.arange(0, 32)
        mask = offs < 24
        Q: tl.constexpr = 24
        D: tl.constexpr = 96
        stride_l = 3 * H * D
        stride_3 = H * D
        stride_h = D

        # --- Q (idx 0) ---
        q_base = qkv_ptr + l_idx * stride_l + 0 * stride_3 + h * stride_h
        qs_base = q_scale_ptr + h * D

        qu0 = tl.load(q_base + 0 * Q + offs, mask=mask, other=0.0)
        qv0 = tl.load(q_base + 1 * Q + offs, mask=mask, other=0.0)
        qu1 = tl.load(q_base + 2 * Q + offs, mask=mask, other=0.0)
        qv1 = tl.load(q_base + 3 * Q + offs, mask=mask, other=0.0)

        qu0_f, qv0_f = qu0.to(tl.float32), qv0.to(tl.float32)
        qu1_f, qv1_f = qu1.to(tl.float32), qv1.to(tl.float32)
        q_sq = tl.sum(qu0_f * qu0_f + qv0_f * qv0_f + qu1_f * qu1_f + qv1_f * qv1_f, axis=0)
        q_rrms = tl.rsqrt(q_sq * (1.0 / 96.0) + eps)

        qsu0 = tl.load(qs_base + 0 * Q + offs, mask=mask, other=0.0).to(qu0.dtype).to(tl.float32)
        qsv0 = tl.load(qs_base + 1 * Q + offs, mask=mask, other=0.0).to(qu0.dtype).to(tl.float32)
        qsu1 = tl.load(qs_base + 2 * Q + offs, mask=mask, other=0.0).to(qu0.dtype).to(tl.float32)
        qsv1 = tl.load(qs_base + 3 * Q + offs, mask=mask, other=0.0).to(qu0.dtype).to(tl.float32)

        qu0_n = (qu0_f * q_rrms).to(qu0.dtype).to(tl.float32) * qsu0
        qv0_n = (qv0_f * q_rrms).to(qu0.dtype).to(tl.float32) * qsv0
        qu1_n = (qu1_f * q_rrms).to(qu0.dtype).to(tl.float32) * qsu1
        qv1_n = (qv1_f * q_rrms).to(qu0.dtype).to(tl.float32) * qsv1

        # --- K (idx 1) ---
        k_base = qkv_ptr + l_idx * stride_l + 1 * stride_3 + h * stride_h
        ks_base = k_scale_ptr + h * D

        ku0 = tl.load(k_base + 0 * Q + offs, mask=mask, other=0.0)
        kv0 = tl.load(k_base + 1 * Q + offs, mask=mask, other=0.0)
        ku1 = tl.load(k_base + 2 * Q + offs, mask=mask, other=0.0)
        kv1 = tl.load(k_base + 3 * Q + offs, mask=mask, other=0.0)

        ku0_f, kv0_f = ku0.to(tl.float32), kv0.to(tl.float32)
        ku1_f, kv1_f = ku1.to(tl.float32), kv1.to(tl.float32)
        k_sq = tl.sum(ku0_f * ku0_f + kv0_f * kv0_f + ku1_f * ku1_f + kv1_f * kv1_f, axis=0)
        k_rrms = tl.rsqrt(k_sq * (1.0 / 96.0) + eps)

        ksu0 = tl.load(ks_base + 0 * Q + offs, mask=mask, other=0.0).to(ku0.dtype).to(tl.float32)
        ksv0 = tl.load(ks_base + 1 * Q + offs, mask=mask, other=0.0).to(ku0.dtype).to(tl.float32)
        ksu1 = tl.load(ks_base + 2 * Q + offs, mask=mask, other=0.0).to(ku0.dtype).to(tl.float32)
        ksv1 = tl.load(ks_base + 3 * Q + offs, mask=mask, other=0.0).to(ku0.dtype).to(tl.float32)

        ku0_n = (ku0_f * k_rrms).to(ku0.dtype).to(tl.float32) * ksu0
        kv0_n = (kv0_f * k_rrms).to(ku0.dtype).to(tl.float32) * ksv0
        ku1_n = (ku1_f * k_rrms).to(ku0.dtype).to(tl.float32) * ksu1
        kv1_n = (kv1_f * k_rrms).to(ku0.dtype).to(tl.float32) * ksv1

        if l_idx >= prefix_tokens:
            sp_idx = l_idx - prefix_tokens
            c_base = cos2d_ptr + sp_idx * (2 * Q)
            s_base = sin2d_ptr + sp_idx * (2 * Q)
            c0 = tl.load(c_base + 0 * Q + offs, mask=mask, other=1.0).to(tl.float32)
            c1 = tl.load(c_base + 1 * Q + offs, mask=mask, other=1.0).to(tl.float32)
            s0 = tl.load(s_base + 0 * Q + offs, mask=mask, other=0.0).to(tl.float32)
            s1 = tl.load(s_base + 1 * Q + offs, mask=mask, other=0.0).to(tl.float32)

            tl.store(q_base + 0 * Q + offs, (qu0_n * c0 - qv0_n * s0).to(qu0.dtype), mask=mask)
            tl.store(q_base + 1 * Q + offs, (qv0_n * c0 + qu0_n * s0).to(qu0.dtype), mask=mask)
            tl.store(q_base + 2 * Q + offs, (qu1_n * c1 - qv1_n * s1).to(qu0.dtype), mask=mask)
            tl.store(q_base + 3 * Q + offs, (qv1_n * c1 + qu1_n * s1).to(qu0.dtype), mask=mask)

            tl.store(k_base + 0 * Q + offs, (ku0_n * c0 - kv0_n * s0).to(ku0.dtype), mask=mask)
            tl.store(k_base + 1 * Q + offs, (kv0_n * c0 + ku0_n * s0).to(ku0.dtype), mask=mask)
            tl.store(k_base + 2 * Q + offs, (ku1_n * c1 - kv1_n * s1).to(ku0.dtype), mask=mask)
            tl.store(k_base + 3 * Q + offs, (kv1_n * c1 + ku1_n * s1).to(ku0.dtype), mask=mask)
        else:
            tl.store(q_base + 0 * Q + offs, qu0_n.to(qu0.dtype), mask=mask)
            tl.store(q_base + 1 * Q + offs, qv0_n.to(qu0.dtype), mask=mask)
            tl.store(q_base + 2 * Q + offs, qu1_n.to(qu0.dtype), mask=mask)
            tl.store(q_base + 3 * Q + offs, qv1_n.to(qu0.dtype), mask=mask)

            tl.store(k_base + 0 * Q + offs, ku0_n.to(ku0.dtype), mask=mask)
            tl.store(k_base + 1 * Q + offs, kv0_n.to(ku0.dtype), mask=mask)
            tl.store(k_base + 2 * Q + offs, ku1_n.to(ku0.dtype), mask=mask)
            tl.store(k_base + 3 * Q + offs, kv1_n.to(ku0.dtype), mask=mask)


def fused_qknorm_rope2d_cond_inplace(
    qkv_cond: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cos_cond_2d: torch.Tensor,
    sin_cond_2d: torch.Tensor,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Apply QK RMSNorm + 2D RoPE on `qkv_cond` [B, L, 3, H, 96].
    On GPU with Triton and B==1, executes in-place in a single kernel with zero allocations!
    """
    B, L, _, H, D = qkv_cond.shape
    if use_triton and HAS_TRITON and qkv_cond.is_cuda and B == 1 and D == 96:
        _fused_qknorm_rope2d_cond_inplace_kernel[(L, H)](
            qkv_cond,
            q_scale,
            k_scale,
            cos_cond_2d,
            sin_cond_2d,
            L=L,
            H=H,
            prefix_tokens=5,
            eps=1e-6,
            num_warps=1,
        )
        return qkv_cond

    q_c, k_c, v_c = qkv_cond.unbind(dim=-3)
    q_c = fast_rms_norm(q_c) * q_scale.to(q_c.dtype)
    k_c = fast_rms_norm(k_c) * k_scale.to(k_c.dtype)
    q_c = torch.cat([q_c[:, :5], apply_rope_cached_2d(q_c[:, 5:], cos_cond_2d, sin_cond_2d)], dim=1)
    k_c = torch.cat([k_c[:, :5], apply_rope_cached_2d(k_c[:, 5:], cos_cond_2d, sin_cond_2d)], dim=1)
    return torch.stack([q_c, k_c, v_c], dim=-3)


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


def build_order_caches_fast(
    h_coords: torch.Tensor,
    sizes: List[int],
    orders: List[str],
    order_kwargs: dict,
    seqlen_cache: dict,
    arange_n: torch.Tensor,
    use_triton: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """
    Zero-sync batched Hilbert space-filling curve partition builder:
      - Eliminates 4 blocking GPU->CPU `.item()` synchronizations (`seqlens.max().item()`) per sample
      - Precomputes and caches shared `cu_seqlens` and `max_len` on CPU once for all window orders
      - On CUDA with Triton and standard `orders == ["h", "ht1", "h1", "ht"]`, fuses all 4 Hilbert
        encodings into a single kernel pass (`_encode_hilbert_4_kernel`) and executes a single
        batched `[4, N]` radix sort (`torch.argsort(codes_4, dim=-1)`)!
    """
    from base_models.order import calc_mixed_partition, calc_order_partition, encode_hilbert, encode_z

    device = h_coords.device
    N = h_coords.shape[0]
    patch_size = int(order_kwargs.get("patch_size", 4096))
    adj_first = bool(order_kwargs.get("adj_first", False))
    sliding = bool(order_kwargs.get("sliding", False))

    if not adj_first and all(o in ("z", "zt", "h", "ht", "z1", "zt1", "h1", "ht1") for o in orders):
        # Compute shared `cu_seqlens` and `max_len` purely on Python CPU with zero GPU->CPU sync!
        part_key = ("order_part", device, tuple(sizes), patch_size, sliding)
        cached_part = seqlen_cache.get(part_key)
        if cached_part is None:
            if sliding:
                seqlens_list = [int(s) for s in sizes]
            else:
                seqlens_list = []
                for s in sizes:
                    s_int = int(s)
                    full_chunks = (s_int - 1) // patch_size if s_int > 0 else 0
                    for _ in range(full_chunks):
                        seqlens_list.append(patch_size)
                    if s_int > 0:
                        seqlens_list.append(s_int - full_chunks * patch_size)
            max_len = max(seqlens_list) if seqlens_list else 0
            cu_list = [0]
            acc_s = 0
            for sl in seqlens_list:
                acc_s += sl
                cu_list.append(acc_s)
            cu_seqlens = torch.tensor(cu_list, device=device, dtype=torch.int32)
            cached_part = (cu_seqlens, int(max_len))
            seqlen_cache[part_key] = cached_part
        else:
            cu_seqlens, max_len = cached_part

        if (
            use_triton
            and HAS_TRITON
            and h_coords.is_cuda
            and tuple(orders) == ("h", "ht1", "h1", "ht")
        ):
            codes_4 = h_coords.new_empty((4, N), dtype=torch.int64)
            blk = max(32, min(1024, 1 << max(0, (N - 1).bit_length())))
            grid = ((N + blk - 1) // blk,)
            _encode_hilbert_4_kernel[grid](
                h_coords,
                codes_4,
                N,
                h_coords.stride(0),
                h_coords.stride(1),
                BLK=blk,
            )
            fwd_4 = torch.argsort(codes_4, dim=-1)
            order_caches = []
            for idx_o in range(4):
                fwd_idx = fwd_4[idx_o]
                bwd_idx = torch.empty_like(fwd_idx)
                bwd_idx[fwd_idx] = arange_n
                order_caches.append((fwd_idx, bwd_idx, cu_seqlens, max_len))
            return order_caches

        h_coords_p1: Optional[torch.Tensor] = None
        order_caches = []
        for order in orders:
            if order.endswith("1"):
                if h_coords_p1 is None:
                    h_coords_p1 = h_coords + 1
                xyz_in = h_coords_p1
            else:
                xyz_in = h_coords
            conv = "yxz" if "t" in order else "xyz"
            code = encode_hilbert(xyz_in, conv) if order[0] == "h" else encode_z(xyz_in, conv)
            fwd_idx = torch.argsort(code)
            bwd_idx = torch.empty_like(fwd_idx)
            bwd_idx[fwd_idx] = arange_n
            order_caches.append((fwd_idx, bwd_idx, cu_seqlens, max_len))
        return order_caches

    order_caches = []
    for order in orders:
        if order in ("z", "zt", "h", "ht", "z1", "zt1", "h1", "ht1"):
            fwd_idx, bwd_idx, cu_seqlens, max_len = calc_order_partition(
                h_coords,
                sizes,
                mode=order,
                patch_size=patch_size,
                adj_first=adj_first,
                sliding=sliding,
            )
        elif order in ("zmod", "zdiv", "xmod", "xdiv", "hmod", "hdiv"):
            fwd_idx, bwd_idx, cu_seqlens, max_len = calc_mixed_partition(
                h_coords,
                sizes,
                mode=order,
                bucket_size=order_kwargs.get("bucket_size", 512),
                global_size=order_kwargs.get("global_size", 32),
            )
        else:
            fwd_idx = bwd_idx = arange_n
            max_len = int(max(sizes))
            cu_seqlens = torch.cumsum(
                torch.tensor([0] + sizes, device=device, dtype=torch.int32), dim=0, dtype=torch.int32
            )
        order_caches.append((fwd_idx, bwd_idx, cu_seqlens, int(max_len)))
    return order_caches


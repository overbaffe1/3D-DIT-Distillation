"""Optimized LatDiT execution engine with static geometry hoisting and 1-token CFG condensation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import base_models.sparse as sp
from base_models.fourier_utils import get_time_encoding
from base_models.order import calc_order_partition, calc_mixed_partition
from kandi3d_image23d.models.modeling_lat_dit import LatDitModel
from kandi3d_image23d.runtime import FlowMatchSamplingParams, LatDitConditioning, resolve

from lib.fast_kernels import (
    apply_rope_cached_2d,
    build_conv_indices_fast,
    fast_gate,
    fast_gate_add,
    fast_layer_norm,
    fast_modulate,
    fast_modulate_cond,
    fast_rms_norm,
    fast_sparse_conv3d,
    fast_swiglu,
    flash_cond_self_attn,
    flash_cross_attn,
    flash_varlen_qkvpacked_already_sorted,
    fused_qknorm_rope2d_cond_inplace,
    fused_qknorm_rope3d_gather,
    lookup_rope_3d,
    precompute_rope_2d_cond,
    precompute_rope_3d_table,
    preload_extensions,
)
from lib.fast_sampler import (
    SolverConfig,
    build_schedule,
    get_qaoa_eval_timesteps,
    should_eval_cond,
    should_eval_uncond,
)


class OptimizedLatDitRunner:
    """
    Wraps a `LatDitModel` and executes an accelerated, mathematically faithful
    Flow-Matching inference pipeline.
    """

    def __init__(
        self,
        model: LatDitModel,
        solver_config: SolverConfig,
        weights_dir: Optional[Path] = None,
    ):
        self.model = model
        self.core = model.core
        self.flow = model.core.flow
        self.solver_config = solver_config
        self.weights_dir = weights_dir

        self.dim: int = int(self.core.dim)
        self.latent_dim: int = int(self.core.config.latent_dim)
        self.latent_scale: float = float(self.core.latent_scale)
        self.res: int = int(self.flow.res)
        self.depth: int = len(self.flow.blocks)
        self.cond_depth: int = sum(1 for b in self.flow.blocks if b.cond)
        self.num_heads: int = int(self.flow.blocks[0].num_heads)
        self.head_dim: int = int(self.flow.blocks[0].head_dim)

        # Cached RoPE tables per device
        self._rope_3d_cache: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._rope_2d_cache: Dict[ Tuple[torch.device, int], Tuple[torch.Tensor, torch.Tensor]] = {}

        # Precomputed timestep gates cache: keyed by (device, dtype, round(t_val, 7))
        self._time_gates_cache: Dict[Tuple[torch.device, torch.dtype, float], Dict[str, Any]] = {}

        # Precomputed QAOA GPU tensors cache: keyed by (device, dtype)
        self._qaoa_tensors_cache: Dict[Tuple[torch.device, torch.dtype], Dict[str, Any]] = {}

        # Precomputed static sequence length tensors for batch_size == 1
        self._seqlen_cache: Dict[Tuple[torch.device, int], torch.Tensor] = {}

        # Optional learned per-step calibration scales/shifts (from distillation)
        self.step_calibration: Optional[Dict[str, torch.Tensor]] = None
        if weights_dir is not None:
            cal_path = Path(weights_dir) / "step_calibration.pt"
            if cal_path.is_file():
                try:
                    self.step_calibration = torch.load(cal_path, map_location="cpu", weights_only=True)
                except Exception:
                    self.step_calibration = None

        self._prepare_weights_and_buffers()

    def _runtime_device(self) -> torch.device:
        return self.core._runtime_device()

    def _compute_dtype(self, device: torch.device) -> torch.dtype:
        if device.type == "cuda":
            return torch.bfloat16
        return torch.float32

    def _prepare_weights_and_buffers(self) -> None:
        """
        Pre-cast linear weights to bfloat16 (when enabled) to eliminate 7.8GB of
        on-the-fly FP32->BF16 weight casting per step, while preserving FP32 for
        `ffn_conv` weights/biases required by `libsparse.conv_fwd`.
        Also caches sliced weights for 1-token unconditional condensation.
        """
        device = self._runtime_device()
        target_dtype = self._compute_dtype(device)

        # Preserve full FP32 precision for the 16x1536 (98 KB) output unpatchify projection
        # before any BF16 casting so the 16-channel velocity output has zero BF16 mantissa truncation
        self._unpatchify_w_fp32 = (
            self.flow.unpatchify.weight.detach().to(device=device, dtype=torch.float32).contiguous()
        )

        if self.solver_config.use_bf16_weights and device.type == "cuda":
            for module in self.core.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = module.weight.data.to(device=device, dtype=target_dtype).contiguous()
                    if module.bias is not None:
                        module.bias.data = module.bias.data.to(device=device, dtype=target_dtype).contiguous()
            if hasattr(self.flow.patchify, "weight"):
                self.flow.patchify.weight.data = (
                    self.flow.patchify.weight.data.to(device=device, dtype=target_dtype).contiguous()
                )
            if hasattr(self.flow.unpatchify, "weight"):
                self.flow.unpatchify.weight.data = (
                    self.flow.unpatchify.weight.data.to(device=device, dtype=target_dtype).contiguous()
                )

        D = self.dim
        for block in self.flow.blocks:
            if block.use_ffn_conv:
                block._ffn_conv_w_fp32 = block.ffn_conv.weight.detach().to(device=device, dtype=torch.float32).contiguous()
                block._ffn_conv_b_fp32 = block.ffn_conv.bias.detach().to(device=device, dtype=torch.float32).contiguous()
            if block.rms:
                block.qk_norm._q_scale_cast = block.qk_norm.q_scale.detach().to(device=device, dtype=target_dtype).contiguous()
                block.qk_norm._k_scale_cast = block.qk_norm.k_scale.detach().to(device=device, dtype=target_dtype).contiguous()
                if block.cond:
                    block.cond_qk_norm._q_scale_cast = (
                        block.cond_qk_norm.q_scale.detach().to(device=device, dtype=target_dtype).contiguous()
                    )
                    block.cond_qk_norm._k_scale_cast = (
                        block.cond_qk_norm.k_scale.detach().to(device=device, dtype=target_dtype).contiguous()
                    )
            # Pre-slice weights for 1-token unconditional condensation:
            # In to_qkv_cross ([3*D, D]), slice [D:3*D] is K,V only (skipping Q on N voxels!)
            if block.cond:
                block._to_q_cross_w = block.to_qkv_cross.weight[:D, :].contiguous()
                block._to_q_cross_b = (
                    block.to_qkv_cross.bias[:D].contiguous() if block.to_qkv_cross.bias is not None else None
                )
                block._to_kv_cross_w = block.to_qkv_cross.weight[D:, :].contiguous()
                block._to_kv_cross_b = (
                    block.to_qkv_cross.bias[D:].contiguous() if block.to_qkv_cross.bias is not None else None
                )
                # In cond_to_qkv ([3*D, D]), slice [2*D:3*D] is V only
                block._cond_to_v_w = block.cond_to_qkv.weight[2 * D :, :].contiguous()
                block._cond_to_v_b = (
                    block.cond_to_qkv.bias[2 * D :].contiguous() if block.cond_to_qkv.bias is not None else None
                )
            else:
                # In cond_to_kv_cross ([2*D, D]), slice [D:2*D] is V only
                block._cond_to_v_cross_w = block.cond_to_kv_cross.weight[D:, :].contiguous()
                block._cond_to_v_cross_b = (
                    block.cond_to_kv_cross.bias[D:].contiguous() if block.cond_to_kv_cross.bias is not None else None
                )

        if self.solver_config.precompute_rope_tables:
            self._get_rope_3d_table(device)
            self._get_rope_2d_table(device, 64)

        if self.solver_config.precompute_time_gates and not self.core.use_token:
            t_seq, _, _ = build_schedule(self.solver_config, device=device)
            t_list = t_seq.tolist()
            for t_val in t_list[:-1]:
                self._get_time_gates(float(t_val), device, target_dtype)
            for t_eval in get_qaoa_eval_timesteps(self.solver_config, t_list):
                self._get_time_gates(float(t_eval), device, target_dtype)

        self._mean_f32 = self.core.mean.detach().view(1, -1).to(device=device, dtype=torch.float32).contiguous()
        self._std_f32 = self.core.std.detach().view(1, -1).to(device=device, dtype=torch.float32).contiguous()
        if self.solver_config.qaoa_enabled:
            self._get_qaoa_tensors(device)

    def _get_qaoa_tensors(self, device: torch.device) -> Dict[str, Any]:
        """Pre-materialize all QAOA variational matrices and vectors on the target device once."""
        key = (device, torch.float32)
        cached = self._qaoa_tensors_cache.get(key)
        if cached is not None:
            return cached

        sc = self.solver_config
        D = self.latent_dim
        out: Dict[str, Any] = {}

        if len(sc.qaoa_rk_init_mixer) == D:
            out["b_init_t"] = torch.tensor(sc.qaoa_rk_init_mixer, device=device, dtype=torch.float32).T.contiguous()
        out["neg_fd_vecs"] = [
            torch.tensor(row, device=device, dtype=torch.float32).view(1, -1).contiguous()
            if len(row) == D
            else None
            for row in sc.qaoa_neg_fd_scales
        ]
        out["step_cfg_vecs"] = [
            torch.tensor(row, device=device, dtype=torch.float32).view(1, -1).contiguous()
            if len(row) == D
            else None
            for row in sc.qaoa_step_cfg_channels
        ]
        out["b_cfg_t"] = [
            torch.tensor(mat, device=device, dtype=torch.float32).T.contiguous()
            if len(mat) == D
            else None
            for mat in sc.qaoa_rk_cfg_mixers
        ]
        out["b_pos_t"] = [
            torch.tensor(mat, device=device, dtype=torch.float32).T.contiguous()
            if len(mat) == D
            else None
            for mat in sc.qaoa_rk_pos_mixers
        ]
        out["mom_vecs"] = [
            torch.tensor(
                [
                    (float(sc.qaoa_momentum[k]) if k < len(sc.qaoa_momentum) else 0.0) + float(x)
                    for x in row
                ],
                device=device,
                dtype=torch.float32,
            ).view(1, -1).contiguous()
            if len(row) == D
            else None
            for k, row in enumerate(sc.qaoa_step_momentum_channels)
        ]
        out["acc_vecs"] = [
            torch.tensor(row, device=device, dtype=torch.float32).view(1, -1).contiguous()
            if len(row) == D
            else None
            for row in sc.qaoa_step_accel_channels
        ]
        out["step_ch_vecs"] = [
            torch.tensor(row, device=device, dtype=torch.float32).view(1, -1).contiguous()
            if len(row) == D
            else None
            for row in sc.qaoa_step_channel_scales
        ]
        if len(sc.qaoa_channel_mixer) == D:
            out["ch_mixer_t"] = torch.tensor(sc.qaoa_channel_mixer, device=device, dtype=torch.float32).T.contiguous()
        if len(sc.qaoa_channel_scale) == D:
            out["ch_scale"] = torch.tensor(sc.qaoa_channel_scale, device=device, dtype=torch.float32).view(1, -1).contiguous()
            out["ch_bias"] = torch.tensor(sc.qaoa_channel_bias, device=device, dtype=torch.float32).view(1, -1).contiguous()
        if len(sc.qaoa_kerr_scales) == D:
            out["kerr_vec"] = torch.tensor(sc.qaoa_kerr_scales, device=device, dtype=torch.float32).view(1, -1).contiguous()

        # Check whether the 128x128 Butcher readout matrices are within physical spectral bounds
        # for arbitrary high-res N-voxel clouds (guard against synthetic basis over-conditioning)
        max_rk_abs = 0.0
        for mat in sc.qaoa_rk_pos_mixers + sc.qaoa_rk_cfg_mixers:
            for row in mat:
                for val in row:
                    if abs(float(val)) > max_rk_abs:
                        max_rk_abs = abs(float(val))
        out["rk_spectral_safe"] = max_rk_abs <= 0.15

        self._qaoa_tensors_cache[key] = out
        return out

    def _get_rope_3d_table(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._rope_3d_cache.get(device)
        if cached is None:
            cached = precompute_rope_3d_table(res=self.res, base=128.0, head_dim=self.head_dim, device=device)
            self._rope_3d_cache[device] = cached
        return cached

    def _get_rope_2d_table(self, device: torch.device, cond_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (device, cond_size)
        cached = self._rope_2d_cache.get(key)
        if cached is None:
            cached = precompute_rope_2d_cond(cond_size=cond_size, base=64.0, head_dim=self.head_dim, device=device)
            self._rope_2d_cache[key] = cached
        return cached

    @torch.no_grad()
    def _get_time_gates(
        self,
        t_val: float,
        device: torch.device,
        dtype: torch.dtype,
        clf_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute (or retrieve cached) timestep embedding and all 24 blocks' modulation/gate tensors."""
        can_cache = (not self.core.use_token) and (
            clf_token is None or clf_token.shape[0] == 1
        )
        cache_key = (device, dtype, round(float(t_val), 7))
        if can_cache and cache_key in self._time_gates_cache:
            return self._time_gates_cache[cache_key]

        batch_len = 1 if clf_token is None else clf_token.shape[0]
        t_tensor = torch.full((batch_len,), float(t_val) * 1000.0, device=device, dtype=torch.float32)
        t_enc = get_time_encoding(t_tensor, self.flow.time_pos_features).to(dtype=dtype)
        temb = self.flow.time_embed(t_enc)
        if clf_token is not None and self.core.use_token:
            clf = F.silu(temb + clf_token.to(dtype=dtype))
        else:
            clf = F.silu(temb)

        clf_shared = self.flow.clf_table_map(clf) if self.flow.clf_shared else clf

        block_gates = []
        cond_gates = []
        for b_idx, block in enumerate(self.flow.blocks):
            gb = block.gates.type_as(clf_shared) + clf_shared if block.clf_shared else block.gates(clf)
            bg_tuple = gb.chunk(9, dim=-1)
            if batch_len == 1:
                s_msa, sh_msa, g_msa, s_cr, sh_cr, g_cr, s_mlp, sh_mlp, g_mlp = bg_tuple
                for s_t, sh_t, g_t in ((s_msa, sh_msa, g_msa), (s_cr, sh_cr, g_cr), (s_mlp, sh_mlp, g_mlp)):
                    s_t._w_1ps = (1.0 + s_t.view(-1)).contiguous()
                    sh_t._b_1d = sh_t.view(-1).contiguous()
                    g_t._g_2d = g_t.view(1, -1).contiguous()
            block_gates.append(bg_tuple)
            if block.cond:
                cgb = block.cond_gates.type_as(clf_shared) + clf_shared if block.clf_shared else block.cond_gates(clf)
                cg_tuple = cgb.chunk(9, dim=-1)
                if batch_len == 1:
                    cs_msa, csh_msa, cg_msa, cs_cr, csh_cr, cg_cr, cs_mlp, csh_mlp, cg_mlp = cg_tuple
                    for s_t, sh_t, g_t in ((cs_msa, csh_msa, cg_msa), (cs_cr, csh_cr, cg_cr), (cs_mlp, csh_mlp, cg_mlp)):
                        s_t._w_1ps = (1.0 + s_t.view(-1)).contiguous()
                        sh_t._b_1d = sh_t.view(-1).contiguous()
                        g_t._g_2d = g_t.view(1, -1).contiguous()
                cond_gates.append(cg_tuple)
            else:
                if block.clf_shared:
                    cgb = block.cond_gates.type_as(clf) + clf_shared[:, 3 * self.dim : 5 * self.dim]
                else:
                    cgb = block.cond_gates(clf)
                cg_tuple2 = cgb.chunk(2, dim=-1)
                if batch_len == 1:
                    cs_cr, csh_cr = cg_tuple2
                    cs_cr._w_1ps = (1.0 + cs_cr.view(-1)).contiguous()
                    csh_cr._b_1d = csh_cr.view(-1).contiguous()
                cond_gates.append(cg_tuple2)

        out_scale, out_shift = self.flow.out_gates(clf).chunk(2, dim=-1)
        if batch_len == 1:
            out_scale._w_1ps = (1.0 + out_scale.view(-1)).contiguous()
            out_shift._b_1d = out_shift.view(-1).contiguous()
        result = {
            "clf": clf,
            "block_gates": block_gates,
            "cond_gates": cond_gates,
            "out_scale": out_scale,
            "out_shift": out_shift,
        }
        if can_cache and len(self._time_gates_cache) < 256:
            self._time_gates_cache[cache_key] = result
        return result

    @torch.no_grad()
    def _prepare_sample_geometry(
        self,
        vox_idxs: List[torch.Tensor],
        context: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Precompute all static geometry and attention partitioning ONCE per sample
        instead of repeating them across all 50-88 diffusion steps.
        """
        device = context.device
        coords_in = sp.sparse_coords(vox_idxs).int()
        N = coords_in.shape[0]
        batch_size = len(vox_idxs)

        # Replicate SparseDownsample(patch_size=1) coordinate sorting once
        code = coords_in[:, 1:].long()
        offs = torch.cumprod((code.max(dim=0).values + 1).flip(dims=(-1,)), 0).flip(dims=(-1,)).long()
        offs = F.pad(offs, (0, 1), value=1)
        full_code = coords_in[:, 0].long() * offs[0] + (code * offs[1:]).sum(dim=-1)
        if batch_size == 1:
            # Single-sample voxel coordinates are strictly unique: argsort avoids D2H stream sync of unique()
            perm = torch.argsort(full_code)
            up_widx = torch.empty_like(perm)
            up_widx[perm] = torch.arange(N, device=device, dtype=perm.dtype)
        else:
            windows, up_widx = full_code.unique(return_inverse=True)
            perm = torch.empty_like(up_widx)
            perm[up_widx] = torch.arange(len(up_widx), device=device, dtype=up_widx.dtype)
        h_coords = coords_in[perm].int().contiguous()

        # Layout & sizes per batch item
        if batch_size == 1:
            sizes = [N]
            layout = [slice(0, N)]
        else:
            seq_len = torch.bincount(h_coords[:, 0], minlength=batch_size)
            offset = torch.cumsum(seq_len, dim=0)
            layout = [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(batch_size)]
            sizes = [lay.stop - lay.start for lay in layout]

        # Precompute 3D RoPE cos/sin for all voxels once
        cos_table_3d, sin_table_3d = self._get_rope_3d_table(device)
        cos_vox, sin_vox = lookup_rope_3d(h_coords[:, 1:], cos_table_3d, sin_table_3d)

        # Precompute 2D RoPE cos/sin for image cond tokens
        cond_tokens = context.shape[1]
        cond_size = int(np.round(np.sqrt(cond_tokens - 5)))
        cos_cond_2d, sin_cond_2d = self._get_rope_2d_table(device, cond_size)

        # Precompute all 4 Hilbert space-filling curve attention orders once
        order_caches = []
        orders = self.flow.orders
        order_kwargs = self.flow.order_kwargs
        patch_size = order_kwargs.get("patch_size", 4096)
        for order in orders:
            if order in ("z", "zt", "h", "ht", "z1", "zt1", "h1", "ht1"):
                fwd_idx, bwd_idx, cu_seqlens, max_len = calc_order_partition(
                    h_coords,
                    sizes,
                    mode=order,
                    patch_size=patch_size,
                    adj_first=order_kwargs.get("adj_first", False),
                    sliding=order_kwargs.get("sliding", False),
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
                fwd_idx = bwd_idx = torch.arange(N, device=device, dtype=torch.long)
                max_len = max(sizes)
                cu_seqlens = torch.cumsum(
                    torch.tensor([0] + sizes, device=device, dtype=torch.int32), dim=0, dtype=torch.int32
                )
            order_caches.append((fwd_idx, bwd_idx, cu_seqlens, int(max_len)))

        # Precompute 3D submanifold sparse conv neighbor indices once (replaces 2,112 calls!)
        conv_indices = build_conv_indices_fast(h_coords)

        # Cross-attention sequence lengths
        q_seqlens = torch.tensor([0] + sizes, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        q_maxlen = int(max(sizes))
        kv_full_key = (device, cond_tokens, batch_size)
        kv_seqlens_full = self._seqlen_cache.get(kv_full_key)
        if kv_seqlens_full is None:
            kv_seqlens_full = torch.tensor(
                [0] + [cond_tokens] * batch_size, device=device, dtype=torch.int32
            ).cumsum(0, dtype=torch.int32)
            self._seqlen_cache[kv_full_key] = kv_seqlens_full
        kv_maxlen_full = int(cond_tokens)
        kv_1tok_key = (device, 1, batch_size)
        kv_seqlens_1tok = self._seqlen_cache.get(kv_1tok_key)
        if kv_seqlens_1tok is None:
            kv_seqlens_1tok = torch.tensor(
                [0] + [1] * batch_size, device=device, dtype=torch.int32
            ).cumsum(0, dtype=torch.int32)
            self._seqlen_cache[kv_1tok_key] = kv_seqlens_1tok
        kv_maxlen_1tok = 1

        return {
            "N": N,
            "batch_size": batch_size,
            "sizes": sizes,
            "layout": layout,
            "perm": perm,
            "up_widx": up_widx,
            "h_coords": h_coords,
            "cos_vox": cos_vox,
            "sin_vox": sin_vox,
            "cos_cond_2d": cos_cond_2d,
            "sin_cond_2d": sin_cond_2d,
            "order_caches": order_caches,
            "conv_indices": conv_indices,
            "q_seqlens": q_seqlens,
            "q_maxlen": q_maxlen,
            "kv_seqlens_full": kv_seqlens_full,
            "kv_maxlen_full": kv_maxlen_full,
            "kv_seqlens_1tok": kv_seqlens_1tok,
            "kv_maxlen_1tok": kv_maxlen_1tok,
        }

    @torch.no_grad()
    def forward_flow_fast(
        self,
        xt_sorted: torch.Tensor,
        t_val: float,
        context: torch.Tensor,
        clf_token: torch.Tensor,
        geom: Dict[str, Any],
        *,
        uncond: bool = False,
        cached_cond_block7: Optional[torch.Tensor] = None,
        return_cond_block7: bool = False,
        cached_mid_delta: Optional[torch.Tensor] = None,
        mid_cache_scale: float = 1.0,
        return_mid_delta: bool = False,
        cached_cond_delta: Optional[torch.Tensor] = None,
        return_cond_delta: bool = False,
        cached_ln_cond7: Optional[torch.Tensor] = None,
        return_ln_cond7: bool = False,
    ):
        """
        Execute 24-block FlowModel forward pass using static geometry cache,
        fused Triton kernels, exact 1-token unconditional condensation,
        and QAOA-calibrated 22-block First-Block Cache (FBC) + Delta-DiT Cond Refresh.
        """
        device = xt_sorted.device
        compute_dtype = self._compute_dtype(device)
        N = geom["N"]
        sizes = geom["sizes"]
        batch_size = geom["batch_size"]
        D, H, HD = self.dim, self.num_heads, self.head_dim

        cos_vox, sin_vox = geom["cos_vox"], geom["sin_vox"]
        cos_cond_2d, sin_cond_2d = geom["cos_cond_2d"], geom["sin_cond_2d"]
        order_caches = geom["order_caches"]
        conv_indices = geom["conv_indices"]
        q_seqlens, q_maxlen = geom["q_seqlens"], geom["q_maxlen"]
        kv_seqlens_full, kv_maxlen_full = geom["kv_seqlens_full"], geom["kv_maxlen_full"]
        kv_seqlens_1tok, kv_maxlen_1tok = geom["kv_seqlens_1tok"], geom["kv_maxlen_1tok"]

        tg = self._get_time_gates(t_val, device, compute_dtype, clf_token)
        block_gates = tg["block_gates"]
        cond_gates = tg["cond_gates"]

        # DiTFastAttn (NeurIPS 2024): Exact Block-0 Patchify + 3D Self-Attention Sharing across CFG!
        # At b_idx == 0, `cond` has not interacted with `x` yet, so `x` after Block-0 3D Self-Attention
        # in `uncond` is BIT-FOR-BIT IDENTICAL to `pos` at the same timestep `t_val`!
        reuse_b0_msa = (
            uncond
            and bool(getattr(self.solver_config, "ditfastattn_b0_msa_share", True))
            and getattr(self, "_last_b0_t_val", None) == t_val
            and getattr(self, "_last_b0_x_after_msa", None) is not None
            and self._last_b0_x_after_msa.shape == (N, D)
            and self._last_b0_x_after_msa.device == device
        )
        if reuse_b0_msa:
            x = self._last_b0_x_after_msa
        else:
            # Patchify (for patch_size=1 on pre-sorted coords, this is a single linear projection)
            x = F.linear(xt_sorted.to(dtype=compute_dtype), self.flow.patchify.weight.to(dtype=compute_dtype))

        use_1tok_uncond = uncond and self.solver_config.uncond_1token_condensation and (batch_size == 1)
        if use_1tok_uncond:
            cond = torch.zeros(1, 1, D, device=device, dtype=compute_dtype)
        elif uncond:
            cond = torch.zeros_like(context, dtype=compute_dtype)
        else:
            cond = context.to(dtype=compute_dtype)

        ln_cond_after_7: Optional[torch.Tensor] = cached_ln_cond7
        kv_cc_cache_out: Optional[List[torch.Tensor]] = [] if return_cond_block7 else None
        use_triton = self.solver_config.use_fused_triton_kernels

        num_blocks = len(self.flow.blocks)
        mid_start = int(getattr(self.solver_config, "mid_block_start", 1))
        raw_mid_end = int(getattr(self.solver_config, "mid_block_end", 24))
        mid_end = min(raw_mid_end, num_blocks) if num_blocks >= 8 else min(raw_mid_end, num_blocks - 1)
        can_mid_cache = (0 < mid_start < mid_end)
        use_freqca_crf = can_mid_cache and (mid_end == num_blocks) and bool(getattr(self.solver_config, "freqca_crf_cache", True))
        use_subblock_crf = use_freqca_crf and bool(getattr(self.solver_config, "dbcache_foca_subblock_crf", True))
        foca_gamma = float(getattr(self.solver_config, "foca_fn_heun_gamma", 0.12))
        x_patchify = x
        x_mid_in: Optional[torch.Tensor] = None
        x_b0_msa: Optional[torch.Tensor] = None
        x_b0_cr: Optional[torch.Tensor] = None
        cond_mid_in: Optional[torch.Tensor] = None
        mid_delta_out: Optional[torch.Tensor] = None
        cond_delta_out: Optional[torch.Tensor] = None

        for b_idx, block in enumerate(self.flow.blocks):
            if can_mid_cache:
                if cached_mid_delta is not None and mid_start <= b_idx < mid_end:
                    if b_idx == mid_start:
                        x = x + cached_mid_delta * float(mid_cache_scale)
                        if (not uncond) and use_freqca_crf and x_b0_msa is not None:
                            # Refresh FreqCa Cumulative Residual Feature (CRF) at Step 2 for Step 3!
                            self._last_pos_crf_after_b0_msa = x - x_b0_msa
                        if cached_cond_delta is not None and ln_cond_after_7 is None:
                            ln_cond_after_7 = fast_layer_norm(cond + cached_cond_delta)
                        if mid_end == num_blocks:
                            # True First-Block Cache (FBCache): tail residual already reaches after block 23!
                            break
                    if kv_cc_cache_out is not None:
                        kv_cc_cache_out.append(None)
                    continue
                if b_idx == mid_start:
                    if return_mid_delta and not (uncond and use_freqca_crf):
                        x_mid_in = x
                    if return_cond_delta:
                        cond_mid_in = cond

            s_msa, sh_msa, g_msa, s_cr, sh_cr, g_cr, s_mlp, sh_mlp, g_mlp = block_gates[b_idx]
            fwd_idx, bwd_idx, cu_seqlens, max_len = order_caches[block.order_idx]

            # --- 1. Windowed Self-Attention on 3D Voxels ---
            if b_idx == 0 and reuse_b0_msa:
                x = self._last_b0_x_after_msa
            else:
                h = fast_modulate(x, sizes, s_msa, sh_msa)
                qkv = block.to_qkv(h).view(N, 3, H, HD)
                q_scale = getattr(block.qk_norm, "_q_scale_cast", block.qk_norm.q_scale)
                k_scale = getattr(block.qk_norm, "_k_scale_cast", block.qk_norm.k_scale)
                qkv_sorted = fused_qknorm_rope3d_gather(
                    qkv, q_scale, k_scale, cos_vox, sin_vox, fwd_idx, use_triton=use_triton
                )
                attn_out = flash_varlen_qkvpacked_already_sorted(qkv_sorted, bwd_idx, cu_seqlens, max_len)
                h = block.to_out(attn_out.reshape(N, D))
                x = fast_gate_add(x, h, sizes, g_msa)
                if b_idx == 0 and not uncond:
                    self._last_b0_x_after_msa = x
                    self._last_b0_t_val = t_val

            if b_idx == 0 and use_freqca_crf:
                if uncond:
                    if cached_mid_delta is not None:
                        # FreqCa CRF (NeurIPS 2026) + DiTFastAttn: `cached_mid_delta` for uncond already
                        # spans from `x_after_b0_msa` to `x_after_23` (including Block-0 cross-attn & FFN)!
                        x = x + cached_mid_delta * float(mid_cache_scale)
                        break
                    elif return_mid_delta:
                        x_mid_in = x
                else:
                    if (
                        cached_cond_block7 is not None
                        and cached_mid_delta is not None
                        and getattr(self, "_last_pos_crf_after_b0_msa", None) is not None
                        and self._last_pos_crf_after_b0_msa.shape == (N, D)
                        and self._last_pos_crf_after_b0_msa.device == device
                    ):
                        # FreqCa CRF (NeurIPS 2026) + DBCache-FoCa Heun Calibrator (vipshop/cache-dit 2026)
                        # on fast pos steps (1, 3): apply cumulative residual + Heun first-block increment!
                        cur_fn_res = x - x_patchify
                        x = x + self._last_pos_crf_after_b0_msa * float(mid_cache_scale)
                        if (
                            use_subblock_crf
                            and foca_gamma > 0.0
                            and getattr(self, "_last_b0_fn_res", None) is not None
                            and self._last_b0_fn_res.shape == (N, D)
                        ):
                            x = x + foca_gamma * (cur_fn_res - self._last_b0_fn_res)
                        break
                    else:
                        x_b0_msa = x
                        if use_subblock_crf:
                            self._last_b0_fn_res = x - x_patchify
                            if cached_mid_delta is None:
                                self._last_b0_fn_res_step0 = self._last_b0_fn_res

            # --- 2. Conditioning Self-Attention + Cross-Attention ---
            if block.cond:
                cs_msa, csh_msa, cg_msa, cs_cr, csh_cr, cg_cr, cs_mlp, csh_mlp, cg_mlp = cond_gates[b_idx]
                # Exact Dead-Branch Elimination (FoCa / FBCache): when the next block (b_idx + 1 == mid_start)
                # applies the full tail residual up to block 23 and breaks, updated `cond` after x->cond is never read!
                skip_cond_out = (
                    can_mid_cache
                    and (cached_mid_delta is not None)
                    and (mid_end == num_blocks)
                    and (b_idx == mid_start - 1)
                    and (not return_cond_delta)
                )

                if use_1tok_uncond:
                    # Exact 1-token unconditional self-attention: only V projection needed!
                    h_cond = fast_modulate_cond(cond, cs_msa, csh_msa)
                    v_w = getattr(block, "_cond_to_v_w", None)
                    if v_w is not None:
                        v_cond = F.linear(h_cond, v_w, getattr(block, "_cond_to_v_b", None))
                    else:
                        v_cond = block.cond_to_qkv(h_cond).view(1, 1, 3, D)[:, :, 2]
                    cond = cond + block.cond_to_out(v_cond) * cg_msa[:, None]

                    # Exact 1-token cross-attention:
                    # x -> cond output is identically v_cond_cross for all N voxels!
                    h_cond_cr = fast_modulate_cond(cond, cs_cr, csh_cr)
                    q_cc, _, v_cc = block.cond_to_qkv_cross(h_cond_cr).view(1, 3, H, HD).unbind(dim=-3)
                    h_x_cross = block.to_out_cross(v_cc.reshape(1, D))

                    if not skip_cond_out:
                        # cond -> x only needed if subsequent blocks read `cond` (must use `x` before gate_add!)
                        h_x = fast_modulate(x, sizes, s_cr, sh_cr)
                        kv_w = getattr(block, "_to_kv_cross_w", None)
                        if kv_w is not None:
                            kv_x = F.linear(h_x, kv_w, getattr(block, "_to_kv_cross_b", None)).view(N, 2, H, HD)
                        else:
                            kv_x = block.to_qkv_cross(h_x).view(N, 3, H, HD)[:, 1:]

                        h_c_cross = flash_cross_attn(
                            q_cc, kv_seqlens_1tok, kv_maxlen_1tok, kv_x, q_seqlens, q_maxlen
                        )
                        cond = cond + block.cond_to_out_cross(h_c_cross.view(1, 1, D)) * cg_cr[:, None]

                        # 1-token cond FFN
                        h_cond_mlp = fast_modulate_cond(cond, cs_mlp, csh_mlp)
                        hc, hc_gate = block.cond_ffn0(h_cond_mlp).chunk(2, dim=-1)
                        cond = cond + block.cond_ffn1(fast_swiglu(hc_gate, hc)) * cg_mlp[:, None]

                    x = fast_gate_add(x, h_x_cross, sizes, g_cr)

                elif cached_cond_block7 is not None:
                    # PAB (Pyramid Attention Broadcast, ICLR 2025): on fast FBCache steps, broadcast
                    # cached Block-0 cross-attention output projection directly with current step's gate `g_cr`!
                    if (
                        b_idx == 0
                        and skip_cond_out
                        and bool(getattr(self.solver_config, "pab_cross_attn_broadcast", True))
                        and getattr(self, "_last_b0_cr_proj", None) is not None
                        and self._last_b0_cr_proj.shape == (N, D)
                        and self._last_b0_cr_proj.device == device
                    ):
                        x = fast_gate_add(x, self._last_b0_cr_proj, sizes, g_cr)
                    else:
                        # Reuse cached K,V of conditioning from previous step; only compute Q_x and x->cond cross-attn!
                        h_x = fast_modulate(x, sizes, s_cr, sh_cr)
                        q_w = getattr(block, "_to_q_cross_w", None)
                        if q_w is not None:
                            q_x = F.linear(h_x, q_w, getattr(block, "_to_q_cross_b", None)).view(N, H, HD)
                        else:
                            q_x = block.to_qkv_cross(h_x).view(N, 3, H, HD)[:, 0]
                        kv_cc = cached_cond_block7[b_idx]
                        h_x_out = flash_cross_attn(
                            q_x, q_seqlens, q_maxlen, kv_cc, kv_seqlens_full, kv_maxlen_full
                        )
                        cr_proj_out = block.to_out_cross(h_x_out.reshape(N, D))
                        if b_idx == 0 and not uncond:
                            self._last_b0_cr_proj = cr_proj_out
                        x = fast_gate_add(x, cr_proj_out, sizes, g_cr)
                else:
                    L = cond.shape[1]
                    h_cond = fast_modulate_cond(cond, cs_msa, csh_msa)
                    qkv_cond = block.cond_to_qkv(h_cond).view(batch_size, L, 3, H, HD)
                    cq_scale = getattr(block.cond_qk_norm, "_q_scale_cast", block.cond_qk_norm.q_scale)
                    ck_scale = getattr(block.cond_qk_norm, "_k_scale_cast", block.cond_qk_norm.k_scale)
                    qkv_cond = fused_qknorm_rope2d_cond_inplace(
                        qkv_cond, cq_scale, ck_scale, cos_cond_2d, sin_cond_2d, use_triton=use_triton
                    )
                    h_cond = flash_cond_self_attn(qkv_cond)
                    cond = cond + block.cond_to_out(h_cond.reshape(batch_size, L, D)) * cg_msa[:, None]

                    h_x = fast_modulate(x, sizes, s_cr, sh_cr)
                    h_cond_cr = fast_modulate_cond(cond, cs_cr, csh_cr)
                    qkv_cc_cr = block.cond_to_qkv_cross(h_cond_cr).view(batch_size * L, 3, H, HD)
                    kv_cc = qkv_cc_cr[:, 1:]
                    if kv_cc_cache_out is not None:
                        kv_cc_cache_out.append(kv_cc)

                    if skip_cond_out:
                        # Only Q_x is needed when updated `cond` is not read by subsequent blocks!
                        q_w = getattr(block, "_to_q_cross_w", None)
                        if q_w is not None:
                            q_x = F.linear(h_x, q_w, getattr(block, "_to_q_cross_b", None)).view(N, H, HD)
                        else:
                            q_x = block.to_qkv_cross(h_x).view(N, 3, H, HD)[:, 0]
                        h_x_out = flash_cross_attn(
                            q_x,
                            q_seqlens,
                            q_maxlen,
                            kv_cc,
                            kv_seqlens_full,
                            kv_maxlen_full,
                        )
                        cr_proj_out = block.to_out_cross(h_x_out.reshape(N, D))
                        if b_idx == 0 and not uncond:
                            self._last_b0_cr_proj = cr_proj_out
                        x = fast_gate_add(x, cr_proj_out, sizes, g_cr)
                    else:
                        qkv_x_cr = block.to_qkv_cross(h_x).view(N, 3, H, HD)
                        h_x_out = flash_cross_attn(
                            qkv_x_cr[:, 0],
                            q_seqlens,
                            q_maxlen,
                            kv_cc,
                            kv_seqlens_full,
                            kv_maxlen_full,
                        )
                        cr_proj_out = block.to_out_cross(h_x_out.reshape(N, D))
                        if b_idx == 0 and not uncond:
                            self._last_b0_cr_proj = cr_proj_out
                        x = fast_gate_add(x, cr_proj_out, sizes, g_cr)

                        h_c_out = flash_cross_attn(
                            qkv_cc_cr[:, 0],
                            kv_seqlens_full,
                            kv_maxlen_full,
                            qkv_x_cr[:, 1:],
                            q_seqlens,
                            q_maxlen,
                        )
                        cond = cond + block.cond_to_out_cross(h_c_out.reshape(batch_size, L, D)) * cg_cr[:, None]

                        h_cond_mlp = fast_modulate_cond(cond, cs_mlp, csh_mlp)
                        hc, hc_gate = block.cond_ffn0(h_cond_mlp).chunk(2, dim=-1)
                        cond = cond + block.cond_ffn1(fast_swiglu(hc_gate, hc)) * cg_mlp[:, None]

                if b_idx == self.cond_depth - 1 and cached_cond_block7 is None:
                    if return_cond_delta and cond_mid_in is not None:
                        cond_delta_out = cond - cond_mid_in
                    ln_cond_after_7 = fast_layer_norm(cond)

            else:
                cs_cr, csh_cr = cond_gates[b_idx]

                if use_1tok_uncond:
                    # Exact 1-token cross-attn in blocks 8..23:
                    # Skips modulate(x), to_q_cross(x), FlashAttention, and N-voxel to_out_cross!
                    h_cond_cr = ln_cond_after_7 * (1.0 + cs_cr[:, None]) + csh_cr[:, None]
                    v_w = getattr(block, "_cond_to_v_cross_w", None)
                    if v_w is not None:
                        v_cc = F.linear(h_cond_cr, v_w, getattr(block, "_cond_to_v_cross_b", None)).view(1, D)
                    else:
                        v_cc = block.cond_to_kv_cross(h_cond_cr).view(1, 2, D)[:, 1]
                    x = fast_gate_add(x, block.to_out_cross(v_cc), sizes, g_cr)
                elif cached_cond_block7 is not None:
                    h_x = fast_modulate(x, sizes, s_cr, sh_cr)
                    q_x = block.to_q_cross(h_x).view(N, H, HD)
                    kv_cc = cached_cond_block7[b_idx]
                    h_x_out = flash_cross_attn(
                        q_x, q_seqlens, q_maxlen, kv_cc, kv_seqlens_full, kv_maxlen_full
                    )
                    x = fast_gate_add(x, block.to_out_cross(h_x_out.reshape(N, D)), sizes, g_cr)
                else:
                    L = cond.shape[1]
                    h_cond_cr = ln_cond_after_7 * (1.0 + cs_cr[:, None]) + csh_cr[:, None]
                    h_x = fast_modulate(x, sizes, s_cr, sh_cr)
                    q_x = block.to_q_cross(h_x).view(N, H, HD)
                    kv_cc = block.cond_to_kv_cross(h_cond_cr).view(batch_size * L, 2, H, HD)
                    if kv_cc_cache_out is not None:
                        kv_cc_cache_out.append(kv_cc)
                    h_x_out = flash_cross_attn(
                        q_x, q_seqlens, q_maxlen, kv_cc, kv_seqlens_full, kv_maxlen_full
                    )
                    x = fast_gate_add(x, block.to_out_cross(h_x_out.reshape(N, D)), sizes, g_cr)

            if b_idx == 0 and (not uncond) and use_subblock_crf:
                if cached_mid_delta is not None and (
                    getattr(self, "_last_pos_crf_after_b0_cr", None) is not None
                    and self._last_pos_crf_after_b0_cr.shape == (N, D)
                    and self._last_pos_crf_after_b0_cr.device == device
                ):
                    # DBCache-FoCa Subblock-2 CRF (vipshop/cache-dit 2026 + FreqCa NeurIPS 2026):
                    # On Step 2 (pos), after updating 3D MSA + DINO Cross-Attn in Block 0,
                    # bypass Block 0's heavy FFN0(1536->8192) + 3x3x3 SparseConv + FFN1 and Blocks 1..23!
                    x = x + self._last_pos_crf_after_b0_cr * float(mid_cache_scale)
                    if (
                        foca_gamma > 0.0
                        and x_b0_msa is not None
                        and getattr(self, "_last_b0_fn_res_step0", None) is not None
                        and self._last_b0_fn_res_step0.shape == (N, D)
                    ):
                        x = x + foca_gamma * ((x_b0_msa - x_patchify) - self._last_b0_fn_res_step0)
                    if x_b0_msa is not None:
                        self._last_pos_crf_after_b0_msa = x - x_b0_msa
                    break
                else:
                    x_b0_cr = x

            # --- 3. Main FFN + 3D Submanifold Sparse Conv + SwiGLU ---
            h_mlp = fast_modulate(x, sizes, s_mlp, sh_mlp)
            h_mlp = block.ffn0(h_mlp)
            if block.use_ffn_conv:
                w_fp32 = getattr(block, "_ffn_conv_w_fp32", block.ffn_conv.weight.float().contiguous())
                b_fp32 = getattr(block, "_ffn_conv_b_fp32", block.ffn_conv.bias.float().contiguous())
                h_mlp = fast_sparse_conv3d(h_mlp, w_fp32, b_fp32, conv_indices)
            h1, h_gate = h_mlp.chunk(2, dim=-1)
            h_mlp = block.ffn1(fast_swiglu(h_gate, h1))
            x = fast_gate_add(x, h_mlp, sizes, g_mlp)

            if can_mid_cache and b_idx == (mid_end - 1) and return_mid_delta and x_mid_in is not None:
                mid_delta_out = x - x_mid_in
                if (not uncond) and use_freqca_crf and x_b0_msa is not None:
                    self._last_pos_crf_after_b0_msa = x - x_b0_msa
                if (not uncond) and use_subblock_crf and x_b0_cr is not None:
                    self._last_pos_crf_after_b0_cr = x - x_b0_cr

        # Output LayerNorm + exact FP32 Unpatchify projection (zero BF16 output truncation)
        x = fast_modulate(x, sizes, tg["out_scale"], tg["out_shift"])
        out = F.linear(x.float(), self._unpatchify_w_fp32)

        if return_cond_block7 and return_mid_delta and return_cond_delta:
            return out, kv_cc_cache_out, mid_delta_out, cond_delta_out
        if return_cond_block7 and return_mid_delta:
            return out, kv_cc_cache_out, mid_delta_out
        if return_cond_block7:
            return out, kv_cc_cache_out
        if return_mid_delta and return_ln_cond7:
            return out, mid_delta_out, ln_cond_after_7
        if return_mid_delta:
            return out, mid_delta_out
        return out

    @torch.no_grad()
    def warmup(self) -> None:
        """Run a tiny dummy pass at load time to warm up CUDA/Triton/FlashAttention kernels."""
        device = self._runtime_device()
        if device.type != "cuda":
            return
        try:
            dummy_vox = [torch.arange(64, device=device, dtype=torch.int32).unsqueeze(-1).repeat(1, 3)]
            dummy_dino = torch.zeros(1, 4101, 1280, device=device, dtype=torch.float16)
            dummy_cond = LatDitConditioning(dino_features=dummy_dino, batch_size=1)
            dummy_sampling = FlowMatchSamplingParams(
                guidance_scale=7.0,
                cfg_range=(0.5, 1.0),
                rescale_t=3.0,
                num_inference_steps=50,
            )
            self.sample_fast(dummy_vox, dummy_cond, sampling=dummy_sampling)
            torch.cuda.synchronize(device)
        except Exception:
            pass

    @torch.no_grad()
    def sample_fast(
        self,
        vox_idxs: List[torch.Tensor],
        conditioning: LatDitConditioning,
        *,
        sampling: Optional[FlowMatchSamplingParams] = None,
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        # Fallback if non-standard architecture flags are used
        if (
            self.flow.io_conv
            or self.flow.patchify.patch_size != 1
            or getattr(self.flow.blocks[0], "use_pos_conv", False)
        ):
            return self.core.sample_sparse(
                vox_idxs, conditioning, sampling=sampling, generator=generator
            )

        device = self._runtime_device()
        self.core.device = device
        data_dict = conditioning.to_data_dict()
        data_dict["vox_idxs"] = vox_idxs
        data_dict["eval_mode"] = True

        compute_dtype = self._compute_dtype(device)
        dino_feats = data_dict["dino_features"].to(device=device, dtype=compute_dtype)
        context = self.core.image_map(dino_feats)

        if self.core.use_token and "dino_token" in data_dict:
            clf_token = self.core.token_map(data_dict["dino_token"].to(device=device, dtype=compute_dtype))
        else:
            clf_token = torch.zeros(
                conditioning.batch_size, self.dim, device=device, dtype=compute_dtype
            )

        geom = self._prepare_sample_geometry(vox_idxs, context)
        N = geom["N"]

        noise = torch.randn(
            N,
            self.latent_dim,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        # Permute initial noise into patchify coordinate order once before the ODE loop
        xt = noise[geom["perm"]].contiguous()

        # Resolve sampling overrides
        num_steps_override = None
        rescale_t_override = None
        cfg_range_override = None
        guidance_scale = self.solver_config.guidance_scale

        if sampling is not None:
            guidance_scale = float(sampling.guidance_scale)
            rescale_t_override = float(sampling.rescale_t)
            cfg_range_override = tuple(float(x) for x in sampling.cfg_range)
            if sampling.num_inference_steps != 50:
                num_steps_override = int(sampling.num_inference_steps)

        t_seq, is_guided, n_guided = build_schedule(
            self.solver_config,
            num_steps_override=num_steps_override,
            rescale_t_override=rescale_t_override,
            cfg_range_override=cfg_range_override,
            device=device,
        )
        t_list = t_seq.tolist()
        total_steps = len(is_guided)
        eval_t_list = get_qaoa_eval_timesteps(self.solver_config, t_list)

        use_ab2 = self.solver_config.solver_type == "piecewise_ab2"
        use_qaoa = self.solver_config.qaoa_enabled and (
            self.solver_config.solver_type == "piecewise_qaoa"
            or len(self.solver_config.qaoa_dt_scales) > 0
        )
        ab2_max_ratio = float(self.solver_config.ab2_max_ratio)
        cfg_stride = int(self.solver_config.cfg_stride)
        cfg_mode = self.solver_config.cfg_cache_mode
        cfg_alpha = float(self.solver_config.cfg_extrapolation_alpha)
        cond7_stride = int(self.solver_config.cache_cond_block7_stride)
        eval_cond_steps = self.solver_config.eval_cond_steps
        eval_uncond_steps = self.solver_config.eval_uncond_steps

        prev_v: Optional[torch.Tensor] = None
        prev2_v: Optional[torch.Tensor] = None
        prev_v_pos: Optional[torch.Tensor] = None
        prev_dt: Optional[float] = None
        last_delta: Optional[torch.Tensor] = None
        prev_delta: Optional[torch.Tensor] = None
        last_neg: Optional[torch.Tensor] = None
        last_cond7: Optional[List[torch.Tensor]] = None
        last_mid_delta: Optional[torch.Tensor] = None
        last_cond_delta: Optional[torch.Tensor] = None
        last_uncond_mid_delta: Optional[torch.Tensor] = None
        last_uncond_ln7: Optional[torch.Tensor] = None
        rk_corr: Optional[torch.Tensor] = None
        use_mid_cache = bool(getattr(self.solver_config, "mid_block_cache_enabled", False))
        use_uncond_mid_cache = use_mid_cache and bool(getattr(self.solver_config, "uncond_mid_block_cache", True))
        mid_scales = getattr(self.solver_config, "qaoa_mid_cache_scales", [])
        uncond_mid_sc = float(getattr(self.solver_config, "qaoa_uncond_mid_scale", 0.95))

        qt = self._get_qaoa_tensors(device) if use_qaoa else {}
        apply_rk_readout = use_qaoa and bool(qt.get("rk_spectral_safe", False))
        b_init_t = qt.get("b_init_t") if apply_rk_readout else None
        if b_init_t is not None:
            rk_corr = xt @ b_init_t

        self._last_b0_x_after_msa = None
        self._last_b0_cr_proj = None
        self._last_pos_crf_after_b0_msa = None
        self._last_pos_crf_after_b0_cr = None
        self._last_b0_fn_res = None
        self._last_b0_fn_res_step0 = None

        for step_idx in range(total_steps):
            t_val = float(t_list[step_idx])
            t_next = float(t_list[step_idx + 1])
            t_eval = float(eval_t_list[step_idx])
            dt = t_val - t_next
            guided = (guidance_scale > 0.0) and is_guided[step_idx]

            # Reset multistep history across the t=0.5 CFG boundary jump
            if step_idx == n_guided:
                prev_v = None
                prev2_v = None
                prev_dt = None

            eval_cond = should_eval_cond(step_idx, n_guided, cond7_stride, eval_cond_steps)
            need_cond_cache = (cond7_stride > 1) or (eval_cond_steps is not None and len(eval_cond_steps) > 0)
            mid_sc = float(mid_scales[step_idx]) if step_idx < len(mid_scales) else 1.0

            if (not eval_cond) and (last_cond7 is not None):
                # Fast steps (1, 3): execute only blocks 0 and 23 (skipping 22 middle blocks 1..22)!
                v_pos = self.forward_flow_fast(
                    xt,
                    t_eval,
                    context,
                    clf_token,
                    geom,
                    uncond=False,
                    cached_cond_block7=last_cond7,
                    cached_mid_delta=last_mid_delta if use_mid_cache else None,
                    mid_cache_scale=mid_sc,
                )
            elif need_cond_cache and use_mid_cache and (last_mid_delta is not None):
                # Step 2+: refresh DINO conditioning in block 0 at new t_eval, reconstruct ln_cond_after_7
                # via Delta-DiT `last_cond_delta`, and apply True First-Block Cache (FBCache)!
                v_pos, new_cond7 = self.forward_flow_fast(
                    xt,
                    t_eval,
                    context,
                    clf_token,
                    geom,
                    uncond=False,
                    return_cond_block7=True,
                    cached_mid_delta=last_mid_delta,
                    mid_cache_scale=mid_sc,
                    cached_cond_delta=last_cond_delta,
                )
                if last_cond7 is None:
                    last_cond7 = new_cond7
                elif new_cond7 is not None:
                    for b_i, kv_new in enumerate(new_cond7):
                        if kv_new is not None and b_i < len(last_cond7):
                            last_cond7[b_i] = kv_new
            elif need_cond_cache and use_mid_cache:
                v_pos, last_cond7, last_mid_delta, last_cond_delta = self.forward_flow_fast(
                    xt,
                    t_eval,
                    context,
                    clf_token,
                    geom,
                    uncond=False,
                    return_cond_block7=True,
                    return_mid_delta=True,
                    return_cond_delta=True,
                )
            elif need_cond_cache:
                v_pos, last_cond7 = self.forward_flow_fast(
                    xt, t_eval, context, clf_token, geom, uncond=False, return_cond_block7=True
                )
            else:
                v_pos = self.forward_flow_fast(
                    xt, t_eval, context, clf_token, geom, uncond=False
                )

            if guided:
                eval_neg = (
                    should_eval_uncond(step_idx, n_guided, cfg_stride, cfg_mode, eval_uncond_steps)
                    or (last_delta is None)
                )
                if eval_neg:
                    if use_uncond_mid_cache and last_uncond_mid_delta is None:
                        v_neg, last_uncond_mid_delta, last_uncond_ln7 = self.forward_flow_fast(
                            xt,
                            t_eval,
                            context,
                            clf_token,
                            geom,
                            uncond=True,
                            return_mid_delta=True,
                            return_ln_cond7=True,
                        )
                    elif use_uncond_mid_cache and last_uncond_mid_delta is not None:
                        v_neg = self.forward_flow_fast(
                            xt,
                            t_eval,
                            context,
                            clf_token,
                            geom,
                            uncond=True,
                            cached_mid_delta=last_uncond_mid_delta,
                            mid_cache_scale=uncond_mid_sc,
                            cached_ln_cond7=last_uncond_ln7,
                        )
                    else:
                        v_neg = self.forward_flow_fast(
                            xt, t_eval, context, clf_token, geom, uncond=True
                        )
                    cur_delta = v_pos - v_neg
                    if last_delta is not None:
                        prev_delta = last_delta
                    last_delta = cur_delta
                    last_neg = v_neg
                else:
                    if cfg_mode == "neg_reuse" and last_neg is not None:
                        neg_fd_vecs = qt.get("neg_fd_vecs", [])
                        fd_vec = (
                            neg_fd_vecs[step_idx]
                            if (apply_rk_readout and prev_v_pos is not None and step_idx < len(neg_fd_vecs))
                            else None
                        )
                        if fd_vec is not None:
                            neg_est = torch.addcmul(last_neg, v_pos - prev_v_pos, fd_vec, value=1.0)
                            cur_delta = v_pos - neg_est
                            last_neg = neg_est
                        elif bool(getattr(self.solver_config, "cfg_zero_star_projection", True)):
                            # CFG-Zero* (arXiv:2503.18886): optimal scalar projection s* = <v_pos, v_neg> / ||v_neg||^2
                            dot_pn = (v_pos * last_neg).sum()
                            norm_nn = (last_neg * last_neg).sum().clamp_min(1e-6)
                            s_star = (dot_pn / norm_nn).clamp(0.90, 1.10)
                            cur_delta = v_pos - s_star * last_neg
                        else:
                            cur_delta = v_pos - last_neg
                    elif cfg_mode == "delta_extrapolate" and prev_delta is not None:
                        cur_delta = last_delta + cfg_alpha * (last_delta - prev_delta)
                    else:
                        cur_delta = last_delta

                cfg_mult = 1.0
                if use_qaoa and step_idx < len(self.solver_config.qaoa_cfg_scales):
                    cfg_mult = float(self.solver_config.qaoa_cfg_scales[step_idx])
                step_cfg_vecs = qt.get("step_cfg_vecs", [])
                cfg_ch = step_cfg_vecs[step_idx] if (apply_rk_readout and step_idx < len(step_cfg_vecs)) else None
                if cfg_ch is not None:
                    v = torch.addcmul(v_pos, cur_delta, cfg_ch, value=guidance_scale * cfg_mult)
                else:
                    v = v_pos + (guidance_scale * cfg_mult) * cur_delta

                b_cfg_t_list = qt.get("b_cfg_t", [])
                b_cfg_t = b_cfg_t_list[step_idx] if (apply_rk_readout and step_idx < len(b_cfg_t_list)) else None
                if b_cfg_t is not None:
                    cfg_rk_feat = (cur_delta * v_pos) if (not eval_neg) else cur_delta
                    if rk_corr is None:
                        rk_corr = cfg_rk_feat @ b_cfg_t
                    else:
                        rk_corr = torch.addmm(rk_corr, cfg_rk_feat, b_cfg_t)
            else:
                v = v_pos

            b_pos_t_list = qt.get("b_pos_t", [])
            b_pos_t = b_pos_t_list[step_idx] if (apply_rk_readout and step_idx < len(b_pos_t_list)) else None
            if b_pos_t is not None:
                if rk_corr is None:
                    rk_corr = v_pos @ b_pos_t
                else:
                    rk_corr = torch.addmm(rk_corr, v_pos, b_pos_t)

            prev_v_pos = v_pos

            # Optional learned per-step calibration from distillation
            if self.step_calibration is not None:
                scales = self.step_calibration.get("scales")
                shifts = self.step_calibration.get("shifts")
                if scales is not None and step_idx < scales.shape[0]:
                    s_i = scales[step_idx].to(device=device, dtype=v.dtype).view(1, -1)
                    v = v * s_i
                if shifts is not None and step_idx < shifts.shape[0]:
                    b_i = shifts[step_idx].to(device=device, dtype=v.dtype).view(1, -1)
                    v = v + b_i

            if use_qaoa:
                mom = 0.0
                if step_idx < len(self.solver_config.qaoa_momentum):
                    mom = float(self.solver_config.qaoa_momentum[step_idx])
                mom_vecs = qt.get("mom_vecs", [])
                mom_vec = mom_vecs[step_idx] if (apply_rk_readout and step_idx < len(mom_vecs)) else None
                if prev_v is not None and (abs(mom) > 1e-7 or mom_vec is not None):
                    if mom_vec is not None:
                        v_step = torch.addcmul(v, v - prev_v, mom_vec, value=1.0)
                    else:
                        v_step = (1.0 + mom) * v - mom * prev_v
                else:
                    v_step = v

                acc_vecs = qt.get("acc_vecs", [])
                acc_vec = acc_vecs[step_idx] if (apply_rk_readout and step_idx < len(acc_vecs)) else None
                if prev2_v is not None and prev_v is not None:
                    if acc_vec is not None:
                        v_step = torch.addcmul(v_step, v - 2.0 * prev_v + prev2_v, acc_vec, value=1.0)
                    else:
                        # HiCache Dual-Scaled Hermite 2nd-Order Velocity Forecast + Faster-TRELLIS Directional Gate
                        hermite_sigma = float(getattr(self.solver_config, "hicache_hermite_sigma", 0.18))
                        if hermite_sigma > 0.0:
                            dv_cur = v - prev_v
                            dv_prev = prev_v - prev2_v
                            if float((dv_cur * dv_prev).mean().item()) > 0.0:
                                v_step = v_step + (0.5 * hermite_sigma * hermite_sigma) * (dv_cur - dv_prev)

                dt_scale = 1.0
                if apply_rk_readout and step_idx < len(self.solver_config.qaoa_dt_scales):
                    dt_scale = float(self.solver_config.qaoa_dt_scales[step_idx])
                dt_eff = dt * dt_scale
            elif use_ab2 and (prev_v is not None) and (prev_dt is not None):
                ratio = min(0.5 * (dt / max(prev_dt, 1e-6)), ab2_max_ratio)
                v_step = (1.0 + ratio) * v - ratio * prev_v
                dt_eff = dt
            else:
                v_step = v
                dt_eff = dt

            prev2_v = prev_v
            prev_v = v
            prev_dt = dt
            step_ch_vecs = qt.get("step_ch_vecs", [])
            ch_vec = step_ch_vecs[step_idx] if (apply_rk_readout and step_idx < len(step_ch_vecs)) else None
            if ch_vec is not None:
                xt = torch.addcmul(xt, v_step, ch_vec, value=-dt_eff)
            else:
                xt = xt - dt_eff * v_step

        # Optional QAOA 16x16 cross-channel linear readout mixer + per-channel scale/bias + Kerr cubic term
        if apply_rk_readout:
            ch_mixer_t = qt.get("ch_mixer_t")
            if ch_mixer_t is not None:
                xt = xt @ ch_mixer_t
            ch_scale = qt.get("ch_scale")
            ch_bias = qt.get("ch_bias")
            if ch_scale is not None and ch_bias is not None:
                xt = xt * ch_scale + ch_bias
            if rk_corr is not None:
                xt = xt + rk_corr
            kerr_vec = qt.get("kerr_vec")
            if kerr_vec is not None:
                xt = torch.addcmul(xt, xt * xt * xt, kerr_vec, value=1.0)

        # Unpermute back to original vox_idxs order and denormalize once
        xt_orig = xt[geom["up_widx"]]
        xt_denorm = (xt_orig * self._std_f32 + self._mean_f32) * self.latent_scale

        return [xt_denorm[lay] for lay in geom["layout"]]

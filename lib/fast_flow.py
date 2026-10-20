"""Optimized LatDiT execution engine with static geometry hoisting and 1-token CFG condensation."""

from __future__ import annotations

import math
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
    build_order_caches_fast,
    fast_gate,
    fast_gate_add,
    fast_gate_add_and_ln_cond,
    fast_gate_add_and_modulate,
    fast_gate_add_and_modulate_cond,
    fast_gate_add_cond,
    fast_layer_norm,
    fast_modulate,
    fast_modulate_cond,
    fast_modulate_preln_cond,
    fast_out_modulate_unpatchify,
    fast_rms_norm,
    fast_sparse_conv3d,
    fast_swiglu,
    fast_swiglu_from_packed,
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
    compute_anderson_aag_gain,
    compute_arex_affine_defect,
    compute_cab_nsr_step_ratio,
    compute_cachescope_worldcache_osi_weight,
    compute_exp_rosenbrock_phi_gains,
    compute_flow_unipc_bh2_scale,
    compute_manifold_spectral_fdc_weight,
    compute_postcache_rts_gain,
    compute_racer_trust_shrinkage,
    compute_seacache_softcache_weight,
    compute_spectral_cache_scale,
    compute_stork2_beta,
    compute_stork4_rock4_weights,
    compute_tacache_vde_mcfm_gains,
    compute_gp_svd_subspace_weight,
    compute_slerpflow_strain_vorticity_gains,
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
        if hasattr(self.flow.patchify, "weight"):
            self._patchify_w_cast = (
                self.flow.patchify.weight.detach().to(device=device, dtype=target_dtype).contiguous()
            )
        else:
            self._patchify_w_cast = None

        D = self.dim
        for block in self.flow.blocks:
            if block.clf_shared:
                block._gates_cast = block.gates.detach().to(device=device, dtype=target_dtype).contiguous()
                block._cond_gates_cast = block.cond_gates.detach().to(device=device, dtype=target_dtype).contiguous()
            if block.use_ffn_conv:
                block._ffn_conv_w_fp32 = block.ffn_conv.weight.detach().to(device=device, dtype=torch.float32).contiguous()
                block._ffn_conv_b_fp32 = block.ffn_conv.bias.detach().to(device=device, dtype=torch.float32).contiguous()
            if block.rms:
                block.qk_norm._q_scale_cast = block.qk_norm.q_scale.detach().to(device=device, dtype=target_dtype).contiguous()
                block.qk_norm._k_scale_cast = block.qk_norm.k_scale.detach().to(device=device, dtype=target_dtype).contiguous()
                block.qk_norm._q_scale_cast._qk_stacked = torch.stack(
                    [block.qk_norm._q_scale_cast, block.qk_norm._k_scale_cast], dim=0
                ).unsqueeze(0).contiguous()
                if block.cond:
                    block.cond_qk_norm._q_scale_cast = (
                        block.cond_qk_norm.q_scale.detach().to(device=device, dtype=target_dtype).contiguous()
                    )
                    block.cond_qk_norm._k_scale_cast = (
                        block.cond_qk_norm.k_scale.detach().to(device=device, dtype=target_dtype).contiguous()
                    )
                    block.cond_qk_norm._q_scale_cast._qk_stacked = torch.stack(
                        [block.cond_qk_norm._q_scale_cast, block.cond_qk_norm._k_scale_cast], dim=0
                    ).unsqueeze(0).unsqueeze(0).contiguous()
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
                # In cond_to_qkv_cross ([3*D, D]), slice [:D] is Q_cc only and slice [D:3*D] is K_cc, V_cc only
                block._cond_to_q_cross_slice_w = block.cond_to_qkv_cross.weight[:D, :].contiguous()
                block._cond_to_q_cross_slice_b = (
                    block.cond_to_qkv_cross.bias[:D].contiguous()
                    if block.cond_to_qkv_cross.bias is not None
                    else None
                )
                block._cond_to_kv_cross_slice_w = block.cond_to_qkv_cross.weight[D:, :].contiguous()
                block._cond_to_kv_cross_slice_b = (
                    block.cond_to_qkv_cross.bias[D:].contiguous()
                    if block.cond_to_qkv_cross.bias is not None
                    else None
                )
                # And slice [2*D:3*D] is V_cc only (used when use_1tok_uncond=True and skip_cond_out=True!)
                block._cond_to_v_cross_slice_w = block.cond_to_qkv_cross.weight[2 * D :, :].contiguous()
                block._cond_to_v_cross_slice_b = (
                    block.cond_to_qkv_cross.bias[2 * D :].contiguous()
                    if block.cond_to_qkv_cross.bias is not None
                    else None
                )
                # Algebraic 2-Layer Linear Collapse for 1-Token Uncond Self-Attn & Cross-Attn:
                # 1. `cond_to_out(cond_to_v(h_cond))` -> single GEMM `_cond_1tok_msa_w/b`
                w_out_m = block.cond_to_out.weight.detach().float()
                b_out_m = (
                    block.cond_to_out.bias.detach().float()
                    if block.cond_to_out.bias is not None
                    else torch.zeros(D, device=device, dtype=torch.float32)
                )
                w_v_m = block._cond_to_v_w.detach().float()
                b_v_m = (
                    block._cond_to_v_b.detach().float()
                    if block._cond_to_v_b is not None
                    else torch.zeros(D, device=device, dtype=torch.float32)
                )
                block._cond_1tok_msa_w = (w_out_m @ w_v_m).to(dtype=target_dtype).contiguous()
                block._cond_1tok_msa_b = (w_out_m @ b_v_m + b_out_m).to(dtype=target_dtype).contiguous()

                # 2. `to_out_cross(cond_to_v_cross(h_cond_cr))` -> single GEMM `_cond_1tok_cr_w/b`
                #    plus packed `[Q_cc; to_out_cross(V_cc)]` (`[2*D, D]`) for `not skip_cond_out`!
                w_out_cr = block.to_out_cross.weight.detach().float()
                b_out_cr = (
                    block.to_out_cross.bias.detach().float()
                    if block.to_out_cross.bias is not None
                    else torch.zeros(D, device=device, dtype=torch.float32)
                )
                w_vc = block._cond_to_v_cross_slice_w.detach().float()
                b_vc = (
                    block._cond_to_v_cross_slice_b.detach().float()
                    if block._cond_to_v_cross_slice_b is not None
                    else torch.zeros(D, device=device, dtype=torch.float32)
                )
                w_cr_1tok = w_out_cr @ w_vc
                b_cr_1tok = w_out_cr @ b_vc + b_out_cr
                block._cond_1tok_cr_w = w_cr_1tok.to(dtype=target_dtype).contiguous()
                block._cond_1tok_cr_b = b_cr_1tok.to(dtype=target_dtype).contiguous()

                w_qc = block.cond_to_qkv_cross.weight[:D, :].detach().float()
                b_qc = (
                    block.cond_to_qkv_cross.bias[:D].detach().float()
                    if block.cond_to_qkv_cross.bias is not None
                    else torch.zeros(D, device=device, dtype=torch.float32)
                )
                block._cond_1tok_q_and_out_cr_w = (
                    torch.cat([w_qc, w_cr_1tok], dim=0).to(dtype=target_dtype).contiguous()
                )
                block._cond_1tok_q_and_out_cr_b = (
                    torch.cat([b_qc, b_cr_1tok], dim=0).to(dtype=target_dtype).contiguous()
                )
            else:
                # In cond_to_kv_cross ([2*D, D]), slice [D:2*D] is V only
                block._cond_to_v_cross_w = block.cond_to_kv_cross.weight[D:, :].contiguous()
                block._cond_to_v_cross_b = (
                    block.cond_to_kv_cross.bias[D:].contiguous() if block.cond_to_kv_cross.bias is not None else None
                )
                # Algebraic 2-Layer Linear Collapse in blocks 8..23:
                # `to_out_cross(cond_to_v_cross(h_cond_cr))` -> single GEMM `_cond_1tok_cr_w/b`!
                w_out_cr = block.to_out_cross.weight.detach().float()
                b_out_cr = (
                    block.to_out_cross.bias.detach().float()
                    if block.to_out_cross.bias is not None
                    else torch.zeros(D, device=device, dtype=torch.float32)
                )
                w_vc = block._cond_to_v_cross_w.detach().float()
                b_vc = (
                    block._cond_to_v_cross_b.detach().float()
                    if block._cond_to_v_cross_b is not None
                    else torch.zeros(D, device=device, dtype=torch.float32)
                )
                block._cond_1tok_cr_w = (w_out_cr @ w_vc).to(dtype=target_dtype).contiguous()
                block._cond_1tok_cr_b = (w_out_cr @ b_vc + b_out_cr).to(dtype=target_dtype).contiguous()

        noncond_w_t = [b._cond_1tok_cr_w.t().contiguous() for b in self.flow.blocks if not b.cond]
        noncond_b = [b._cond_1tok_cr_b.view(1, D).contiguous() for b in self.flow.blocks if not b.cond]
        if noncond_w_t:
            self._noncond_1tok_w_t = torch.stack(noncond_w_t, dim=0).contiguous()
            self._noncond_1tok_b = torch.stack(noncond_b, dim=0).contiguous()
        else:
            self._noncond_1tok_w_t = None
            self._noncond_1tok_b = None

        if self.flow.clf_shared and all(b.clf_shared for b in self.flow.blocks):
            self._stacked_block_gates_raw = torch.stack(
                [b._gates_cast.view(9, D) for b in self.flow.blocks], dim=0
            ).contiguous()
            cond_blocks = [b for b in self.flow.blocks if b.cond]
            noncond_blocks = [b for b in self.flow.blocks if not b.cond]
            self._stacked_cond_gates_8_raw = (
                torch.stack([b._cond_gates_cast.view(9, D) for b in cond_blocks], dim=0).contiguous()
                if cond_blocks
                else None
            )
            self._stacked_cond_gates_16_raw = (
                torch.stack([b._cond_gates_cast.view(2, D) for b in noncond_blocks], dim=0).contiguous()
                if noncond_blocks
                else None
            )
        else:
            self._stacked_block_gates_raw = None
            self._stacked_cond_gates_8_raw = None
            self._stacked_cond_gates_16_raw = None

        self._zero_clf_token = torch.zeros(1, self.dim, device=device, dtype=target_dtype).contiguous()

        if self.solver_config.precompute_rope_tables:
            self._get_rope_3d_table(device)
            self._get_rope_2d_table(device, 64)

        if self.solver_config.precompute_time_gates:
            t_seq, _, _ = build_schedule(self.solver_config, device=device)
            t_list = t_seq.tolist()
            for t_val in t_list[:-1]:
                self._get_time_gates(float(t_val), device, target_dtype, self._zero_clf_token)
            for t_eval in get_qaoa_eval_timesteps(self.solver_config, t_list):
                self._get_time_gates(float(t_eval), device, target_dtype, self._zero_clf_token)

        self._mean_f32 = self.core.mean.detach().view(1, -1).to(device=device, dtype=torch.float32).contiguous()
        self._std_f32 = self.core.std.detach().view(1, -1).to(device=device, dtype=torch.float32).contiguous()
        self._denorm_scale = (self._std_f32 * self.latent_scale).contiguous()
        self._denorm_shift = (self._mean_f32 * self.latent_scale).contiguous()
        self._is_identity_denorm = bool(
            torch.all(self._denorm_scale == 1.0).item() and torch.all(self._denorm_shift == 0.0).item()
        )
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
        block0_only: bool = False,
    ) -> Dict[str, Any]:
        """Compute (or retrieve cached) timestep embedding and block modulation/gate tensors."""
        batch_len = 1 if clf_token is None else clf_token.shape[0]
        is_zero_clf = (
            (not self.core.use_token)
            or clf_token is None
            or (clf_token is getattr(self, "_zero_clf_token", None))
        )
        can_cache = batch_len == 1 and is_zero_clf
        t_key = round(float(t_val), 7)
        full_cache_key = (device, dtype, t_key, False)
        if can_cache:
            if full_cache_key in self._time_gates_cache:
                return self._time_gates_cache[full_cache_key]
            if block0_only:
                b0_cache_key = (device, dtype, t_key, True)
                if b0_cache_key in self._time_gates_cache:
                    return self._time_gates_cache[b0_cache_key]

        t_tensor = torch.full((batch_len,), float(t_val) * 1000.0, device=device, dtype=torch.float32)
        t_enc = get_time_encoding(t_tensor, self.flow.time_pos_features).to(dtype=dtype)
        temb = self.flow.time_embed(t_enc)
        if clf_token is not None and self.core.use_token and not is_zero_clf:
            clf = F.silu(temb + clf_token.to(dtype=dtype))
        else:
            clf = F.silu(temb)

        clf_shared = self.flow.clf_table_map(clf) if self.flow.clf_shared else clf

        block_gates = []
        cond_gates = []
        D = self.dim
        use_batched_gates = (
            batch_len == 1
            and bool(getattr(self.solver_config, "batched_gate_modulate_stack", True))
            and getattr(self, "_stacked_block_gates_raw", None) is not None
        )
        if use_batched_gates:
            if block0_only:
                gb0 = self._stacked_block_gates_raw[0] + clf_shared.view(9, D)
                gb0_w = gb0[(0, 3, 6), :] + 1.0
                bg_tuple = tuple(gb0[j].unsqueeze(0) for j in range(9))
                for k, j_base in enumerate((0, 3, 6)):
                    s_t, sh_t, g_t = bg_tuple[j_base], bg_tuple[j_base + 1], bg_tuple[j_base + 2]
                    s_t._w_1ps = gb0_w[k]
                    sh_t._b_1d = gb0[j_base + 1]
                    g_t._g_2d = g_t
                block_gates.append(bg_tuple)
                cond_gates.append(())
                nc_cs_w_fast = None
                nc_csh_b_fast = None
                nc_g_cr_fast = None
            else:
                gb_all = self._stacked_block_gates_raw + clf_shared.view(1, 9, D)
                gb_w_all = gb_all[:, (0, 3, 6), :] + 1.0
                cgb_8 = (
                    self._stacked_cond_gates_8_raw + clf_shared.view(1, 9, D)
                    if self._stacked_cond_gates_8_raw is not None
                    else None
                )
                cgb_8_w = cgb_8[:, (0, 3, 6), :] + 1.0 if cgb_8 is not None else None
                cgb_16 = (
                    self._stacked_cond_gates_16_raw + clf_shared[:, 3 * D : 5 * D].view(1, 2, D)
                    if self._stacked_cond_gates_16_raw is not None
                    else None
                )
                cgb_16_w = cgb_16[:, 0, :] + 1.0 if cgb_16 is not None else None
                c8_idx = 0
                c16_idx = 0
                nc_g_cr_list = []
                for b_i, block in enumerate(self.flow.blocks):
                    gb_i = gb_all[b_i]
                    gb_w_i = gb_w_all[b_i]
                    bg_tuple = tuple(gb_i[j].unsqueeze(0) for j in range(9))
                    for k, j_base in enumerate((0, 3, 6)):
                        s_t, sh_t, g_t = bg_tuple[j_base], bg_tuple[j_base + 1], bg_tuple[j_base + 2]
                        s_t._w_1ps = gb_w_i[k]
                        sh_t._b_1d = gb_i[j_base + 1]
                        g_t._g_2d = g_t
                    block_gates.append(bg_tuple)
                    if block.cond and cgb_8 is not None and cgb_8_w is not None:
                        cg_i = cgb_8[c8_idx]
                        cg_w_i = cgb_8_w[c8_idx]
                        c8_idx += 1
                        cg_tuple = tuple(cg_i[j].unsqueeze(0) for j in range(9))
                        for k, j_base in enumerate((0, 3, 6)):
                            s_t, sh_t, g_t = cg_tuple[j_base], cg_tuple[j_base + 1], cg_tuple[j_base + 2]
                            s_t._w_1ps = cg_w_i[k]
                            sh_t._b_1d = cg_i[j_base + 1]
                            g_t._g_2d = g_t
                            g_t._g_3d = g_t.unsqueeze(1)
                        cond_gates.append(cg_tuple)
                    elif cgb_16 is not None and cgb_16_w is not None:
                        cg2_i = cgb_16[c16_idx]
                        cg2_w_i = cgb_16_w[c16_idx]
                        c16_idx += 1
                        cs_cr = cg2_i[0].unsqueeze(0)
                        csh_cr = cg2_i[1].unsqueeze(0)
                        cs_cr._w_1ps = cg2_w_i
                        csh_cr._b_1d = cg2_i[1]
                        cs_cr._w_3d = cg2_w_i.view(1, 1, -1)
                        csh_cr._b_3d = cg2_i[1].view(1, 1, -1)
                        cond_gates.append((cs_cr, csh_cr))
                        nc_g_cr_list.append(gb_i[5])
                nc_cs_w_fast = cgb_16_w
                nc_csh_b_fast = cgb_16[:, 1, :] if cgb_16 is not None else None
                nc_g_cr_fast = (
                    gb_all[c8_idx:, 5, :]
                    if c8_idx + c16_idx == len(self.flow.blocks) and c16_idx > 0
                    else (torch.stack(nc_g_cr_list, dim=0) if nc_g_cr_list else None)
                )
        else:
            nc_cs_w_fast = None
            nc_csh_b_fast = None
            nc_g_cr_fast = None
            blocks_to_eval = self.flow.blocks[:1] if block0_only else self.flow.blocks
            for block in blocks_to_eval:
                g_param = getattr(block, "_gates_cast", None)
                if block.clf_shared:
                    gb = (g_param if g_param is not None else block.gates.type_as(clf_shared)) + clf_shared
                else:
                    gb = block.gates(clf)
                bg_tuple = gb.chunk(9, dim=-1)
                if batch_len == 1:
                    s_msa, sh_msa, g_msa, s_cr, sh_cr, g_cr, s_mlp, sh_mlp, g_mlp = bg_tuple
                    for s_t, sh_t, g_t in ((s_msa, sh_msa, g_msa), (s_cr, sh_cr, g_cr), (s_mlp, sh_mlp, g_mlp)):
                        s_t._w_1ps = (1.0 + s_t.view(-1)).contiguous()
                        sh_t._b_1d = sh_t.view(-1).contiguous()
                        g_t._g_2d = g_t.view(1, -1).contiguous()
                block_gates.append(bg_tuple)
                if block0_only:
                    cond_gates.append(())
                    continue
                cg_param = getattr(block, "_cond_gates_cast", None)
                if block.cond:
                    if block.clf_shared:
                        cgb = (cg_param if cg_param is not None else block.cond_gates.type_as(clf_shared)) + clf_shared
                    else:
                        cgb = block.cond_gates(clf)
                    cg_tuple = cgb.chunk(9, dim=-1)
                    if batch_len == 1:
                        cs_msa, csh_msa, cg_msa, cs_cr, csh_cr, cg_cr, cs_mlp, csh_mlp, cg_mlp = cg_tuple
                        for s_t, sh_t, g_t in ((cs_msa, csh_msa, cg_msa), (cs_cr, csh_cr, cg_cr), (cs_mlp, csh_mlp, cg_mlp)):
                            s_t._w_1ps = (1.0 + s_t.view(-1)).contiguous()
                            sh_t._b_1d = sh_t.view(-1).contiguous()
                            g_t._g_2d = g_t.view(1, -1).contiguous()
                            g_t._g_3d = g_t._g_2d.unsqueeze(1)
                    cond_gates.append(cg_tuple)
                else:
                    if block.clf_shared:
                        cgb = (
                            (cg_param if cg_param is not None else block.cond_gates.type_as(clf))
                            + clf_shared[:, 3 * self.dim : 5 * self.dim]
                        )
                    else:
                        cgb = block.cond_gates(clf)
                    cg_tuple2 = cgb.chunk(2, dim=-1)
                    if batch_len == 1:
                        cs_cr, csh_cr = cg_tuple2
                        cs_cr._w_1ps = (1.0 + cs_cr.view(-1)).contiguous()
                        csh_cr._b_1d = csh_cr.view(-1).contiguous()
                        cs_cr._w_3d = cs_cr._w_1ps.view(1, 1, -1)
                        csh_cr._b_3d = csh_cr._b_1d.view(1, 1, -1)
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
        if not block0_only and batch_len == 1:
            if nc_cs_w_fast is not None and nc_csh_b_fast is not None and nc_g_cr_fast is not None:
                result["nc_cs_w"] = nc_cs_w_fast
                result["nc_csh_b"] = nc_csh_b_fast
                result["nc_g_cr"] = nc_g_cr_fast
            else:
                nc_cs_w = [cond_gates[i][0]._w_1ps for i, b in enumerate(self.flow.blocks) if not b.cond]
                nc_csh_b = [cond_gates[i][1]._b_1d for i, b in enumerate(self.flow.blocks) if not b.cond]
                nc_g_cr = [block_gates[i][5].view(-1) for i, b in enumerate(self.flow.blocks) if not b.cond]
                if nc_cs_w:
                    result["nc_cs_w"] = torch.stack(nc_cs_w, dim=0).contiguous()
                    result["nc_csh_b"] = torch.stack(nc_csh_b, dim=0).contiguous()
                    result["nc_g_cr"] = torch.stack(nc_g_cr, dim=0).contiguous()
        if can_cache and len(self._time_gates_cache) < 256:
            cache_key = (device, dtype, t_key, bool(block0_only))
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
        arange_n = torch.arange(N, device=device, dtype=torch.long)

        # Bit-packed lexicographic key for 3D coordinates (x, y, z) in [0, 1023]^3:
        # Mathematically and bit-for-bit identical to SparseDownsample(patch_size=1) sort order,
        # eliminating 6 intermediate GPU kernels (max, 2x flip, cumprod, pad, mul+sum)!
        if batch_size == 1 and self.res <= 1024:
            c_long = coords_in.long()
            full_code = (c_long[:, 1] << 20) | (c_long[:, 2] << 10) | c_long[:, 3]
            perm = torch.argsort(full_code)
            up_widx = torch.empty_like(perm)
            up_widx[perm] = arange_n
        else:
            code = coords_in[:, 1:].long()
            offs = torch.cumprod((code.max(dim=0).values + 1).flip(dims=(-1,)), 0).flip(dims=(-1,)).long()
            offs = F.pad(offs, (0, 1), value=1)
            full_code = coords_in[:, 0].long() * offs[0] + (code * offs[1:]).sum(dim=-1)
            if batch_size == 1:
                perm = torch.argsort(full_code)
                up_widx = torch.empty_like(perm)
                up_widx[perm] = arange_n
            else:
                windows, up_widx = full_code.unique(return_inverse=True)
                perm = torch.empty_like(up_widx)
                perm[up_widx] = arange_n
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

        # Precompute 3D RoPE cos/sin for all voxels once (or keep 4KB L1 table for direct Triton lookup!)
        cos_table_3d, sin_table_3d = self._get_rope_3d_table(device)
        use_l1_table = (
            self.solver_config.use_fused_triton_kernels
            and bool(getattr(self.solver_config, "l1_table_rope3d", True))
            and device.type == "cuda"
        )
        if use_l1_table:
            cos_vox, sin_vox = cos_table_3d, sin_table_3d
        else:
            cos_vox, sin_vox = lookup_rope_3d(h_coords[:, 1:], cos_table_3d, sin_table_3d)

        # Precompute 2D RoPE cos/sin for image cond tokens
        cond_tokens = context.shape[1]
        cond_size = int(np.round(np.sqrt(cond_tokens - 5)))
        cos_cond_2d, sin_cond_2d = self._get_rope_2d_table(device, cond_size)

        # Precompute all 4 Hilbert space-filling curve attention orders once (zero-sync batched fast path)
        order_caches = build_order_caches_fast(
            h_coords,
            sizes,
            self.flow.orders,
            self.flow.order_kwargs,
            self._seqlen_cache,
            arange_n,
            use_triton=self.solver_config.use_fused_triton_kernels,
        )

        # Precompute 3D submanifold sparse conv neighbor indices once (replaces 2,112 calls!)
        conv_indices = build_conv_indices_fast(
            h_coords, already_sorted=(batch_size == 1 and self.res <= 1024)
        )

        # Cross-attention sequence lengths
        q_seq_key = (device, "q_seq", tuple(sizes))
        q_seqlens = self._seqlen_cache.get(q_seq_key)
        if q_seqlens is None:
            q_seqlens = torch.tensor([0] + sizes, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
            self._seqlen_cache[q_seq_key] = q_seqlens
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
        h_coords = geom["h_coords"]
        cos_cond_2d, sin_cond_2d = geom["cos_cond_2d"], geom["sin_cond_2d"]
        order_caches = geom["order_caches"]
        conv_indices = geom["conv_indices"]
        q_seqlens, q_maxlen = geom["q_seqlens"], geom["q_maxlen"]
        kv_seqlens_full, kv_maxlen_full = geom["kv_seqlens_full"], geom["kv_maxlen_full"]
        kv_seqlens_1tok, kv_maxlen_1tok = geom["kv_seqlens_1tok"], geom["kv_maxlen_1tok"]

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
        num_blocks = len(self.flow.blocks)
        mid_start = int(getattr(self.solver_config, "mid_block_start", 1))
        raw_mid_end = int(getattr(self.solver_config, "mid_block_end", 24))
        mid_end = min(raw_mid_end, num_blocks) if num_blocks >= 2 else num_blocks
        can_mid_cache = (0 < mid_start < mid_end)
        use_freqca_crf = can_mid_cache and (mid_end == num_blocks) and bool(getattr(self.solver_config, "freqca_crf_cache", True))
        use_subblock_crf = use_freqca_crf and bool(getattr(self.solver_config, "dbcache_foca_subblock_crf", True))
        use_hetcache_b0 = bool(getattr(self.solver_config, "hetcache_b0_cond_msa_reuse", True))
        use_pab_step2 = use_subblock_crf and bool(getattr(self.solver_config, "pab_step2_cr_gate_broadcast", True))
        use_sencache = bool(getattr(self.solver_config, "sencache_sensitivity_scale", True))
        foca_gamma = float(getattr(self.solver_config, "foca_fn_heun_gamma", 0.12))
        bary_beta = float(getattr(self.solver_config, "barycache_crf_beta", 0.10))
        qmc_stride = int(getattr(self.solver_config, "hilbert_qmc_uncond_stride", 4))
        pos_qmc_stride = int(getattr(self.solver_config, "hilbert_qmc_pos_stride", 4))
        nc_dino_stride = int(getattr(self.solver_config, "noncond_dino_qmc_stride", 4))
        use_b7_fuse = bool(getattr(self.solver_config, "fused_b7_gate_ln_cond", True))
        use_dicache_dcta = bool(getattr(self.solver_config, "dicache_dcta_probe", True))
        pab_nc_stride = int(getattr(self.solver_config, "pab_noncond_cr_stride", 2))
        cond_msa_stride = int(getattr(self.solver_config, "hetcache_cond_msa_stride", 2))
        cond_ffn_stride = int(getattr(self.solver_config, "hetcache_cond_ffn_stride", 2))
        cond_kv_qmc_stride = int(getattr(self.solver_config, "cond_dino_kv_qmc_stride", 4))
        use_koopman_obs = bool(getattr(self.solver_config, "koopman_shallow_observer", True))
        koopman_gain = float(getattr(self.solver_config, "koopman_observer_gain", 0.035))
        uncond_1tok_ffn_stride = int(getattr(self.solver_config, "uncond_1tok_cond_ffn_stride", 2))
        cond_cr_stride = int(getattr(self.solver_config, "refadapt_cond_cr_stride", 2))
        use_racer = bool(getattr(self.solver_config, "racer_disagreement_shrinkage", True))
        use_spec_scale = bool(getattr(self.solver_config, "spectral_norm_scaling", True))
        spec_gamma = float(getattr(self.solver_config, "spectral_norm_gamma", 0.035))
        use_b7_carve = bool(getattr(self.solver_config, "b7_terminal_token_carving", True))
        use_pisa_comp = bool(getattr(self.solver_config, "pisa_centroid_value_comp", True))
        pisa_alpha = float(getattr(self.solver_config, "pisa_centroid_alpha", 0.15))
        nc_vox_stride = int(getattr(self.solver_config, "layercache_noncond_vox_stride", 2))
        use_seacache = bool(getattr(self.solver_config, "seacache_wiener_probe", True))
        use_softcache = bool(getattr(self.solver_config, "softcache_headroom_modulation", True))
        softcache_tau = float(getattr(self.solver_config, "softcache_tau", 1.35))
        softcache_gamma = float(getattr(self.solver_config, "softcache_gamma", 0.18))
        cond_vox_stride = int(getattr(self.solver_config, "layercache_cond_vox_stride", 2))
        use_manifold_fdc = bool(getattr(self.solver_config, "manifold_spectral_fdc", True))
        manifold_beta_l = float(getattr(self.solver_config, "manifold_beta_l", 0.25))
        use_b0_carve = bool(getattr(self.solver_config, "b0_post_msa_token_carving", True))
        uncond_1tok_rcr_stride = int(getattr(self.solver_config, "uncond_1tok_rcr_stride", 2))
        use_cs_osi = bool(getattr(self.solver_config, "cachescope_worldcache_osi", True))
        use_hier_stride = bool(getattr(self.solver_config, "hierarchical_even_cond_nc_stride", True))
        use_gp_svd_refiner = bool(getattr(self.solver_config, "gp_svd_subspace_refiner", True))
        dmd_lambda = float(getattr(self.solver_config, "dmd_exp_decay_lambda", 0.015))
        use_1tok_collapse = bool(getattr(self.solver_config, "fused_1tok_linear_collapse", True))
        eff_mid_scale = (
            float(mid_cache_scale) * math.exp(-dmd_lambda * max(0.0, 1.0 - float(t_val)))
            if dmd_lambda > 0.0
            else float(mid_cache_scale)
        )

        # Single-Gate Fast Path (`block0_only`): on cached steps that exit in Block 0 before `block.cond`,
        # skip computing 23 unused `block_gates[1..23]` and 24 unused `cond_gates[0..23]`!
        fast_b0_exit = (
            bool(getattr(self.solver_config, "single_gate_fast_path", True))
            and use_freqca_crf
            and (cached_mid_delta is not None)
            and (
                uncond
                or (
                    cached_cond_block7 is not None
                    and getattr(self, "_last_pos_crf_after_b0_msa", None) is not None
                )
                or (
                    use_pab_step2
                    and getattr(self, "_last_b0_cr_proj", None) is not None
                    and getattr(self, "_last_pos_crf_after_b0_cr", None) is not None
                )
            )
        )
        tg = self._get_time_gates(t_val, device, compute_dtype, clf_token, block0_only=fast_b0_exit)
        block_gates = tg["block_gates"]
        cond_gates = tg["cond_gates"]
        use_triton = self.solver_config.use_fused_triton_kernels

        if reuse_b0_msa:
            x = self._last_b0_x_after_msa
            if (
                uncond
                and use_freqca_crf
                and cached_mid_delta is not None
                and not return_cond_block7
                and not return_mid_delta
                and not return_cond_delta
                and not return_ln_cond7
            ):
                # Fast Uncond CRF Short-Circuit (FreqCa + DiTFastAttn + DMD Exponential Decay):
                # Single-kernel `torch.add(..., alpha=...)` + Fused OutModulate & Unpatchify!
                x = torch.add(x, cached_mid_delta, alpha=eff_mid_scale)
                return fast_out_modulate_unpatchify(
                    x, sizes, tg["out_scale"], tg["out_shift"], self._unpatchify_w_fp32, use_triton=use_triton
                )
        else:
            # Patchify (for patch_size=1 on pre-sorted coords, this is a single linear projection)
            w_patch = getattr(self, "_patchify_w_cast", None)
            if w_patch is None or w_patch.dtype != compute_dtype:
                w_patch = self.flow.patchify.weight.to(dtype=compute_dtype)
            x = F.linear(xt_sorted.to(dtype=compute_dtype), w_patch)

        use_1tok_uncond = uncond and self.solver_config.uncond_1token_condensation and (batch_size == 1)
        skip_cond_alloc = (
            (not uncond)
            and use_freqca_crf
            and (cached_mid_delta is not None)
            and (
                (cached_cond_block7 is not None and getattr(self, "_last_pos_crf_after_b0_msa", None) is not None)
                or (
                    use_pab_step2
                    and getattr(self, "_last_b0_cr_proj", None) is not None
                    and getattr(self, "_last_pos_crf_after_b0_cr", None) is not None
                )
            )
        )
        if skip_cond_alloc:
            cond = context
        elif use_1tok_uncond:
            cond = torch.zeros(1, 1, D, device=device, dtype=compute_dtype)
        elif uncond:
            cond = torch.zeros_like(context, dtype=compute_dtype)
        else:
            cond = context.to(dtype=compute_dtype)

        ln_cond_after_7: Optional[torch.Tensor] = cached_ln_cond7
        need_full_kv_cc = return_cond_block7 and not (use_freqca_crf and use_subblock_crf)
        kv_cc_cache_out: Optional[List[torch.Tensor]] = [] if need_full_kv_cc else ([None] if return_cond_block7 else None)
        use_triton = self.solver_config.use_fused_triton_kernels
        use_carrier_share = bool(getattr(self.solver_config, "uncond_carrier_share", True))
        use_interblock_fuse = bool(getattr(self.solver_config, "interblock_gate_modulate_fusion", True))

        if (
            use_1tok_uncond
            and use_carrier_share
            and return_mid_delta
            and (cached_mid_delta is None)
            and reuse_b0_msa
            and getattr(self, "_last_pos_carrier_t_val", None) == t_val
            and getattr(self, "_last_pos_x_without_cr", None) is not None
            and self._last_pos_x_without_cr.shape == (N, D)
            and self._last_pos_x_without_cr.device == device
            and len(getattr(self, "_last_pos_kv_x_list", ())) == self.cond_depth
        ):
            # Exact 24-Block 1-Token Cross-Attention Carrier-Wave Decomposition:
            # All 24 blocks of 1-token unconditional conditioning execute on a [1, 1, 1536] tensor
            # while sharing the pure 3D voxel geometry carrier (MSA + FFN + SparseConv3d) from Step-0 pos,
            # preserving all 24 exact cross-attention contrasts `g_cr[b] * (cr_pos[b] - cr_uncond[b])`
            # and bypassing 23 heavy 3D voxel MSA + FFN + SparseConv3d blocks!
            use_bmm_nc = (
                bool(getattr(self.solver_config, "batched_1tok_noncond_bmm", True))
                and use_1tok_collapse
                and getattr(self, "_noncond_1tok_w_t", None) is not None
                and "nc_cs_w" in tg
            )
            use_1tok_ffn_stride = uncond_1tok_ffn_stride > 1
            use_1tok_rcr_stride = uncond_1tok_rcr_stride > 1
            prev_1tok_msa_out: Optional[torch.Tensor] = None
            prev_1tok_hx_cross: Optional[torch.Tensor] = None
            prev_1tok_rcr_proj: Optional[torch.Tensor] = None
            prev_1tok_ffn_out: Optional[torch.Tensor] = None
            uncond_cr_sum: Optional[torch.Tensor] = None
            h_cond_msa_pre: Optional[torch.Tensor] = None
            for b_idx, block in enumerate(self.flow.blocks):
                g_cr = block_gates[b_idx][5]
                g_cr_2d = getattr(g_cr, "_g_2d", g_cr.view(1, -1))
                if block.cond:
                    cs_msa, csh_msa, cg_msa, cs_cr, csh_cr, cg_cr, cs_mlp, csh_mlp, cg_mlp = cond_gates[b_idx]
                    will_reuse_1tok_ffn = (
                        use_1tok_ffn_stride
                        and (b_idx > 0)
                        and (b_idx < self.cond_depth - 1)
                        and (b_idx % uncond_1tok_ffn_stride != 0)
                        and (prev_1tok_msa_out is not None)
                        and (prev_1tok_hx_cross is not None)
                        and (prev_1tok_rcr_proj is not None)
                        and (prev_1tok_ffn_out is not None)
                    )
                    will_reuse_1tok_rcr = (
                        use_1tok_rcr_stride
                        and (not will_reuse_1tok_ffn)
                        and (b_idx > 0)
                        and (b_idx < self.cond_depth - 1)
                        and ((b_idx // max(1, uncond_1tok_ffn_stride)) % uncond_1tok_rcr_stride != 0)
                        and (prev_1tok_rcr_proj is not None)
                        and (prev_1tok_ffn_out is not None)
                    )
                    if (
                        will_reuse_1tok_ffn
                        and prev_1tok_msa_out is not None
                        and prev_1tok_hx_cross is not None
                        and prev_1tok_rcr_proj is not None
                        and prev_1tok_ffn_out is not None
                    ):
                        # SCR (arXiv:2609.29505) + HetCache (arXiv:2603.24260) + RefAdapt-DiT (arXiv:2609.32415)
                        # Zero-GEMM 1-Token Odd-Block Conditioning Reuse:
                        # In odd intermediate conditioning blocks (1, 3, 5), reuse the previous even block's
                        # 1-token MSA, forward cross-attention, reverse cross-attention, and FFN residuals
                        # modulated by exact current-block gates `cg_msa`, `g_cr_2d`, `cg_cr`, `cg_mlp`!
                        h_cond_msa_pre = None
                        h_x_cross = prev_1tok_hx_cross
                        cond = fast_gate_add_cond(cond, prev_1tok_msa_out, cg_msa)
                        cond = fast_gate_add_cond(cond, prev_1tok_rcr_proj, cg_cr)
                        cond_ffn_out = prev_1tok_ffn_out
                    else:
                        h_cond = (
                            csh_msa.view(1, 1, D)
                            if b_idx == 0
                            else (
                                h_cond_msa_pre
                                if h_cond_msa_pre is not None
                                else fast_modulate_cond(cond, cs_msa, csh_msa)
                            )
                        )
                        h_cond_msa_pre = None
                        m1_w = getattr(block, "_cond_1tok_msa_w", None) if use_1tok_collapse else None
                        if m1_w is not None:
                            cond_msa_out = F.linear(h_cond, m1_w, getattr(block, "_cond_1tok_msa_b", None))
                        else:
                            v_w = getattr(block, "_cond_to_v_w", None)
                            if v_w is not None:
                                v_cond = F.linear(h_cond, v_w, getattr(block, "_cond_to_v_b", None))
                            else:
                                v_cond = block.cond_to_qkv(h_cond).view(1, 1, 3, D)[:, :, 2]
                            cond_msa_out = block.cond_to_out(v_cond)
                        if use_1tok_ffn_stride:
                            prev_1tok_msa_out = cond_msa_out
                        cond, h_cond_cr = fast_gate_add_and_modulate_cond(
                            cond, cond_msa_out, cg_msa, cs_cr, csh_cr, use_triton=use_triton
                        )
                        if will_reuse_1tok_rcr and prev_1tok_rcr_proj is not None and prev_1tok_ffn_out is not None:
                            # PISA (arXiv:2602.01077) + ToMA (arXiv:2509.10918) Even-Block 1-Token Reverse-CR & FFN Reuse:
                            # In intermediate even blocks (2, 6), compute exact current-block `cond_msa_out` and
                            # forward 1-token `h_x_cross = F.linear(h_cond_cr, _cond_1tok_cr_w)` for 3D voxels,
                            # while reusing `prev_1tok_rcr_proj` and `prev_1tok_ffn_out` with exact current gates!
                            c1_w = getattr(block, "_cond_1tok_cr_w", None) if use_1tok_collapse else None
                            if c1_w is not None:
                                h_x_cross = F.linear(
                                    h_cond_cr.view(1, D), c1_w, getattr(block, "_cond_1tok_cr_b", None)
                                )
                            else:
                                v_w = getattr(block, "_cond_to_v_cross_w", None)
                                if v_w is not None:
                                    v_cc = F.linear(h_cond_cr, v_w, getattr(block, "_cond_to_v_cross_b", None)).view(1, D)
                                else:
                                    v_cc = block.cond_to_qkv_cross(h_cond_cr).view(1, 3, D)[:, 2]
                                h_x_cross = block.to_out_cross(v_cc)
                            if use_1tok_ffn_stride:
                                prev_1tok_hx_cross = h_x_cross
                            cond = fast_gate_add_cond(cond, prev_1tok_rcr_proj, cg_cr)
                            cond_ffn_out = prev_1tok_ffn_out
                        else:
                            qo_w = getattr(block, "_cond_1tok_q_and_out_cr_w", None) if use_1tok_collapse else None
                            if qo_w is not None:
                                q_cc_flat, h_x_cross = (
                                    F.linear(
                                        h_cond_cr.view(1, D), qo_w, getattr(block, "_cond_1tok_q_and_out_cr_b", None)
                                    )
                                    .view(2, 1, D)
                                    .unbind(0)
                                )
                                q_cc = q_cc_flat.view(1, H, HD)
                            else:
                                q_cc, _, v_cc = block.cond_to_qkv_cross(h_cond_cr).view(1, 3, H, HD).unbind(dim=-3)
                                h_x_cross = block.to_out_cross(v_cc.reshape(1, D))
                            if use_1tok_ffn_stride:
                                prev_1tok_hx_cross = h_x_cross
                            kv_x, q_seq_use, q_max_use = self._last_pos_kv_x_list[b_idx]
                            h_c_cross = flash_cross_attn(
                                q_cc, kv_seqlens_1tok, kv_maxlen_1tok, kv_x, q_seq_use, q_max_use
                            )
                            cond_rcr_proj = block.cond_to_out_cross(h_c_cross.view(1, 1, D))
                            if use_1tok_ffn_stride or use_1tok_rcr_stride:
                                prev_1tok_rcr_proj = cond_rcr_proj
                            cond, h_cond_mlp = fast_gate_add_and_modulate_cond(
                                cond,
                                cond_rcr_proj,
                                cg_cr,
                                cs_mlp,
                                csh_mlp,
                                use_triton=use_triton,
                            )
                            cond_ffn_out = block.cond_ffn1(
                                fast_swiglu_from_packed(block.cond_ffn0(h_cond_mlp), use_triton=use_triton)
                            )
                            if use_1tok_ffn_stride or use_1tok_rcr_stride:
                                prev_1tok_ffn_out = cond_ffn_out
                    next_reuses_1tok = (
                        use_1tok_ffn_stride
                        and (b_idx + 1 > 0)
                        and (b_idx + 1 < self.cond_depth - 1)
                        and ((b_idx + 1) % uncond_1tok_ffn_stride != 0)
                    )
                    if use_interblock_fuse and (b_idx + 1 < self.cond_depth) and (not next_reuses_1tok):
                        cs_msa_nxt, csh_msa_nxt = cond_gates[b_idx + 1][0], cond_gates[b_idx + 1][1]
                        cond, h_cond_msa_pre = fast_gate_add_and_modulate_cond(
                            cond, cond_ffn_out, cg_mlp, cs_msa_nxt, csh_msa_nxt, use_triton=use_triton
                        )
                    elif use_b7_fuse and b_idx == self.cond_depth - 1:
                        _, ln_cond_after_7 = fast_gate_add_and_ln_cond(
                            cond, cond_ffn_out, cg_mlp, use_triton=use_triton, return_cond=False
                        )
                    else:
                        cond = fast_gate_add_cond(cond, cond_ffn_out, cg_mlp)
                    if b_idx == self.cond_depth - 1:
                        if ln_cond_after_7 is None:
                            ln_cond_after_7 = fast_layer_norm(cond)
                        if use_bmm_nc and ln_cond_after_7 is not None:
                            if uncond_cr_sum is None:
                                uncond_cr_sum = h_x_cross * g_cr_2d
                            else:
                                uncond_cr_sum = torch.addcmul(uncond_cr_sum, h_x_cross, g_cr_2d, value=1.0)
                            # Batched 16-Block 1-Token Linear Collapse (blocks 8..23 in 1 TensorCore BMM!):
                            h_nc = torch.addcmul(
                                tg["nc_csh_b"], ln_cond_after_7.view(1, D), tg["nc_cs_w"]
                            ).unsqueeze(1)
                            hx_nc = torch.baddbmm(
                                self._noncond_1tok_b, h_nc, self._noncond_1tok_w_t
                            ).squeeze(1)
                            nc_sum = (hx_nc * tg["nc_g_cr"]).sum(dim=0, keepdim=True)
                            uncond_cr_sum = uncond_cr_sum + nc_sum
                            break
                else:
                    cs_cr, csh_cr = cond_gates[b_idx]
                    h_cond_cr = fast_modulate_preln_cond(ln_cond_after_7, cs_cr, csh_cr)
                    c1_w = getattr(block, "_cond_1tok_cr_w", None) if use_1tok_collapse else None
                    if c1_w is not None:
                        h_x_cross = F.linear(h_cond_cr.view(1, D), c1_w, getattr(block, "_cond_1tok_cr_b", None))
                    else:
                        v_w = getattr(block, "_cond_to_v_cross_w", None)
                        if v_w is not None:
                            v_cc = F.linear(h_cond_cr, v_w, getattr(block, "_cond_to_v_cross_b", None)).view(1, D)
                        else:
                            v_cc = block.cond_to_kv_cross(h_cond_cr).view(1, 2, D)[:, 1]
                        h_x_cross = block.to_out_cross(v_cc)
                if uncond_cr_sum is None:
                    uncond_cr_sum = h_x_cross * g_cr_2d
                else:
                    uncond_cr_sum = torch.addcmul(uncond_cr_sum, h_x_cross, g_cr_2d, value=1.0)
            x = self._last_pos_x_without_cr + uncond_cr_sum
            mid_delta_out = x - self._last_b0_x_after_msa
            out = fast_out_modulate_unpatchify(
                x, sizes, tg["out_scale"], tg["out_shift"], self._unpatchify_w_fp32, use_triton=use_triton
            )
            if return_ln_cond7:
                return out, mid_delta_out, ln_cond_after_7
            return out, mid_delta_out

        x_patchify = x
        x_mid_in: Optional[torch.Tensor] = None
        x_b0_msa: Optional[torch.Tensor] = None
        x_b0_cr: Optional[torch.Tensor] = None
        cond_mid_in: Optional[torch.Tensor] = None
        mid_delta_out: Optional[torch.Tensor] = None
        cond_delta_out: Optional[torch.Tensor] = None
        h_msa_precomputed: Optional[torch.Tensor] = None
        h_cond_msa_precomputed: Optional[torch.Tensor] = None
        record_pos_carrier = (
            (not uncond)
            and use_carrier_share
            and return_mid_delta
            and (cached_mid_delta is None)
        )
        use_fused_cr_accum = record_pos_carrier and bool(
            getattr(self.solver_config, "fused_cr_carrier_accum", True)
        )
        pos_cr_sum: Optional[torch.Tensor] = torch.empty_like(x) if use_fused_cr_accum else None
        pos_kv_x_list: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = [] if record_pos_carrier else None
        use_pab_nc_cr = (
            (not uncond)
            and (pab_nc_stride > 1)
            and return_mid_delta
            and (not need_full_kv_cc)
            and (cached_cond_block7 is None)
        )
        use_cond_msa_stride = (
            (not uncond)
            and (cond_msa_stride > 1)
            and return_mid_delta
            and (not need_full_kv_cc)
            and (cached_cond_block7 is None)
        )
        use_cond_ffn_stride = (
            (not uncond)
            and (cond_ffn_stride > 1)
            and return_mid_delta
            and (not need_full_kv_cc)
            and (cached_cond_block7 is None)
        )
        use_cond_cr_stride = (
            (not uncond)
            and (cond_cr_stride > 1)
            and return_mid_delta
            and (not need_full_kv_cc)
            and (cached_cond_block7 is None)
        )
        use_nc_vox_stride = (
            (not uncond)
            and (nc_vox_stride > 1)
            and return_mid_delta
            and (not need_full_kv_cc)
            and (cached_cond_block7 is None)
        )
        use_cond_vox_stride = (
            (not uncond)
            and (cond_vox_stride > 1)
            and return_mid_delta
            and (not need_full_kv_cc)
            and (cached_cond_block7 is None)
        )
        prev_nc_msa_out: Optional[torch.Tensor] = None
        prev_nc_cr_out: Optional[torch.Tensor] = None
        prev_nc_mlp_out: Optional[torch.Tensor] = None
        prev_cond_vox_msa_out: Optional[torch.Tensor] = None
        prev_cond_vox_mlp_out: Optional[torch.Tensor] = None
        prev_cond_msa_proj: Optional[torch.Tensor] = None
        prev_cond_rcr_proj: Optional[torch.Tensor] = None
        prev_cond_ffn_out: Optional[torch.Tensor] = None
        prev_cond_x_cr_out: Optional[torch.Tensor] = None
        prev_pos_kv_x: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None

        for b_idx, block in enumerate(self.flow.blocks):
            if can_mid_cache:
                if cached_mid_delta is not None and mid_start <= b_idx < mid_end:
                    if b_idx == mid_start:
                        x = torch.add(x, cached_mid_delta, alpha=eff_mid_scale)
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
            will_reuse_cond_cr = (
                use_cond_cr_stride
                and block.cond
                and (b_idx > 0)
                and (b_idx < self.cond_depth - 1)
                and (b_idx % cond_cr_stride != 0)
                and (prev_cond_x_cr_out is not None)
                and use_cond_msa_stride
                and (prev_cond_msa_proj is not None)
                and use_cond_ffn_stride
                and (prev_cond_rcr_proj is not None)
                and (prev_cond_ffn_out is not None)
                and (pos_kv_x_list is None or prev_pos_kv_x is not None)
            )
            will_reuse_cond_vox = (
                use_cond_vox_stride
                and block.cond
                and (b_idx > 0)
                and (b_idx < self.cond_depth - 1)
                and (b_idx % cond_vox_stride != 0)
                and will_reuse_cond_cr
                and (prev_cond_vox_msa_out is not None)
                and (prev_cond_vox_mlp_out is not None)
            )
            if (
                will_reuse_cond_vox
                and prev_cond_vox_msa_out is not None
                and prev_cond_x_cr_out is not None
                and prev_cond_vox_mlp_out is not None
                and prev_cond_msa_proj is not None
                and prev_cond_rcr_proj is not None
                and prev_cond_ffn_out is not None
            ):
                # LayerCache (arXiv:2604.16492, Sec. 3.2) + SpectralCache (arXiv:2603.05315, Sec. 3.2)
                # + ManifoldCache (arXiv:2610.04510, Theorem 2) + RefAdapt-DiT (arXiv:2609.32415)
                # Interleaved Shallow-Group 3D Voxel & DINO Subblock Reuse (blocks 1, 3, 5):
                # Even blocks (0, 2, 4, 6) serve as full error-correcting checkpoints traversing all 4
                # Hilbert space-filling curves, while odd blocks (1, 3, 5) reuse the previous even block's
                # 3D windowed MSA, cross-attention, 3D FFN+SparseConv3D, and DINO residuals modulated by
                # exact current-block AdaLN-Zero gates `g_msa`, `g_cr`, `g_mlp`, `cg_msa`, `cg_cr`, `cg_mlp`!
                h_msa_precomputed = None
                h_cond_msa_precomputed = None
                x = fast_gate_add(x, prev_cond_vox_msa_out, sizes, g_msa)
                if record_pos_carrier:
                    g_cr_2d = getattr(g_cr, "_g_2d", g_cr.view(1, -1))
                    if pos_cr_sum is None:
                        pos_cr_sum = prev_cond_x_cr_out * g_cr_2d
                    else:
                        pos_cr_sum.addcmul_(prev_cond_x_cr_out, g_cr_2d, value=1.0)
                if pos_kv_x_list is not None and prev_pos_kv_x is not None:
                    pos_kv_x_list.append(prev_pos_kv_x)
                x = fast_gate_add(x, prev_cond_x_cr_out, sizes, g_cr)
                cs_msa, csh_msa, cg_msa, cs_cr, csh_cr, cg_cr, cs_mlp, csh_mlp, cg_mlp = cond_gates[b_idx]
                cond = fast_gate_add_cond(cond, prev_cond_msa_proj, cg_msa)
                cond = fast_gate_add_cond(cond, prev_cond_rcr_proj, cg_cr)
                next_reuses_cond_msa = (
                    use_cond_msa_stride
                    and (b_idx + 1 > 0)
                    and (b_idx + 1 < self.cond_depth - 1)
                    and (
                        ((b_idx + 1) % cond_msa_stride != 0)
                        or (
                            use_hier_stride
                            and use_cond_vox_stride
                            and (((b_idx + 1) // cond_vox_stride) % cond_msa_stride != 0)
                        )
                    )
                )
                if (
                    use_interblock_fuse
                    and (b_idx + 1 < self.cond_depth)
                    and (cached_mid_delta is None)
                    and (not next_reuses_cond_msa)
                ):
                    cs_msa_nxt, csh_msa_nxt = cond_gates[b_idx + 1][0], cond_gates[b_idx + 1][1]
                    cond, h_cond_msa_precomputed = fast_gate_add_and_modulate_cond(
                        cond, prev_cond_ffn_out, cg_mlp, cs_msa_nxt, csh_msa_nxt, use_triton=use_triton
                    )
                else:
                    cond = fast_gate_add_cond(cond, prev_cond_ffn_out, cg_mlp)
                next_reuses_cond_vox = (
                    use_cond_vox_stride
                    and (b_idx + 1 > 0)
                    and (b_idx + 1 < self.cond_depth - 1)
                    and ((b_idx + 1) % cond_vox_stride != 0)
                )
                if use_interblock_fuse and (b_idx + 1 < num_blocks) and (not next_reuses_cond_vox):
                    s_msa_nxt, sh_msa_nxt = block_gates[b_idx + 1][0], block_gates[b_idx + 1][1]
                    x, h_msa_precomputed = fast_gate_add_and_modulate(
                        x, prev_cond_vox_mlp_out, sizes, g_mlp, s_msa_nxt, sh_msa_nxt, use_triton=use_triton
                    )
                else:
                    x = fast_gate_add(x, prev_cond_vox_mlp_out, sizes, g_mlp)
                continue

            will_reuse_nc_vox = (
                use_nc_vox_stride
                and (not block.cond)
                and ((b_idx - self.cond_depth) % nc_vox_stride != 0)
                and (b_idx < num_blocks - 2)
                and (prev_nc_msa_out is not None)
                and (prev_nc_cr_out is not None)
                and (prev_nc_mlp_out is not None)
            )
            if (
                will_reuse_nc_vox
                and prev_nc_msa_out is not None
                and prev_nc_cr_out is not None
                and prev_nc_mlp_out is not None
            ):
                # LayerCache (arXiv:2604.16492) + BlockDance (CVPR, arXiv:2603.17642) + Faster-DiT (arXiv:2610.22369)
                # Non-Conditioning 3D Voxel Subblock Reuse with Exact Current-Block AdaLN-Zero Gating:
                # In odd intermediate non-conditioning blocks (9, 11, 13, 15, 17, 19, 21), reuse the adjacent
                # even block's 3D windowed MSA, cross-attention, and 3D FFN+SparseConv3D outputs modulated by
                # exact current-block gates `g_msa[b_idx]`, `g_cr[b_idx]`, `g_mlp[b_idx]`!
                h_msa_precomputed = None
                x = fast_gate_add(x, prev_nc_msa_out, sizes, g_msa)
                if record_pos_carrier:
                    g_cr_2d = getattr(g_cr, "_g_2d", g_cr.view(1, -1))
                    if pos_cr_sum is None:
                        pos_cr_sum = prev_nc_cr_out * g_cr_2d
                    else:
                        pos_cr_sum.addcmul_(prev_nc_cr_out, g_cr_2d, value=1.0)
                x = fast_gate_add(x, prev_nc_cr_out, sizes, g_cr)
                next_reuses_nc_vox = (
                    use_nc_vox_stride
                    and ((b_idx + 1) >= self.cond_depth)
                    and (((b_idx + 1) - self.cond_depth) % nc_vox_stride != 0)
                    and ((b_idx + 1) < num_blocks - 2)
                )
                if use_interblock_fuse and (b_idx + 1 < num_blocks) and (not next_reuses_nc_vox):
                    s_msa_nxt, sh_msa_nxt = block_gates[b_idx + 1][0], block_gates[b_idx + 1][1]
                    x, h_msa_precomputed = fast_gate_add_and_modulate(
                        x, prev_nc_mlp_out, sizes, g_mlp, s_msa_nxt, sh_msa_nxt, use_triton=use_triton
                    )
                else:
                    x = fast_gate_add(x, prev_nc_mlp_out, sizes, g_mlp)
                continue

            if use_cond_vox_stride and block.cond and (b_idx < self.cond_depth - 1) and len(order_caches) > 0:
                eff_order_idx = (b_idx // cond_vox_stride) % len(order_caches)
            elif use_nc_vox_stride and (not block.cond) and (b_idx < num_blocks - 1) and len(order_caches) > 0:
                eff_order_idx = ((b_idx - self.cond_depth) // nc_vox_stride) % len(order_caches)
            else:
                eff_order_idx = block.order_idx
            fwd_idx, bwd_idx, cu_seqlens, max_len = order_caches[eff_order_idx]
            will_reuse_nc_cr = (
                use_pab_nc_cr
                and (not block.cond)
                and (
                    (
                        ((b_idx - self.cond_depth) % pab_nc_stride != 0)
                        and (b_idx < num_blocks - 2)
                    )
                    or (
                        use_hier_stride
                        and use_nc_vox_stride
                        and (((b_idx - self.cond_depth) // nc_vox_stride) % pab_nc_stride != 0)
                        and (b_idx < num_blocks - 1)
                    )
                )
                and (prev_nc_cr_out is not None)
            )

            # --- 1. Windowed Self-Attention on 3D Voxels ---
            h_x_precomputed: Optional[torch.Tensor] = None
            if b_idx == 0 and reuse_b0_msa:
                x = self._last_b0_x_after_msa
            else:
                h = h_msa_precomputed if h_msa_precomputed is not None else fast_modulate(x, sizes, s_msa, sh_msa)
                h_msa_precomputed = None
                qkv = block.to_qkv(h).view(N, 3, H, HD)
                q_scale = getattr(block.qk_norm, "_q_scale_cast", block.qk_norm.q_scale)
                k_scale = getattr(block.qk_norm, "_k_scale_cast", block.qk_norm.k_scale)
                qkv_sorted = fused_qknorm_rope3d_gather(
                    qkv,
                    q_scale,
                    k_scale,
                    cos_vox,
                    sin_vox,
                    fwd_idx,
                    coords=h_coords,
                    use_triton=use_triton,
                )
                attn_out = flash_varlen_qkvpacked_already_sorted(qkv_sorted, bwd_idx, cu_seqlens, max_len)
                h = block.to_out(attn_out.reshape(N, D))
                if use_cond_vox_stride and block.cond:
                    prev_cond_vox_msa_out = h
                elif use_nc_vox_stride and (not block.cond):
                    prev_nc_msa_out = h
                if (
                    (b_idx > 0 or cached_mid_delta is None)
                    and (not use_1tok_uncond)
                    and (not will_reuse_nc_cr)
                    and (not will_reuse_cond_cr)
                ):
                    # Fused MSA Gate-Add + Cross-Attention LayerNorm & AdaLN Modulation in 1 Triton kernel!
                    x, h_x_precomputed = fast_gate_add_and_modulate(
                        x, h, sizes, g_msa, s_cr, sh_cr, use_triton=use_triton
                    )
                else:
                    x = fast_gate_add(x, h, sizes, g_msa)
                if b_idx == 0 and not uncond:
                    self._last_b0_x_after_msa = x
                    self._last_b0_t_val = t_val

            if b_idx == 0 and use_freqca_crf:
                if uncond:
                    if cached_mid_delta is not None:
                        # FreqCa CRF (NeurIPS 2026) + DiTFastAttn: `cached_mid_delta` for uncond already
                        # spans from `x_after_b0_msa` to `x_after_23` (including Block-0 cross-attn & FFN)!
                        x = torch.add(x, cached_mid_delta, alpha=eff_mid_scale)
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
                        # + BaryCache/L2P-Cache 2-Anchor Extrapolation (arXiv:2608.28670 / CVPR 2026)
                        # + DiCache Dynamic Cache Trajectory Alignment (ICLR 2026, arXiv:2508.17356)
                        # + RACER Dual-Forecast Disagreement Shrinkage (arXiv:2608.01740, Aug 2026)
                        # + SpectralCache Singular-Value Norm Scaling & FDC/TADS (arXiv:2610.02660 / arXiv:2603.05315)
                        # + SeaCache Spectral-Evolution-Aware Wiener Filter (CVPR 2026 Oral, arXiv:2602.18993v2)
                        # + SoftCache Continuous Probe Headroom Modulation (arXiv:2610.22369, Oct 2026)
                        # + ManifoldCache Normal/Tangential Drift Envelope (arXiv:2610.04510, Oct 2026):
                        cur_fn_res = x - x_patchify
                        crf_base = self._last_pos_crf_after_b0_msa
                        crf_s0 = getattr(self, "_last_pos_crf_after_b0_msa_step0", None)
                        eff_mid_step_scale = eff_mid_scale
                        has_fn_res = (
                            getattr(self, "_last_b0_fn_res", None) is not None
                            and self._last_b0_fn_res.shape == (N, D)
                        )
                        has_fn_step0 = (
                            has_fn_res
                            and getattr(self, "_last_b0_fn_res_step0", None) is not None
                            and self._last_b0_fn_res is not self._last_b0_fn_res_step0
                            and self._last_b0_fn_res_step0.shape == (N, D)
                        )
                        diff_cur_fn: Optional[torch.Tensor] = (
                            (cur_fn_res - self._last_b0_fn_res) if has_fn_res else None
                        )
                        y_prev_fn: Optional[torch.Tensor] = (
                            (self._last_b0_fn_res - self._last_b0_fn_res_step0) if has_fn_step0 else None
                        )
                        if has_fn_res and (
                            (use_spec_scale and spec_gamma > 0.0)
                            or (bary_beta > 0.0 and use_dicache_dcta and has_fn_step0)
                        ):
                            # Single-Sync Batched L1/L2 Probe Reduction (eliminates 5-9 separate .item() GPU syncs
                            # and avoids allocating a 46MB [N, D] temporary tensor via zero-copy FDC channel slices!):
                            d_half = D // 2
                            has_x_patch_prev = (
                                getattr(self, "_last_b0_x_patchify", None) is not None
                                and self._last_b0_x_patchify.shape == (N, D)
                            )
                            dx_cur_fn: Optional[torch.Tensor] = (
                                (x_patchify - self._last_b0_x_patchify) if has_x_patch_prev else diff_cur_fn
                            )
                            if has_fn_step0 and diff_cur_fn is not None and y_prev_fn is not None:
                                stats_list = torch.stack(
                                    [
                                        cur_fn_res.abs().mean(),
                                        self._last_b0_fn_res.abs().mean(),
                                        diff_cur_fn.abs().mean(),
                                        y_prev_fn.abs().mean(),
                                        diff_cur_fn[:, :d_half].abs().mean(),
                                        diff_cur_fn[:, d_half:].abs().mean(),
                                        y_prev_fn[:, :d_half].abs().mean(),
                                        y_prev_fn[:, d_half:].abs().mean(),
                                        dx_cur_fn.abs().mean() if dx_cur_fn is not None else diff_cur_fn.abs().mean(),
                                        (diff_cur_fn * y_prev_fn).mean(),
                                        (y_prev_fn * y_prev_fn).mean(),
                                    ]
                                ).tolist()
                                (
                                    norm_cur_l1,
                                    norm_anc_l1,
                                    num_l1,
                                    den_l1,
                                    lf_cur,
                                    hf_cur,
                                    lf_anc,
                                    hf_anc,
                                    dx_l1,
                                    dot_dy,
                                    norm_dy_prev_sq,
                                ) = (float(v_s) for v_s in stats_list)
                            else:
                                stats_list = torch.stack(
                                    [
                                        cur_fn_res.abs().mean(),
                                        self._last_b0_fn_res.abs().mean(),
                                    ]
                                ).tolist()
                                norm_cur_l1 = float(stats_list[0])
                                norm_anc_l1 = float(stats_list[1])
                                num_l1 = den_l1 = lf_cur = hf_cur = lf_anc = hf_anc = dx_l1 = dot_dy = norm_dy_prev_sq = 0.0
                            if use_spec_scale and spec_gamma > 0.0:
                                eff_mid_step_scale = eff_mid_step_scale * compute_spectral_cache_scale(
                                    norm_cur_l1, norm_anc_l1, gamma=spec_gamma
                                )
                        else:
                            num_l1 = den_l1 = lf_cur = hf_cur = lf_anc = hf_anc = dx_l1 = dot_dy = norm_dy_prev_sq = 0.0
                        if (
                            bary_beta > 0.0
                            and crf_s0 is not None
                            and crf_s0 is not crf_base
                            and crf_s0.shape == (N, D)
                        ):
                            eff_beta = bary_beta * eff_mid_step_scale if use_sencache else bary_beta
                            if use_dicache_dcta and has_fn_step0:
                                dcta_ratio = min(1.35, max(0.65, num_l1 / max(1e-6, den_l1)))
                                eff_beta = eff_beta * dcta_ratio
                                if use_racer:
                                    disagree_r = abs(num_l1 - den_l1) / max(1e-6, den_l1)
                                    eff_beta = eff_beta * compute_racer_trust_shrinkage(disagree_r, horizon_k=1)
                                if use_seacache and use_softcache:
                                    eff_beta = eff_beta * compute_seacache_softcache_weight(
                                        t_val,
                                        lf_cur,
                                        hf_cur,
                                        den_l1,
                                        tau=softcache_tau,
                                        gamma_soft=softcache_gamma,
                                    )
                                if use_manifold_fdc:
                                    eff_beta = eff_beta * compute_manifold_spectral_fdc_weight(
                                        t_val,
                                        lf_cur,
                                        hf_cur,
                                        lf_anc,
                                        hf_anc,
                                        beta_l=manifold_beta_l,
                                    )
                                if use_cs_osi:
                                    eff_beta = eff_beta * compute_cachescope_worldcache_osi_weight(
                                        num_l1,
                                        dx_l1,
                                        dot_dy,
                                        norm_dy_prev_sq,
                                        rfc_io_ratio=num_l1 / max(1e-6, dx_l1),
                                    )
                                if use_gp_svd_refiner:
                                    eff_beta = eff_beta * compute_gp_svd_subspace_weight(
                                        lf_cur,
                                        hf_cur,
                                        lf_anc,
                                        hf_anc,
                                        num_l1,
                                        den_l1,
                                    )
                            x = torch.add(x, crf_base, alpha=eff_mid_step_scale * (1.0 + eff_beta))
                            x = torch.add(x, crf_s0, alpha=-eff_mid_step_scale * eff_beta)
                        else:
                            x = torch.add(x, crf_base, alpha=eff_mid_step_scale)
                        if use_subblock_crf and foca_gamma > 0.0 and diff_cur_fn is not None:
                            if use_koopman_obs and koopman_gain > 0.0 and y_prev_fn is not None:
                                # Koopman-Luenberger Shallow-to-Deep Observer Innovation (ETH/MPI/MIT, arXiv:2610.10366, Oct 2026):
                                # Algebraic fusion: foca_gamma * y_cur + koopman_gain * (y_cur - y_prev)
                                # = (foca_gamma + koopman_gain) * diff_cur_fn - koopman_gain * y_prev_fn!
                                x = torch.add(x, diff_cur_fn, alpha=foca_gamma + koopman_gain)
                                x = torch.add(x, y_prev_fn, alpha=-koopman_gain)
                            else:
                                x = torch.add(x, diff_cur_fn, alpha=foca_gamma)
                        break
                    else:
                        x_b0_msa = x
                        self._last_b0_x_patchify = x_patchify
                        if use_subblock_crf:
                            self._last_b0_fn_res = x - x_patchify
                            if cached_mid_delta is None:
                                self._last_b0_fn_res_step0 = self._last_b0_fn_res
                        if (
                            use_pab_step2
                            and cached_mid_delta is not None
                            and getattr(self, "_last_b0_cr_proj", None) is not None
                            and self._last_b0_cr_proj.shape == (N, D)
                            and self._last_b0_cr_proj.device == device
                            and getattr(self, "_last_pos_crf_after_b0_cr", None) is not None
                            and self._last_pos_crf_after_b0_cr.shape == (N, D)
                            and self._last_pos_crf_after_b0_cr.device == device
                        ):
                            # PAB + T-GATE Step-2 Cross-Attention Gate Broadcast + Subblock-2 CRF:
                            # Apply cached Block-0 cross-attn projection with Step-2 gate `g_cr(t_2)`,
                            # fuse Subblock-2 CRF + FoCa Heun increment, refresh unscaled `_last_pos_crf_after_b0_msa`,
                            # and bypass Block-0 DINO Self-Attn, Cross-Attn, FFN+SparseConv3D, and Blocks 1..23!
                            x_cr_s2 = fast_gate_add(x, self._last_b0_cr_proj, sizes, g_cr)
                            self._last_pos_crf_after_b0_msa = (
                                (x_cr_s2 - x_b0_msa) + self._last_pos_crf_after_b0_cr
                            )
                            x = torch.add(x_cr_s2, self._last_pos_crf_after_b0_cr, alpha=eff_mid_scale)
                            if (
                                foca_gamma > 0.0
                                and getattr(self, "_last_b0_fn_res_step0", None) is not None
                                and self._last_b0_fn_res_step0.shape == (N, D)
                            ):
                                x = torch.add(
                                    x,
                                    (x_b0_msa - x_patchify) - self._last_b0_fn_res_step0,
                                    alpha=foca_gamma,
                                )
                            break

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
                    # Exact 1-token unconditional self-attention: algebraic 2-layer collapse `cond_to_out(cond_to_v(h_cond))`!
                    h_cond = fast_modulate_cond(cond, cs_msa, csh_msa)
                    m1_w = getattr(block, "_cond_1tok_msa_w", None) if use_1tok_collapse else None
                    if m1_w is not None:
                        cond_msa_out = F.linear(h_cond, m1_w, getattr(block, "_cond_1tok_msa_b", None))
                    else:
                        v_w = getattr(block, "_cond_to_v_w", None)
                        if v_w is not None:
                            v_cond = F.linear(h_cond, v_w, getattr(block, "_cond_to_v_b", None))
                        else:
                            v_cond = block.cond_to_qkv(h_cond).view(1, 1, 3, D)[:, :, 2]
                        cond_msa_out = block.cond_to_out(v_cond)

                    # Fused `cond = cond + cond_msa_out * cg_msa` + `h_cond_cr = LayerNorm(cond) * (1 + cs_cr) + csh_cr`!
                    cond, h_cond_cr = fast_gate_add_and_modulate_cond(
                        cond, cond_msa_out, cg_msa, cs_cr, csh_cr, use_triton=use_triton
                    )
                    if skip_cond_out:
                        c1_w = getattr(block, "_cond_1tok_cr_w", None) if use_1tok_collapse else None
                        if c1_w is not None:
                            h_x_cross = F.linear(h_cond_cr.view(1, D), c1_w, getattr(block, "_cond_1tok_cr_b", None))
                        else:
                            vc_w = getattr(block, "_cond_to_v_cross_slice_w", None)
                            if vc_w is not None:
                                v_cc = F.linear(
                                    h_cond_cr, vc_w, getattr(block, "_cond_to_v_cross_slice_b", None)
                                ).view(1, D)
                            else:
                                v_cc = block.cond_to_qkv_cross(h_cond_cr).view(1, 3, D)[:, 2]
                            h_x_cross = block.to_out_cross(v_cc)
                    else:
                        qo_w = getattr(block, "_cond_1tok_q_and_out_cr_w", None) if use_1tok_collapse else None
                        if qo_w is not None:
                            q_cc_flat, h_x_cross = (
                                F.linear(
                                    h_cond_cr.view(1, D), qo_w, getattr(block, "_cond_1tok_q_and_out_cr_b", None)
                                )
                                .view(2, 1, D)
                                .unbind(0)
                            )
                            q_cc = q_cc_flat.view(1, H, HD)
                        else:
                            q_cc, _, v_cc = block.cond_to_qkv_cross(h_cond_cr).view(1, 3, H, HD).unbind(dim=-3)
                            h_x_cross = block.to_out_cross(v_cc.reshape(1, D))

                    if not skip_cond_out:
                        # cond -> x only needed if subsequent blocks read `cond` (must use `x` before gate_add!)
                        if return_mid_delta and batch_size == 1 and N >= 64 and qmc_stride > 1:
                            # Hilbert-Curve QMC Strided Reverse Cross-Attention (MC-Sparse arXiv:2610.06801 / JiT CVPR 2026):
                            # Since 3D voxels `x` are ordered along a continuous Hilbert space-filling curve,
                            # strided subsampling `x[::qmc_stride]` forms a quasi-Monte Carlo quadrature for the
                            # 1-token global softmax pooling `cond -> x`, cutting 75% of reverse cross-attn FLOPs!
                            x_sub = x[::qmc_stride]
                            n_sub = x_sub.shape[0]
                            h_x = fast_modulate(x_sub, [n_sub], s_cr, sh_cr)
                            q_seq_use = geom.get(f"q_seqlens_qmc_{qmc_stride}")
                            if q_seq_use is None:
                                q_seq_use = torch.tensor([0, n_sub], device=device, dtype=torch.int32)
                                geom[f"q_seqlens_qmc_{qmc_stride}"] = q_seq_use
                            q_max_use = n_sub
                        else:
                            n_sub = N
                            h_x = fast_modulate(x, sizes, s_cr, sh_cr)
                            q_seq_use = q_seqlens
                            q_max_use = q_maxlen

                        kv_w = getattr(block, "_to_kv_cross_w", None)
                        if kv_w is not None:
                            kv_x = F.linear(h_x, kv_w, getattr(block, "_to_kv_cross_b", None)).view(n_sub, 2, H, HD)
                        else:
                            kv_x = block.to_qkv_cross(h_x).view(n_sub, 3, H, HD)[:, 1:]

                        h_c_cross = flash_cross_attn(
                            q_cc, kv_seqlens_1tok, kv_maxlen_1tok, kv_x, q_seq_use, q_max_use
                        )
                        cond, h_cond_mlp = fast_gate_add_and_modulate_cond(
                            cond,
                            block.cond_to_out_cross(h_c_cross.view(1, 1, D)),
                            cg_cr,
                            cs_mlp,
                            csh_mlp,
                            use_triton=use_triton,
                        )

                        # 1-token cond FFN with Fused Packed SwiGLU
                        cond = fast_gate_add_cond(
                            cond,
                            block.cond_ffn1(fast_swiglu_from_packed(block.cond_ffn0(h_cond_mlp), use_triton=use_triton)),
                            cg_mlp,
                        )

                    cr_out_for_x = h_x_cross

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
                        cr_out_for_x = self._last_b0_cr_proj
                    else:
                        # Reuse cached K,V of conditioning from previous step; only compute Q_x and x->cond cross-attn!
                        h_x = h_x_precomputed if h_x_precomputed is not None else fast_modulate(x, sizes, s_cr, sh_cr)
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
                        cr_out_for_x = cr_proj_out
                else:
                    L = cond.shape[1]
                    will_reuse_cond_msa = (
                        use_cond_msa_stride
                        and (b_idx > 0)
                        and (b_idx < self.cond_depth - 1)
                        and (
                            (b_idx % cond_msa_stride != 0)
                            or (
                                use_hier_stride
                                and use_cond_vox_stride
                                and ((b_idx // cond_vox_stride) % cond_msa_stride != 0)
                            )
                        )
                        and (prev_cond_msa_proj is not None)
                    )
                    if (
                        will_reuse_cond_cr
                        and prev_cond_x_cr_out is not None
                        and prev_cond_msa_proj is not None
                        and prev_cond_rcr_proj is not None
                        and prev_cond_ffn_out is not None
                    ):
                        # RefAdapt-DiT (arXiv:2609.32415, Sep 2026) + PAB (ICLR 2025) + SCR (arXiv:2609.29505)
                        # Block-Adaptive Reference-Conditioned Cross-Attention & DINO Residual Reuse:
                        # In odd intermediate conditioning blocks (1, 3, 5), reuse the previous even block's
                        # forward `x -> cond` cross-attention projection `prev_cond_x_cr_out` modulated by exact
                        # current gate `g_cr[b_idx]`, and accumulate `prev_cond_msa_proj`, `prev_cond_rcr_proj`,
                        # and `prev_cond_ffn_out` on `cond` with exact current gates `cg_msa`, `cg_cr`, `cg_mlp`,
                        # bypassing all intermediate LayerNorms, QKV GEMMs, and FlashCrossAttn in blocks 1, 3, 5!
                        h_cond_msa_precomputed = None
                        cr_out_for_x = prev_cond_x_cr_out
                        if pos_kv_x_list is not None and prev_pos_kv_x is not None:
                            pos_kv_x_list.append(prev_pos_kv_x)
                        cond = fast_gate_add_cond(cond, prev_cond_msa_proj, cg_msa)
                        cond = fast_gate_add_cond(cond, prev_cond_rcr_proj, cg_cr)
                        cond_ffn_out = prev_cond_ffn_out
                        next_reuses_cond_msa = (
                            use_cond_msa_stride
                            and (b_idx + 1 > 0)
                            and (b_idx + 1 < self.cond_depth - 1)
                            and ((b_idx + 1) % cond_msa_stride != 0)
                        )
                        if (
                            use_interblock_fuse
                            and (b_idx + 1 < self.cond_depth)
                            and (cached_mid_delta is None)
                            and (not next_reuses_cond_msa)
                        ):
                            cs_msa_nxt, csh_msa_nxt = cond_gates[b_idx + 1][0], cond_gates[b_idx + 1][1]
                            cond, h_cond_msa_precomputed = fast_gate_add_and_modulate_cond(
                                cond, cond_ffn_out, cg_mlp, cs_msa_nxt, csh_msa_nxt, use_triton=use_triton
                            )
                        else:
                            cond = fast_gate_add_cond(cond, cond_ffn_out, cg_mlp)
                    else:
                        if (
                            b_idx == 0
                            and (not uncond)
                            and skip_cond_out
                            and use_hetcache_b0
                            and getattr(self, "_last_b0_cond_msa_out", None) is not None
                            and self._last_b0_cond_msa_out.shape == (batch_size, L, D)
                            and self._last_b0_cond_msa_out.device == device
                        ):
                            # HetCache (arXiv:2603.24260, 2026): static DINO context tokens in Block 0
                            # reuse their O(L^2) self-attention output projection modulated by current `cg_msa(t)`!
                            cond, h_cond_cr = fast_gate_add_and_modulate_cond(
                                cond, self._last_b0_cond_msa_out, cg_msa, cs_cr, csh_cr, use_triton=use_triton
                            )
                        elif will_reuse_cond_msa and prev_cond_msa_proj is not None:
                            # HetCache (arXiv:2603.24260) + DuCa (ToCa-v2) Alternating Conditioning Self-Attention Reuse:
                            # In odd intermediate conditioning blocks (1, 3, 5), reuse the previous even block's
                            # O(L^2) DINO self-attention projection with the current block's exact gate `cg_msa[b_idx]`!
                            h_cond_msa_precomputed = None
                            cond, h_cond_cr = fast_gate_add_and_modulate_cond(
                                cond, prev_cond_msa_proj, cg_msa, cs_cr, csh_cr, use_triton=use_triton
                            )
                        else:
                            h_cond = (
                                h_cond_msa_precomputed
                                if h_cond_msa_precomputed is not None
                                else fast_modulate_cond(cond, cs_msa, csh_msa)
                            )
                            h_cond_msa_precomputed = None
                            qkv_cond = block.cond_to_qkv(h_cond).view(batch_size, L, 3, H, HD)
                            cq_scale = getattr(block.cond_qk_norm, "_q_scale_cast", block.cond_qk_norm.q_scale)
                            ck_scale = getattr(block.cond_qk_norm, "_k_scale_cast", block.cond_qk_norm.k_scale)
                            qkv_cond = fused_qknorm_rope2d_cond_inplace(
                                qkv_cond, cq_scale, ck_scale, cos_cond_2d, sin_cond_2d, use_triton=use_triton
                            )
                            h_cond = flash_cond_self_attn(qkv_cond)
                            cond_msa_proj = block.cond_to_out(h_cond.reshape(batch_size, L, D))
                            if b_idx == 0 and not uncond:
                                self._last_b0_cond_msa_out = cond_msa_proj
                            if not locals().get("b7_carved", False):
                                b7_carved = False
                            if (
                                (not b7_carved)
                                and (
                                    (use_b0_carve and b_idx == 0 and cached_cond_block7 is None)
                                    or (use_b7_carve and b_idx == self.cond_depth - 1)
                                )
                                and (not uncond)
                                and (nc_dino_stride > 1)
                                and (cond_kv_qmc_stride == nc_dino_stride)
                                and return_mid_delta
                                and (not need_full_kv_cc)
                                and (not return_cond_delta)
                                and batch_size == 1
                                and L > 16
                            ):
                                # ToPi (arXiv:2602.01609) + CoReDiT (arXiv:2605.14191) + Fast-SAM3D (arXiv:2602.05293, ICML 2026)
                                # + PISA (arXiv:2602.01077, ICML 2026) Position-Preserving Post-Block-0 2D QMC DINO Token Carving:
                                # Immediately after Block-0's full 4101x4101 2D DINO self-attention aggregates global context,
                                # carve `cond` and `cond_msa_proj` to the 5 global tokens + every `nc_dino_stride`-th patch
                                # with zeroth-order PISA centroid value compensation and position-preserving 2D RoPE slicing!
                                c_patches_all = cond[:, 5:]
                                c_patches_sub = c_patches_all[:, ::nc_dino_stride]
                                m_patches_all = cond_msa_proj[:, 5:]
                                m_patches_sub = m_patches_all[:, ::nc_dino_stride]
                                if use_pisa_comp and pisa_alpha > 0.0 and c_patches_sub.shape[1] > 0:
                                    c_patches_sub = c_patches_sub + pisa_alpha * (
                                        c_patches_all.mean(dim=1, keepdim=True) - c_patches_sub.mean(dim=1, keepdim=True)
                                    )
                                    m_patches_sub = m_patches_sub + pisa_alpha * (
                                        m_patches_all.mean(dim=1, keepdim=True) - m_patches_sub.mean(dim=1, keepdim=True)
                                    )
                                cond = torch.cat([cond[:, :5], c_patches_sub], dim=1).contiguous()
                                cond_msa_proj = torch.cat([cond_msa_proj[:, :5], m_patches_sub], dim=1).contiguous()
                                L = cond.shape[1]
                                if b_idx == 0:
                                    rope2d_sub_key = f"rope2d_sub_{nc_dino_stride}_{L - 5}"
                                    rope2d_sub = geom.get(rope2d_sub_key)
                                    if rope2d_sub is None:
                                        rope2d_sub = (
                                            cos_cond_2d[:, ::nc_dino_stride].contiguous(),
                                            sin_cond_2d[:, ::nc_dino_stride].contiguous(),
                                        )
                                        geom[rope2d_sub_key] = rope2d_sub
                                    cos_cond_2d, sin_cond_2d = rope2d_sub
                                b7_carved = True
                            if use_cond_msa_stride:
                                prev_cond_msa_proj = cond_msa_proj
                            cond, h_cond_cr = fast_gate_add_and_modulate_cond(
                                cond, cond_msa_proj, cg_msa, cs_cr, csh_cr, use_triton=use_triton
                            )

                        if not locals().get("b7_carved", False):
                            b7_carved = False
                        h_x = h_x_precomputed if h_x_precomputed is not None else fast_modulate(x, sizes, s_cr, sh_cr)
                        if L != kv_maxlen_full:
                            c_key = f"kv_seqlens_nc_{L}"
                            kv_seq_cond_cr = geom.get(c_key)
                            if kv_seq_cond_cr is None:
                                kv_seq_cond_cr = torch.arange(
                                    0, (batch_size + 1) * L, L, device=device, dtype=torch.int32
                                )
                                geom[c_key] = kv_seq_cond_cr
                            kv_max_cond_cr = L
                            q_seq_cond_cr = kv_seq_cond_cr
                            q_max_cond_cr = kv_max_cond_cr
                        else:
                            kv_seq_cond_cr = kv_seqlens_full
                            kv_max_cond_cr = kv_maxlen_full
                            q_seq_cond_cr = kv_seqlens_full
                            q_max_cond_cr = kv_maxlen_full
                        q_cc_cr: Optional[torch.Tensor] = None
                        will_reuse_cond_ffn = (
                            use_cond_ffn_stride
                            and (b_idx > 0)
                            and (b_idx < self.cond_depth - 1)
                            and (
                                (b_idx % cond_ffn_stride != 0)
                                or (
                                    use_hier_stride
                                    and use_cond_vox_stride
                                    and ((b_idx // cond_vox_stride) % cond_ffn_stride != 0)
                                )
                            )
                            and (prev_cond_rcr_proj is not None)
                            and (prev_cond_ffn_out is not None)
                            and (pos_kv_x_list is None or prev_pos_kv_x is not None)
                        )
                        if skip_cond_out or will_reuse_cond_ffn:
                            # Exact Cross-KV Slice: when skip_cond_out=True or will_reuse_cond_ffn=True, Q_cc ([L, D]) is never used!
                            # Compute only K_cc, V_cc ([L, 2*D]), cutting 33.3% of DINO cross-QKV GEMM FLOPs!
                            kvc_w = getattr(block, "_cond_to_kv_cross_slice_w", None)
                            if kvc_w is not None:
                                kv_cc = F.linear(
                                    h_cond_cr, kvc_w, getattr(block, "_cond_to_kv_cross_slice_b", None)
                                ).view(batch_size * L, 2, H, HD)
                            else:
                                kv_cc = block.cond_to_qkv_cross(h_cond_cr).view(batch_size * L, 3, H, HD)[:, 1:]
                        elif (
                            (b_idx >= 1)
                            and (not b7_carved)
                            and (cond_kv_qmc_stride > 1)
                            and return_mid_delta
                            and (not need_full_kv_cc)
                            and batch_size == 1
                            and L > 16
                            and getattr(block, "_cond_to_q_cross_slice_w", None) is not None
                            and getattr(block, "_cond_to_kv_cross_slice_w", None) is not None
                        ):
                            # ClusCa (arXiv:2509.10312) + MC-Sparse (arXiv:2610.06801) Conditioning DINO Cross-KV 2D QMC Stride
                            # + PISA (arXiv:2602.01077, ICML 2026) Zeroth-Order Centroid Value Compensation:
                            # In blocks 1..7 on Step 0 pos, compute full-L Q_cc for reverse `cond -> x` pooling (only on active FFN blocks),
                            # and 2D QMC strided K_cc, V_cc (5 global tokens + every `cond_kv_qmc_stride`-th patch)
                            # for forward `x -> cond` cross-attention, cutting 50%-83% of cond_to_qkv_cross GEMM FLOPs
                            # and 75% of forward x->cond FlashAttention FLOPs!
                            if not will_reuse_cond_ffn:
                                q_cc_cr = F.linear(
                                    h_cond_cr,
                                    block._cond_to_q_cross_slice_w,
                                    getattr(block, "_cond_to_q_cross_slice_b", None),
                                ).view(batch_size * L, H, HD)
                            cr_patches_all = h_cond_cr[:, 5:]
                            cr_patches_sub = cr_patches_all[:, ::cond_kv_qmc_stride]
                            if use_pisa_comp and pisa_alpha > 0.0 and cr_patches_sub.shape[1] > 0:
                                cr_patches_sub = cr_patches_sub + pisa_alpha * (
                                    cr_patches_all.mean(dim=1, keepdim=True) - cr_patches_sub.mean(dim=1, keepdim=True)
                                )
                            h_cond_cr_sub = torch.cat([h_cond_cr[:, :5], cr_patches_sub], dim=1).contiguous()
                            L_sub_cr = h_cond_cr_sub.shape[1]
                            kv_cc = F.linear(
                                h_cond_cr_sub.view(batch_size * L_sub_cr, D),
                                block._cond_to_kv_cross_slice_w,
                                getattr(block, "_cond_to_kv_cross_slice_b", None),
                            ).view(batch_size * L_sub_cr, 2, H, HD)
                            c_key = f"kv_seqlens_nc_{L_sub_cr}"
                            kv_seq_cond_cr = geom.get(c_key)
                            if kv_seq_cond_cr is None:
                                kv_seq_cond_cr = torch.arange(
                                    0, (batch_size + 1) * L_sub_cr, L_sub_cr, device=device, dtype=torch.int32
                                )
                                geom[c_key] = kv_seq_cond_cr
                            kv_max_cond_cr = L_sub_cr
                        else:
                            qkv_cc_cr = block.cond_to_qkv_cross(h_cond_cr).view(batch_size * L, 3, H, HD)
                            q_cc_cr = qkv_cc_cr[:, 0]
                            kv_cc = qkv_cc_cr[:, 1:]
                        if need_full_kv_cc and kv_cc_cache_out is not None:
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
                            cr_out_for_x = cr_proj_out
                        elif will_reuse_cond_ffn and prev_cond_rcr_proj is not None and prev_cond_ffn_out is not None:
                            # SCR (Spectral-Guided Diffusion, arXiv:2609.29505, Aug 2026) + HetCache (arXiv:2603.24260)
                            # + LayerCache (arXiv:2604.16492): in odd intermediate conditioning blocks (1, 3, 5),
                            # compute exact forward `x -> cond` cross-attention for 3D voxels while reusing the
                            # adjacent even block's reverse `cond <- x` projection and 4101-token DINO FFN residual
                            # modulated by the current block's exact gates `cg_cr[b_idx]` and `cg_mlp[b_idx]`!
                            q_w = getattr(block, "_to_q_cross_w", None)
                            if q_w is not None:
                                q_x = F.linear(h_x, q_w, getattr(block, "_to_q_cross_b", None)).view(N, H, HD)
                            else:
                                q_x = block.to_qkv_cross(h_x).view(N, 3, H, HD)[:, 0]
                            if pos_kv_x_list is not None and prev_pos_kv_x is not None:
                                pos_kv_x_list.append(prev_pos_kv_x)
                            h_x_out = flash_cross_attn(
                                q_x,
                                q_seqlens,
                                q_maxlen,
                                kv_cc,
                                kv_seq_cond_cr,
                                kv_max_cond_cr,
                            )
                            cr_proj_out = block.to_out_cross(h_x_out.reshape(N, D))
                            if use_cond_cr_stride:
                                prev_cond_x_cr_out = cr_proj_out
                            cr_out_for_x = cr_proj_out
                            cond = fast_gate_add_cond(cond, prev_cond_rcr_proj, cg_cr)
                            cond_ffn_out = prev_cond_ffn_out
                            next_reuses_cond_msa = (
                                use_cond_msa_stride
                                and (b_idx + 1 > 0)
                                and (b_idx + 1 < self.cond_depth - 1)
                                and (
                                    ((b_idx + 1) % cond_msa_stride != 0)
                                    or (
                                        use_hier_stride
                                        and use_cond_vox_stride
                                        and (((b_idx + 1) // cond_vox_stride) % cond_msa_stride != 0)
                                    )
                                )
                            )
                            if (
                                use_interblock_fuse
                                and (b_idx + 1 < self.cond_depth)
                                and (cached_mid_delta is None)
                                and (not next_reuses_cond_msa)
                            ):
                                cs_msa_nxt, csh_msa_nxt = cond_gates[b_idx + 1][0], cond_gates[b_idx + 1][1]
                                cond, h_cond_msa_precomputed = fast_gate_add_and_modulate_cond(
                                    cond, cond_ffn_out, cg_mlp, cs_msa_nxt, csh_msa_nxt, use_triton=use_triton
                                )
                            else:
                                cond = fast_gate_add_cond(cond, cond_ffn_out, cg_mlp)
                        else:
                            if (
                                return_mid_delta
                                and (not need_full_kv_cc)
                                and batch_size == 1
                                and N >= 64
                                and pos_qmc_stride > 1
                            ):
                                # Dual-Asymmetric Cross-Attention (MC-Sparse arXiv:2610.06801 / JiT CVPR 2026):
                                # 1. Exact full-N Q_x ([N, H, HD]) for voxel -> DINO cross-attention (`x -> cond`)
                                # 2. Hilbert-curve QMC strided K_x, V_x (`h_x[::pos_qmc_stride]`) for global DINO
                                #    context pooling (`cond -> x`), cutting 50% of to_qkv_cross GEMM FLOPs and 75%
                                #    of reverse FlashAttention FLOPs across all 8 conditioning blocks!
                                q_w = getattr(block, "_to_q_cross_w", None)
                                kv_w = getattr(block, "_to_kv_cross_w", None)
                                if q_w is not None and kv_w is not None:
                                    q_x = F.linear(h_x, q_w, getattr(block, "_to_q_cross_b", None)).view(N, H, HD)
                                    h_x_sub = h_x[::pos_qmc_stride]
                                    n_sub_pos = h_x_sub.shape[0]
                                    kv_x = F.linear(h_x_sub, kv_w, getattr(block, "_to_kv_cross_b", None)).view(
                                        n_sub_pos, 2, H, HD
                                    )
                                    q_seq_pos = geom.get(f"q_seqlens_qmc_{pos_qmc_stride}")
                                    if q_seq_pos is None:
                                        q_seq_pos = torch.tensor([0, n_sub_pos], device=device, dtype=torch.int32)
                                        geom[f"q_seqlens_qmc_{pos_qmc_stride}"] = q_seq_pos
                                    q_max_pos = n_sub_pos
                                else:
                                    qkv_x_cr = block.to_qkv_cross(h_x).view(N, 3, H, HD)
                                    q_x = qkv_x_cr[:, 0]
                                    kv_x = qkv_x_cr[:, 1:]
                                    q_seq_pos = q_seqlens
                                    q_max_pos = q_maxlen
                            else:
                                qkv_x_cr = block.to_qkv_cross(h_x).view(N, 3, H, HD)
                                q_x = qkv_x_cr[:, 0]
                                kv_x = qkv_x_cr[:, 1:]
                                q_seq_pos = q_seqlens
                                q_max_pos = q_maxlen

                            cur_pos_kv_x = (kv_x, q_seq_pos, int(q_max_pos))
                            if pos_kv_x_list is not None:
                                pos_kv_x_list.append(cur_pos_kv_x)
                            if use_cond_ffn_stride:
                                prev_pos_kv_x = cur_pos_kv_x
                            h_x_out = flash_cross_attn(
                                q_x,
                                q_seqlens,
                                q_maxlen,
                                kv_cc,
                                kv_seq_cond_cr,
                                kv_max_cond_cr,
                            )
                            cr_proj_out = block.to_out_cross(h_x_out.reshape(N, D))
                            if b_idx == 0 and not uncond:
                                self._last_b0_cr_proj = cr_proj_out
                            if use_cond_cr_stride:
                                prev_cond_x_cr_out = cr_proj_out
                            cr_out_for_x = cr_proj_out

                            h_c_out = flash_cross_attn(
                                q_cc_cr,
                                q_seq_cond_cr,
                                q_max_cond_cr,
                                kv_x,
                                q_seq_pos,
                                q_max_pos,
                            )
                            cond_rcr_proj = block.cond_to_out_cross(h_c_out.reshape(batch_size, L, D))
                            if use_cond_ffn_stride:
                                prev_cond_rcr_proj = cond_rcr_proj
                            cond, h_cond_mlp = fast_gate_add_and_modulate_cond(
                                cond,
                                cond_rcr_proj,
                                cg_cr,
                                cs_mlp,
                                csh_mlp,
                                use_triton=use_triton,
                            )

                            cond_ffn_out = block.cond_ffn1(
                                fast_swiglu_from_packed(block.cond_ffn0(h_cond_mlp), use_triton=use_triton)
                            )
                            if use_cond_ffn_stride:
                                prev_cond_ffn_out = cond_ffn_out
                            next_reuses_cond_msa = (
                                use_cond_msa_stride
                                and (b_idx + 1 > 0)
                                and (b_idx + 1 < self.cond_depth - 1)
                                and (
                                    ((b_idx + 1) % cond_msa_stride != 0)
                                    or (
                                        use_hier_stride
                                        and use_cond_vox_stride
                                        and (((b_idx + 1) // cond_vox_stride) % cond_msa_stride != 0)
                                    )
                                )
                            )
                            if (
                                use_interblock_fuse
                                and (b_idx + 1 < self.cond_depth)
                                and (cached_mid_delta is None)
                                and (not next_reuses_cond_msa)
                            ):
                                cs_msa_nxt, csh_msa_nxt = cond_gates[b_idx + 1][0], cond_gates[b_idx + 1][1]
                                cond, h_cond_msa_precomputed = fast_gate_add_and_modulate_cond(
                                    cond, cond_ffn_out, cg_mlp, cs_msa_nxt, csh_msa_nxt, use_triton=use_triton
                                )
                            elif use_b7_fuse and (b_idx == self.cond_depth - 1) and (cached_cond_block7 is None):
                                need_c_out = bool(return_cond_delta and cond_mid_in is not None)
                                cond_new, ln_cond_after_7 = fast_gate_add_and_ln_cond(
                                    cond, cond_ffn_out, cg_mlp, use_triton=use_triton, return_cond=need_c_out
                                )
                                if need_c_out and cond_new is not None:
                                    cond = cond_new
                            else:
                                cond = fast_gate_add_cond(cond, cond_ffn_out, cg_mlp)

                if b_idx == self.cond_depth - 1 and cached_cond_block7 is None:
                    if return_cond_delta and cond_mid_in is not None:
                        cond_delta_out = cond - cond_mid_in
                    if ln_cond_after_7 is None:
                        ln_cond_after_7 = fast_layer_norm(cond)
                    if (
                        (not uncond)
                        and (not locals().get("b7_carved", False))
                        and (nc_dino_stride > 1)
                        and return_mid_delta
                        and (not need_full_kv_cc)
                        and batch_size == 1
                        and ln_cond_after_7 is not None
                        and ln_cond_after_7.shape[1] > 16
                    ):
                        # ClusCa (ACM MM 2025, arXiv:2509.10312) + MC-Sparse (arXiv:2610.06801)
                        # + PISA (arXiv:2602.01077, ICML 2026) Non-Conditioning DINO Context 2D QMC Striding for Blocks 8..23:
                        # Retain 100% of the 5 global tokens (CLS + 4 registers) and every `nc_dino_stride`-th
                        # patch token after 8 layers of dense 2D self+cross attention aggregation!
                        ln_patches_all = ln_cond_after_7[:, 5:]
                        ln_patches_sub = ln_patches_all[:, ::nc_dino_stride]
                        if use_pisa_comp and pisa_alpha > 0.0 and ln_patches_sub.shape[1] > 0:
                            ln_patches_sub = ln_patches_sub + pisa_alpha * (
                                ln_patches_all.mean(dim=1, keepdim=True) - ln_patches_sub.mean(dim=1, keepdim=True)
                            )
                        ln_cond_after_7 = torch.cat([ln_cond_after_7[:, :5], ln_patches_sub], dim=1).contiguous()

            else:
                cs_cr, csh_cr = cond_gates[b_idx]

                if use_1tok_uncond:
                    # Exact 1-token cross-attn in blocks 8..23:
                    # Algebraic 2-layer collapse `to_out_cross(cond_to_v_cross(h_cond_cr))` in 1 GEMM!
                    h_cond_cr = fast_modulate_preln_cond(ln_cond_after_7, cs_cr, csh_cr)
                    c1_w = getattr(block, "_cond_1tok_cr_w", None) if use_1tok_collapse else None
                    if c1_w is not None:
                        h_x_cross = F.linear(h_cond_cr.view(1, D), c1_w, getattr(block, "_cond_1tok_cr_b", None))
                    else:
                        v_w = getattr(block, "_cond_to_v_cross_w", None)
                        if v_w is not None:
                            v_cc = F.linear(h_cond_cr, v_w, getattr(block, "_cond_to_v_cross_b", None)).view(1, D)
                        else:
                            v_cc = block.cond_to_kv_cross(h_cond_cr).view(1, 2, D)[:, 1]
                        h_x_cross = block.to_out_cross(v_cc)
                    cr_out_for_x = h_x_cross
                elif cached_cond_block7 is not None:
                    h_x = h_x_precomputed if h_x_precomputed is not None else fast_modulate(x, sizes, s_cr, sh_cr)
                    q_x = block.to_q_cross(h_x).view(N, H, HD)
                    kv_cc = cached_cond_block7[b_idx]
                    h_x_out = flash_cross_attn(
                        q_x, q_seqlens, q_maxlen, kv_cc, kv_seqlens_full, kv_maxlen_full
                    )
                    cr_out_for_x = block.to_out_cross(h_x_out.reshape(N, D))
                elif will_reuse_nc_cr and prev_nc_cr_out is not None:
                    # PAB (ICLR 2025, arXiv:2408.12558) + DuCa (ToCa-v2) Non-Conditioning Cross-Attention Broadcast:
                    # In frozen-context blocks 8..23 (where `ln_cond_after_7` is static), odd intermediate blocks
                    # reuse the adjacent block's cross-attention output with the current block's exact gate `g_cr[b_idx]`,
                    # bypassing LayerNorm+Modulate(x), to_q_cross, cond_to_kv_cross, FlashCrossAttn, and to_out_cross!
                    cr_out_for_x = prev_nc_cr_out
                else:
                    L_nc = ln_cond_after_7.shape[1]
                    h_cond_cr = fast_modulate_preln_cond(ln_cond_after_7, cs_cr, csh_cr)
                    h_x = h_x_precomputed if h_x_precomputed is not None else fast_modulate(x, sizes, s_cr, sh_cr)
                    q_x = block.to_q_cross(h_x).view(N, H, HD)
                    kv_cc = block.cond_to_kv_cross(h_cond_cr.view(batch_size * L_nc, D)).view(batch_size * L_nc, 2, H, HD)
                    if need_full_kv_cc and kv_cc_cache_out is not None:
                        kv_cc_cache_out.append(kv_cc)
                    if L_nc == kv_maxlen_full:
                        kv_seq_nc, kv_max_nc = kv_seqlens_full, kv_maxlen_full
                    else:
                        nc_key = f"kv_seqlens_nc_{L_nc}"
                        kv_seq_nc = geom.get(nc_key)
                        if kv_seq_nc is None:
                            kv_seq_nc = torch.arange(
                                0, (batch_size + 1) * L_nc, L_nc, device=device, dtype=torch.int32
                            )
                            geom[nc_key] = kv_seq_nc
                        kv_max_nc = L_nc
                    h_x_out = flash_cross_attn(
                        q_x, q_seqlens, q_maxlen, kv_cc, kv_seq_nc, kv_max_nc
                    )
                    cr_out_for_x = block.to_out_cross(h_x_out.reshape(N, D))
                    if use_pab_nc_cr or use_nc_vox_stride:
                        prev_nc_cr_out = cr_out_for_x

            if record_pos_carrier and not use_fused_cr_accum:
                g_cr_2d = getattr(g_cr, "_g_2d", g_cr.view(1, -1))
                if pos_cr_sum is None:
                    pos_cr_sum = cr_out_for_x * g_cr_2d
                else:
                    pos_cr_sum = torch.addcmul(pos_cr_sum, cr_out_for_x, g_cr_2d, value=1.0)

            if b_idx == 0 and (not uncond) and use_subblock_crf:
                if cached_mid_delta is not None and (
                    getattr(self, "_last_pos_crf_after_b0_cr", None) is not None
                    and self._last_pos_crf_after_b0_cr.shape == (N, D)
                    and self._last_pos_crf_after_b0_cr.device == device
                ):
                    x = fast_gate_add(x, cr_out_for_x, sizes, g_cr)
                    # DBCache-FoCa Subblock-2 CRF (vipshop/cache-dit 2026 + FreqCa NeurIPS 2026):
                    # On Step 2 (pos), after updating 3D MSA + DINO Cross-Attn in Block 0,
                    # bypass Block 0's heavy FFN0(1536->8192) + 3x3x3 SparseConv + FFN1 and Blocks 1..23!
                    if x_b0_msa is not None:
                        self._last_pos_crf_after_b0_msa = (
                            (x - x_b0_msa) + self._last_pos_crf_after_b0_cr
                        )
                    x = torch.add(x, self._last_pos_crf_after_b0_cr, alpha=eff_mid_scale)
                    if (
                        foca_gamma > 0.0
                        and x_b0_msa is not None
                        and getattr(self, "_last_b0_fn_res_step0", None) is not None
                        and self._last_b0_fn_res_step0.shape == (N, D)
                    ):
                        x = torch.add(
                            x,
                            (x_b0_msa - x_patchify) - self._last_b0_fn_res_step0,
                            alpha=foca_gamma,
                        )
                    break
                else:
                    # Fused Block-0 Cross-Attention Gate-Add + FFN LayerNorm & AdaLN Modulation in 1 Triton kernel!
                    x, h_mlp = fast_gate_add_and_modulate(
                        x,
                        cr_out_for_x,
                        sizes,
                        g_cr,
                        s_mlp,
                        sh_mlp,
                        use_triton=use_triton,
                        acc_inout=pos_cr_sum if use_fused_cr_accum else None,
                        init_acc=(b_idx == 0),
                    )
                    x_b0_cr = x
            else:
                # Fused Cross-Attention Gate-Add + FFN LayerNorm & AdaLN Modulation in 1 Triton kernel!
                x, h_mlp = fast_gate_add_and_modulate(
                    x,
                    cr_out_for_x,
                    sizes,
                    g_cr,
                    s_mlp,
                    sh_mlp,
                    use_triton=use_triton,
                    acc_inout=pos_cr_sum if use_fused_cr_accum else None,
                    init_acc=(b_idx == 0),
                )

            # --- 3. Main FFN + 3D Submanifold Sparse Conv + Fused Packed SwiGLU ---
            h_mlp = block.ffn0(h_mlp)
            if block.use_ffn_conv:
                w_fp32 = getattr(block, "_ffn_conv_w_fp32", block.ffn_conv.weight.float().contiguous())
                b_fp32 = getattr(block, "_ffn_conv_b_fp32", block.ffn_conv.bias.float().contiguous())
                h_mlp = fast_sparse_conv3d(h_mlp, w_fp32, b_fp32, conv_indices)
            h_mlp = block.ffn1(fast_swiglu_from_packed(h_mlp, use_triton=use_triton))
            if use_cond_vox_stride and block.cond:
                prev_cond_vox_mlp_out = h_mlp
            elif use_nc_vox_stride and (not block.cond):
                prev_nc_mlp_out = h_mlp
            will_skip_next = (
                can_mid_cache
                and (cached_mid_delta is not None)
                and (mid_start <= (b_idx + 1) < mid_end)
            )
            next_reuses_cond_vox = (
                use_cond_vox_stride
                and (b_idx + 1 > 0)
                and (b_idx + 1 < self.cond_depth - 1)
                and ((b_idx + 1) % cond_vox_stride != 0)
            )
            next_reuses_nc_vox = (
                use_nc_vox_stride
                and ((b_idx + 1) >= self.cond_depth)
                and (((b_idx + 1) - self.cond_depth) % nc_vox_stride != 0)
                and ((b_idx + 1) < num_blocks - 2)
            )
            if (
                use_interblock_fuse
                and (b_idx + 1 < num_blocks)
                and (not will_skip_next)
                and (not next_reuses_cond_vox)
                and (not next_reuses_nc_vox)
            ):
                s_msa_nxt, sh_msa_nxt = block_gates[b_idx + 1][0], block_gates[b_idx + 1][1]
                x, h_msa_precomputed = fast_gate_add_and_modulate(
                    x, h_mlp, sizes, g_mlp, s_msa_nxt, sh_msa_nxt, use_triton=use_triton
                )
            else:
                x = fast_gate_add(x, h_mlp, sizes, g_mlp)

            if can_mid_cache and b_idx == (mid_end - 1) and return_mid_delta and x_mid_in is not None:
                if (not uncond) and use_freqca_crf and x_b0_msa is not None:
                    self._last_pos_crf_after_b0_msa = x - x_b0_msa
                    if cached_mid_delta is None:
                        self._last_pos_crf_after_b0_msa_step0 = self._last_pos_crf_after_b0_msa
                if (not uncond) and use_subblock_crf and x_b0_cr is not None:
                    self._last_pos_crf_after_b0_cr = x - x_b0_cr
                if record_pos_carrier and pos_cr_sum is not None and pos_kv_x_list is not None:
                    self._last_pos_x_without_cr = x - pos_cr_sum
                    self._last_pos_kv_x_list = pos_kv_x_list
                    self._last_pos_carrier_t_val = t_val
                if (not uncond) and use_subblock_crf and getattr(self, "_last_pos_crf_after_b0_msa", None) is not None:
                    mid_delta_out = self._last_pos_crf_after_b0_msa
                else:
                    mid_delta_out = x - x_mid_in

        # Fused Output LayerNorm + AdaLN modulation + exact FP32 Unpatchify projection
        out = fast_out_modulate_unpatchify(
            x, sizes, tg["out_scale"], tg["out_shift"], self._unpatchify_w_fp32, use_triton=use_triton
        )

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
        elif (
            conditioning.batch_size == 1
            and getattr(self, "_zero_clf_token", None) is not None
            and self._zero_clf_token.device == device
            and self._zero_clf_token.dtype == compute_dtype
        ):
            clf_token = self._zero_clf_token
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
        if not apply_rk_readout:
            uncond_mid_sc = 1.0
        b_init_t = qt.get("b_init_t") if apply_rk_readout else None
        if b_init_t is not None:
            rk_corr = xt @ b_init_t

        self._last_b0_x_after_msa = None
        self._last_b0_x_patchify = None
        self._last_b0_cr_proj = None
        self._last_b0_cond_msa_out = None
        self._last_pos_crf_after_b0_msa = None
        self._last_pos_crf_after_b0_msa_step0 = None
        self._last_pos_crf_after_b0_cr = None
        self._last_b0_fn_res = None
        self._last_b0_fn_res_step0 = None
        self._last_pos_x_without_cr = None
        self._last_pos_kv_x_list = None
        self._last_pos_carrier_t_val = None
        last_neg_x0: Optional[torch.Tensor] = None
        at_blend = float(getattr(self.solver_config, "autotarget_clean_endpoint_blend", 0.08))
        erta_alpha = float(getattr(self.solver_config, "ertacache_euler_alpha", 0.04))
        cd_omega = float(getattr(self.solver_config, "counterdiabatic_berry_omega", 0.025))
        magnus_gamma = float(getattr(self.solver_config, "magnus_commutator_gamma", 0.015))
        use_cheb_pade = bool(getattr(self.solver_config, "chebyshev_pade_resolvent", True))
        pade_eta = float(getattr(self.solver_config, "pade_resolvent_eta", 0.018))
        use_symp_verlet = bool(getattr(self.solver_config, "symplectic_verlet_kick", True))
        hap_omega = float(getattr(self.solver_config, "hamiltonian_hap_omega", 0.016))
        use_stork2 = bool(getattr(self.solver_config, "stork2_stabilized_rk", True))
        stork2_stages = int(getattr(self.solver_config, "stork2_stages", 9))
        stork2_beta = compute_stork2_beta(stork2_stages) if use_stork2 else 0.0
        use_romberg = bool(getattr(self.solver_config, "richardson_romberg_extrapolation", True))
        use_fastercache_cfg = bool(getattr(self.solver_config, "fastercache_freq_decoupled_cfg", True))
        fc_hf_boost = float(getattr(self.solver_config, "fastercache_hf_boost", 0.035))
        use_cayley_lie = bool(getattr(self.solver_config, "cayley_lie_orthogonal_step", True))
        use_flow_unipc = bool(getattr(self.solver_config, "flow_unipc_bh2_integrator", True))
        use_cab_defect = bool(getattr(self.solver_config, "cab_extrapolation_defect", True))
        cab_gamma = float(getattr(self.solver_config, "cab_defect_gamma", 0.022))
        use_stork4 = bool(getattr(self.solver_config, "stork4_rock4_finishing", True))
        stork4_gamma = float(getattr(self.solver_config, "stork4_gamma", 0.018))
        use_postcache_rts = bool(getattr(self.solver_config, "postcache_rts_smoothing", True))
        postcache_gamma = float(getattr(self.solver_config, "postcache_rts_gamma", 0.020))
        use_arex_affine = bool(getattr(self.solver_config, "arex_affine_integrator", True))
        arex_gamma = float(getattr(self.solver_config, "arex_affine_gamma", 0.016))
        use_racer_shrink = bool(getattr(self.solver_config, "racer_disagreement_shrinkage", True))
        racer_eta = float(getattr(self.solver_config, "racer_trust_eta", 0.15))
        use_anderson_aag = bool(getattr(self.solver_config, "anderson_aag_acceleration", True))
        anderson_gamma = float(getattr(self.solver_config, "anderson_aag_gamma", 0.020))
        use_exp_rosenbrock = bool(getattr(self.solver_config, "exp_rosenbrock_integrator", True))
        exp_rosenbrock_gamma = float(getattr(self.solver_config, "exp_rosenbrock_gamma", 0.018))
        use_tacache_povd = bool(getattr(self.solver_config, "tacache_vde_povd", True))
        tacache_gamma = float(getattr(self.solver_config, "tacache_povd_gamma", 0.016))
        mcfm_gamma = float(getattr(self.solver_config, "mcfm_strang_prox_gamma", 0.012))
        use_slerp_sv = bool(getattr(self.solver_config, "slerpflow_strain_vorticity", True))
        slerp_gamma = float(getattr(self.solver_config, "slerpflow_centripetal_gamma", 0.014))
        strain_alpha = float(getattr(self.solver_config, "strain_suppression_alpha", 0.015))
        prev_v_pos_step: Optional[torch.Tensor] = None
        prev2_v_pos_step: Optional[torch.Tensor] = None
        prev_xt_step: Optional[torch.Tensor] = None
        prev_x0_hat_step: Optional[torch.Tensor] = None
        num_blocks = len(self.flow.blocks)
        raw_mid_end = int(getattr(self.solver_config, "mid_block_end", 24))
        mid_end_eff = min(raw_mid_end, num_blocks) if num_blocks >= 2 else num_blocks
        use_freqca_tail = (
            use_mid_cache
            and (mid_end_eff == num_blocks)
            and bool(getattr(self.solver_config, "freqca_crf_cache", True))
        )
        need_cond_delta = not use_freqca_tail
        need_uncond_ln7 = not use_freqca_tail

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
                prev_xt_step = None
                prev_x0_hat_step = None

            eval_cond = should_eval_cond(step_idx, n_guided, cond7_stride, eval_cond_steps)
            need_cond_cache = (cond7_stride > 1) or (eval_cond_steps is not None and len(eval_cond_steps) > 0)
            mid_sc = float(mid_scales[step_idx]) if (apply_rk_readout and step_idx < len(mid_scales)) else 1.0

            if (not eval_cond) and (last_cond7 is not None):
                # Fast steps (1, 3): execute only Block 0 Subblock-1 (skipping 23 tail blocks 1..23)!
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
                # Step 2+: refresh DINO conditioning in block 0 at new t_eval and apply True First-Block Cache!
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
                    cached_cond_delta=last_cond_delta if need_cond_delta else None,
                )
                if last_cond7 is None:
                    last_cond7 = new_cond7
                elif new_cond7 is not None:
                    for b_i, kv_new in enumerate(new_cond7):
                        if kv_new is not None and b_i < len(last_cond7):
                            last_cond7[b_i] = kv_new
            elif need_cond_cache and use_mid_cache:
                if need_cond_delta:
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
                else:
                    v_pos, last_cond7, last_mid_delta = self.forward_flow_fast(
                        xt,
                        t_eval,
                        context,
                        clf_token,
                        geom,
                        uncond=False,
                        return_cond_block7=True,
                        return_mid_delta=True,
                        return_cond_delta=False,
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
                        if need_uncond_ln7:
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
                        else:
                            v_neg, last_uncond_mid_delta = self.forward_flow_fast(
                                xt,
                                t_eval,
                                context,
                                clf_token,
                                geom,
                                uncond=True,
                                return_mid_delta=True,
                                return_ln_cond7=False,
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
                            cached_ln_cond7=last_uncond_ln7 if need_uncond_ln7 else None,
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
                    if at_blend > 0.0:
                        last_neg_x0 = xt - t_eval * v_neg
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
                            # AutoTarget Clean-Endpoint Reparameterization (arXiv:2610.03577) + CFG-Zero* (arXiv:2503.18886)
                            neg_cand = last_neg
                            if at_blend > 0.0 and last_neg_x0 is not None and t_eval > 0.25:
                                v_neg_ep = (xt - last_neg_x0) * (1.0 / t_eval)
                                neg_cand = (1.0 - at_blend) * last_neg + at_blend * v_neg_ep
                            dot_pn = (v_pos * neg_cand).sum()
                            norm_nn = (neg_cand * neg_cand).sum().clamp_min(1e-6)
                            s_star = (dot_pn / norm_nn).clamp(0.90, 1.10)
                            cur_delta = v_pos - s_star * neg_cand
                            if (not apply_rk_readout) and use_fastercache_cfg and fc_hf_boost > 0.0 and N > 1:
                                # FasterCache Frequency-Decoupled CFG Reuse (ICLR 2025, Vchitect/FasterCache):
                                # Decompose CFG delta into low-frequency DC mean and high-frequency spatial detail,
                                # boosting high-frequency surface detail on skipped unconditional steps!
                                delta_dc = cur_delta.mean(dim=0, keepdim=True)
                                hf_gain = fc_hf_boost * max(0.0, 1.0 - t_eval)
                                cur_delta = torch.add(cur_delta, cur_delta - delta_dc, alpha=hf_gain)
                        else:
                            cur_delta = v_pos - last_neg
                    elif cfg_mode == "delta_extrapolate" and prev_delta is not None:
                        cur_delta = last_delta + cfg_alpha * (last_delta - prev_delta)
                    else:
                        cur_delta = last_delta

                cfg_mult = 1.0
                if apply_rk_readout and step_idx < len(self.solver_config.qaoa_cfg_scales):
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
                if apply_rk_readout and step_idx < len(self.solver_config.qaoa_momentum):
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

                if (
                    (not apply_rk_readout)
                    and use_stork2
                    and stork2_beta > 0.0
                    and (prev_v is not None)
                    and (prev_dt is not None)
                ):
                    # STORK-2 Stabilized 9-Stage Orthogonal Runge-Kutta (arXiv:2505.24210):
                    # Closed-form S-stage Gegenbauer-Chebyshev recurrence (beta_9 = 0.46283549)
                    # with Padé-bounded step ratio!
                    stork_ratio = min(dt / max(prev_dt, 1e-6), ab2_max_ratio)
                    unipc_phi2 = compute_flow_unipc_bh2_scale(t_val, t_next) if use_flow_unipc else 1.0
                    stork_coeff = 0.06 * stork2_beta * stork_ratio * unipc_phi2
                    dv_stork = v - prev_v
                    v_norm_s = (v * v).sum().clamp_min(1e-6)
                    stork_res = 1.0 / (1.0 + pade_eta * ((dv_stork * dv_stork).sum() / v_norm_s))
                    v_step = torch.addcmul(v_step, dv_stork, stork_res, value=stork_coeff)

                acc_vecs = qt.get("acc_vecs", [])
                acc_vec = acc_vecs[step_idx] if (apply_rk_readout and step_idx < len(acc_vecs)) else None
                if prev2_v is not None and prev_v is not None:
                    if acc_vec is not None:
                        v_step = torch.addcmul(v_step, v - 2.0 * prev_v + prev2_v, acc_vec, value=1.0)
                    else:
                        # HiCache Dual-Scaled Hermite 2nd-Order Velocity Forecast + Zero-Sync GPU Directional Gate
                        hermite_sigma = float(getattr(self.solver_config, "hicache_hermite_sigma", 0.18))
                        if hermite_sigma > 0.0:
                            dv_cur = v - prev_v
                            dv_prev = prev_v - prev2_v
                            dir_gate = ((dv_cur * dv_prev).sum() > 0).to(v_step.dtype)
                            v_step = torch.addcmul(
                                v_step,
                                dv_cur - dv_prev,
                                dir_gate,
                                value=0.5 * hermite_sigma * hermite_sigma,
                            )
                        if use_cab_defect and cab_gamma > 0.0 and step_idx >= 2:
                            # CAB Extrapolation-Defect Correction in Noise-to-Signal Ratio (NSR) Coordinates
                            # (Roy & Nair, arXiv:2605.16736v2, May 2026):
                            # Feed back the Lagrange extrapolation defect `v - ((1 + r_{i-1})*v_{i-1} - r_{i-1}*v_{i-2})`
                            # on the rectified rho_t = t / (1 - t) grid with Padé resolvent stabilization!
                            r_cab = compute_cab_nsr_step_ratio(
                                float(t_list[step_idx - 2]), float(t_list[step_idx - 1]), t_val
                            )
                            cab_defect = v - (1.0 + r_cab) * prev_v + r_cab * prev2_v
                            v_norm_cab = (v * v).sum().clamp_min(1e-6)
                            cab_res = (1.0 / (1.0 + pade_eta * ((cab_defect * cab_defect).sum() / v_norm_cab))).to(
                                dtype=v_step.dtype
                            )
                            cab_phi2 = compute_flow_unipc_bh2_scale(t_val, t_next) if use_flow_unipc else 1.0
                            v_step = torch.addcmul(
                                v_step,
                                cab_defect,
                                cab_res,
                                value=cab_gamma * cab_phi2 * (r_cab / (1.0 + r_cab)),
                            )
                        if use_stork4 and stork4_gamma > 0.0 and step_idx >= 2:
                            # STORK-4 / ROCK4 4-Stage Composition Finishing Integrator
                            # (Tan, Wang, Bertozzi & Ryu, ICLR 2026, arXiv:2505.24210v2 / Abdulle 2002)
                            # + RACER Dual-Forecast Disagreement Shrinkage (Li et al., arXiv:2608.01740, Aug 2026):
                            # Non-uniform 3-point Lagrange second-derivative curvature correction with closed-loop
                            # disagreement shrinkage when CAB and ROCK4 curvature forecasts diverge!
                            h_p2 = float(t_list[step_idx - 2]) - float(t_list[step_idx - 1])
                            h_p1 = float(t_list[step_idx - 1]) - t_val
                            w_c, w_p1, w_p2 = compute_stork4_rock4_weights(h_p2, h_p1, dt)
                            rock4_curv = w_c * v + w_p1 * prev_v + w_p2 * prev2_v
                            v_norm_r4 = (v * v).sum().clamp_min(1e-6)
                            rock4_res = (1.0 / (1.0 + pade_eta * ((rock4_curv * rock4_curv).sum() / v_norm_r4))).to(
                                dtype=v_step.dtype
                            )
                            if use_racer_shrink and use_cab_defect and cab_gamma > 0.0 and racer_eta > 0.0:
                                racer_diff = cab_defect - rock4_curv
                                racer_trust = (
                                    1.0 / (1.0 + racer_eta * ((racer_diff * racer_diff).sum() / v_norm_r4))
                                ).to(dtype=v_step.dtype)
                                rock4_res = rock4_res * racer_trust
                            v_step = torch.addcmul(
                                v_step,
                                rock4_curv,
                                rock4_res,
                                value=stork4_gamma,
                            )
                        if use_anderson_aag and anderson_gamma > 0.0 and step_idx >= 2:
                            # AAg(1) Modified Anderson Acceleration with Sharp Contraction Gain
                            # (arXiv:2605.17664, May 2026, Sec. 3, Eqs. 3.6-3.9 & Theorem 3.2)
                            # + Fast3Dcache Spatiotemporal Voxel Stability Criterion (CVPR 2026, arXiv:2511.22533v2):
                            # Minimize ||g_k + theta * dg_k||_2^2 over 3-step velocity residuals g_k = v - prev_v,
                            # dg_k = v - 2*prev_v + prev2_v, weighted by exact contraction gain (1 - gamma_{AA, k})!
                            g_aa = v - prev_v
                            dg_aa = g_aa - (prev_v - prev2_v)
                            norm_g_aa = (g_aa * g_aa).sum().clamp_min(1e-6)
                            norm_dg_aa = (dg_aa * dg_aa).sum().clamp_min(1e-6)
                            dot_aa = (g_aa * dg_aa).sum()
                            theta_aa = (-dot_aa / norm_dg_aa).clamp(-0.85, 0.85)
                            cos_sq_aa = ((dot_aa * dot_aa) / (norm_g_aa * norm_dg_aa)).clamp(0.0, 1.0)
                            gain_aa = 1.0 - torch.sqrt((1.0 - cos_sq_aa).clamp_min(0.0))
                            aa_corr = g_aa + theta_aa * dg_aa
                            v_norm_aa = (v * v).sum().clamp_min(1e-6)
                            aa_res = (
                                gain_aa / (1.0 + pade_eta * ((aa_corr * aa_corr).sum() / v_norm_aa))
                            ).to(dtype=v_step.dtype)
                            v_step = torch.addcmul(
                                v_step,
                                aa_corr,
                                aa_res,
                                value=anderson_gamma,
                            )

                if (
                    (not apply_rk_readout)
                    and (not eval_cond)
                    and erta_alpha > 0.0
                    and prev_v_pos is not None
                ):
                    # ERTACache Euler Trajectory Error-Adjustment (ICLR 2026, bytedance/ERTACache):
                    # Compensate accumulated caching truncation error on skipped conditioning steps!
                    v_step = torch.add(v_step, v_pos - prev_v_pos, alpha=erta_alpha)

                if (
                    (not apply_rk_readout)
                    and eval_cond
                    and use_postcache_rts
                    and postcache_gamma > 0.0
                    and step_idx >= 2
                    and prev_v_pos_step is not None
                    and prev_dt is not None
                ):
                    # PoSTCache Postdictive Rauch-Tung-Striebel (RTS) Trajectory Smoother
                    # (Tsinghua Univ., arXiv:2610.25664, Oct 19 2026, Eqs. 7-9):
                    # At future anchor b = step_idx, use the anchor innovation r_b = v_pos_b - v_pos_k
                    # weighted by RTS smoothing gain w_{k|a,b} = (t_a - t_k) / (t_a - t_b) to
                    # retrospectively revise the skipped interval's integration error with 0 extra NFE!
                    rts_gain = compute_postcache_rts_gain(
                        float(t_list[step_idx - 2]),
                        float(t_list[step_idx - 1]),
                        t_val,
                        prev_dt,
                        dt,
                    )
                    rts_innov = v_pos - prev_v_pos_step
                    v_norm_rts = (v_pos * v_pos).sum().clamp_min(1e-6)
                    rts_res = (1.0 / (1.0 + pade_eta * ((rts_innov * rts_innov).sum() / v_norm_rts))).to(
                        dtype=v_step.dtype
                    )
                    v_step = torch.addcmul(
                        v_step,
                        rts_innov,
                        rts_res,
                        value=postcache_gamma * rts_gain,
                    )

                if (
                    (not apply_rk_readout)
                    and use_arex_affine
                    and arex_gamma > 0.0
                    and prev_v is not None
                ):
                    # AREX Moment-Matched Affine-Residual Exponential Integrator
                    # (arXiv:2610.03483, Oct 2026, Proposition 3, Eqs. 5-9):
                    # Analytically propagate the L2-optimal Gaussian affine backbone F(t_{n+1}, t_n)
                    # and compensate the Euler affine discretization defect with Padé stabilization!
                    arex_def, _ = compute_arex_affine_defect(t_val, t_next)
                    if abs(arex_def) > 1e-6:
                        arex_vec = xt.to(dtype=v_step.dtype) - (t_val * v).to(dtype=v_step.dtype)
                        v_norm_ax = (v * v).sum().clamp_min(1e-6)
                        arex_res = (1.0 / (1.0 + pade_eta * ((arex_vec * arex_vec).sum() / v_norm_ax))).to(
                            dtype=v_step.dtype
                        )
                        v_step = torch.addcmul(
                            v_step,
                            arex_vec,
                            arex_res,
                            value=arex_gamma * arex_def,
                        )

                if (
                    (not apply_rk_readout)
                    and use_exp_rosenbrock
                    and exp_rosenbrock_gamma > 0.0
                    and prev_xt_step is not None
                    and prev_x0_hat_step is not None
                ):
                    # Exponential Rosenbrock-Euler / ExpRK Stiff Integrator (Hochbruck, Ostermann & Schweitzer /
                    # Ando, Cangiotti & Sensi, arXiv:2507.04024, Feb 2026, Eqs. 3-4)
                    # + DaTMe Tweedie Clean-Manifold Residual (arXiv:2610.22370, Oct 2026, Eqs. 4-6)
                    # + TACache Parallel-Orthogonal Velocity Decomposition (POVD / TASU, arXiv:2605.16789, May 2026)
                    # + VDE Input-Adaptive Velocity Decomposition (arXiv:2605.23381v1, May 2026)
                    # + MCFM Strang-Prox Metriplectic Conservative-Dissipative Shrinkage (arXiv:2509.19526v2, Theorem 2):
                    xt_f = xt.to(dtype=v_step.dtype)
                    x0_hat_cur = xt_f - (t_val * v).to(dtype=v_step.dtype)
                    dx0_hat = x0_hat_cur - prev_x0_hat_step
                    dxt_cur = xt_f - prev_xt_step
                    has_ta = use_tacache_povd and tacache_gamma > 0.0 and prev_v is not None and prev_dt is not None
                    if has_ta:
                        dv_cur = v - prev_v
                        a_ta = dv_cur * (1.0 / max(prev_dt, 1e-6))
                        er_norms = torch.stack(
                            [
                                (dx0_hat * dx0_hat).sum(),
                                (dxt_cur * dxt_cur).sum(),
                                (v * v).sum(),
                                (a_ta * v).sum(),
                                (a_ta * a_ta).sum(),
                                (v * xt_f).sum(),
                                (xt_f * xt_f).sum(),
                                (dv_cur * dxt_cur).sum(),
                            ]
                        ).tolist()
                    else:
                        er_norms = torch.stack(
                            [
                                (dx0_hat * dx0_hat).sum(),
                                (dxt_cur * dxt_cur).sum(),
                                (v * v).sum(),
                            ]
                        ).tolist()
                    n_dx0_sq = max(0.0, float(er_norms[0]))
                    n_dxt_sq = max(1e-12, float(er_norms[1]))
                    n_v_sq = max(1e-6, float(er_norms[2]))
                    _, phi1_er, phi2_er = compute_exp_rosenbrock_phi_gains(
                        t_val,
                        t_next,
                        math.sqrt(n_dx0_sq),
                        math.sqrt(n_dxt_sq),
                    )
                    er_res = 1.0 / (1.0 + pade_eta * (n_dx0_sq / n_v_sq))
                    er_coeff = exp_rosenbrock_gamma * phi1_er * (1.0 + phi2_er) * er_res
                    v_step = torch.add(v_step, dx0_hat, alpha=-er_coeff)
                    if has_ta:
                        dot_av_s = float(er_norms[3])
                        norm_a_sq_s = max(0.0, float(er_norms[4]))
                        dot_vx_s = float(er_norms[5])
                        norm_x_sq_s = max(1e-8, float(er_norms[6]))
                        dot_dv_dx_s = float(er_norms[7])
                        k_ta_s = dot_av_s / n_v_sq
                        norm_r_perp_sq = max(0.0, norm_a_sq_s - k_ta_s * dot_av_s)
                        g_par_ta, d_perp_ta, shrink_mcfm = compute_tacache_vde_mcfm_gains(
                            t_val,
                            t_next,
                            dot_av_s,
                            n_v_sq,
                            math.sqrt(norm_r_perp_sq),
                            dot_vx_s,
                            norm_x_sq_s,
                            mcfm_gamma=mcfm_gamma,
                        )
                        r_perp_ta = a_ta - k_ta_s * v
                        ta_res = 1.0 / (1.0 + pade_eta * (norm_r_perp_sq / n_v_sq))
                        v_step = v_step * (shrink_mcfm * (1.0 + tacache_gamma * (g_par_ta - 1.0)))
                        v_step = torch.add(v_step, r_perp_ta, alpha=tacache_gamma * d_perp_ta * ta_res)
                        if use_slerp_sv and (slerp_gamma > 0.0 or strain_alpha > 0.0):
                            c_slerp, g_strain = compute_slerpflow_strain_vorticity_gains(
                                t_val,
                                t_next,
                                n_v_sq,
                                dot_vx_s,
                                norm_x_sq_s,
                                dot_dv_dx_s,
                                n_dxt_sq,
                                slerp_gamma=slerp_gamma,
                                strain_alpha=strain_alpha,
                            )
                            if g_strain < 0.999999:
                                v_step = v_step * g_strain
                            if c_slerp > 1e-6 and dt > 1e-6:
                                v_step = torch.add(v_step, xt_f, alpha=c_slerp / max(dt, 1e-6))

                if (not apply_rk_readout) and prev_v is not None and (cd_omega > 0.0 or magnus_gamma > 0.0):
                    # Quantum Counterdiabatic Berry-Gauge Driving (Berry 2009 / Sels & Polkovnikov 2017)
                    # + 4th-Order Magnus Lie-Commutator Integrator (Magnus 1954 / Blanes et al. 2009):
                    # Orthogonal Fubini-Study / Berry connection projection of flow acceleration `dv = v - prev_v`
                    # onto the normal subspace of `v`, rotating the step along geodesic curvature with zero
                    # longitudinal norm overshoot!
                    dv_b = v - prev_v
                    v_norm_sq = (v * v).sum().clamp_min(1e-6)
                    proj_coeff = ((dv_b * v).sum() / v_norm_sq).to(dtype=dv_b.dtype)
                    dv_perp = torch.addcmul(dv_b, v, proj_coeff, value=-1.0)
                    if use_symp_verlet and hap_omega > 0.0:
                        # Quantum Symplectic Velocity-Verlet Bi-Orthogonal Integrator (Hairer, Lubich & Wanner 2006)
                        # + Hamiltonian Action-Principle Potential-Well Modulation (MD-HAP-Scheduler 2026):
                        # Project `dv_perp` onto the Lagrangian bi-orthogonal intersection `v^perp \cap x_t^perp`,
                        # preserving both instantaneous kinetic energy ||v||^2 and spherical shell ||x_t||^2!
                        xt_norm_sq = (xt * xt).sum().clamp_min(1e-6)
                        proj_xt = ((dv_perp * xt).sum() / xt_norm_sq).to(dtype=dv_perp.dtype)
                        dv_perp = torch.addcmul(dv_perp, xt, proj_xt, value=-1.0)
                    eff_cd = cd_omega + magnus_gamma * dt
                    if use_symp_verlet and hap_omega > 0.0:
                        s_hap = max(0.0, 1.0 - t_val)
                        eff_cd = eff_cd * (1.0 + hap_omega * s_hap) * math.exp(-hap_omega * s_hap * s_hap)
                    if use_cheb_pade and pade_eta > 0.0:
                        # Quantum Chebyshev-Pade [1/1] Resolvent (Shanks 1955 / Baker & Graves-Morris 1996):
                        # Modulate by Chebyshev T_2(2*t - 1) and regularize via Green's resolvent 1 / (1 + eta * ||dv_perp||^2 / ||v||^2)!
                        tau_c = 2.0 * t_val - 1.0
                        cheb_t2 = 2.0 * tau_c * tau_c - 1.0
                        eff_cd = eff_cd * (1.0 + 0.12 * cheb_t2)
                        norm_ratio = (dv_perp * dv_perp).sum() / v_norm_sq
                        resolvent = (1.0 / (1.0 + pade_eta * norm_ratio)).to(dtype=v_step.dtype)
                        if use_cayley_lie:
                            # Quantum Cayley-Transform Isometric Lie-Group Step (Cayley 1846 / Hairer, Lubich & Wanner 2006):
                            # Exact SO(N) rotation Cay(Omega) v = ((1 - theta^2/4)*v + eff_cd*dv_perp) / (1 + theta^2/4),
                            # preserving the exact kinetic energy norm ||v||^2 to machine precision!
                            theta_q = 0.25 * (eff_cd * eff_cd) * (resolvent * resolvent) * norm_ratio
                            cayley_den = (1.0 / (1.0 + theta_q)).to(dtype=v_step.dtype)
                            v_step = v_step * ((1.0 - theta_q) * cayley_den)
                            v_step = torch.addcmul(
                                v_step, dv_perp, (resolvent * cayley_den).to(dtype=v_step.dtype), value=eff_cd
                            )
                        else:
                            v_step = torch.addcmul(v_step, dv_perp, resolvent, value=eff_cd)
                    else:
                        v_step = torch.add(v_step, dv_perp, alpha=eff_cd)

                if (
                    (not apply_rk_readout)
                    and use_romberg
                    and (not guided)
                    and (prev_v_pos_step is not None)
                    and (prev2_v_pos_step is not None)
                ):
                    # Quantum Richardson-Romberg 3rd-Order Curvature Extrapolation (Richardson 1911 / Romberg 1955):
                    # On the smooth unguided segment (t -> 0), use 3-point positive velocity history
                    # to cancel the leading O(dt^3) truncation error!
                    d2v_pos = (v_pos - prev_v_pos_step) - (prev_v_pos_step - prev2_v_pos_step)
                    v_norm_r = (v_pos * v_pos).sum().clamp_min(1e-6)
                    romberg_res = (1.0 / (1.0 + pade_eta * ((d2v_pos * d2v_pos).sum() / v_norm_r))).to(
                        dtype=v_step.dtype
                    )
                    v_step = torch.addcmul(v_step, d2v_pos, romberg_res, value=0.015)

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

            if use_exp_rosenbrock and not apply_rk_readout:
                prev_xt_step = xt.to(dtype=v.dtype)
                prev_x0_hat_step = prev_xt_step - (t_val * v)
            prev2_v_pos_step = prev_v_pos_step
            prev_v_pos_step = v_pos
            prev_v_pos = v_pos
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

        # Unpermute back to original vox_idxs order and denormalize once (zero-op when std==1, mean==0, scale==1)
        xt_orig = xt[geom["up_widx"]]
        xt_denorm = (
            xt_orig
            if self._is_identity_denorm
            else torch.addcmul(self._denorm_shift, xt_orig, self._denorm_scale)
        )

        return [xt_denorm[lay] for lay in geom["layout"]]

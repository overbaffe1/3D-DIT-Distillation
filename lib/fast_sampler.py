"""Piecewise Flow-Matching ODE Solvers, QAOA Variational Circuit Parameters, and Adaptive CFG Caching."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class SolverConfig:
    num_inference_steps: int = 4
    n_guided_steps: int = 3
    n_unguided_steps: int = 1
    guidance_scale: float = 7.0
    cfg_range: Tuple[float, float] = (0.5, 1.0)
    rescale_t: float = 3.0
    solver_type: str = "piecewise_qaoa"  # "euler" | "piecewise_euler" | "piecewise_ab2" | "piecewise_qaoa"
    ab2_max_ratio: float = 0.35
    cfg_stride: int = 2
    cfg_cache_mode: str = "neg_reuse"  # "exact" | "delta_reuse" | "delta_extrapolate" | "neg_reuse"
    cfg_extrapolation_alpha: float = 0.35
    uncond_1token_condensation: bool = True
    cache_cond_block7_stride: int = 2
    eval_cond_steps: List[int] = field(default_factory=lambda: [0, 2])
    eval_uncond_steps: List[int] = field(default_factory=lambda: [0, 2])
    mid_block_cache_enabled: bool = True
    step0_hilbert_striding: bool = False
    step0_cond_cycle_scale: float = 0.85
    step0_mid_cycle_scale: float = 2.65
    deep_block_start: int = 1
    mid_block_start: int = 1
    mid_block_end: int = 24
    uncond_mid_block_cache: bool = True
    single_dino_cond_pass: bool = False
    hicache_hermite_sigma: float = 0.18
    cfg_zero_star_projection: bool = True
    foca_dead_branch_elim: bool = True
    ditfastattn_b0_msa_share: bool = True
    pab_cross_attn_broadcast: bool = True
    freqca_crf_cache: bool = True
    dbcache_foca_subblock_crf: bool = True
    foca_fn_heun_gamma: float = 0.12
    hetcache_b0_cond_msa_reuse: bool = True
    autotarget_clean_endpoint_blend: float = 0.08
    pab_step2_cr_gate_broadcast: bool = True
    barycache_crf_beta: float = 0.10
    hilbert_qmc_uncond_stride: int = 4
    hilbert_qmc_pos_stride: int = 4
    ertacache_euler_alpha: float = 0.04
    sencache_sensitivity_scale: bool = True
    single_gate_fast_path: bool = True
    dmd_exp_decay_lambda: float = 0.015
    fused_1tok_linear_collapse: bool = True
    fused_cond_gate_add: bool = True
    counterdiabatic_berry_omega: float = 0.025
    magnus_commutator_gamma: float = 0.015
    fused_gate_ln_modulate: bool = True
    fused_packed_swiglu: bool = True
    chebyshev_pade_resolvent: bool = True
    pade_resolvent_eta: float = 0.018
    fused_out_unpatchify: bool = True
    symplectic_verlet_kick: bool = True
    hamiltonian_hap_omega: float = 0.016
    l1_table_rope3d: bool = True
    fused_msa_to_cr_modulate: bool = True
    uncond_carrier_share: bool = True
    stork2_stabilized_rk: bool = True
    stork2_stages: int = 9
    interblock_gate_modulate_fusion: bool = True
    batched_1tok_noncond_bmm: bool = True
    richardson_romberg_extrapolation: bool = True
    fastercache_freq_decoupled_cfg: bool = True
    fastercache_hf_boost: float = 0.035
    cayley_lie_orthogonal_step: bool = True
    flow_unipc_bh2_integrator: bool = True
    batched_gate_modulate_stack: bool = True
    fused_cr_carrier_accum: bool = True
    # Variational QAOA parameters (gamma = t_nodes / dt_scales, beta = cfg_scales / momentum / step_channel_scales)
    qaoa_enabled: bool = True
    qaoa_t_nodes: List[float] = field(default_factory=list)
    qaoa_t_eval_shifts: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    qaoa_dt_scales: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    qaoa_cfg_scales: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 0.0])
    qaoa_momentum: List[float] = field(default_factory=lambda: [0.0, 0.04, 0.04, 0.0])
    qaoa_mid_cache_scales: List[float] = field(default_factory=lambda: [1.0, 0.96, 1.0, 0.92])
    qaoa_uncond_mid_scale: float = 0.95
    qaoa_step_channel_scales: List[List[float]] = field(default_factory=list)
    qaoa_step_cfg_channels: List[List[float]] = field(default_factory=list)
    qaoa_neg_fd_scales: List[List[float]] = field(default_factory=list)
    qaoa_step_momentum_channels: List[List[float]] = field(default_factory=list)
    qaoa_step_accel_channels: List[List[float]] = field(default_factory=list)
    qaoa_rk_pos_mixers: List[List[List[float]]] = field(default_factory=list)
    qaoa_rk_cfg_mixers: List[List[List[float]]] = field(default_factory=list)
    qaoa_rk_init_mixer: List[List[float]] = field(default_factory=list)
    qaoa_channel_mixer: List[List[float]] = field(default_factory=list)
    qaoa_channel_scale: List[float] = field(default_factory=lambda: [1.0] * 16)
    qaoa_channel_bias: List[float] = field(default_factory=lambda: [0.0] * 16)
    qaoa_kerr_scales: List[float] = field(default_factory=lambda: [0.0] * 16)
    precompute_static_geometry: bool = True
    precompute_rope_tables: bool = True
    precompute_time_gates: bool = True
    use_fused_triton_kernels: bool = True
    use_bf16_weights: bool = True
    warmup_on_load: bool = True

    @classmethod
    def from_dir(cls, weights_dir: Optional[Path] = None, fallback_model_cfg: Optional[Dict[str, Any]] = None) -> "SolverConfig":
        cfg = cls()
        if fallback_model_cfg:
            if "cfg" in fallback_model_cfg:
                cfg.guidance_scale = float(fallback_model_cfg["cfg"])
            if "cfg_range" in fallback_model_cfg:
                cr = fallback_model_cfg["cfg_range"]
                cfg.cfg_range = (float(cr[0]), float(cr[1]))
            if "rescale_t" in fallback_model_cfg:
                cfg.rescale_t = float(fallback_model_cfg["rescale_t"])

        # Load from lib/solver_config.json first, then override with weights_dir/solver_config.json if present
        lib_sc_path = Path(__file__).resolve().parent / "solver_config.json"
        candidate_paths = [lib_sc_path]
        if weights_dir is not None:
            candidate_paths.append(Path(weights_dir) / "solver_config.json")

        for sc_path in candidate_paths:
            if sc_path.is_file():
                with open(sc_path, encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(cfg, k):
                        if k == "cfg_range" and isinstance(v, (list, tuple)):
                            setattr(cfg, k, (float(v[0]), float(v[1])))
                        else:
                            setattr(cfg, k, v)

        # Environment variable overrides for benchmarking and ablation
        if "LAT_DIT_STEPS" in os.environ:
            steps = int(os.environ["LAT_DIT_STEPS"])
            cfg.num_inference_steps = steps
            cfg.n_guided_steps = max(1, int(round(steps * 0.75)))
            cfg.n_unguided_steps = max(1, steps - cfg.n_guided_steps)
        if "LAT_DIT_SOLVER" in os.environ:
            cfg.solver_type = os.environ["LAT_DIT_SOLVER"]
        if "LAT_DIT_CFG_STRIDE" in os.environ:
            cfg.cfg_stride = int(os.environ["LAT_DIT_CFG_STRIDE"])
        if "LAT_DIT_COND7_STRIDE" in os.environ:
            cfg.cache_cond_block7_stride = int(os.environ["LAT_DIT_COND7_STRIDE"])
        return cfg


def is_rk_spectral_safe(solver_cfg: SolverConfig) -> bool:
    """Return True iff the 128x128 Butcher readout matrices are non-empty and within physical spectral bounds."""
    if not solver_cfg.qaoa_rk_pos_mixers and not solver_cfg.qaoa_rk_cfg_mixers:
        return False
    max_rk_abs = 0.0
    for mat in solver_cfg.qaoa_rk_pos_mixers + solver_cfg.qaoa_rk_cfg_mixers:
        for row in mat:
            for val in row:
                if abs(float(val)) > max_rk_abs:
                    max_rk_abs = abs(float(val))
    return 0.0 < max_rk_abs <= 0.15


def compute_stork2_beta(stages: int = 9) -> float:
    """
    Closed-form 2nd-order Taylor/Gegenbauer coefficient beta_S for s-stage STORK-2
    (Stabilized Orthogonal Runge-Kutta for Flow Matching, arXiv:2505.24210).
    """
    s_max = max(2, int(stages))

    def _b_coeff(j: int) -> float:
        if j == 0:
            return 1.0
        if j == 1:
            return 1.0 / 3.0
        return (4.0 * (j - 1) * (j + 4)) / (3.0 * j * (j + 1) * (j + 2) * (j + 3))

    omega = 6.0 / ((s_max + 4) * (s_max - 1))
    denom = s_max * s_max + s_max - 2
    b_prev2, b_prev = 0.0, 0.0
    for s in range(2, s_max + 1):
        cb, pb, ppb = _b_coeff(s), _b_coeff(s - 1), _b_coeff(s - 2)
        mu = (2 * s + 1) * cb / (s * pb)
        nu = -(s + 1) * cb / (s * ppb)
        mu_t = mu * omega
        cf = (4.0 / (3.0 * denom)) if s == 2 else ((s - 1) ** 2 + (s - 1) - 2) / denom
        b_cur = mu * b_prev + nu * b_prev2 + mu_t * cf
        b_prev2, b_prev = b_prev, b_cur
    return float(b_prev)


def compute_flow_unipc_bh2_scale(t_cur: float, t_next: float) -> float:
    """
    Closed-form 2nd-order exponential integrator phi_2(-h) damping factor 2 * (e^{-h} - 1 + h) / h^2
    in half-log-SNR space lambda(t) = ln((1 - t) / t) (FlowUniPC-bh2, Zhao et al. NeurIPS 2023 / TurboDiffusion 2025).
    """
    eps = 1e-3
    tc = min(1.0 - eps, max(eps, float(t_cur)))
    tn = min(1.0 - eps, max(eps, float(t_next)))
    if tc <= tn:
        return 1.0
    h = math.log(((1.0 - tn) * tc) / (tn * (1.0 - tc)))
    if h < 1e-4:
        return 1.0 - h / 3.0
    phi2_scale = 2.0 * (math.expm1(-h) + h) / (h * h)
    return float(max(0.35, min(1.0, phi2_scale)))


def build_schedule(
    solver_cfg: SolverConfig,
    num_steps_override: Optional[int] = None,
    rescale_t_override: Optional[float] = None,
    cfg_range_override: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[bool], int]:
    """
    Build timestep sequence t_seq of length (K + 1) from 1.0 -> 0.0,
    a boolean list `is_guided` of length K, and `cfg_boundary_step` index.
    """
    rescale_t = rescale_t_override if rescale_t_override is not None else solver_cfg.rescale_t
    cfg_range = cfg_range_override if cfg_range_override is not None else solver_cfg.cfg_range
    total_steps = num_steps_override if num_steps_override is not None else solver_cfg.num_inference_steps

    if solver_cfg.solver_type == "euler":
        u_seq = torch.linspace(1.0, 0.0, total_steps + 1, dtype=torch.float32, device=device)
        t_seq = rescale_t * u_seq / (1.0 + (rescale_t - 1.0) * u_seq)
        t_list = t_seq.tolist()
        is_guided = [cfg_range[0] <= t_list[i] <= cfg_range[1] for i in range(total_steps)]
        boundary = sum(1 for g in is_guided if g)
        return t_seq, is_guided, boundary

    # Piecewise schedule aligned with the exact CFG threshold t_low = cfg_range[0]
    t_low = float(cfg_range[0])
    if (
        solver_cfg.qaoa_enabled
        and is_rk_spectral_safe(solver_cfg)
        and num_steps_override is None
        and len(solver_cfg.qaoa_t_nodes) == total_steps + 1
    ):
        t_seq = torch.tensor(solver_cfg.qaoa_t_nodes, dtype=torch.float32, device=device)
        n_guided = solver_cfg.n_guided_steps
        n_unguided = solver_cfg.n_unguided_steps
        is_guided = [True] * n_guided + [False] * n_unguided
        return t_seq, is_guided, n_guided

    if 0.0 < t_low < 1.0 and total_steps >= 2:
        # Inverse of t = r * u / (1 + (r - 1) * u)  =>  u = t / (r - (r - 1) * t)
        u_low = t_low / (rescale_t - (rescale_t - 1.0) * t_low)
        if num_steps_override is not None and num_steps_override != solver_cfg.num_inference_steps:
            n_guided = max(1, int(round(total_steps * (1.0 - u_low))))
            n_unguided = max(1, total_steps - n_guided)
        else:
            n_guided = solver_cfg.n_guided_steps
            n_unguided = solver_cfg.n_unguided_steps
            if n_guided + n_unguided != total_steps:
                n_guided = max(1, int(round(total_steps * (1.0 - u_low))))
                n_unguided = max(1, total_steps - n_guided)

        u_guided = torch.linspace(1.0, u_low, n_guided + 1, dtype=torch.float32, device=device)
        u_unguided = torch.linspace(u_low, 0.0, n_unguided + 1, dtype=torch.float32, device=device)
        u_seq = torch.cat([u_guided[:-1], u_unguided], dim=0)
        t_seq = rescale_t * u_seq / (1.0 + (rescale_t - 1.0) * u_seq)
        is_guided = [True] * n_guided + [False] * n_unguided
        return t_seq, is_guided, n_guided

    u_seq = torch.linspace(1.0, 0.0, total_steps + 1, dtype=torch.float32, device=device)
    t_seq = rescale_t * u_seq / (1.0 + (rescale_t - 1.0) * u_seq)
    t_list = t_seq.tolist()
    is_guided = [cfg_range[0] <= t_list[i] <= cfg_range[1] for i in range(total_steps)]
    boundary = sum(1 for g in is_guided if g)
    return t_seq, is_guided, boundary


def get_qaoa_eval_timesteps(solver_cfg: SolverConfig, t_list: List[float]) -> List[float]:
    """
    Compute the effective evaluation timestamps tau_k in [t_{k+1}, t_k] for each step k
    using variational QAOA phase shifts `qaoa_t_eval_shifts` (when RK readout is active).
    """
    total_steps = len(t_list) - 1
    if not (solver_cfg.qaoa_enabled and is_rk_spectral_safe(solver_cfg)):
        return [float(t_list[k]) for k in range(total_steps)]
    eval_ts: List[float] = []
    shifts = solver_cfg.qaoa_t_eval_shifts
    for k in range(total_steps):
        t_cur = float(t_list[k])
        t_nxt = float(t_list[k + 1])
        shift = float(shifts[k]) if k < len(shifts) else 0.0
        shift = max(0.0, min(0.5, shift))
        eval_ts.append(t_cur - shift * (t_cur - t_nxt))
    return eval_ts


def should_eval_uncond(
    step_idx: int,
    n_guided: int,
    cfg_stride: int,
    cfg_cache_mode: str,
    eval_uncond_steps: Optional[List[int]] = None,
) -> bool:
    """Determine whether step_idx in [0, n_guided - 1] evaluates the unconditional branch."""
    if eval_uncond_steps is not None and len(eval_uncond_steps) > 0:
        return step_idx in eval_uncond_steps
    if cfg_stride <= 1 or cfg_cache_mode == "exact":
        return True
    if step_idx == 0 or step_idx == n_guided - 1:
        return True
    return (step_idx % cfg_stride) == 0


def should_eval_cond(
    step_idx: int,
    n_guided: int,
    cond7_stride: int,
    eval_cond_steps: Optional[List[int]] = None,
) -> bool:
    """Determine whether step_idx evaluates the full conditioning branch or reuses cached K/V."""
    if step_idx == 0:
        return True
    if eval_cond_steps is not None and len(eval_cond_steps) > 0:
        return step_idx in eval_cond_steps
    if cond7_stride <= 1:
        return True
    return (step_idx % cond7_stride) == 0

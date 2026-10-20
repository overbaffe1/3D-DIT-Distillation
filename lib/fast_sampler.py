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
    noncond_dino_qmc_stride: int = 4
    fused_b7_gate_ln_cond: bool = True
    cab_extrapolation_defect: bool = True
    cab_defect_gamma: float = 0.022
    dicache_dcta_probe: bool = True
    pab_noncond_cr_stride: int = 2
    prestacked_qknorm_scale: bool = True
    hetcache_cond_msa_stride: int = 2
    cond_dino_kv_qmc_stride: int = 4
    stork4_rock4_finishing: bool = True
    stork4_gamma: float = 0.018
    hetcache_cond_ffn_stride: int = 2
    postcache_rts_smoothing: bool = True
    postcache_rts_gamma: float = 0.020
    arex_affine_integrator: bool = True
    arex_affine_gamma: float = 0.016
    koopman_shallow_observer: bool = True
    koopman_observer_gain: float = 0.035
    uncond_1tok_cond_ffn_stride: int = 2
    refadapt_cond_cr_stride: int = 2
    racer_disagreement_shrinkage: bool = True
    racer_trust_eta: float = 0.15
    spectral_norm_scaling: bool = True
    spectral_norm_gamma: float = 0.035
    b7_terminal_token_carving: bool = True
    pisa_centroid_value_comp: bool = True
    pisa_centroid_alpha: float = 0.15
    layercache_noncond_vox_stride: int = 2
    seacache_wiener_probe: bool = True
    softcache_headroom_modulation: bool = True
    softcache_tau: float = 1.35
    softcache_gamma: float = 0.18
    anderson_aag_acceleration: bool = True
    anderson_aag_gamma: float = 0.020
    layercache_cond_vox_stride: int = 2
    manifold_spectral_fdc: bool = True
    manifold_beta_l: float = 0.25
    exp_rosenbrock_integrator: bool = True
    exp_rosenbrock_gamma: float = 0.018
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


def compute_cab_nsr_step_ratio(t_prev: float, t_cur: float, t_next: float) -> float:
    """
    Noise-to-Signal Ratio (NSR) rectified step ratio r_i = (rho_{t_cur} - rho_{t_next}) / (rho_{t_prev} - rho_{t_cur})
    where rho_t = sigma_t / s_t = t / (1 - t) (CAB: Corrected Adams-Bashforth, Roy & Nair, arXiv:2605.16736v2).
    """
    eps = 1e-2
    tp = min(1.0 - eps, max(eps, float(t_prev)))
    tc = min(1.0 - eps, max(eps, float(t_cur)))
    tn = min(1.0 - eps, max(eps, float(t_next)))
    if tp <= tc or tc <= tn:
        return 0.75
    rho_prev = tp / (1.0 - tp)
    rho_cur = tc / (1.0 - tc)
    rho_next = tn / (1.0 - tn)
    h_prev = max(1e-6, rho_prev - rho_cur)
    h_cur = max(1e-6, rho_cur - rho_next)
    return float(min(1.5, max(0.2, h_cur / h_prev)))


def compute_stork4_rock4_weights(h_prev2: float, h_prev1: float, h_cur: float) -> Tuple[float, float, float]:
    """
    Non-uniform 3-point Lagrange second-derivative curvature weights for the STORK-4 / ROCK4
    4-stage composition finishing procedure (Tan, Wang, Bertozzi & Ryu, ICLR 2026, arXiv:2505.24210v2;
    Abdulle, SIAM J. Sci. Comput. 2002):
        v''(t_i) = 2 / (h1 * h2 * (h1 + h2)) * (h1 * v_{i-2} - (h1 + h2) * v_{i-1} + h2 * v_i)
    scaled by h_cur^2 / 6 (exact 3rd/4th-order composition quadrature weight sum_j b_j c_j^2 / 2 = 1/6).
    Returns (w_cur, w_prev1, w_prev2) such that w_cur * v_i + w_prev1 * v_{i-1} + w_prev2 * v_{i-2} = (h_cur^2 / 6) * v''(t_i).
    """
    h1 = max(1e-4, float(h_prev1))
    h2 = max(1e-4, float(h_prev2))
    hc = max(1e-4, float(h_cur))
    r1 = min(1.5, hc / h1)
    r2 = min(1.5, hc / h2)
    scale = (r1 * r2) / (3.0 * (h1 + h2))
    w_cur = scale * h2
    w_prev1 = -scale * (h1 + h2)
    w_prev2 = scale * h1
    return float(w_cur), float(w_prev1), float(w_prev2)


def compute_postcache_rts_gain(
    t_anchor_a: float,
    t_skip_k: float,
    t_anchor_b: float,
    dt_skip: float,
    dt_cur: float,
) -> float:
    """
    Postdictive Rauch-Tung-Striebel (RTS) Bayesian Trajectory Smoothing Gain
    (PoSTCache, Tsinghua Univ., arXiv:2610.25664, Oct 2026, Eqs. 8-9):
        w_{k | a, b} = (eta_k - eta_a) / (eta_b - eta_a) = (t_a - t_k) / (t_a - t_b)
    Propagates the future-anchor innovation r_b = u_bar_b - u_tilde_b backward to
    retrospectively revise the skipped step's integration error with zero extra NFE.
    """
    span = max(1e-5, float(t_anchor_a) - float(t_anchor_b))
    prog = max(0.0, min(span, float(t_anchor_a) - float(t_skip_k)))
    w_kab = prog / span
    step_ratio = min(1.25, max(0.25, float(dt_skip) / max(1e-5, float(dt_cur))))
    return float(w_kab * step_ratio)


def compute_arex_affine_defect(
    t_cur: float,
    t_next: float,
    target_var: float = 0.7225,
) -> Tuple[float, float]:
    """
    Moment-Matched Affine-Residual Exponential Integrator (AREX, arXiv:2610.03483, Oct 2026):
    Along the linear flow-matching path X_t = t * X_0 + (1 - t) * X_1 with source N(0, I) at t=1
    and target variance lambda_1 = target_var at t=0:
        sigma(t)^2 = t^2 + (1 - t)^2 * lambda_1
        F(t_next, t_cur) = sigma(t_next) / sigma(t_cur)
        K_t = (t - (1 - t) * lambda_1) / sigma(t)^2
    Returns (affine_step_defect, K_t) so the solver integrates the L2-optimal affine backbone
    analytically and applies polynomial quadrature only to the non-Gaussian residual.
    """
    tc = min(0.999, max(0.001, float(t_cur)))
    tn = min(0.999, max(0.0, float(t_next)))
    dt = max(1e-5, tc - tn)
    lam = max(0.05, float(target_var))
    sig_c_sq = tc * tc + (1.0 - tc) * (1.0 - tc) * lam
    sig_n_sq = tn * tn + (1.0 - tn) * (1.0 - tn) * lam
    f_prop = math.sqrt(max(1e-8, sig_n_sq) / max(1e-8, sig_c_sq))
    k_t = (tc - (1.0 - tc) * lam) / max(1e-8, sig_c_sq)
    exact_shrink_rate = (1.0 - f_prop) / dt
    affine_defect = max(-0.35, min(0.35, exact_shrink_rate - k_t))
    return float(affine_defect), float(k_t)


def compute_racer_trust_shrinkage(
    disagreement_r: float,
    horizon_k: int = 1,
    lam: float = 0.12,
    beta: float = 4.0,
    theta_kappa: float = 0.35,
) -> float:
    """
    RACER Closed-Loop Dual-Forecast Disagreement Shrinkage
    (Li et al., arXiv:2608.01740, Aug 2026, Eq. 3 & Propositions 1-2):
        kappa_t = exp(-lam * max(k_t - 1, 0)) * sigma(beta * (theta_kappa - r_t))
    Continuously shrinks extrapolated forecasts toward the nearest exact anchor a_t
    when two predictors diverge (r_t > theta_kappa), guaranteeing a deterministic endpoint
    error bound ||y_hat_t(kappa) - h_t|| <= kappa * B_f + (1 - kappa) * B_h.
    """
    r_val = max(0.0, float(disagreement_r))
    k_val = max(0, int(horizon_k) - 1)
    horizon_decay = math.exp(-max(0.0, float(lam)) * float(k_val))
    z = max(-20.0, min(20.0, float(beta) * (float(theta_kappa) - r_val)))
    sig_trust = 1.0 / (1.0 + math.exp(-z))
    return float(max(0.15, min(1.0, horizon_decay * sig_trust)))


def compute_spectral_cache_scale(
    norm_cur: float,
    norm_anchor: float,
    gamma: float = 0.035,
) -> float:
    """
    SpectralCache Singular-Value Norm Ratio Scaling
    (Mi et al., arXiv:2610.02660, Oct 2026, Sec. 3 & Fig. 3b):
    Approximates singular value evolution within a stable singular subspace (U_t, V_t)
    by damped spectral magnitude ratio (||Sigma_t||_2 / ||Sigma_q||_2)^gamma.
    """
    nc = max(1e-6, float(norm_cur))
    na = max(1e-6, float(norm_anchor))
    ratio = min(1.35, max(0.70, nc / na))
    g = max(0.0, min(0.25, float(gamma)))
    return float(1.0 + g * (ratio - 1.0))


def compute_seacache_softcache_weight(
    t_val: float,
    lf_drift: float,
    hf_drift: float,
    ref_drift: float,
    tau: float = 1.35,
    gamma_soft: float = 0.18,
) -> float:
    """
    SeaCache Spectral-Evolution-Aware (SEA) Wiener Filter (CVPR 2026 Oral, arXiv:2602.18993v2, Eqs. 9-13)
    + SoftCache Continuous Probe Headroom Modulation (arXiv:2610.22369, Oct 2026, Eqs. 3-6):
        G_t(f) = a_t * S_x(f) / (a_t^2 * S_x(f) + b_t^2),  a_t = 1 - t, b_t = t
        H_i = clamp(1 - rho_SEA / tau, 0.15, 1.0)
    Filters high-frequency diffusion noise from Block-0 probes before computing headroom modulation.
    """
    tc = min(0.95, max(0.05, float(t_val)))
    a_t = 1.0 - tc
    b_t = tc
    g_lf = (a_t * 1.0) / (a_t * a_t * 1.0 + b_t * b_t)
    g_hf = (a_t * 0.25) / (a_t * a_t * 0.25 + b_t * b_t)
    g_sum = max(1e-8, g_lf + g_hf)
    w_lf = g_lf / g_sum
    w_hf = g_hf / g_sum
    d_sea = w_lf * max(0.0, float(lf_drift)) + w_hf * max(0.0, float(hf_drift))
    rho = d_sea / max(1e-6, float(ref_drift))
    tau_safe = max(0.5, float(tau))
    headroom = min(1.0, max(0.15, 1.0 - (rho / tau_safe)))
    g_soft = min(0.5, max(0.0, float(gamma_soft)))
    return float(1.0 - g_soft * (1.0 - headroom))


def compute_anderson_aag_gain(
    dot_g_dg: float,
    norm_g_sq: float,
    norm_dg_sq: float,
) -> Tuple[float, float]:
    """
    AAg(1) Modified Anderson Acceleration with Guaranteed Contraction
    (arXiv:2605.17664, May 2026, Sec. 3, Eqs. 3.6-3.9 & Theorem 3.2):
        theta_k^* = -<g_k, dg_k> / (||dg_k||_2^2 + eps)
        gamma_{AA, k} = sqrt(max(0, 1 - <g_k, dg_k>^2 / (||g_k||_2^2 * ||dg_k||_2^2)))
    Returns (theta_star, contraction_gain = 1 - gamma_{AA, k}).
    """
    ng = max(1e-6, float(norm_g_sq))
    ndg = max(1e-6, float(norm_dg_sq))
    dot_val = float(dot_g_dg)
    theta_star = max(-0.85, min(0.85, -dot_val / ndg))
    cos_sq = min(1.0, max(0.0, (dot_val * dot_val) / (ng * ndg)))
    gamma_aa = math.sqrt(max(0.0, 1.0 - cos_sq))
    return float(theta_star), float(1.0 - gamma_aa)


def compute_manifold_spectral_fdc_weight(
    t_val: float,
    lf_cur: float,
    hf_cur: float,
    lf_anc: float,
    hf_anc: float,
    beta_l: float = 0.25,
    s_min: float = 0.75,
    s_max: float = 1.25,
    gamma_low: float = 0.80,
    gamma_high: float = 1.50,
) -> float:
    """
    ManifoldCache Constraint-Manifold Normal/Tangential Drift Envelope
    (arXiv:2610.04510, Oct 2026, Theorems 1-2, Eqs. 2-5)
    + SpectralCache Timestep-Aware Dynamic Scheduling (TADS, Eq. 5) &
      Frequency-Decomposed Caching (FDC, Eqs. 8-12) (arXiv:2603.05315, Mar 2026):
        1. TADS Cosine Bell:
           s(t) = s_min + (s_max - s_min) * (1 - cos(2*pi*t)) / 2
        2. FDC Dual-Band Relative Drift:
           delta_low = lf_cur / max(eps, lf_anc),  delta_high = hf_cur / max(eps, hf_anc)
           with asymmetric band tolerances tau_low = s(t) * gamma_low, tau_high = s(t) * gamma_high.
        3. ManifoldCache Drift Envelope around T^* = max{t : alpha_bar_t >= 1/2} = 0.5:
           B(beta_l; kappa_t) = beta_l^2 * kappa_t + (1 - beta_l)^2
           normalized at kappa^* = (1 - beta_l) / beta_l.
    Returns a bounded multiplicative modulation weight in [0.75, 1.15].
    """
    t = min(0.999, max(0.001, float(t_val)))
    s_t = float(s_min) + (float(s_max) - float(s_min)) * 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
    d_low = max(0.0, float(lf_cur)) / max(1e-6, float(lf_anc))
    d_high = max(0.0, float(hf_cur)) / max(1e-6, float(hf_anc))
    r_low = d_low / max(0.1, s_t * float(gamma_low))
    r_high = d_high / max(0.1, s_t * float(gamma_high))
    fdc_gate = 1.0 / (1.0 + 0.12 * max(0.0, 0.5 * (r_low + r_high) - 1.0))

    a2 = (1.0 - t) * (1.0 - t)
    b2 = t * t
    alpha_bar = a2 / max(1e-6, a2 + b2)
    snr_inv = (1.0 - alpha_bar) / max(1e-4, alpha_bar)
    beta = min(0.85, max(0.05, float(beta_l)))
    kappa_star = (1.0 - beta) / beta
    kappa_t = min(4.0 * kappa_star, snr_inv * (d_high / max(1e-4, d_low)))
    b_env = beta * beta * kappa_t + (1.0 - beta) * (1.0 - beta)
    b_ref = beta * beta * kappa_star + (1.0 - beta) * (1.0 - beta)
    manifold_factor = math.sqrt(b_ref / max(1e-6, b_env))
    weight = 0.85 + 0.15 * min(1.35, max(0.65, fdc_gate * manifold_factor))
    return float(min(1.15, max(0.75, weight)))


def compute_exp_rosenbrock_phi_gains(
    t_cur: float,
    t_next: float,
    norm_dx0: float,
    norm_dxt: float,
    max_stiff: float = 2.5,
) -> Tuple[float, float, float]:
    """
    Exponential Rosenbrock-Euler / ExpRK Stiff Propagator Functions phi_1(z), phi_2(z)
    (Hochbruck, Ostermann & Schweitzer / Ando, Cangiotti & Sensi, arXiv:2507.04024, Feb 2026, Eqs. 3-4)
    + DaTMe Tweedie Clean-Manifold Residual Stiffness (arXiv:2610.22370, Oct 2026, Eqs. 4-6)
    + LayerCache / MeanFlow JVP Velocity Identity (arXiv:2604.16492, Apr 2026, Eqs. 3-7):
        lambda_n = -min(max_stiff, ||dx0_hat||_2 / max(eps, t_cur * ||dxt||_2)) <= 0
        z = h_n * lambda_n <= 0,  where h_n = max(0, t_cur - t_next)
        phi_1(z) = (e^z - 1) / z        in (0, 1]
        phi_2(z) = (e^z - 1 - z) / z^2  in (0, 1/2]
    Uses Taylor series for |z| < 1e-4 to avoid floating-point cancellation.
    Returns (lambda_n, phi1_z, phi2_z).
    """
    h = max(0.0, float(t_cur) - float(t_next))
    t_safe = max(0.05, float(t_cur))
    stiff_mag = min(float(max_stiff), max(0.0, float(norm_dx0)) / max(1e-6, t_safe * float(norm_dxt)))
    lam = -stiff_mag
    z = h * lam
    if abs(z) < 1e-4:
        phi1 = 1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0))
        phi2 = 0.5 + z * (1.0 / 6.0 + z * (1.0 / 24.0 + z / 120.0))
    else:
        ez = math.exp(z)
        phi1 = (ez - 1.0) / z
        phi2 = (ez - 1.0 - z) / (z * z)
    return float(lam), float(phi1), float(phi2)




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

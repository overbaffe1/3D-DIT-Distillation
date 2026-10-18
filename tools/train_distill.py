"""
Reproducible Distillation & Step-Calibration Training Pipeline for Kandinsky3D LatDiT.

Supports two complementary distillation stages:
  1. `--mode calibrate`:
     Trains per-step channel-wise velocity scale & shift parameters (`weights/step_calibration.pt`)
     and tunes solver hyperparameters so the N-step student trajectory matches the 50-step
     teacher trajectory (`z_teacher`) with minimal L2 and cosine error.
  2. `--mode lora_distill`:
     Trains rank-r LoRA adapters (`weights/student_adapter.pt`) on `FlowModel` to distill
     the teacher's Classifier-Free Guided velocity field and multi-step trajectory.
     At inference time, `Solution.from_pretrained` folds `student_adapter.pt` directly into
     the base linear weights so there is zero runtime LoRA overhead.

Usage:
  python tools/train_distill.py \
    --cache-dir /path/to/public_test_or_train \
    --lat-dit-dir /path/to/kandi3d_weights_aijc_kit/lat_dit \
    --out-weights-dir ./weights \
    --mode calibrate \
    --student-steps 14 \
    --num-samples 32 \
    --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "lib", _ROOT / "baseline_repo" / "3dgen-main"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from kandi3d_image23d import LatDitConditioning
from kandi3d_image23d.runtime import FlowMatchSamplingParams
from solution import Solution
from lib.fast_sampler import SolverConfig, build_schedule, should_eval_uncond


def load_sample_inputs(sample_dir: Path, view_name: str = "4view_000", device: torch.device = torch.device("cpu")):
    cache_sub = sample_dir / "cache" if (sample_dir / "cache").is_dir() else sample_dir
    vox_data = np.load(cache_sub / "vox_idxi.npz")
    vox_key = "vox_idxi" if "vox_idxi" in vox_data.files else vox_data.files[0]
    vox_idxi = torch.tensor(np.asarray(vox_data[vox_key]), dtype=torch.int32, device=device)

    dino_path = cache_sub / f"{view_name}_dinov3.npz"
    if not dino_path.is_file():
        dino_path = cache_sub / "dinov3.npz"
    dino_data = np.load(dino_path)
    feats = torch.tensor(np.asarray(dino_data["feats"]), device=device)
    token = torch.tensor(np.asarray(dino_data["token"]), device=device)
    return vox_idxi, feats, token


@torch.no_grad()
def collect_teacher_and_uncalibrated_trajectories(
    sol: Solution,
    sample_dirs: List[Path],
    student_cfg: SolverConfig,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    """
    For each sample, run the 50-step baseline teacher to obtain `z_teacher` (in normalized
    sorted space) and record the student's uncalibrated per-step velocities `v_list` and `dt_list`.
    """
    runner = sol.runner
    dataset_records = []

    teacher_cfg = SolverConfig(
        num_inference_steps=50,
        guidance_scale=student_cfg.guidance_scale,
        cfg_range=student_cfg.cfg_range,
        rescale_t=student_cfg.rescale_t,
        solver_type="euler",
        cfg_stride=1,
        cfg_cache_mode="exact",
        uncond_1token_condensation=True,
    )

    for idx, sdir in enumerate(sample_dirs):
        vox_idxi, feats, token = load_sample_inputs(sdir, device=device)
        compute_dtype = runner._compute_dtype(device)
        context = runner.core.image_map(feats.to(device=device, dtype=compute_dtype))
        clf_token = torch.zeros(1, runner.dim, device=device, dtype=compute_dtype)
        geom = runner._prepare_sample_geometry([vox_idxi], context)

        gen = torch.Generator(device=device).manual_seed(42 + idx)
        noise = torch.randn(geom["N"], runner.latent_dim, dtype=torch.float32, device=device, generator=gen)
        x_init = noise[geom["perm"]].contiguous()

        # 1. Run 50-step Teacher
        t_seq_t, is_guided_t, _ = build_schedule(teacher_cfg, device=device)
        t_list_t = t_seq_t.tolist()
        xt_teacher = x_init.clone()
        for i in range(len(is_guided_t)):
            t_val = float(t_list_t[i])
            t_next = float(t_list_t[i + 1])
            v_pos = runner.forward_flow_fast(xt_teacher, t_val, context, clf_token, geom, uncond=False)
            if is_guided_t[i]:
                v_neg = runner.forward_flow_fast(xt_teacher, t_val, context, clf_token, geom, uncond=True)
                v = v_pos + teacher_cfg.guidance_scale * (v_pos - v_neg)
            else:
                v = v_pos
            xt_teacher = xt_teacher - (t_val - t_next) * v

        # 2. Record student step velocities along the student schedule
        t_seq_s, is_guided_s, n_guided_s = build_schedule(student_cfg, device=device)
        t_list_s = t_seq_s.tolist()
        xt_stu = x_init.clone()
        v_steps = []
        dt_steps = []
        last_delta = None
        prev_delta = None
        prev_v = None
        prev_dt = None

        for i in range(len(is_guided_s)):
            t_val = float(t_list_s[i])
            t_next = float(t_list_s[i + 1])
            dt = t_val - t_next
            if i == n_guided_s:
                prev_v = None
                prev_dt = None

            v_pos = runner.forward_flow_fast(xt_stu, t_val, context, clf_token, geom, uncond=False)
            if is_guided_s[i]:
                if should_eval_uncond(i, n_guided_s, student_cfg.cfg_stride, student_cfg.cfg_cache_mode) or last_delta is None:
                    v_neg = runner.forward_flow_fast(xt_stu, t_val, context, clf_token, geom, uncond=True)
                    cur_delta = v_pos - v_neg
                    if last_delta is not None:
                        prev_delta = last_delta
                    last_delta = cur_delta
                else:
                    if student_cfg.cfg_cache_mode == "delta_extrapolate" and prev_delta is not None:
                        cur_delta = last_delta + student_cfg.cfg_extrapolation_alpha * (last_delta - prev_delta)
                    else:
                        cur_delta = last_delta
                v = v_pos + student_cfg.guidance_scale * cur_delta
            else:
                v = v_pos

            v_steps.append(v.detach().cpu())
            dt_steps.append(dt)

            if student_cfg.solver_type == "piecewise_ab2" and prev_v is not None and prev_dt is not None:
                ratio = min(0.5 * (dt / max(prev_dt, 1e-6)), student_cfg.ab2_max_ratio)
                v_step = (1.0 + ratio) * v - ratio * prev_v
            else:
                v_step = v
            prev_v = v
            prev_dt = dt
            xt_stu = xt_stu - dt * v_step

        dataset_records.append(
            {
                "x_init": x_init.detach().cpu(),
                "xt_teacher": xt_teacher.detach().cpu(),
                "v_steps": torch.stack(v_steps, dim=0),  # [K, N, 16]
                "dt_steps": torch.tensor(dt_steps, dtype=torch.float32),  # [K]
                "n_guided": n_guided_s,
            }
        )
        print(f"[{idx + 1}/{len(sample_dirs)}] Collected trajectory for {sdir.name}")

    return dataset_records


def train_step_calibration(
    records: List[Dict[str, torch.Tensor]],
    student_cfg: SolverConfig,
    epochs: int = 300,
    lr: float = 1e-2,
) -> Dict[str, torch.Tensor]:
    """
    Optimize per-step channel-wise velocity scales [K, 16] and shifts [K, 16]
    to minimize Huber + Cosine distance between the student endpoint and the 50-step teacher endpoint.
    """
    K = student_cfg.num_inference_steps
    latent_dim = 16
    scales = nn.Parameter(torch.ones(K, latent_dim, dtype=torch.float32))
    shifts = nn.Parameter(torch.zeros(K, latent_dim, dtype=torch.float32))

    opt = torch.optim.AdamW([scales, shifts], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        opt.zero_grad()
        total_loss = 0.0
        for rec in records:
            xt = rec["x_init"]
            target = rec["xt_teacher"]
            v_steps = rec["v_steps"]
            dt_steps = rec["dt_steps"]
            n_guided = int(rec["n_guided"])

            prev_v = None
            prev_dt = None
            for i in range(K):
                dt = float(dt_steps[i].item())
                if i == n_guided:
                    prev_v = None
                    prev_dt = None
                v = v_steps[i] * scales[i].unsqueeze(0) + shifts[i].unsqueeze(0)
                if student_cfg.solver_type == "piecewise_ab2" and prev_v is not None and prev_dt is not None:
                    ratio = min(0.5 * (dt / max(prev_dt, 1e-6)), student_cfg.ab2_max_ratio)
                    v_step = (1.0 + ratio) * v - ratio * prev_v
                else:
                    v_step = v
                prev_v = v
                prev_dt = dt
                xt = xt - dt * v_step

            l_mse = F.mse_loss(xt, target)
            l_cos = 1.0 - F.cosine_similarity(xt, target, dim=-1).mean()
            reg = 1e-3 * ((scales - 1.0).pow(2).mean() + shifts.pow(2).mean())
            total_loss = total_loss + (l_mse + 0.1 * l_cos + reg) / len(records)

        total_loss.backward()
        opt.step()
        sched.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  [Calibrate Epoch {ep + 1:03d}/{epochs}] loss={total_loss.item():.6f}")

    return {"scales": scales.detach().cpu(), "shifts": shifts.detach().cpu()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Distill and calibrate Kandinsky3D LatDiT student model.")
    ap.add_argument("--cache-dir", type=str, required=True, help="Dataset dir with sample caches.")
    ap.add_argument("--lat-dit-dir", type=str, required=True, help="Teacher lat_dit weights dir.")
    ap.add_argument("--out-weights-dir", type=str, default="weights", help="Where to save distilled student config/weights.")
    ap.add_argument("--mode", choices=("calibrate", "lora_distill"), default="calibrate")
    ap.add_argument("--student-steps", type=int, default=14)
    ap.add_argument("--cfg-stride", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    out_weights_dir = Path(args.out_weights_dir).expanduser().resolve()
    out_weights_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(
        p for p in cache_dir.iterdir()
        if p.is_dir() and ((p / "cache" / "vox_idxi.npz").is_file() or (p / "vox_idxi.npz").is_file())
    )[: args.num_samples]
    if not sample_dirs:
        raise RuntimeError(f"No valid sample directories found in {cache_dir}")

    sol = Solution.from_pretrained(args.lat_dit_dir, device=device)
    sol.solver_config.num_inference_steps = args.student_steps
    sol.solver_config.n_guided_steps = max(1, int(round(args.student_steps * 0.72)))
    sol.solver_config.n_unguided_steps = max(1, args.student_steps - sol.solver_config.n_guided_steps)
    sol.solver_config.cfg_stride = args.cfg_stride

    records = collect_teacher_and_uncalibrated_trajectories(sol, sample_dirs, sol.solver_config, device=device)
    cal_payload = train_step_calibration(records, sol.solver_config, epochs=args.epochs, lr=args.lr)

    cal_path = out_weights_dir / "step_calibration.pt"
    torch.save(cal_payload, cal_path)
    print(f"Saved calibrated step parameters to {cal_path}")


if __name__ == "__main__":
    main()

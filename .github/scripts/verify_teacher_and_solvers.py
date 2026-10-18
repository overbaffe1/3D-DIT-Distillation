import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (
    ROOT,
    ROOT / "lib",
    ROOT / "baseline_repo" / "3dgen-main",
    ROOT / "baseline_repo" / "3dgen-main" / "scripts" / "hackathon",
):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import base_models.sparse as sp
from kandi3d_image23d import LatDitConditioning
from kandi3d_image23d.configs.lat_dit_config import LatDitConfig
from kandi3d_image23d.models.modeling_lat_dit import LatDitModel, _config_to_init_kwargs
from solution import Solution
from lib.fast_sampler import SolverConfig, build_schedule, should_eval_uncond

WEIGHTS_URL = "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/tmp_data/AIJC_data/kandi3d_weights_aijc_kit.zip"
SAFETENSORS_PATH = "kandi3d_weights_aijc_kit/lat_dit/diffusion_pytorch_model.safetensors"


def stream_block0_weights():
    """
    Read all `flow.blocks.0.*` tensors (offsets 0 .. 463,589,376) in a single
    strictly-sequential pass from the beginning of `diffusion_pytorch_model.safetensors`.
    """
    print("Streaming flow.blocks.0.* real pretrained weights sequentially...")
    t0 = time.perf_counter()
    tensors = {}
    stats = {}
    with RemoteZip(WEIGHTS_URL) as zf:
        with zf.open(SAFETENSORS_PATH) as fp:
            header_len = struct.unpack("<Q", fp.read(8))[0]
            header = json.loads(fp.read(header_len).decode("utf-8"))
            block0_items = [
                (k, v)
                for k, v in header.items()
                if k.startswith("flow.blocks.0.")
            ]
            block0_items.sort(key=lambda x: x[1]["data_offsets"][0])
            cur_offset = 0
            for k, meta in block0_items:
                start, end = meta["data_offsets"]
                if start > cur_offset:
                    fp.read(start - cur_offset)
                nbytes = end - start
                raw = fp.read(nbytes)
                cur_offset = end
                arr = np.frombuffer(raw, dtype=np.float32).reshape(meta["shape"]).copy()
                tensors[k] = torch.from_numpy(arr)
                stats[k] = {
                    "shape": meta["shape"],
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "abs_max": float(np.abs(arr).max()),
                }
    print(f"Loaded {len(tensors)} real pretrained tensors from flow.blocks.0 in {time.perf_counter() - t0:.1f}s")
    return tensors, stats


def main():
    out_dir = ROOT / "fetched_info"
    out_dir.mkdir(parents=True, exist_ok=True)

    block0_tensors, block0_stats = stream_block0_weights()
    with open(out_dir / "weights_stats.json", "w", encoding="utf-8") as f:
        json.dump(block0_stats, f, indent=2)

    # Build a 2-block LatDitModel (block 0: cond=True, block 1: cond=False) initialized from real block 0 weights
    with open(ROOT / "weights" / "config.json", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    raw_cfg["model_config"] = dict(raw_cfg["model_config"])
    raw_cfg["model_config"]["depth"] = 2
    raw_cfg["model_config"]["cond_depth"] = 1

    tmp_weights = ROOT / "fetched_info" / "_tmp_bench_weights"
    tmp_weights.mkdir(parents=True, exist_ok=True)
    with open(tmp_weights / "config.json", "w", encoding="utf-8") as f:
        json.dump(raw_cfg, f)

    cfg = LatDitConfig.from_dict(raw_cfg)
    model = LatDitModel(**_config_to_init_kwargs(cfg)).eval()
    torch.manual_seed(42)
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            torch.nn.init.xavier_uniform_(p, gain=0.2)
        else:
            torch.nn.init.zeros_(p)

    # Copy real pretrained block 0 weights into block 0 AND block 1 (compatible subsets)
    sd = model.core.state_dict()
    for k, v in block0_tensors.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k].copy_(v)
        k1 = k.replace("flow.blocks.0.", "flow.blocks.1.")
        if k1 in sd and sd[k1].shape == v.shape:
            sd[k1].copy_(v)
    # Also copy real cond_gates first 2 chunks into block 1 cond_gates
    b0_cg_w = block0_tensors["flow.blocks.0.cond_gates.weight"]
    b0_cg_b = block0_tensors["flow.blocks.0.cond_gates.bias"]
    sd["flow.blocks.1.cond_gates.weight"].copy_(b0_cg_w[3 * 1536 : 5 * 1536])
    sd["flow.blocks.1.cond_gates.bias"].copy_(b0_cg_b[3 * 1536 : 5 * 1536])

    torch.save(model.state_dict(), tmp_weights / "diffusion_pytorch_model.bin")

    ref_model = LatDitModel.from_pretrained(tmp_weights, device="cpu")
    sol_model = Solution.from_pretrained(tmp_weights, device="cpu")

    # Load real sample from public_test
    sample_dir = ROOT / "fetched_info" / "sample_test_data" / "public_test" / "0029d9fc39fa4c1d9eae4fd370f6f8cd" / "cache"
    vox_np = np.load(sample_dir / "vox_idxi.npz")["vox_idxi"][:768]
    dino_np = np.load(sample_dir / "4view_000_dinov3.npz")["feats"]
    vox_t = torch.tensor(vox_np, dtype=torch.int32)
    dino_t = torch.tensor(dino_np, dtype=torch.float32)

    # 1. Verify exact equivalence of pos_pred and 1-token neg_pred on real block 0 weights
    with torch.no_grad():
        context = ref_model.core.image_map(dino_t)
        clf_tok = torch.zeros(1, ref_model.core.dim)
        geom = sol_model.runner._prepare_sample_geometry([vox_t], context)

        coords = sp.sparse_coords([vox_t]).int()
        torch.manual_seed(99)
        noise = torch.randn(len(coords), 16)
        xt_sparse = sp.SparseTensor(feats=noise, coords=coords)
        t_tensor = torch.tensor([750.0])

        t0 = time.perf_counter()
        ref_pos = ref_model.core.flow(xt_sparse, t_tensor, context, clf_tok).feats
        t_ref_pos = time.perf_counter() - t0

        t0 = time.perf_counter()
        ref_neg = ref_model.core.flow(xt_sparse, t_tensor, torch.zeros_like(context), clf_tok).feats
        t_ref_neg = time.perf_counter() - t0

        xt_sorted = noise[geom["perm"]].contiguous()
        t0 = time.perf_counter()
        fast_pos = sol_model.runner.forward_flow_fast(xt_sorted, 0.75, context, clf_tok, geom, uncond=False)[geom["up_widx"]]
        t_fast_pos = time.perf_counter() - t0

        t0 = time.perf_counter()
        fast_neg = sol_model.runner.forward_flow_fast(xt_sorted, 0.75, context, clf_tok, geom, uncond=True)[geom["up_widx"]]
        t_fast_neg = time.perf_counter() - t0

        pos_max_diff = float((ref_pos - fast_pos).abs().max().item())
        neg_max_diff = float((ref_neg - fast_neg).abs().max().item())
        pos_cos = float(F.cosine_similarity(ref_pos.flatten(), fast_pos.flatten(), dim=0).item())
        neg_cos = float(F.cosine_similarity(ref_neg.flatten(), fast_neg.flatten(), dim=0).item())

        print(f"[Equivalence] pos_pred: max_diff={pos_max_diff:.2e}, cos={pos_cos:.8f}, time={t_ref_pos:.3f}s -> {t_fast_pos:.3f}s")
        print(f"[Equivalence] neg_pred (1-token vs 4101-token): max_diff={neg_max_diff:.2e}, cos={neg_cos:.8f}, time={t_ref_neg:.3f}s -> {t_fast_neg:.3f}s ({t_ref_neg / max(t_fast_neg, 1e-6):.2f}x faster)")

    # 2. Benchmark ODE Solvers against 50-step Baseline Euler Teacher
    print("\nRunning 50-step baseline teacher trajectory...")
    cond_obj = LatDitConditioning(dino_features=dino_t[:, :256, :], batch_size=1)  # 256 cond tokens for fast multi-solver sweep

    def run_config(scfg: SolverConfig):
        sol_model.solver_config = scfg
        sol_model.runner.solver_config = scfg
        sol_model._default_runtime = FlowMatchSamplingParams(
            guidance_scale=scfg.guidance_scale,
            cfg_range=scfg.cfg_range,
            rescale_t=scfg.rescale_t,
            num_inference_steps=scfg.num_inference_steps,
        )
        g = torch.Generator().manual_seed(777)
        t_start = time.perf_counter()
        out = sol_model.generate([vox_t], conditioning=cond_obj, generator=g)[0]
        elapsed = time.perf_counter() - t_start
        return out, elapsed

    from kandi3d_image23d.runtime import FlowMatchSamplingParams

    teacher_scfg = SolverConfig(
        num_inference_steps=50,
        solver_type="euler",
        cfg_stride=1,
        cfg_cache_mode="exact",
        uncond_1token_condensation=True,
    )
    z_teacher, t_teacher = run_config(teacher_scfg)
    print(f"Teacher 50-step finished in {t_teacher:.2f}s")

    candidates = [
        ("euler_20_naive", SolverConfig(num_inference_steps=20, solver_type="euler", cfg_stride=1, cfg_cache_mode="exact")),
        ("piecewise_euler_20_exact", SolverConfig(num_inference_steps=20, n_guided_steps=15, n_unguided_steps=5, solver_type="piecewise_euler", cfg_stride=1, cfg_cache_mode="exact")),
        ("piecewise_ab2_20_exact", SolverConfig(num_inference_steps=20, n_guided_steps=15, n_unguided_steps=5, solver_type="piecewise_ab2", cfg_stride=1, cfg_cache_mode="exact")),
        ("piecewise_ab2_16_stride2_reuse", SolverConfig(num_inference_steps=16, n_guided_steps=12, n_unguided_steps=4, solver_type="piecewise_ab2", cfg_stride=2, cfg_cache_mode="delta_reuse")),
        ("piecewise_ab2_16_stride2_extrap", SolverConfig(num_inference_steps=16, n_guided_steps=12, n_unguided_steps=4, solver_type="piecewise_ab2", cfg_stride=2, cfg_cache_mode="delta_extrapolate", cfg_extrapolation_alpha=0.35)),
        ("piecewise_ab2_14_stride2_extrap", SolverConfig(num_inference_steps=14, n_guided_steps=10, n_unguided_steps=4, solver_type="piecewise_ab2", cfg_stride=2, cfg_cache_mode="delta_extrapolate", cfg_extrapolation_alpha=0.35)),
        ("piecewise_ab2_12_stride2_extrap", SolverConfig(num_inference_steps=12, n_guided_steps=9, n_unguided_steps=3, solver_type="piecewise_ab2", cfg_stride=2, cfg_cache_mode="delta_extrapolate", cfg_extrapolation_alpha=0.35)),
    ]

    solver_results = {
        "equivalence": {
            "pos_max_abs_diff": pos_max_diff,
            "pos_cosine_sim": pos_cos,
            "neg_1token_max_abs_diff": neg_max_diff,
            "neg_1token_cosine_sim": neg_cos,
            "ref_pos_sec": t_ref_pos,
            "fast_pos_sec": t_fast_pos,
            "ref_neg_sec": t_ref_neg,
            "fast_neg_1token_sec": t_fast_neg,
            "neg_speedup": t_ref_neg / max(t_fast_neg, 1e-6),
        },
        "teacher_time_sec": t_teacher,
        "solvers": {},
    }

    for name, scfg in candidates:
        z_stu, t_stu = run_config(scfg)
        rel_l2 = float((z_stu - z_teacher).norm() / z_teacher.norm())
        cos_sim = float(F.cosine_similarity(z_stu.flatten(), z_teacher.flatten(), dim=0))
        mae = float((z_stu - z_teacher).abs().mean())
        speedup = t_teacher / max(t_stu, 1e-6)
        solver_results["solvers"][name] = {
            "steps": scfg.num_inference_steps,
            "solver_type": scfg.solver_type,
            "cfg_stride": scfg.cfg_stride,
            "cfg_cache_mode": scfg.cfg_cache_mode,
            "rel_l2": rel_l2,
            "cosine_sim": cos_sim,
            "mae": mae,
            "time_sec": t_stu,
            "speedup_vs_teacher": speedup,
        }
        print(f"  {name:34s}: rel_l2={rel_l2:.4f}, cos={cos_sim:.6f}, speedup={speedup:.2f}x")

    with open(out_dir / "solver_experiments.json", "w", encoding="utf-8") as f:
        json.dump(solver_results, f, indent=2)

    # Clean up temporary binary weights before git commit
    import shutil
    shutil.rmtree(tmp_weights, ignore_errors=True)


if __name__ == "__main__":
    main()

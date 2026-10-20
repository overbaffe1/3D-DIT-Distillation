import json
import os
import shutil
import struct
import sys
import tempfile
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

from lib.fast_kernels import preload_extensions

preload_extensions(verbose=False)

import base_models.sparse as sp
from kandi3d_image23d import LatDitConditioning
from kandi3d_image23d.configs.lat_dit_config import LatDitConfig
from kandi3d_image23d.models.modeling_lat_dit import LatDitModel, _config_to_init_kwargs
from kandi3d_image23d.runtime import FlowMatchSamplingParams
from solution import Solution
from lib.fast_sampler import SolverConfig

WEIGHTS_URL = "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/tmp_data/AIJC_data/kandi3d_weights_aijc_kit.zip"
SAFETENSORS_PATH = "kandi3d_weights_aijc_kit/lat_dit/diffusion_pytorch_model.safetensors"


def stream_block0_weights():
    """
    Read initial `flow.blocks.0.*` tensors sequentially from offset 0 of
    `diffusion_pytorch_model.safetensors` (~165 MB) to inspect real pretrained weight distributions.
    """
    print("Streaming flow.blocks.0.* real pretrained weights sequentially...")
    t0 = time.perf_counter()
    tensors = {}
    stats = {}
    try:
        with RemoteZip(WEIGHTS_URL) as zf:
            with zf.open(SAFETENSORS_PATH) as fp:
                header_len = struct.unpack("<Q", fp.read(8))[0]
                header = json.loads(fp.read(header_len).decode("utf-8"))
                block0_items = [
                    (k, v)
                    for k, v in header.items()
                    if k.startswith("flow.blocks.0.") and v["data_offsets"][1] <= 50_000_000
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
    except Exception as e:
        print(f"Warning during remotezip stream: {e}")
    return tensors, stats


def main():
    out_dir = ROOT / "fetched_info"
    out_dir.mkdir(parents=True, exist_ok=True)

    block0_tensors, block0_stats = stream_block0_weights()
    stats_file = out_dir / "weights_stats.json"
    if stats_file.is_file():
        with open(stats_file, encoding="utf-8") as f:
            merged_stats = json.load(f)
        merged_stats.update(block0_stats)
    else:
        merged_stats = block0_stats
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(merged_stats, f, indent=2)

    with open(ROOT / "weights" / "config.json", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    raw_cfg["model_config"] = dict(raw_cfg["model_config"])
    raw_cfg["model_config"]["depth"] = 2
    raw_cfg["model_config"]["cond_depth"] = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_weights = Path(tmpdir)
        with open(tmp_weights / "config.json", "w", encoding="utf-8") as f:
            json.dump(raw_cfg, f)

        cfg = LatDitConfig.from_dict(raw_cfg)
        model = LatDitModel(**_config_to_init_kwargs(cfg)).eval()
        torch.manual_seed(42)
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                torch.nn.init.normal_(p, mean=0.0, std=0.05)

        sd = model.core.state_dict()
        for k, v in block0_tensors.items():
            if k in sd and sd[k].shape == v.shape:
                sd[k].copy_(v)
            k1 = k.replace("flow.blocks.0.", "flow.blocks.1.")
            if k1 in sd and sd[k1].shape == v.shape:
                sd[k1].copy_(v)
        if "flow.blocks.0.cond_gates.weight" in block0_tensors:
            b0_cg_w = block0_tensors["flow.blocks.0.cond_gates.weight"]
            b0_cg_b = block0_tensors["flow.blocks.0.cond_gates.bias"]
            sd["flow.blocks.1.cond_gates.weight"].copy_(b0_cg_w[3 * 1536 : 5 * 1536])
            sd["flow.blocks.1.cond_gates.bias"].copy_(b0_cg_b[3 * 1536 : 5 * 1536])

        torch.save(model.state_dict(), tmp_weights / "diffusion_pytorch_model.bin")

        ref_model = LatDitModel.from_pretrained(tmp_weights, device="cpu")
        sol_model = Solution.from_pretrained(tmp_weights, device="cpu")

        # Load real sample from public_test
        sample_dir = ROOT / "fetched_info" / "sample_test_data" / "public_test" / "22a9a0954ea14062b20dc3f826081811" / "cache"
        vox_np = np.load(sample_dir / "vox_idxi.npz")["vox_idxi"][:384]
        dino_np = np.load(sample_dir / "4view_000_dinov3.npz")["feats"]
        vox_t = torch.tensor(vox_np, dtype=torch.int32)
        dino_t = torch.tensor(dino_np, dtype=torch.float32)

        # 1. Verify exact equivalence of pos_pred and 1-token neg_pred on real block 0 weights + 4101 cond tokens
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
            ref_pos = ref_model.core.flow(xt_sparse, t_tensor, clf_tok, context).feats
            t_ref_pos = time.perf_counter() - t0

            t0 = time.perf_counter()
            ref_neg = ref_model.core.flow(xt_sparse, t_tensor, clf_tok, torch.zeros_like(context)).feats
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
        vox_bench = vox_t[:128]
        cond_obj = LatDitConditioning(dino_features=dino_t[:, :69, :], batch_size=1)  # 5 + 8x8 = 69 cond tokens

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
            out = sol_model.generate([vox_bench], conditioning=cond_obj, generator=g)[0]
            elapsed = time.perf_counter() - t_start
            return out, elapsed

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


if __name__ == "__main__":
    main()

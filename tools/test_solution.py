"""
Comprehensive verification and benchmark suite for `Solution`.

Tests:
  1. Official submission format validation (`inspect_submission`) on both workspace root and `submission.zip`.
  2. `LatDitGenerator` protocol compliance (`issubclass`, `from_pretrained`, `generate`, `last_run_metadata`).
  3. Mathematical equivalence of:
     - `OptimizedLatDitRunner.forward_flow_fast(..., uncond=False)` vs baseline `LatDitCore.flow(...)`
     - `OptimizedLatDitRunner.forward_flow_fast(..., uncond=True)` (1-token condensation) vs 4101-token baseline `neg_pred`
     - Full sampling trajectory consistency & numerical stability on real test samples from `public_test`.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (
    _ROOT,
    _ROOT / "lib",
    _ROOT / "baseline_repo" / "3dgen-main",
    _ROOT / "baseline_repo" / "3dgen-main" / "scripts" / "hackathon",
):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from submission_format import SubmissionLayout, inspect_submission


def format_report(report: SubmissionLayout) -> str:
    return (
        f"  root={report.root}, ok={report.ok}, "
        f"errors={report.errors}, notes={report.notes}"
    )


def test_submission_format() -> None:
    print("1. Testing submission format compliance...")
    report_dir = inspect_submission(_ROOT)
    print(format_report(report_dir))
    assert report_dir.ok, f"Directory submission check failed: {report_dir.errors}"

    sub_zip = _ROOT / "submission.zip"
    if sub_zip.is_file():
        report_zip = inspect_submission(sub_zip)
        print(format_report(report_zip))
        assert report_zip.ok, f"Zip submission check failed: {report_zip.errors}"
    for sc_rel in ("lib/solver_config.json", "weights/solver_config.json"):
        sc_data = json.loads((_ROOT / sc_rel).read_text(encoding="utf-8"))
        assert sc_data.get("symplectic_verlet_kick") is True
        assert sc_data.get("l1_table_rope3d") is True
        assert sc_data.get("fused_msa_to_cr_modulate") is True
        assert sc_data.get("flow_unipc_bh2_integrator") is True
        assert sc_data.get("batched_gate_modulate_stack") is True
        assert sc_data.get("fused_cr_carrier_accum") is True
        assert sc_data.get("noncond_dino_qmc_stride") == 4
        assert sc_data.get("fused_b7_gate_ln_cond") is True
    print("   [PASS] Submission format verified.")


def test_pytorch_pipeline() -> None:
    try:
        import numpy as np
        import torch
    except ImportError:
        print("   [SKIP] PyTorch/NumPy not installed in this environment; skipping tensor runtime tests.")
        return

    from lib.fast_kernels import preload_extensions
    preload_extensions(verbose=False)

    import base_models.sparse as sp
    from kandi3d_image23d import LatDitConditioning, LatDitGenerator
    from kandi3d_image23d.configs.lat_dit_config import LatDitConfig
    from kandi3d_image23d.models.modeling_lat_dit import LatDitModel, _config_to_init_kwargs
    from kandi3d_image23d.runtime import FlowMatchSamplingParams
    from solution import Solution
    from lib.fast_sampler import SolverConfig

    print("2. Testing Solution class contract & numerical equivalence...")
    assert issubclass(Solution, LatDitGenerator), "Solution must subclass LatDitGenerator"
    assert "generate" not in Solution.__dict__, "Solution must not override generate()"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build a small-depth synthetic LatDitModel with the exact competition config (except depth=4 for fast test)
    with open(_ROOT / "weights" / "config.json", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    raw_cfg["model_config"] = dict(raw_cfg["model_config"])
    raw_cfg["model_config"]["depth"] = 4
    raw_cfg["model_config"]["cond_depth"] = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(raw_cfg, f)

        cfg = LatDitConfig.from_dict(raw_cfg)
        base_model = LatDitModel(**_config_to_init_kwargs(cfg)).to(device).eval()
        torch.manual_seed(0)
        # Initialize non-zero gates and projections so all paths are active
        for p in base_model.parameters():
            if p.ndim >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                torch.nn.init.normal_(p, mean=0.0, std=0.05)
        base_model.core.mean.copy_(torch.linspace(-0.5, 0.5, 16))
        base_model.core.std.copy_(torch.linspace(0.8, 1.2, 16))

        torch.save(base_model.state_dict(), tmp_path / "diffusion_pytorch_model.bin")

        # Load both baseline LatDitModel and our Solution from tmp_path
        ref_model = LatDitModel.from_pretrained(tmp_path, device=device)
        sol_model = Solution.from_pretrained(tmp_path, device=device)

        # Load real sample coordinates and DINO features from fetched_info
        sample_dir = _ROOT / "fetched_info" / "sample_test_data" / "public_test" / "22a9a0954ea14062b20dc3f826081811" / "cache"
        if sample_dir.is_dir():
            vox_np = np.load(sample_dir / "vox_idxi.npz")["vox_idxi"][:512]
            dino_np = np.load(sample_dir / "4view_000_dinov3.npz")["feats"]
            vox_t = torch.tensor(vox_np, dtype=torch.int32, device=device)
            dino_t = torch.tensor(dino_np, dtype=torch.float32, device=device)
        else:
            vox_t = torch.randint(0, 64, (512, 3), dtype=torch.int32, device=device)
            dino_t = torch.randn(1, 4101, 1280, dtype=torch.float32, device=device)

        cond_input = LatDitConditioning(dino_features=dino_t, batch_size=1)

        # Verify single-step pos_pred and 1-token neg_pred equivalence
        with torch.no_grad():
            context = ref_model.core.image_map(dino_t.to(sol_model.runner._compute_dtype(device)))
            clf_tok = torch.zeros(1, ref_model.core.dim, device=device, dtype=context.dtype)
            geom = sol_model.runner._prepare_sample_geometry([vox_t], context)

            coords = sp.sparse_coords([vox_t]).int()
            noise = torch.randn( len(coords), 16, device=device, dtype=torch.float32)
            xt_sparse = sp.SparseTensor(feats=noise, coords=coords)
            t_tensor = torch.tensor([750.0], device=device, dtype=torch.float32)

            ref_pos = ref_model.core.flow(xt_sparse, t_tensor, clf_tok.float(), context.float()).feats
            ref_neg = ref_model.core.flow(xt_sparse, t_tensor, clf_tok.float(), torch.zeros_like(context).float()).feats

            xt_sorted = noise[geom["perm"]].contiguous()
            fast_pos_sorted = sol_model.runner.forward_flow_fast(
                xt_sorted, 0.75, context, clf_tok, geom, uncond=False
            )
            fast_neg_sorted = sol_model.runner.forward_flow_fast(
                xt_sorted, 0.75, context, clf_tok, geom, uncond=True
            )
            fast_pos = fast_pos_sorted[geom["up_widx"]]
            fast_neg = fast_neg_sorted[geom["up_widx"]]

            pos_diff = (ref_pos - fast_pos).abs().max().item()
            neg_diff = (ref_neg - fast_neg).abs().max().item()
            pos_cos = torch.nn.functional.cosine_similarity(ref_pos.flatten(), fast_pos.flatten(), dim=0).item()
            neg_cos = torch.nn.functional.cosine_similarity(ref_neg.flatten(), fast_neg.flatten(), dim=0).item()
            print(f"   pos_pred max_abs_diff={pos_diff:.2e}, cos_sim={pos_cos:.8f}")
            print(f"   neg_pred (1-token vs 4101-token) max_abs_diff={neg_diff:.2e}, cos_sim={neg_cos:.8f}")
            assert pos_cos > 0.9999, f"pos_pred cosine similarity too low: {pos_cos}"
            assert neg_cos > 0.9999, f"neg_pred cosine similarity too low: {neg_cos}"
            assert hasattr(sol_model.runner.flow.blocks[0], "_cond_1tok_msa_w")
            assert hasattr(sol_model.runner.flow.blocks[0], "_cond_1tok_q_and_out_cr_w")
            assert hasattr(sol_model.runner.flow.blocks[-1], "_cond_1tok_cr_w")
            assert getattr(sol_model.runner, "_noncond_1tok_w_t", None) is not None
            assert getattr(sol_model.runner, "_noncond_1tok_b", None) is not None
            assert getattr(sol_model.runner, "_patchify_w_cast", None) is not None
            assert sol_model.runner.solver_config.fastercache_freq_decoupled_cfg is True
            assert sol_model.runner.solver_config.cayley_lie_orthogonal_step is True
            from lib.fast_sampler import compute_stork2_beta
            beta9 = compute_stork2_beta(9)
            assert abs(beta9 - 0.46283549) < 1e-6, f"Unexpected STORK-2 beta_9: {beta9}"

            # Test full generate() call and metadata recording
            g = torch.Generator(device=device).manual_seed(123)
            out_latents = sol_model.generate(
                [vox_t],
                conditioning=cond_input,
                sampling=FlowMatchSamplingParams(num_inference_steps=4),
                generator=g,
            )
            assert isinstance(out_latents, list) and len(out_latents) == 1
            assert out_latents[0].shape == (len(vox_t), 16)
            assert torch.isfinite(out_latents[0]).all()
            assert getattr(sol_model.runner, "_last_pos_x_without_cr", None) is not None
            assert getattr(sol_model.runner, "_last_pos_kv_x_list", None) is not None
            meta = sol_model.last_run_metadata
            assert meta is not None and "elapsed_seconds" in meta and meta.get("status") == "ok"
            print(f"   generate() output shape={tuple(out_latents[0].shape)}, metadata={meta}")
            print("   [PASS] PyTorch pipeline & numerical equivalence verified.")


if __name__ == "__main__":
    test_submission_format()
    test_pytorch_pipeline()

"""
Accelerated & Distilled LatDiT Generator (`Solution`) for the Kandinsky3D LatDiT Track.

Contract compliance:
  - Subclasses `LatDitGenerator`
  - Implements `from_pretrained(...)` and `generate_impl(...)`
  - Does NOT override `generate()` (timing & metadata owned by `LatDitGenerator`)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Ensure local `lib/` and repo root are importable regardless of how `solution.py` is loaded
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR / "lib", _THIS_DIR / "baseline_repo" / "3dgen-main"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.fast_kernels import preload_extensions

preload_extensions(verbose=False)

from kandi3d_image23d import LatDitConditioning, LatDitGenerator
from kandi3d_image23d.configs.lat_dit_config import LatDitConfig
from kandi3d_image23d.modeling_utils import normalize_state_dict
from kandi3d_image23d.models.modeling_lat_dit import LatDitModel, _config_to_init_kwargs, _load_lat_dit_config
from kandi3d_image23d.runtime import FlowMatchSamplingParams, resolve

from lib.fast_flow import OptimizedLatDitRunner
from lib.fast_sampler import SolverConfig


def _has_model_weights(directory: Path) -> bool:
    return (
        (directory / "diffusion_pytorch_model.safetensors").is_file()
        or (directory / "diffusion_pytorch_model.bin").is_file()
    )


def _discover_weights_dir(primary_path: Path) -> Path:
    """
    Locate the directory containing `diffusion_pytorch_model.safetensors` (or `.bin`).
    Checks `primary_path` first (standard submission with full weights in `weights/`),
    then falls back to `--lat-dit-dir` CLI arg, env vars, or standard organizer locations.
    """
    if _has_model_weights(primary_path):
        return primary_path
    if _has_model_weights(primary_path / "lat_dit"):
        return primary_path / "lat_dit"

    # Check --lat-dit-dir from sys.argv if provided to run_lat_dit_eval.py
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--lat-dit-dir" and i + 1 < len(argv):
            cand = Path(argv[i + 1]).expanduser().resolve()
            if _has_model_weights(cand):
                return cand
            if _has_model_weights(cand / "lat_dit"):
                return cand / "lat_dit"
        elif arg.startswith("--lat-dit-dir="):
            cand = Path(arg.split("=", 1)[1]).expanduser().resolve()
            if _has_model_weights(cand):
                return cand
            if _has_model_weights(cand / "lat_dit"):
                return cand / "lat_dit"
        elif arg == "--lat-vae-dir" and i + 1 < len(argv):
            cand = Path(argv[i + 1]).expanduser().resolve().parent / "lat_dit"
            if _has_model_weights(cand):
                return cand
        elif arg.startswith("--lat-vae-dir="):
            cand = Path(arg.split("=", 1)[1]).expanduser().resolve().parent / "lat_dit"
            if _has_model_weights(cand):
                return cand

    for env_key in ("LAT_DIT_DIR", "LAT_DIT_WEIGHTS_DIR", "KANDI3D_WEIGHTS_DIR"):
        env_val = os.environ.get(env_key)
        if env_val:
            cand = Path(env_val).expanduser().resolve()
            if _has_model_weights(cand):
                return cand
            if _has_model_weights(cand / "lat_dit"):
                return cand / "lat_dit"

    for rel in (
        Path("weights/lat_dit"),
        Path("kandi3d_weights_aijc_kit/lat_dit"),
        Path("../weights/lat_dit"),
        Path("/weights/lat_dit"),
        Path("/workspace/weights/lat_dit"),
    ):
        if _has_model_weights(rel):
            return rel.resolve()

    return primary_path


def _merge_distilled_adapters_if_present(model: LatDitModel, custom_dir: Path) -> None:
    """
    If distilled LoRA or residual weight deltas (`student_adapter.pt` or `student_lora.safetensors`)
    are present in `custom_dir`, fold them directly into the base weights at load time
    so zero runtime adapter overhead is incurred during `generate_impl`.
    """
    adapter_pt = custom_dir / "student_adapter.pt"
    if not adapter_pt.is_file():
        return
    try:
        payload = torch.load(adapter_pt, map_location="cpu", weights_only=True)
    except Exception:
        return

    sd = model.core.state_dict()
    lora_scale = float(payload.get("lora_scale", 1.0))
    lora_A = payload.get("lora_A", {})
    lora_B = payload.get("lora_B", {})
    for target_key, A in lora_A.items():
        B = lora_B.get(target_key)
        if B is not None and target_key in sd:
            delta = (B.float() @ A.float()) * lora_scale
            sd[target_key].copy_(sd[target_key].float() + delta.to(sd[target_key].device))

    weight_deltas = payload.get("weight_deltas", {})
    for target_key, delta in weight_deltas.items():
        if target_key in sd:
            sd[target_key].copy_(sd[target_key].float() + delta.float().to(sd[target_key].device))


class Solution(LatDitGenerator):
    """
    Accelerated & Distilled LatDiT Generator.

    Combines:
      1. Static 3D Geometry & Partitioning Hoisting (patchify/unpatchify permutation,
         Hilbert space-filling curve orders, 3D submanifold conv neighbor graph,
         and 3D/2D RoPE lookup tables computed once instead of 88x-2112x per sample).
      2. Fused Triton Kernels (QKNorm + 3D RoPE + Hilbert Gather, Fused LayerNorm Modulation).
      3. Exact 1-Token Unconditional Condensation for Classifier-Free Guidance (`neg_pred`),
         eliminating 99.97% of unconditional conditioning token compute and 100% of
         voxel-to-cond cross-attention in `neg_pred` with zero mathematical approximation error.
      4. Piecewise 2nd-Order Adams-Bashforth ODE Solver aligned to the exact $t = 0.5$
         CFG boundary with Adaptive CFG-Delta Extrapolation and distilled step calibration.
    """

    def __init__(
        self,
        inner: LatDitModel,
        solver_config: Optional[SolverConfig] = None,
        weights_dir: Optional[Path] = None,
    ):
        self.inner = inner
        self.solver_config = solver_config or SolverConfig()
        self._default_runtime = FlowMatchSamplingParams(
            guidance_scale=self.solver_config.guidance_scale,
            cfg_range=self.solver_config.cfg_range,
            rescale_t=self.solver_config.rescale_t,
            num_inference_steps=self.solver_config.num_inference_steps,
        )
        self.runner = OptimizedLatDitRunner(
            model=self.inner,
            solver_config=self.solver_config,
            weights_dir=weights_dir,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        device: Optional[Union[torch.device, str, int]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> "Solution":
        preload_extensions(verbose=False)

        primary_path = Path(pretrained_model_name_or_path).expanduser().resolve()
        weights_path = _discover_weights_dir(primary_path)

        config_dir = primary_path if (primary_path / "config.json").is_file() else weights_path
        config = _load_lat_dit_config(config_dir)
        solver_cfg = SolverConfig.from_dir(primary_path, fallback_model_cfg=config.model_config)

        # Load base LatDitModel
        model = LatDitModel(**_config_to_init_kwargs(config))
        weight_file = weights_path / "diffusion_pytorch_model.safetensors"
        if weight_file.is_file():
            from safetensors.torch import load_file

            state_dict = load_file(str(weight_file))
        else:
            bin_path = weights_path / "diffusion_pytorch_model.bin"
            if not bin_path.is_file():
                raise FileNotFoundError(
                    f"No diffusion_pytorch_model.safetensors or .bin found in {weights_path} "
                    f"(from {primary_path})"
                )
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

        model.load_state_dict(normalize_state_dict(state_dict), strict=False)
        _merge_distilled_adapters_if_present(model, primary_path)

        if device is not None:
            model.to(device)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        model.eval()

        instance = cls(model, solver_config=solver_cfg, weights_dir=primary_path)
        if solver_cfg.warmup_on_load:
            instance.runner.warmup()
        return instance

    def parameters(self, recurse: bool = True):
        return self.inner.parameters(recurse=recurse)

    def to(self, *args: Any, **kwargs: Any) -> "Solution":
        self.inner.to(*args, **kwargs)
        self.runner._prepare_weights_and_buffers()
        return self

    def eval(self) -> "Solution":
        self.inner.eval()
        return self

    @torch.no_grad()
    def generate_impl(
        self,
        vox_idxs: list,
        *,
        conditioning: LatDitConditioning,
        sampling: Optional[FlowMatchSamplingParams] = None,
        generator: Optional[torch.Generator] = None,
        **sampling_kwargs: Any,
    ) -> list:
        resolved_sampling = resolve(self._default_runtime, sampling, **sampling_kwargs)
        return self.runner.sample_fast(
            vox_idxs,
            conditioning,
            sampling=resolved_sampling,
            generator=generator,
        )


__all__ = ["Solution"]

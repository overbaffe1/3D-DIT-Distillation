from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401

from kandi3d_image23d import LatDitGenerator
from kandi3d_image23d.lat_dit_stage import LatDitStage

from submission_format import inspect_dir


def parse_impl_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError("Expected format 'module_or_file.py:ClassName'")
    mod, cls = spec.split(":", 1)
    if not mod or not cls:
        raise ValueError("Invalid --generator-impl spec")
    return mod, cls


def _is_blank(value: Optional[str]) -> bool:
    return value is None or not str(value).strip()


def resolve_existing_dir(value: Optional[str], *, flag: str) -> Optional[Path]:
    """Return a directory Path only if the flag is set and the path exists."""
    if _is_blank(value):
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{flag} is set but is not an existing directory: {path}")
    return path


def resolve_generator_impl_spec(spec: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Use --generator-impl only if it is set and resolvable:
      - file spec: the .py path must exist
      - dotted module: importlib must find the module
    """
    if _is_blank(spec):
        return None
    module_or_file, cls_name = parse_impl_spec(spec.strip())
    candidate = Path(module_or_file).expanduser()
    looks_like_file = (
        module_or_file.endswith(".py")
        or candidate.suffix == ".py"
        or "/" in module_or_file
        or "\\" in module_or_file
        or candidate.exists()
    )
    if looks_like_file:
        path = candidate.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"--generator-impl is set but file does not exist: {path}"
            )
        return str(path), cls_name
    if importlib.util.find_spec(module_or_file) is None:
        raise ModuleNotFoundError(
            f"--generator-impl is set but module {module_or_file!r} is not importable"
        )
    return module_or_file, cls_name


def load_class_from_file(module_or_file: str, cls_name: str) -> Type[Any]:
    if os.path.exists(module_or_file) and module_or_file.endswith(".py"):
        path = Path(module_or_file)
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import from file: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        if not hasattr(mod, cls_name):
            raise AttributeError(f"{cls_name} not found in {path}")
        return getattr(mod, cls_name)
    mod = importlib.import_module(module_or_file)
    if not hasattr(mod, cls_name):
        raise AttributeError(f"{cls_name} not found in module {module_or_file}")
    return getattr(mod, cls_name)


def load_vox_idxi(path: Path) -> np.ndarray:
    data = np.load(path)
    if "vox_idxi" in data.files:
        return np.asarray(data["vox_idxi"])
    if "arr_0" in data.files:
        return np.asarray(data["arr_0"])
    raise KeyError(f"No vox indices in {path}; keys={data.files}")


def load_dinov3(
    sample_cache_dir: Path,
    view_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    for combined in (
        sample_cache_dir / f"{view_name}_dinov3.npz",
        sample_cache_dir / "dinov3.npz",
    ):
        if combined.exists():
            data = np.load(combined)
            return np.asarray(data["feats"]), np.asarray(data["token"])

    # Backward-compatible fallback for older caches.
    feats_pt = sample_cache_dir / "dinov3_feats.pt"
    token_pt = sample_cache_dir / "dinov3_token.pt"
    if feats_pt.exists() and token_pt.exists():
        feats = torch.load(feats_pt, map_location="cpu").numpy()
        token = torch.load(token_pt, map_location="cpu").numpy()
        return np.asarray(feats), np.asarray(token)

    raise FileNotFoundError(
        f"Missing DINOv3 cache in {sample_cache_dir} "
        f"(expected {view_name}_dinov3.npz, dinov3.npz or *.pt)"
    )


def resolve_sample_cache_dir(sample_root: Path) -> Path:
    in_place = sample_root / "cache"
    return in_place if in_place.is_dir() else sample_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, required=True, help="prepare_cache output dir.")
    ap.add_argument("--out-dir", type=str, required=True, help="Where to save pred meshes + CSV.")
    ap.add_argument(
        "--lat-dit-dir",
        type=str,
        default=None,
        help="Baseline lat_dit hub dir (weights). Optional when --submission-dir is set.",
    )
    ap.add_argument(
        "--lat-vae-dir",
        type=str,
        default=None,
        help="lat_vae hub dir for mesh decode. Omit with --no-decode.",
    )
    ap.add_argument("--no-decode", action="store_true", help="Skip lat_vae mesh decode.")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--torch-dtype", type=str, default=None, help="float16|bfloat16 or None.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--view-name",
        type=str,
        default="4view_000",
        help="Cached frame to evaluate (default: 4view_000).",
    )
    ap.add_argument(
        "--submission-dir",
        type=str,
        default=None,
        help="Unpacked participant folder with solution.py + weights/. Overrides --generator-*.",
    )
    ap.add_argument("--generator-impl", type=str, default=None, help="module_or_file.py:ClassName")
    ap.add_argument(
        "--generator-weights-dir",
        type=str,
        default=None,
        help="Optional weights path for contestant generator. Default: --lat-dit-dir",
    )
    args = ap.parse_args()

    if not args.no_decode and args.lat_vae_dir is None:
        raise ValueError("Provide --lat-vae-dir or pass --no-decode")

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = None
    if args.torch_dtype is not None:
        dtype = getattr(torch, args.torch_dtype)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    if args.submission_dir:
        if args.generator_impl or args.generator_weights_dir:
            raise ValueError("Do not combine --submission-dir with --generator-impl / --generator-weights-dir")
        layout = inspect_dir(Path(args.submission_dir))
        layout.raise_if_invalid()
        for note in layout.notes:
            print(f"Submission fallback: {note}")
        if layout.solution_py is not None:
            args.generator_impl = layout.generator_impl_spec()
        if layout.weights_dir is not None:
            args.generator_weights_dir = str(layout.weights_dir)
        lib_dir = layout.root / "lib"
        if lib_dir.is_dir():
            lib_path = str(lib_dir.resolve())
            if lib_path not in sys.path:
                sys.path.append(lib_path)
        print(f"Submission root: {layout.root}")
        print(
            f"  code: {layout.solution_py if layout.solution_py is not None else 'baseline LatDitModel'}"
        )
        print(
            f"  weights: {layout.weights_dir if layout.weights_dir is not None else 'organizer --lat-dit-dir'}"
        )

    lat_dit_dir = resolve_existing_dir(args.lat_dit_dir, flag="--lat-dit-dir")
    custom_weights_dir = resolve_existing_dir(
        args.generator_weights_dir,
        flag="--generator-weights-dir",
    )
    impl_spec = resolve_generator_impl_spec(args.generator_impl)

    if lat_dit_dir is None and custom_weights_dir is None:
        raise ValueError("Provide --lat-dit-dir or a submission with non-empty weights/")

    generator_cls: Optional[Type[Any]] = None
    lat_dit_weights_dir = str(custom_weights_dir or lat_dit_dir)
    if impl_spec is not None:
        module_or_file, cls_name = impl_spec
        try:
            gen_cls = load_class_from_file(module_or_file, cls_name)
            if gen_cls is None:
                raise TypeError(f"{cls_name} resolved to None in {module_or_file}")
            if not hasattr(gen_cls, "from_pretrained"):
                raise TypeError(f"{gen_cls} has no from_pretrained()")
            if not isinstance(gen_cls, type) or not issubclass(gen_cls, LatDitGenerator):
                raise TypeError(
                    f"{cls_name} must inherit LatDitGenerator and implement generate_impl()"
                )
            generator_cls = gen_cls
            print(f"Using custom generator: {generator_cls.__module__}.{generator_cls.__qualname__}")
        except Exception as exc:
            print(
                f"WARNING: could not load {cls_name} from {module_or_file} ({exc}); "
                "falling back to baseline LatDitModel"
            )
            generator_cls = None
            impl_spec = None
    if generator_cls is None:
        print("Using baseline LatDitModel")

    print(f"Generator weights dir: {lat_dit_weights_dir}")

    stage = LatDitStage.from_pretrained(
        lat_dit_weights_dir,
        lat_vae_dir=None if args.no_decode else args.lat_vae_dir,
        device=device,
        torch_dtype=dtype,
        fix_seed=True,
        generator_cls=generator_cls,
    )
    if getattr(stage, "lat_dit", None) is None:
        raise RuntimeError("LatDit generator was not loaded into memory")

    sample_ids = sorted(
        p.name
        for p in cache_dir.iterdir()
        if p.is_dir()
        and (resolve_sample_cache_dir(p) / "vox_idxi.npz").is_file()
        and (
            (resolve_sample_cache_dir(p) / f"{args.view_name}_dinov3.npz").is_file()
            or (resolve_sample_cache_dir(p) / "dinov3.npz").is_file()
            or (resolve_sample_cache_dir(p) / "dinov3_feats.pt").is_file()
        )
    )
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
    if not sample_ids:
        raise RuntimeError(f"No sample dirs found under {cache_dir}")

    pred_dir = out_dir / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for i, sample_id in enumerate(sample_ids):
        sample_dir = resolve_sample_cache_dir(cache_dir / sample_id)
        vox_npz = sample_dir / "vox_idxi.npz"
        if not vox_npz.exists():
            raise FileNotFoundError(f"Missing {vox_npz}")

        vox_idxi = torch.tensor(load_vox_idxi(vox_npz), dtype=torch.int32, device=device)
        feats_np, token_np = load_dinov3(sample_dir, args.view_name)
        dinov3_feats = torch.tensor(feats_np, device=device)
        dinov3_token = torch.tensor(token_np, device=device)

        out = stage.run(
            vox_idxi=vox_idxi,
            dinov3_feats=dinov3_feats,
            dinov3_token=dinov3_token,
            decode_mesh=not args.no_decode,
        )

        lat_dit_stage = out["lat_dit_stage"]
        mesh = out["mesh"]

        sample_pred_dir = pred_dir / sample_id
        sample_pred_dir.mkdir(parents=True, exist_ok=True)

        mesh_path = ""
        if mesh is not None:
            mesh_path = str(sample_pred_dir / "mesh.npz")
            np.savez_compressed(
                mesh_path,
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.int32),
            )

        rows.append(
            {
                "id": sample_id,
                "view": args.view_name,
                "mesh_path": mesh_path,
                "lat_dit_stage": float(lat_dit_stage),
                "run_metadata": json.dumps(out["run_metadata"], sort_keys=True),
            }
        )
        print(f"[{i+1}/{len(sample_ids)}] pred {sample_id} lat_dit_stage={lat_dit_stage:.4f}s")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved pred summary: {csv_path}")


if __name__ == "__main__":
    main()

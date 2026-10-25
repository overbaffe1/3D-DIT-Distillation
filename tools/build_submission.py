"""
Build and validate a competition-compliant `submission.zip` archive.

Usage:
  # Build lightweight code+config archive (or full archive if weights/ contains .safetensors):
  python3 tools/build_submission.py --output submission.zip

  # Build full standalone archive bundling teacher/student weights from --lat-dit-dir:
  python3 tools/build_submission.py --lat-dit-dir /path/to/kandi3d_weights_aijc_kit/lat_dit --output submission.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_HACKATHON_SCRIPTS = _ROOT / "baseline_repo" / "3dgen-main" / "scripts" / "hackathon"
if _HACKATHON_SCRIPTS.is_dir() and str(_HACKATHON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_HACKATHON_SCRIPTS))

from submission_format import SubmissionLayout, inspect_submission


def format_report(report: SubmissionLayout) -> str:
    lines = [
        f"root: {report.root}",
        f"ok: {report.ok}",
        f"solution_py: {report.solution_py}",
        f"weights_dir: {report.weights_dir}",
        f"errors: {report.errors}",
        f"notes: {report.notes}",
    ]
    return "\n".join(lines)


def build_submission_zip(
    root_dir: Path,
    output_zip: Path,
    external_lat_dit_dir: Path | None = None,
    include_weights_always: bool = False,
) -> Path:
    output_zip = output_zip.expanduser().resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    solution_py = root_dir / "solution.py"
    if not solution_py.is_file():
        raise FileNotFoundError(f"Missing {solution_py}")

    weights_dir = root_dir / "weights"
    lib_dir = root_dir / "lib"

    has_binary_weights = False
    if weights_dir.is_dir():
        for fname in (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "model.safetensors",
            "adapter_deltas.safetensors",
            "adapter_deltas.pt",
        ):
            if (weights_dir / fname).is_file():
                has_binary_weights = True
                break

    include_weights = include_weights_always or has_binary_weights or (external_lat_dit_dir is not None)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. solution.py
        zf.write(solution_py, arcname="solution.py")

        # 2. lib/
        if lib_dir.is_dir():
            for p in sorted(lib_dir.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                    rel = p.relative_to(root_dir).as_posix()
                    zf.write(p, arcname=rel)

        # 2b. agent_toolkit/ (portable AI-agent instructions, CI scripts, RemoteZip/chunking & ArXiv toolkit)
        toolkit_dir = root_dir / "agent_toolkit"
        if toolkit_dir.is_dir():
            for p in sorted(toolkit_dir.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                    rel = p.relative_to(root_dir).as_posix()
                    zf.write(p, arcname=rel)

        # 3. weights/ (only included when actual model/adapter binary weights are present,
        # so that when submitting a lightweight archive without 7.8 GB weights,
        # run_lat_dit_eval.py automatically passes organizer --lat-dit-dir to Solution.from_pretrained)
        added_weights = set()
        if include_weights and weights_dir.is_dir():
            for p in sorted(weights_dir.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("."):
                    rel = p.relative_to(root_dir).as_posix()
                    zf.write(p, arcname=rel)
                    added_weights.add(p.name)

        # 4. Optionally bundle model weights from --lat-dit-dir if not already in weights/
        if external_lat_dit_dir is not None:
            ext_dir = external_lat_dit_dir.expanduser().resolve()
            if (ext_dir / "lat_dit").is_dir():
                ext_dir = ext_dir / "lat_dit"
            for fname in (
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.bin",
                "weights_meta.json",
                "config.json",
            ):
                if fname not in added_weights and (ext_dir / fname).is_file():
                    zf.write(ext_dir / fname, arcname=f"weights/{fname}")
                    added_weights.add(fname)

    return output_zip


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and verify submission.zip for the LatDiT track.")
    ap.add_argument("--output", type=str, default="submission.zip", help="Output .zip file path")
    ap.add_argument(
        "--lat-dit-dir",
        type=str,
        default=None,
        help="Optional directory with diffusion_pytorch_model.safetensors to bundle into weights/",
    )
    args = ap.parse_args()

    ext_dir = Path(args.lat_dit_dir) if args.lat_dit_dir else None
    out_zip = build_submission_zip(_ROOT, Path(args.output), external_lat_dit_dir=ext_dir)

    print("=== Validating Workspace Root ===")
    dir_report = inspect_submission(_ROOT)
    print(format_report(dir_report))
    if not dir_report.ok:
        sys.exit(1)

    print("\n=== Validating Archive ===")
    zip_report = inspect_submission(out_zip)
    print(format_report(zip_report))
    if not zip_report.ok:
        sys.exit(1)

    print(f"\nSuccessfully built and validated: {out_zip} ({out_zip.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

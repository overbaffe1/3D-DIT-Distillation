"""
Reproducible dataset & weights downloader for the Kandinsky3D LatDiT Distillation track.

Usage:
  python tools/download_data.py --target all --out-dir ./data
  python tools/download_data.py --target weights --out-dir ./weights_kit
  python tools/download_data.py --target public_test --out-dir ./data/public_test
  python tools/download_data.py --target train --train-chunks 1 2 --out-dir ./data/train
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/tmp_data/AIJC_data"
URLS = {
    "code": f"{BASE_URL}/3dgen-main.zip",
    "weights": f"{BASE_URL}/kandi3d_weights_aijc_kit.zip",
    "public_test": f"{BASE_URL}/objhq_splits_clean/public_test.zip",
}


def train_chunk_url(chunk_idx: int) -> str:
    return f"{BASE_URL}/objhq_splits_clean/train_archives/train_part_{chunk_idx:03d}.tar"


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"[skip] {dest} already exists ({dest.stat().st_size / (1024**2):.1f} MB)")
        return dest
    print(f"[download] {url} -> {dest}")
    if shutil.which("curl"):
        subprocess.run(["curl", "-L", "--retry", "3", "-C", "-", url, "-o", str(dest)], check=True)
    else:
        urllib.request.urlretrieve(url, str(dest))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Download competition code, weights, and datasets.")
    ap.add_argument(
        "--target",
        choices=("code", "weights", "public_test", "train", "all"),
        default="weights",
    )
    ap.add_argument("--out-dir", type=str, default="data_downloads")
    ap.add_argument("--train-chunks", type=int, nargs="+", default=[1], help="Train chunk indices (1..15)")
    ap.add_argument("--populate-submission-weights", action="store_true", help="Copy lat_dit weights into ./weights/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [args.target] if args.target != "all" else ["code", "weights", "public_test"]

    if "code" in targets:
        zpath = download_file(URLS["code"], out_dir / "3dgen-main.zip")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out_dir / "code")
        print(f"Extracted baseline code to {out_dir / 'code'}")

    if "weights" in targets:
        zpath = download_file(URLS["weights"], out_dir / "kandi3d_weights_aijc_kit.zip")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out_dir)
        lat_dit_extracted = out_dir / "kandi3d_weights_aijc_kit" / "lat_dit"
        print(f"Extracted weights to {out_dir / 'kandi3d_weights_aijc_kit'}")
        if args.populate_submission_weights and lat_dit_extracted.is_dir():
            sub_weights = Path("weights").resolve()
            sub_weights.mkdir(parents=True, exist_ok=True)
            for fname in ("diffusion_pytorch_model.safetensors", "weights_meta.json"):
                src = lat_dit_extracted / fname
                if src.is_file():
                    shutil.copy2(src, sub_weights / fname)
                    print(f"Copied {src} -> {sub_weights / fname}")

    if "public_test" in targets:
        zpath = download_file(URLS["public_test"], out_dir / "public_test.zip")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out_dir)
        print(f"Extracted public_test to {out_dir / 'public_test'}")

    if "train" in targets:
        train_dir = out_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        for idx in args.train_chunks:
            url = train_chunk_url(idx)
            tpath = download_file(url, train_dir / f"train_part_{idx:03d}.tar")
            with tarfile.open(tpath, "r") as tf:
                tf.extractall(train_dir)
            print(f"Extracted train chunk {idx:03d} to {train_dir}")


if __name__ == "__main__":
    main()

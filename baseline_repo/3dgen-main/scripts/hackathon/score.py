from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torchvision.transforms.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import _bootstrap  # noqa: F401

from rendering import render_mesh_previews


def compare_two_meshes(
    mesh_path_gt: str,
    mesh_path_pred: str,
    *,
    res: int = 512,
    num_views: int = 4,
    lpips: Optional[LearnedPerceptualImagePatchSimilarity] = None,
) -> float:
    if lpips is None:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")

    images_gt = render_mesh_previews(mesh_path=mesh_path_gt, res=res)
    images_pred = render_mesh_previews(mesh_path=mesh_path_pred, res=res)

    width, height = images_gt.size
    if width % num_views != 0:
        raise RuntimeError(f"Unexpected preview width={width} for num_views={num_views}")

    part_width = width // num_views
    lpips_values = []
    for i in range(num_views):
        left = i * part_width
        right = (i + 1) * part_width
        top = 0
        bottom = height

        view_gt = images_gt.crop((left, top, right, bottom))
        view_pred = images_pred.crop((left, top, right, bottom))

        view_gt_pt = (F.pil_to_tensor(view_gt).unsqueeze(0).float() / 255.0 - 0.5) * 2.0
        view_pred_pt = (F.pil_to_tensor(view_pred).unsqueeze(0).float() / 255.0 - 0.5) * 2.0

        lp = lpips(view_gt_pt, view_pred_pt)
        lpips_values.append(float(lp))

    return float(np.mean(lpips_values))


def _mesh_exists(path: object) -> bool:
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return False
    text = str(path).strip()
    if not text or text.lower() == "nan":
        return False
    return Path(text).is_file()


def resolve_mesh(csv_path: Path, sample_id: object, mesh_path_col: object) -> Optional[Path]:
    """Prefer eval layout next to the CSV: pred/{id}/mesh.npz. Fallback to mesh_path column."""
    by_id = csv_path.parent / "pred" / str(sample_id).strip() / "mesh.npz"
    if by_id.is_file():
        return by_id
    if _mesh_exists(mesh_path_col):
        return Path(str(mesh_path_col)).expanduser().resolve()
    return None

def resolve_mesh_gt(csv_path: Path, sample_id: object, mesh_path_col: object) -> Optional[Path]:
    """Prefer eval layout next to the CSV: pred/{id}/mesh.npz. Fallback to mesh_path column."""
    by_id = csv_path.parent / str(sample_id).strip() / "mesh.npz"
    if by_id.is_file():
        return by_id
    if _mesh_exists(mesh_path_col):
        return Path(str(mesh_path_col)).expanduser().resolve()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Score pred meshes vs GT by id. Meshes are read from "
            "<csv_dir>/pred/{id}/mesh.npz (mesh_path in CSV is only a fallback). "
            "Missing / failed pred samples score 0 and are included in the mean "
            "over the full GT id list."
        )
    )
    ap.add_argument("--gt-csv", type=str, required=True)
    ap.add_argument("--pred-csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default="summary.csv")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--num-views", type=int, default=4)
    ap.add_argument("--q-min", type=float, default=0.25)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N rows of --gt-csv (CSV order). Default: all GT ids.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_csv = Path(args.gt_csv).expanduser().resolve()
    pred_csv = Path(args.pred_csv).expanduser().resolve()
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)
    if "id" not in gt_df.columns or "id" not in pred_df.columns:
        raise SystemExit("Both CSVs must contain an 'id' column")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        gt_df = gt_df.head(args.limit)
        print(f"Scoring first {len(gt_df)} GT rows (--limit {args.limit})")
    print("PRED_DF: ", pred_df, "\n")
    print("GT_DF: ", gt_df, "\n")
    

    pred_df = pred_df.drop_duplicates(subset=["id"], keep="last")
    merged_df = pd.merge(
        gt_df,
        pred_df,
        on="id",
        how="left",
        suffixes=("_gt", "_pred"),
        indicator=True,
    )
    print("MERGED_DF: ", merged_df)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")

    statuses: list[str] = []
    lpips_list: list[float] = []
    quality_list: list[float] = []
    time_list: list[float] = []
    total_list: list[float] = []

    for _, row in merged_df.iterrows():
        missing_pred = row["_merge"] != "both"
        sample_id = row["id"]
        pred_mesh = None if missing_pred else resolve_mesh(
            pred_csv, sample_id, row.get("mesh_path_pred")
        )
        gt_mesh = resolve_mesh_gt(gt_csv, sample_id, row.get("mesh_path_gt"))
        pred_time = row.get("lat_dit_stage_pred")
        time_ok = pd.notna(pred_time) and float(pred_time) > 0

        if missing_pred:
            status = "missing_pred"
        elif pred_mesh is None:
            status = "missing_pred_mesh"
        elif gt_mesh is None:
            status = "missing_gt_mesh"
        elif not time_ok:
            status = "invalid_pred_time"
        else:
            status = "ok"
        print("STATUS: ", status)

        if status != "ok":
            print("Status not OK: ", status)
            statuses.append(status)
            lpips_list.append(np.nan)
            quality_list.append(0.0)
            time_list.append(np.nan)
            total_list.append(0.0)
            continue
            
        try:
            lp = compare_two_meshes(
                str(gt_mesh),
                str(pred_mesh),
                res=args.res,
                num_views=args.num_views,
                lpips=lpips,
            )
            print("LP: ", lp)
        except Exception:
            print("some exception")
            statuses.append("compare_failed")
            lpips_list.append(np.nan)
            quality_list.append(0.0)
            time_list.append(np.nan)
            total_list.append(0.0)
            continue

        quality = 0.0 if lp > args.q_min else (1.0 - lp)
        time_factor = float(row.lat_dit_stage_gt) / float(row.lat_dit_stage_pred)
        statuses.append("ok")
        lpips_list.append(lp)
        quality_list.append(float(quality))
        time_list.append(float(time_factor))
        total_list.append(float(quality * time_factor))

    merged_df["status"] = statuses
    merged_df["lpips"] = lpips_list
    merged_df["Quality_factor"] = quality_list
    merged_df["Time_factor"] = time_list
    merged_df["Total_factor"] = total_list
    merged_df = merged_df.drop(columns=["_merge"])

    n_all = len(merged_df)
    n_ok = int((merged_df["status"] == "ok").sum())
    n_zero = n_all - n_ok
    quality_metric = float(merged_df["Quality_factor"].mean()) if n_all else 0.0
    time_metric = float(merged_df["Time_factor"].fillna(0.0).mean()) if n_all else 0.0
    total_metric = float(merged_df["Total_factor"].mean()) if n_all else 0.0

    out_csv_path = out_dir / args.out_csv
    merged_df.to_csv(out_csv_path, index=False)
    print(f"Wrote {out_csv_path} ({n_all} gt samples, {n_ok} scored, {n_zero} zero-penalty)")

    aggregates = {
        "Quality_factor": quality_metric,
        "Time_factor": time_metric,
        "Total_factor": total_metric,
    }
    for name, value in aggregates.items():
        txt_path = out_dir / f"{name}.txt"
        txt_path.write_text(f"{value}", encoding="utf-8")
        print(f"Wrote {txt_path}: {value}")


if __name__ == "__main__":
    main()

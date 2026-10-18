# Hackathon: lat_dit participant kit

This folder is a **slim** extract of Image23D for the `lat_dit` track only.
It contains baseline `lat_dit` + `lat_vae` code/weights and eval scripts.
It does **not** include the full image→3D pipeline, `pcd_*` stages, DINO weights,
or cache-building tools.

## Layout

```
kandi3d_image23d/          # LatDitGenerator, LatDitStage, LatDitModel, LatVaeModel
models/                    # legacy lat_dit / fc_vae modules
base_models/               # sparse ops helpers + mesh postprocess
ops/                       # CUDA JIT sources (libmesh, libsparse, curope)
scripts/hackathon/         # run_lat_dit_eval.py, score.py, submission_format.py
examples/my_lat_dit_stub.py
weights/lat_dit/
weights/lat_vae/
```

## Install

```
pip install -r requirements-hackathon.txt
```

You still need the CUDA stack used by the full repo (flash-attn, spconv, nvdiffrast,
torch_scatter, etc.). See the organizer env notes if compilation of `ops/*` fails.

## Cache contract (input)

Organizers provide a dataset with precomputed cache per sample:

```
<data-dir>/{id}/cache/
  vox_idxi.npz                 # key: vox_idxi, [N,3] int32
  4view_000_dinov3.npz         # feats [1,T,1280] + token [1,1280], float16
```

You do **not** run DINO or `pcd_*`.

## Baseline eval

```
python scripts/hackathon/run_lat_dit_eval.py \
  --cache-dir /path/to/data \
  --out-dir pred_out \
  --lat-dit-dir weights/lat_dit \
  --lat-vae-dir weights/lat_vae
```

## Custom generator

Implement `LatDitGenerator` (see `examples/my_lat_dit_stub.py`), then:

```
python scripts/hackathon/run_lat_dit_eval.py \
  --cache-dir /path/to/data \
  --out-dir pred_out \
  --lat-dit-dir weights/lat_dit \
  --lat-vae-dir weights/lat_vae \
  --generator-impl examples/my_lat_dit_stub.py:MyLatDit
```

Optional custom weights directory:

```
  --generator-weights-dir path/to/weights_dir
```

Outputs:

- `pred_out/results_summary.csv` (`lat_dit_stage`, `run_metadata`)
- `pred_out/pred/{id}/mesh.npz`

## Contract reminder

Subclass `LatDitGenerator` and implement `generate_impl(...)`. Do **not** override
`generate()` — the ABC owns CUDA sync, timing, and JSON-safe `last_run_metadata`.

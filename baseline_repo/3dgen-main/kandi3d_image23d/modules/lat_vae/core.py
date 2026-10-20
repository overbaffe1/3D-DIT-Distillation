"""Native LatVae core (sparse encoder/decoder + mesh extraction)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from nvdiffrast import torch as nvt

from kandi3d_image23d._repo_path import ensure_repo_root_on_path

ensure_repo_root_on_path()

from models.encoders_2D.fc_vae import (
    SparseDecoder,
    SparseEncoder,
    fc_vae,
)
from ops.fc_sparse import FlexiCubes

from kandi3d_image23d.configs.lat_vae_config import LatVaeConfig
from kandi3d_image23d.runtime import LatVaeDecodeParams, MeshOutput


class LatVaeCore(nn.Module):
    """Sparse latent VAE; decode/encode reuse fc_vae implementations."""

    decode = fc_vae.decode
    encode = fc_vae.encode

    @torch.no_grad()
    def decode_mesh(
        self,
        vox_feats: list,
        vox_idxs: list,
        *,
        runtime: Optional[LatVaeDecodeParams] = None,
    ) -> MeshOutput:
        data_dict = {"vox_feats": vox_feats, "vox_idxs": vox_idxs}
        data_dict, _ = self.decode(data_dict, {}, 0, target="decode")
        return MeshOutput(
            vertices=data_dict["mesh_vertices"][0],
            triangles=data_dict["mesh_triangles"][0],
        )

    def forward(
        self,
        data_dict: Dict[str, Any],
        crit_dict: Optional[Dict[str, Any]] = None,
        total_iteration: int = 0,
        target: str = "all",
    ):
        crit_dict = crit_dict if crit_dict is not None else {}
        if target in ("encode", "train", "all") and (
            "vox_feats" not in data_dict or data_dict.get("vae_force", False)
        ):
            data_dict, crit_dict = self.encode(data_dict, crit_dict, total_iteration)
        if target in ("decode", "train", "all", "gt"):
            data_dict, crit_dict = self.decode(data_dict, crit_dict, total_iteration, target)
        return data_dict, crit_dict

    def __init__(
        self,
        config: LatVaeConfig,
        full_config: Optional[Dict[str, Any]] = None,
        build_encoder: bool = True,
    ):
        super().__init__()
        self.config = dict(full_config) if full_config else {}
        self.config.setdefault("training", {"fp16": True, "fp16_type": "float16"})
        self.config.setdefault("generators", {"lat_vae": config.model_config})
        self.config.setdefault("representations", config.rep_config)
        self.model_config = dict(config.model_config)
        self.rep_config = dict(config.rep_config)
        self.latres = config.res
        self.res = self.latres << (len(config.up_dims) - 1)
        self.extractor_type = config.extractor
        self.render_type = config.renderer
        self.render_res = config.render_res
        self.n_renders = config.n_renders
        self.ctx = None

        mc = dict(self.model_config)
        if build_encoder:
            self.encoder = SparseEncoder(
                **mc,
                use_edges=config.edge_sampling,
                grid_res=self.res,
            )

        self.decoder = SparseDecoder(
            sdf_separate=config.sdf_separate,
            **mc,
        )

        if self.extractor_type == "fc":
            self.fc = FlexiCubes(device="cpu")

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.device = torch.device(device) if isinstance(device, int) else device
            self.decoder.to(device, *args, **kwargs)
            if hasattr(self, "encoder"):
                self.encoder.to(device, *args, **kwargs)
            if self.extractor_type == "fc":
                self.fc = FlexiCubes(device=self.device)
            if self.render_type == "nvt" and self.ctx is None:
                self.ctx = nvt.RasterizeCudaContext(device=self.device)
        return super().to(device, *args, **kwargs)

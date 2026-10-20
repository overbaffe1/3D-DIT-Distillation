"""Native LatDit core: sparse voxel flow matching with image conditioning."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional

from kandi3d_image23d._repo_path import ensure_repo_root_on_path

ensure_repo_root_on_path()

import base_models.sparse as sp
from base_models.optimization_utils import expand_as
from models.generators.lat_dit import FlowModel, SimpleTransformerBlock

from kandi3d_image23d.configs.lat_dit_config import LatDitConfig
from kandi3d_image23d.runtime import FlowMatchSamplingParams, LatDitConditioning


class LatDitCore(nn.Module):
    """Sparse latent DiT (inference sample + training denoise)."""

    def __init__(self, config: LatDitConfig, full_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config
        self._full_config = full_config or {}
        mc = config.model_config
        self.model_config = mc
        self.cond = list(config.cond)
        self.use_token = config.use_token

        self.flow = FlowModel(**mc)
        self.register_buffer("mean", torch.zeros(config.latent_dim, dtype=torch.float32))
        self.register_buffer("std", torch.ones(config.latent_dim, dtype=torch.float32))

        self.train_stats = False
        self.latent_scale = config.latent_scale
        self._init_stats()

        self.dim = config.dim
        self.pcd_pos_features = config.encoder_pos_features

        if config.focal_cond:
            self.focal_map = nn.Linear(128, self.dim)

        if "image_last" in self.cond:
            self.image_map = nn.Linear(1280, self.dim)
            if self.use_token:
                self.token_map = nn.Linear(1280, self.dim)

    def _init_stats(self) -> None:
        stat_path = self.config.stat_path
        if stat_path is not None and os.path.exists(stat_path):
            data = np.load(stat_path)
            self.mean.copy_(torch.tensor(data["mean"], device=self.mean.device, dtype=torch.float32))
            self.std.copy_(torch.tensor(data["std"], device=self.std.device, dtype=torch.float32))
            self.train_stats = False

    def _runtime_device(self) -> torch.device:
        try:
            return next(self.flow.parameters()).device
        except StopIteration:
            return self.device

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.device = torch.device(device) if isinstance(device, int) else device
        return super().to(device, *args, **kwargs)

    @torch.no_grad()
    def normalize(self, x: sp.SparseTensor, train: bool = False, alpha: float = 0.9999) -> sp.SparseTensor:
        feats = x.feats.float()
        return x.replace(
            (feats - self.mean.view(1, -1)) / self.std.view(1, -1).clamp(min=1e-6) / self.latent_scale)

    @torch.no_grad()
    def denormalize(self, x: sp.SparseTensor) -> sp.SparseTensor:
        return x.replace((x.feats * self.std.view(1, -1) + self.mean.view(1, -1)) * self.latent_scale)

    def _build_context(
        self,
        data_dict: Dict[str, Any],
        cond: Tuple[str, ...],
        total_iteration: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        context = None
        clf_tokens = []

        if "image_last" in cond:
            image_cond = data_dict["dino_features"].float()
            image_cond = self.image_map(image_cond)
            context = image_cond
            if self.use_token:
                clf_tokens.append(self.token_map(data_dict["dino_token"].float()))

        if self.use_token:
            clf_token = torch.stack(clf_tokens, dim=0).sum(dim=0)
        else:
            clf_token = torch.zeros(
                data_dict["batch_size"],
                self.dim,
                device=self.device,
                dtype=torch.float32,
            )

        return context, clf_token

    @torch.no_grad()
    def sample_sparse(
        self,
        vox_idxs: list,
        conditioning: LatDitConditioning,
        *,
        sampling: Optional[FlowMatchSamplingParams] = None,
        generator: Optional[torch.Generator] = None,
    ) -> list:
        if sampling is None:
            sampling = FlowMatchSamplingParams.from_yaml_section(self.model_config)

        device = self._runtime_device()
        self.device = device
        data_dict = conditioning.to_data_dict()
        data_dict["vox_idxs"] = vox_idxs
        data_dict["eval_mode"] = True
        if generator is not None:
            data_dict["generator"] = generator

        context, clf_token  = self._build_context(data_dict, tuple(self.cond), 0)

        coords = sp.sparse_coords(vox_idxs)
        gen = generator
        noise = torch.randn(
            len(coords),
            self.config.latent_dim,
            dtype=torch.float32,
            device=coords.device,
            generator=gen,
        )
        xt = sp.SparseTensor(noise, coords.int())

        cfg_range = sampling.cfg_range
        guidance_scale = sampling.guidance_scale
        t_seq = torch.linspace(
            1,
            0,
            sampling.num_inference_steps + 1,
            device=xt.device,
            dtype=xt.dtype,
        )
        rescale_t = sampling.rescale_t
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        batch_len = context.shape[0] if isinstance(context, torch.Tensor) else len(context)

        for t, t_next in zip(t_seq[:-1], t_seq[1:]):
            do_cfg = (guidance_scale > 0.0) and (cfg_range[0] <= t.item() <= cfg_range[1])
            pred = self.flow(xt, t.repeat_interleave(batch_len) * 1000, clf_token, context)

            if do_cfg:
                neg_pred = self.flow(
                    xt,
                    t.repeat_interleave(batch_len) * 1000,
                    clf_token,
                    torch.zeros_like(context),
                )
                pred = pred.replace(
                    pred.feats.float() + guidance_scale * (pred.feats.float() - neg_pred.feats.float())
                )

            xt = xt.replace(xt.feats - (t - t_next) * pred.feats)

        xt = self.denormalize(xt)
        return [xt.feats[lay] for lay in xt.layout]

    def forward(
        self,
        data_dict: Dict[str, Any],
        crit_dict: Optional[Dict[str, Any]] = None,
        total_iteration: int = 0,
        mode: str = "sample",
    ):
        crit_dict = crit_dict if crit_dict is not None else {}
        device = self._runtime_device()
        self.device = device
        cfg = self.config.cfg
        cond = (
            data_dict.get("eval_cond", self.cond)
            if data_dict.get("eval_mode", False)
            else [self.cond[total_iteration % len(self.cond)]]
        )
        context, clf_token = self._build_context(data_dict, tuple(cond), total_iteration)

        if data_dict.get("eval_mode", False) or mode == "sample":
            conditioning = LatDitConditioning(
                dino_features=data_dict["dino_features"],
                dino_token=data_dict.get("dino_token"),
                batch_size=data_dict.get("batch_size", 1),
                nfaces=data_dict.get("nfaces"),
            )
            sampling = FlowMatchSamplingParams.from_yaml_section(self.model_config)
            data_dict["vox_feats"] = self.sample_sparse(
                data_dict["vox_idxs"],
                conditioning,
                sampling=sampling,
                generator=data_dict.get("generator"),
            )
        else:
            batch_len = context.shape[0] if isinstance(context, torch.Tensor) else len(context)
            if cfg > 0.0:
                pos_mask = torch.rand(data_dict["batch_size"], device=context.device, dtype=torch.float32) < 0.9
                context = torch.where(expand_as(pos_mask, context), context, torch.zeros_like(context))

            sparse_coords = sp.sparse_coords(data_dict["vox_idxs"])
            sparse_feats = torch.cat(data_dict["vox_feats"])
            x0 = sp.SparseTensor(sparse_feats.float().detach(), sparse_coords)
            x0 = self.normalize(x0, train=True, alpha=1 - max(1 / (total_iteration + 1), 1e-4))
            noise = torch.randn_like(x0.feats)
            target = noise - x0.feats

            time_sched = self.config.time_schedule
            if time_sched == "lognorm":
                t = torch.sigmoid(torch.randn(batch_len, device=context.device, dtype=context.dtype))
            elif time_sched == "shape":
                t = torch.sigmoid(torch.randn(batch_len, device=context.device, dtype=context.dtype) + 1.0)
            else:
                t = torch.rand(batch_len, device=context.device, dtype=context.dtype)
            t = t.clamp(min=0.003)

            xt = x0.replace(
                torch.cat([noise[lay] * tval + x0.feats[lay] * (1 - tval) for lay, tval in zip(x0.layout, t)])
            )

            pred = self.flow(xt, t * 1000, clf_token, context)

            crit_dict["diffusion_loss"] = torch.nn.functional.mse_loss(pred.feats.float(), target)

        return data_dict, crit_dict

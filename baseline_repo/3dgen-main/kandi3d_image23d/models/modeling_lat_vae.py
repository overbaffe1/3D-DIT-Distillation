"""Native LatVaeModel (ModelMixin)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from kandi3d_image23d.configs.lat_vae_config import LatVaeConfig
from kandi3d_image23d.modeling_utils import (
    CompileMixin,
    GradientCheckpointingMixin,
    filter_state_dict_by_prefix,
    load_checkpoint_state,
    normalize_state_dict,
)
from kandi3d_image23d.modules.lat_vae.core import LatVaeCore
from kandi3d_image23d.runtime import LatVaeDecodeParams, MeshOutput, resolve

_LAT_VAE_INIT_FIELDS = (
    "res", "latent_dim", "up_dims", "extractor", "renderer",
    "render_res", "n_renders", "edge_sampling", "sdf_separate",
)


def _config_to_init_kwargs(config: LatVaeConfig) -> Dict[str, Any]:
    kw = {k: getattr(config, k) for k in _LAT_VAE_INIT_FIELDS}
    kw["model_config"] = dict(config.model_config)
    kw["rep_config"] = dict(config.rep_config)
    return kw


def _load_lat_vae_config(path: Path) -> LatVaeConfig:
    with open(path / "config.json", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("model_type") == "lat_vae":
        return LatVaeConfig.from_dict(data)
    from kandi3d_image23d.configuration_image23d import SubmoduleConfig

    return LatVaeConfig.from_geo_config(SubmoduleConfig.from_dict(data).geo_config)


class LatVaeModel(CompileMixin, GradientCheckpointingMixin, ModelMixin, ConfigMixin):
    """Sparse voxel VAE decoder path for inference (encoder dropped after load)."""

    COMPILE_TARGETS = ("decoder",)

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        res: int = 128,
        latent_dim: int = 16,
        up_dims: Optional[list] = None,
        extractor: str = "fc",
        renderer: str = "nvt",
        render_res: int = 1024,
        n_renders: int = 16,
        edge_sampling: bool = False,
        sdf_separate: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        rep_config: Optional[Dict[str, Any]] = None,
        geo_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        vae_config = LatVaeConfig(
            res=res,
            latent_dim=latent_dim,
            up_dims=up_dims,
            extractor=extractor,
            renderer=renderer,
            render_res=render_res,
            n_renders=n_renders,
            edge_sampling=edge_sampling,
            sdf_separate=sdf_separate,
            model_config=model_config or {},
            rep_config=rep_config or {},
        )
        full = geo_config if geo_config is not None else {
            "generators": {"lat_vae": vae_config.model_config},
            "representations": vae_config.rep_config,
        }
        self.core = LatVaeCore(vae_config, full_config=full, build_encoder=True)
        self._default_runtime = LatVaeDecodeParams()

    @classmethod
    def from_geo_config(cls, geo_config: Dict[str, Any]) -> "LatVaeModel":
        cfg = LatVaeConfig.from_geo_config(geo_config)
        model = cls(**_config_to_init_kwargs(cfg), geo_config=geo_config)
        model._default_runtime = LatVaeDecodeParams.from_geo_config(geo_config)
        return model

    def _drop_encoder(self) -> None:
        if hasattr(self.core, "encoder"):
            del self.core.encoder

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        device: Optional[Union[str, int, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "LatVaeModel":
        path = Path(pretrained_model_name_or_path)
        config = _load_lat_vae_config(path)
        model = cls(**_config_to_init_kwargs(config))

        weight_file = path / "diffusion_pytorch_model.safetensors"
        if weight_file.exists():
            from safetensors.torch import load_file

            state_dict = load_file(str(weight_file))
        else:
            bin_path = path / "diffusion_pytorch_model.bin"
            if not bin_path.exists():
                raise FileNotFoundError(f"No weights in {path}")
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)

        model.load_state_dict(normalize_state_dict(state_dict), strict=False)
        if device is not None:
            model.to(device)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        model._drop_encoder()
        model.eval()
        return model

    @classmethod
    def from_legacy_checkpoint(
        cls,
        ckpt_path: Union[str, Path],
        geo_config: Dict[str, Any],
        device: Union[int, str, torch.device] = "cuda",
        weights_only: bool = False,
        decoder_only: bool = False,
    ) -> "LatVaeModel":
        model = cls.from_geo_config(geo_config)
        state = normalize_state_dict(load_checkpoint_state(ckpt_path, weights_only=weights_only))
        if decoder_only:
            state = filter_state_dict_by_prefix(state, ("decoder.", "fc."))
        model.load_state_dict(state, strict=False)
        model.to(device)
        model._drop_encoder()
        model.eval()
        return model

    def state_dict(self, *args, **kwargs):
        return self.core.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, *args, **kwargs):
        from base_models.optimization_utils import fix_state_dict

        state_dict = normalize_state_dict(state_dict)
        state_dict = fix_state_dict(state_dict, self.core.state_dict())
        return self.core.load_state_dict(state_dict, strict=strict, *args, **kwargs)

    def forward(
        self,
        vox_feats: list,
        vox_idxs: list,
        *,
        runtime: Optional[LatVaeDecodeParams] = None,
        **runtime_kwargs: Any,
    ) -> MeshOutput:
        params = resolve(self._default_runtime, runtime, **runtime_kwargs)
        return self.core.decode_mesh(vox_feats, vox_idxs, runtime=params)

    def forward_legacy(
        self,
        data_dict: Dict[str, Any],
        crit_dict: Optional[Dict[str, Any]] = None,
        total_iteration: int = 0,
        target: str = "decode",
    ):
        return self.core.forward(data_dict, crit_dict, total_iteration, target)


__all__ = ["LatVaeModel"]
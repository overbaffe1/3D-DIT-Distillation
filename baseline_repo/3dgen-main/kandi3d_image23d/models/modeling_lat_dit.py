"""Native LatDitModel (ModelMixin)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from kandi3d_image23d.configs.lat_dit_config import LatDitConfig
from kandi3d_image23d.modeling_utils import CompileMixin, GradientCheckpointingMixin, load_checkpoint_state, normalize_state_dict
from kandi3d_image23d.modules.lat_dit.core import LatDitCore
from kandi3d_image23d.runtime import FlowMatchSamplingParams, LatDitConditioning, resolve
from kandi3d_image23d.lat_dit_protocol import LatDitGenerator

__all__ = ["LatDitModel"]

_LAT_DIT_INIT_FIELDS = (
    "dim", "latent_dim", "latent_scale", "encoder_pos_features",
    "focal_cond", "use_token", "stat_path", "time_schedule",
    "cfg", "cond",
)


def _config_to_init_kwargs(config: LatDitConfig) -> Dict[str, Any]:
    kw = {k: getattr(config, k) for k in _LAT_DIT_INIT_FIELDS}
    kw["model_config"] = dict(config.model_config)
    return kw


def _load_lat_dit_config(path: Path) -> LatDitConfig:
    with open(path / "config.json", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("model_type") == "lat_dit":
        return LatDitConfig.from_dict(data)
    from kandi3d_image23d.configuration_image23d import SubmoduleConfig

    return LatDitConfig.from_geo_config(SubmoduleConfig.from_dict(data).geo_config)


class LatDitModel(LatDitGenerator, CompileMixin, GradientCheckpointingMixin, ModelMixin, ConfigMixin):
    """Sparse voxel latent flow matching (DINO-conditioned)."""

    COMPILE_TARGETS = ("flow",)

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        dim: int = 1024,
        latent_dim: int = 8,
        latent_scale: float = 1.0,
        encoder_pos_features: int = 54,
        focal_cond: bool = False,
        use_token: bool = False,
        stat_path: Optional[str] = None,
        time_schedule: str = "lognorm",
        cfg: float = 5.0,
        cond: Optional[list] = None,
        model_config: Optional[Dict[str, Any]] = None,
        geo_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        dit_config = LatDitConfig(
            dim=dim,
            latent_dim=latent_dim,
            latent_scale=latent_scale,
            encoder_pos_features=encoder_pos_features,
            focal_cond=focal_cond,
            use_token=use_token,
            stat_path=stat_path,
            time_schedule=time_schedule,
            cfg=cfg,
            cond=cond,
            model_config=model_config or {},
        )
        full = geo_config if geo_config is not None else {
            "generators": {"lat_dit": dit_config.model_config},
        }
        self.core = LatDitCore(dit_config, full_config=full)
        self._default_runtime = FlowMatchSamplingParams.from_yaml_section(dit_config.model_config)

    @classmethod
    def from_geo_config(cls, geo_config: Dict[str, Any]) -> "LatDitModel":
        cfg = LatDitConfig.from_geo_config(geo_config)
        model = cls(**_config_to_init_kwargs(cfg), geo_config=geo_config)
        model._default_runtime = FlowMatchSamplingParams.from_yaml_section(
            geo_config.get("generators", {}).get("lat_dit", {})
        )
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        device: Optional[Union[str, int, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "LatDitModel":
        path = Path(pretrained_model_name_or_path)
        config = _load_lat_dit_config(path)
        model = cls(**_config_to_init_kwargs(config))

        weight_file = path / "diffusion_pytorch_model.safetensors"
        if weight_file.exists():
            from safetensors.torch import load_file

            state_dict = load_file(str(weight_file))
        else:
            bin_path = path / "diffusion_pytorch_model.bin"
            if not bin_path.exists():
                raise FileNotFoundError(f"No weights in {path}")
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

        model.load_state_dict(normalize_state_dict(state_dict), strict=False)
        if device is not None:
            model.to(device)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        model.eval()
        return model

    @classmethod
    def from_legacy_checkpoint(
        cls,
        ckpt_path: Union[str, Path],
        geo_config: Dict[str, Any],
        device: Union[int, str, torch.device] = "cuda",
        weights_only: bool = True,
    ) -> "LatDitModel":
        model = cls.from_geo_config(geo_config)
        state = normalize_state_dict(load_checkpoint_state(ckpt_path, weights_only=weights_only))
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model

    def state_dict(self, *args, **kwargs):
        return self.core.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, *args, **kwargs):
        from base_models.optimization_utils import fix_state_dict

        state_dict = normalize_state_dict(state_dict)
        state_dict = fix_state_dict(state_dict, self.core.state_dict())
        return self.core.load_state_dict(state_dict, strict=strict, *args, **kwargs)

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        if args:
            dev = args[0]
            self.core.device = (
                torch.device(dev) if isinstance(dev, (str, int)) else dev
            )
        return out

    def forward(
        self,
        vox_idxs: list,
        *,
        conditioning: LatDitConditioning,
        sampling: Optional[FlowMatchSamplingParams] = None,
        generator: Optional[torch.Generator] = None,
        **sampling_kwargs: Any,
    ) -> list:
        return self.generate(
            vox_idxs,
            conditioning=conditioning,
            sampling=sampling,
            generator=generator,
            **sampling_kwargs,
        )

    def generate_impl(
        self,
        vox_idxs: list,
        *,
        conditioning: LatDitConditioning,
        sampling: Optional[FlowMatchSamplingParams] = None,
        generator: Optional[torch.Generator] = None,
        **sampling_kwargs: Any,
    ) -> list:
        params = resolve(self._default_runtime, sampling, **sampling_kwargs)
        return self.core.sample_sparse(
            vox_idxs,
            conditioning,
            sampling=params,
            generator=generator,
        )

    def forward_legacy(
        self,
        data_dict: Dict[str, Any],
        crit_dict: Optional[Dict[str, Any]] = None,
        total_iteration: int = 0,
        mode: str = "sample",
    ):
        return self.core.forward(data_dict, crit_dict, total_iteration, mode)

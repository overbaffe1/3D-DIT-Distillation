"""Configuration for LatDit (sparse voxel flow matching)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from transformers import PretrainedConfig


class LatDitConfig(PretrainedConfig):
    model_type = "lat_dit"

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
        cond: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.latent_dim = latent_dim
        self.latent_scale = latent_scale
        self.encoder_pos_features = encoder_pos_features
        self.focal_cond = focal_cond
        self.use_token = use_token
        self.stat_path = stat_path
        self.time_schedule = time_schedule
        self.cfg = cfg
        self.cond = list(cond) if cond is not None else []
        self.model_config = dict(model_config or {})
        self._apply_model_config_fields()

    def _apply_model_config_fields(self) -> None:
        mc = self.model_config
        if not mc:
            return
        self.dim = mc.get("dim", self.dim)
        self.latent_dim = mc.get("latent_dim", self.latent_dim)
        self.latent_scale = mc.get("latent_scale", self.latent_scale)
        self.encoder_pos_features = mc.get("encoder_pos_features", self.encoder_pos_features)
        self.focal_cond = mc.get("focal_cond", self.focal_cond)
        self.use_token = mc.get("use_token", self.use_token)
        self.stat_path = mc.get("stat_path", self.stat_path)
        self.time_schedule = mc.get("time_schedule", self.time_schedule)
        self.cfg = mc.get("cfg", self.cfg)
        if "cond" in mc:
            self.cond = list(mc["cond"])

    @property
    def cond_tuple(self) -> Tuple[str, ...]:
        return tuple(self.cond)

    @classmethod
    def from_geo_config(cls, geo_config: Dict[str, Any]) -> "LatDitConfig":
        gen = dict(geo_config.get("generators", {}).get("lat_dit", {}))
        return cls(
            dim=gen.get("dim", 1024),
            latent_dim=gen.get("latent_dim", 8),
            latent_scale=gen.get("latent_scale", 1.0),
            encoder_pos_features=gen.get("encoder_pos_features", 54),
            focal_cond=gen.get("focal_cond", False),
            use_token=gen.get("use_token", False),
            stat_path=gen.get("stat_path"),
            time_schedule=gen.get("time_schedule", "lognorm"),
            cfg=gen.get("cfg", 5.0),
            cond=gen.get("cond", []),
            model_config=gen,
        )

"""Configuration for LatVae (sparse FlexiCubes VAE)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig


class LatVaeConfig(PretrainedConfig):
    model_type = "lat_vae"

    def __init__(
        self,
        res: int = 128,
        latent_dim: int = 16,
        up_dims: Optional[List[int]] = None,
        extractor: str = "fc",
        renderer: str = "nvt",
        render_res: int = 1024,
        n_renders: int = 16,
        edge_sampling: bool = False,
        sdf_separate: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
        rep_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.res = res
        self.latent_dim = latent_dim
        self.up_dims = list(up_dims) if up_dims is not None else [512, 256, 128, 64]
        self.extractor = extractor
        self.renderer = renderer
        self.render_res = render_res
        self.n_renders = n_renders
        self.edge_sampling = edge_sampling
        self.sdf_separate = sdf_separate
        self.model_config = dict(model_config or {})
        self.rep_config = dict(rep_config or {})
        self._apply_nested_config_fields()

    def _apply_nested_config_fields(self) -> None:
        mc = self.model_config
        if mc:
            self.res = mc.get("res", self.res)
            self.latent_dim = mc.get("latent_dim", self.latent_dim)
            if "up_dims" in mc:
                self.up_dims = list(mc["up_dims"])
        rep = self.rep_config
        if rep:
            self.extractor = rep.get("extractor", self.extractor)
            self.renderer = rep.get("renderer", self.renderer)
            self.render_res = rep.get("render_res", self.render_res)
            self.n_renders = rep.get("n_renders", self.n_renders)
            self.edge_sampling = rep.get("edge_sampling", self.edge_sampling)
            self.sdf_separate = rep.get("sdf_separate", self.sdf_separate)

    @classmethod
    def from_geo_config(cls, geo_config: Dict[str, Any]) -> "LatVaeConfig":
        gen = dict(geo_config.get("generators", {}).get("lat_vae", {}))
        rep = dict(geo_config.get("representations", {}))
        return cls(
            res=gen.get("res", 128),
            latent_dim=gen.get("latent_dim", 16),
            up_dims=list(gen.get("up_dims", [512, 256, 128, 64])),
            extractor=rep.get("extractor", "fc"),
            renderer=rep.get("renderer", "nvt"),
            render_res=rep.get("render_res", 1024),
            n_renders=rep.get("n_renders", 16),
            edge_sampling=rep.get("edge_sampling", False),
            sdf_separate=rep.get("sdf_separate", False),
            model_config=gen,
            rep_config=rep,
        )

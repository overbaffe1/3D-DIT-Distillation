"""Runtime / sampling parameters for Image23D inference (separate from architecture config)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Optional, Tuple, TypeVar

T = TypeVar("T", bound="RuntimeParams")


@dataclass(frozen=True)
class RuntimeParams:
    """Base runtime params with None-aware merge."""

    def merge(self: T, other: Optional[T] = None, **kwargs: Any) -> T:
        base = asdict(self)
        if other is not None:
            for k, v in asdict(other).items():
                if v is not None:
                    base[k] = v
        for k, v in kwargs.items():
            if v is not None and k in base:
                if k == "cfg_range" and isinstance(v, (list, tuple)):
                    base[k] = tuple(float(x) for x in v)
                else:
                    base[k] = v
        return type(self)(**base)


@dataclass(frozen=True)
class FlowMatchSamplingParams(RuntimeParams):
    guidance_scale: float = 2.5
    cfg_range: Tuple[float, float] = (0.5, 1.0)
    rescale_t: float = 3.0
    num_inference_steps: int = 50

    @classmethod
    def from_yaml_section(cls, section: Dict[str, Any]) -> "FlowMatchSamplingParams":
        cfg_range = section.get("cfg_range", [0.5, 1.0])
        if isinstance(cfg_range, (list, tuple)):
            cfg_range = tuple(float(x) for x in cfg_range)
        else:
            cfg_range = (0.5, 1.0)
        return cls(
            guidance_scale=float(section.get("cfg", section.get("guidance_scale", 2.5))),
            cfg_range=cfg_range,
            rescale_t=float(section.get("rescale_t", 3.0)),
            num_inference_steps=int(section.get("num_inference_steps", 50)),
        )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FlowMatchSamplingParams":
        if not data:
            return cls()
        cfg_range = data.get("cfg_range", (0.5, 1.0))
        if isinstance(cfg_range, list):
            cfg_range = tuple(cfg_range)
        return cls(
            guidance_scale=float(data.get("guidance_scale", 2.5)),
            cfg_range=tuple(cfg_range),
            rescale_t=float(data.get("rescale_t", 3.0)),
            num_inference_steps=int(data.get("num_inference_steps", 50)),
        )


@dataclass(frozen=True)
class PcdVaeDecodeParams(RuntimeParams):
    """Placeholder for future PcdVae decode-time runtime parameters."""

    @classmethod
    def from_geo_config(cls, geo_config: Dict[str, Any]) -> "PcdVaeDecodeParams":
        return cls()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PcdVaeDecodeParams":
        return cls()


@dataclass(frozen=True)
class PcdRepGeometryParams(RuntimeParams):
    max_chunk: Optional[int] = None
    hdecode_levels: int = 3

    @classmethod
    def from_geo_config(cls, geo_config: Dict[str, Any]) -> "PcdRepGeometryParams":
        gen = geo_config.get("generators", {}).get("pcd_vae", {})
        return cls(max_chunk=int(gen.get("max_rep_length", 262144)))

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PcdRepGeometryParams":
        if not data:
            return cls()
        return cls(
            max_chunk=data.get("max_chunk"),
            hdecode_levels=int(data.get("hdecode_levels", 3)),
        )


@dataclass(frozen=True)
class LatDitConditioning:
    """Image conditioning bundle for LatDit (replaces data_dict cond keys)."""

    dino_features: "torch.Tensor"
    dino_token: Optional["torch.Tensor"] = None
    batch_size: int = 1
    nfaces: Optional["torch.Tensor"] = None

    def to_data_dict(self) -> Dict[str, Any]:
        dd: Dict[str, Any] = {
            "dino_features": self.dino_features,
            "batch_size": self.batch_size,
        }
        if self.dino_token is not None:
            dd["dino_token"] = self.dino_token
        if self.nfaces is not None:
            dd["nfaces"] = self.nfaces
        return dd


@dataclass(frozen=True)
class LatVaeDecodeParams(RuntimeParams):
    """Placeholder for future LatVae decode-time runtime parameters."""

    @classmethod
    def from_geo_config(cls, geo_config: Dict[str, Any]) -> "LatVaeDecodeParams":
        return cls()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LatVaeDecodeParams":
        return cls()


@dataclass(frozen=True)
class MeshOutput:
    vertices: "torch.Tensor"
    triangles: "torch.Tensor"


@dataclass(frozen=True)
class Image23DRuntimeParams:
    """Pipeline-level runtime defaults for all stages."""

    pcd_dit: FlowMatchSamplingParams = field(default_factory=FlowMatchSamplingParams)
    pcd_vae: PcdVaeDecodeParams = field(default_factory=PcdVaeDecodeParams)
    pcd_rep: PcdRepGeometryParams = field(default_factory=PcdRepGeometryParams)
    lat_dit: FlowMatchSamplingParams = field(default_factory=FlowMatchSamplingParams)
    lat_vae: LatVaeDecodeParams = field(default_factory=LatVaeDecodeParams)

    @classmethod
    def from_geo_config(cls, geo_config: Dict[str, Any]) -> "Image23DRuntimeParams":
        gen = geo_config.get("generators", {})
        return cls(
            pcd_dit=FlowMatchSamplingParams.from_yaml_section(gen.get("pcd_dit", {})),
            pcd_vae=PcdVaeDecodeParams.from_geo_config(geo_config),
            pcd_rep=PcdRepGeometryParams.from_geo_config(geo_config),
            lat_dit=FlowMatchSamplingParams.from_yaml_section(gen.get("lat_dit", {})),
            lat_vae=LatVaeDecodeParams.from_geo_config(geo_config),
        )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Image23DRuntimeParams":
        if not data:
            return cls.from_geo_config({})
        return cls(
            pcd_dit=FlowMatchSamplingParams.from_dict(data.get("pcd_dit")),
            pcd_vae=PcdVaeDecodeParams.from_dict(data.get("pcd_vae")),
            pcd_rep=PcdRepGeometryParams.from_dict(data.get("pcd_rep")),
            lat_dit=FlowMatchSamplingParams.from_dict(data.get("lat_dit")),
            lat_vae=LatVaeDecodeParams.from_dict(data.get("lat_vae")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pcd_dit": asdict(self.pcd_dit),
            "pcd_vae": asdict(self.pcd_vae),
            "pcd_rep": asdict(self.pcd_rep),
            "lat_dit": asdict(self.lat_dit),
            "lat_vae": asdict(self.lat_vae),
        }

    def merge(
        self,
        other: Optional["Image23DRuntimeParams"] = None,
        **kwargs: Any,
    ) -> "Image23DRuntimeParams":
        base = self
        if other is not None:
            base = Image23DRuntimeParams(
                pcd_dit=base.pcd_dit.merge(other.pcd_dit),
                pcd_vae=base.pcd_vae.merge(other.pcd_vae),
                pcd_rep=base.pcd_rep.merge(other.pcd_rep),
                lat_dit=base.lat_dit.merge(other.lat_dit),
                lat_vae=base.lat_vae.merge(other.lat_vae),
            )
        updates: Dict[str, Any] = {}
        for stage in ("pcd_dit", "pcd_vae", "pcd_rep", "lat_dit", "lat_vae"):
            prefixed = {
                k[len(stage) + 1 :]: v
                for k, v in kwargs.items()
                if k.startswith(f"{stage}_") and v is not None
            }
            if prefixed:
                updates[stage] = getattr(base, stage).merge(None, **prefixed)
        if updates:
            base = replace(base, **updates)
        return base


def resolve(
    default: T,
    override: Optional[T] = None,
    **kwargs: Any,
) -> T:
    """Merge runtime params: default <- override <- kwargs (None values ignored)."""
    return default.merge(override, **kwargs)


def wire_module_runtimes(modules: Dict[str, Any], runtime: Image23DRuntimeParams) -> None:
    """Attach pipeline runtime defaults to each model submodule."""
    modules["pcd_dit"]._default_runtime = runtime.pcd_dit
    modules["pcd_vae"]._default_runtime = runtime.pcd_vae
    modules["pcd_rep"]._default_runtime = runtime.pcd_rep
    modules["lat_dit"]._default_runtime = runtime.lat_dit
    modules["lat_vae"]._default_runtime = runtime.lat_vae

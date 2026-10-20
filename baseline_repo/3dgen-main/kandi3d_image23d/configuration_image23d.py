"""Configuration for Image23D pipeline and sub-models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from transformers import PretrainedConfig

from kandi3d_image23d.runtime import Image23DRuntimeParams


def _to_serializable(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


class Image23DPipelineConfig(PretrainedConfig):
    model_type = "kandi3d_image23d"

    def __init__(
        self,
        geo_config: Optional[Dict[str, Any]] = None,
        fix_seed: bool = True,
        dmc_res: int = 512,
        dino2_model_id: str = "facebook/dinov2-giant",
        dino3_model_id: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        runtime: Optional[Union[Image23DRuntimeParams, Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.geo_config = geo_config or {}
        self.fix_seed = fix_seed
        self.dmc_res = dmc_res
        self.dino2_model_id = dino2_model_id
        self.dino3_model_id = dino3_model_id
        if isinstance(runtime, Image23DRuntimeParams):
            self.runtime = runtime
        elif isinstance(runtime, dict):
            self.runtime = Image23DRuntimeParams.from_dict(runtime)
        else:
            self.runtime = Image23DRuntimeParams.from_geo_config(self.geo_config)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], **overrides) -> "Image23DPipelineConfig":
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(yaml_path))
        for key, value in overrides.items():
            cfg[key] = value
        geo = OmegaConf.to_container(cfg, resolve=True)
        apply_inference_config_overrides(geo)
        return cls(geo_config=geo)

    @classmethod
    def from_geo_dict(cls, geo_config: Dict[str, Any], **kwargs) -> "Image23DPipelineConfig":
        return cls(geo_config=_to_serializable(geo_config), **kwargs)

    def get_submodule_config(self, section: str) -> Dict[str, Any]:
        """Return full geo_config dict expected by legacy BaseTrainable modules."""
        return dict(self.geo_config)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out["runtime"] = self.runtime.to_dict()
        return out

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config_dict = dict(config_dict)
        runtime = config_dict.pop("runtime", None)
        obj = super().from_dict(config_dict, **kwargs)
        if runtime is not None:
            obj.runtime = Image23DRuntimeParams.from_dict(runtime)
        elif not hasattr(obj, "runtime") or obj.runtime is None:
            obj.runtime = Image23DRuntimeParams.from_geo_config(obj.geo_config)
        return obj


class SubmoduleConfig(PretrainedConfig):
    """Per-component config stored in hub subfolders."""

    model_type = "kandi3d_image23d_submodule"

    def __init__(self, geo_config: Optional[Dict[str, Any]] = None, submodule_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.geo_config = geo_config or {}
        self.submodule_name = submodule_name

    @classmethod
    def from_pipeline_config(
        cls, pipeline_config: Image23DPipelineConfig, submodule_name: str
    ) -> "SubmoduleConfig":
        return cls(
            geo_config=pipeline_config.geo_config,
            submodule_name=submodule_name,
        )


def save_pipeline_config(config: Image23DPipelineConfig, save_directory: Union[str, Path]) -> None:
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(save_directory)
    with open(save_directory / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)


def apply_inference_config_overrides(geo_config: Dict[str, Any]) -> None:
    """Match runtime overrides from legacy pipeline_image23d.py."""
    generators = geo_config.setdefault("generators", {})
    lat_dit = generators.setdefault("lat_dit", {})
    lat_dit["mp"] = False


def load_pipeline_config(pretrained_model_name_or_path: Union[str, Path]) -> Image23DPipelineConfig:
    path = Path(pretrained_model_name_or_path)
    pipeline_json = path / "pipeline_config.json"
    if pipeline_json.exists():
        with open(pipeline_json, encoding="utf-8") as f:
            data = json.load(f)
        return Image23DPipelineConfig.from_dict(data)
    return Image23DPipelineConfig.from_pretrained(path)

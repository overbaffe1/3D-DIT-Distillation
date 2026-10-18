"""Hackathon participant kit: lat_dit track only (no pcd_* / full pipeline)."""

from kandi3d_image23d._repo_path import ensure_repo_root_on_path

ensure_repo_root_on_path()

from kandi3d_image23d.lat_dit_protocol import LatDitGenerator
from kandi3d_image23d.lat_dit_stage import LatDitStage, LatDitStageOutput
from kandi3d_image23d.models.modeling_lat_dit import LatDitModel
from kandi3d_image23d.models.modeling_lat_vae import LatVaeModel
from kandi3d_image23d.runtime import (
    FlowMatchSamplingParams,
    LatDitConditioning,
    MeshOutput,
)

__all__ = [
    "LatDitGenerator",
    "LatDitStage",
    "LatDitStageOutput",
    "LatDitModel",
    "LatVaeModel",
    "FlowMatchSamplingParams",
    "LatDitConditioning",
    "MeshOutput",
]

__version__ = "0.1.0-hackathon"

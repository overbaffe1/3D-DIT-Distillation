from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union

import torch
import trimesh

from base_models.mesh_utils import postprocess_mesh
from kandi3d_image23d.lat_dit_protocol import LatDitGenerator
from kandi3d_image23d.modeling_utils import stage_autocast
from kandi3d_image23d.models.modeling_lat_vae import LatVaeModel
from kandi3d_image23d.runtime import FlowMatchSamplingParams, LatDitConditioning


def rotate_vertices(verts: torch.Tensor) -> torch.Tensor:
    # Keep identical to `pipeline_image23d.rotate_vertices()` for consistent orientation.
    rot = torch.tensor(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        device=verts.device,
        dtype=torch.float32,
    )
    verts = verts @ rot.transpose(0, 1)
    rot = torch.tensor(
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        device=verts.device,
        dtype=torch.float32,
    )
    verts = verts @ rot.transpose(0, 1)
    return verts


@dataclass(frozen=True)
class LatDitStageOutput:
    vox_feats: list
    lat_dit_stage: float
    run_metadata: Dict[str, Any]
    mesh: Optional[trimesh.Trimesh] = None


class LatDitStage:
    """
    Isolated "lat_dit track" runner.

    Inputs are pre-cached:
      - `vox_idxi` (sparse voxel indices)
      - `dinov3_feats` and `dinov3_token` (DINOv3 conditioning)
    """

    def __init__(
        self,
        *,
        lat_dit: LatDitGenerator,
        lat_vae: Optional[LatVaeModel] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        fix_seed: bool = True,
        default_sampling: Optional[FlowMatchSamplingParams] = None,
    ):
        self.lat_dit = lat_dit
        self.lat_vae = lat_vae
        self.fix_seed = fix_seed
        self._device = torch.device(device) if device is not None else None
        self._default_sampling = default_sampling

    @classmethod
    def from_pretrained(
        cls,
        lat_dit_dir: str,
        *,
        device: Optional[Union[torch.device, str, int]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        lat_vae_dir: Optional[str] = None,
        generator_cls: Optional[Type[LatDitGenerator]] = None,
        lat_vae_cls: Type[LatVaeModel] = LatVaeModel,
        fix_seed: bool = True,
    ) -> "LatDitStage":
        # Default generator is the repo baseline model.
        from kandi3d_image23d.models.modeling_lat_dit import LatDitModel

        generator_cls = generator_cls or LatDitModel
        lat_dit = generator_cls.from_pretrained(
            lat_dit_dir,
            device=device,
            torch_dtype=torch_dtype,
        )
        if not isinstance(lat_dit, LatDitGenerator):
            raise TypeError(
                f"{generator_cls.__name__}.from_pretrained() must return a LatDitGenerator"
            )
        lat_vae = None
        if lat_vae_dir is not None:
            lat_vae = lat_vae_cls.from_pretrained(
                lat_vae_dir,
                device=device,
                torch_dtype=torch_dtype,
            )

        default_sampling = getattr(lat_dit, "_default_runtime", None)
        return cls(
            lat_dit=lat_dit,
            lat_vae=lat_vae,
            device=device,
            fix_seed=fix_seed,
            default_sampling=default_sampling,
        )

    def set_generator(self, generator: LatDitGenerator) -> None:
        if not isinstance(generator, LatDitGenerator):
            raise TypeError("generator must inherit LatDitGenerator")
        self.lat_dit = generator

    @torch.no_grad()
    def run(
        self,
        *,
        vox_idxi: torch.Tensor,  # [N, 3] int
        dinov3_feats: torch.Tensor,  # [1, T, 1280] float
        dinov3_token: Optional[torch.Tensor] = None,  # [1, 1280]
        generator: Optional[torch.Generator] = None,
        sampling: Optional[FlowMatchSamplingParams] = None,
        decode_mesh: bool = True,
        **sampling_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Returns a dict for CLI convenience or LatDitStageOutput for typed usage.
        """
        device = self._device
        if device is None:
            if hasattr(self.lat_dit, "parameters"):
                try:
                    device = next(self.lat_dit.parameters()).device  # type: ignore[arg-type]
                except StopIteration:
                    device = vox_idxi.device
            else:
                device = vox_idxi.device

        if sampling is None:
            sampling = self._default_sampling
        if sampling is None:
            sampling = FlowMatchSamplingParams()

        if self.fix_seed:
            generator = generator or torch.Generator(device).manual_seed(42)

        conditioning = LatDitConditioning(
            dino_features=dinov3_feats.to(device=device),
            dino_token=dinov3_token.to(device=device) if dinov3_token is not None else None,
            batch_size=1,
        )

        vox_idxs = [vox_idxi.to(device=device)]

        with stage_autocast("lat_dit", device):
            vox_feats = self.lat_dit.generate(
                vox_idxs,
                conditioning=conditioning,
                sampling=sampling,
                generator=generator,
                **sampling_kwargs,
            )
        run_metadata = self.lat_dit.last_run_metadata
        lat_dit_stage = float(run_metadata["elapsed_seconds"])

        mesh = None
        if decode_mesh and self.lat_vae is not None:
            with stage_autocast("lat_vae", device):
                mesh_out = self.lat_vae(vox_feats, vox_idxs)
            verts, faces = postprocess_mesh(
                mesh_out.vertices,
                mesh_out.triangles,
                n_faces=600000,
            )
            verts, faces = verts.float(), faces.int()
            verts = rotate_vertices(verts)

            mesh = trimesh.Trimesh(
                verts.detach().cpu().numpy(),
                faces.detach().cpu().numpy(),
                process=False,
                validate=False,
            )

        return {
            "vox_feats": vox_feats,
            "lat_dit_stage": lat_dit_stage,
            "run_metadata": run_metadata,
            "mesh": mesh,
        }


__all__ = ["LatDitStage", "LatDitStageOutput"]


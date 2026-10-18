"""Minimal participant stub wrapping baseline LatDitModel.

Usage:
  python scripts/hackathon/run_lat_dit_eval.py \
    --cache-dir /path/to/cache \
    --out-dir pred_out \
    --lat-dit-dir weights/lat_dit \
    --lat-vae-dir weights/lat_vae \
    --generator-impl examples/my_lat_dit_stub.py:MyLatDit
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from kandi3d_image23d import LatDitConditioning, LatDitGenerator
from kandi3d_image23d.models.modeling_lat_dit import LatDitModel
from kandi3d_image23d.runtime import FlowMatchSamplingParams


class MyLatDit(LatDitGenerator):
    def __init__(self, inner: LatDitModel):
        self.inner = inner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: Optional[Union[torch.device, str, int]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        inner = LatDitModel.from_pretrained(
            pretrained_model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        return cls(inner)

    def generate_impl(
        self,
        vox_idxs: list,
        *,
        conditioning: LatDitConditioning,
        sampling: Optional[FlowMatchSamplingParams] = None,
        generator: Optional[torch.Generator] = None,
        **sampling_kwargs,
    ) -> list:
        return self.inner.generate_impl(
            vox_idxs,
            conditioning=conditioning,
            sampling=sampling,
            generator=generator,
            **sampling_kwargs,
        )

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
import time
from typing import Any, Dict, Optional, Union

import torch

from kandi3d_image23d.runtime import FlowMatchSamplingParams, LatDitConditioning


_OMIT = object()


def _json_safe(value: Any) -> Any:
    """Convert serializable values and omit tensors/opaque runtime objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            converted = _json_safe(item)
            if converted is not _OMIT:
                result[str(key)] = converted
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            converted = _json_safe(item)
            if converted is not _OMIT:
                result.append(converted)
        return result
    return _OMIT


class LatDitGenerator(ABC):
    """
    Minimal contract for the hackathon "lat_dit track".

    Participants implement this interface so their model can be swapped in without touching the rest
    of the pipeline (cached voxels + cached DINOv3 conditioning + fixed lat_vae decoder).

    `generate()` is a mandatory, non-overridable wrapper. Implement `generate_impl()` only.
    The wrapper synchronizes CUDA, measures generation wall time and stores JSON-safe metadata in
    `last_run_metadata`. Cached voxel indices and DINO tensors are never recorded.
    """

    last_run_metadata: Dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "generate" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override LatDitGenerator.generate(); "
                "implement generate_impl() instead"
            )

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: Optional[Union[torch.device, str, int]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "LatDitGenerator":
        raise NotImplementedError

    def generate(
        self,
        vox_idxs: list,
        *,
        conditioning: LatDitConditioning,
        sampling: Optional[FlowMatchSamplingParams] = None,
        generator: Optional[torch.Generator] = None,
        **sampling_kwargs,
    ) -> list:
        """
        Args:
          vox_idxs: list of tensors with shape [N, 3] (integer voxel indices).
          conditioning: DINOv3 conditioning bundle.
          sampling: Flow-matching sampling params.
          generator: optional RNG.

        Returns:
          list of sparse latent feature tensors compatible with `LatVaeModel`:
          each element corresponds to one batch entry and has shape [N, latent_dim].
        """

        resolved_sampling = sampling or getattr(
            self, "_default_runtime", FlowMatchSamplingParams()
        )
        serializable_kwargs = _json_safe(sampling_kwargs)
        metadata: Dict[str, Any] = {
            "generator_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "inputs": {
                "sampling": _json_safe(resolved_sampling),
                "sampling_kwargs": serializable_kwargs,
                "batch_size": int(conditioning.batch_size),
                "num_voxels": [int(vox.shape[0]) for vox in vox_idxs],
                "generator_seed": int(generator.initial_seed()) if generator is not None else None,
            },
        }

        cuda_device = None
        for vox in vox_idxs:
            if torch.is_tensor(vox) and vox.is_cuda:
                cuda_device = vox.device
                break

        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        started = time.perf_counter()
        try:
            result = self.generate_impl(
                vox_idxs,
                conditioning=conditioning,
                sampling=sampling,
                generator=generator,
                **sampling_kwargs,
            )
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            metadata["elapsed_seconds"] = float(time.perf_counter() - started)
            metadata["status"] = "ok"
            self.last_run_metadata = metadata
            return result
        except Exception as exc:
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            metadata["elapsed_seconds"] = float(time.perf_counter() - started)
            metadata["status"] = "error"
            metadata["error_type"] = type(exc).__name__
            self.last_run_metadata = metadata
            raise

    @abstractmethod
    def generate_impl(
        self,
        vox_idxs: list,
        *,
        conditioning: LatDitConditioning,
        sampling: Optional[FlowMatchSamplingParams] = None,
        generator: Optional[torch.Generator] = None,
        **sampling_kwargs: Any,
    ) -> list:
        """Generate sparse latents; timing and metadata are handled by `generate()`."""
        raise NotImplementedError


__all__ = ["LatDitGenerator"]


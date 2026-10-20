"""Shared utilities for diffusers ModelMixin wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_checkpoint_state(
    path: Union[str, Path], weights_only: bool = True
) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=weights_only)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def filter_state_dict_by_prefix(
    state_dict: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]
) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if k.startswith(prefixes)}


class CompileMixin:
    """Runtime torch.compile toggles on core submodules (off by default)."""

    COMPILE_TARGETS: Tuple[str, ...] = ()

    def compile(
        self,
        targets: Optional[Tuple[str, ...]] = None,
        mode: str = "default",
        **kwargs: Any,
    ) -> None:
        if not hasattr(self, "_compile_originals"):
            self._compile_originals: Dict[str, Any] = {}
        core = self.core
        for name in targets or self.COMPILE_TARGETS:
            if name in self._compile_originals:
                continue
            original = getattr(core, name)
            self._compile_originals[name] = original
            setattr(core, name, torch.compile(original, mode=mode, **kwargs))

    def uncompile(self, targets: Optional[Tuple[str, ...]] = None) -> None:
        originals: Dict[str, Any] = getattr(self, "_compile_originals", {})
        if not originals:
            return
        core = self.core
        names = list(targets) if targets is not None else list(originals.keys())
        for name in names:
            if name not in originals:
                continue
            setattr(core, name, originals.pop(name))
        if not originals:
            self._compile_originals = {}

    @property
    def is_compiled(self) -> bool:
        return bool(getattr(self, "_compile_originals", {}))

    def compiled_targets(self) -> Tuple[str, ...]:
        return tuple(getattr(self, "_compile_originals", {}).keys())


class GradientCheckpointingMixin:
    """Map diffusers enable_gradient_checkpointing() to legacy .gc flags."""

    _supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Optional[Callable] = None,
    ) -> None:
        if gradient_checkpointing_func is None:
            gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

        is_set = False
        for _, module in self.named_modules():
            if hasattr(module, "gc"):
                module.gc = enable
                is_set = True
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = enable
                is_set = True
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_set = True

        if not is_set:
            raise ValueError(
                f"{self.__class__.__name__} does not contain modules that support gradient checkpointing."
            )


STAGE_COMPUTE_DTYPES: Dict[str, torch.dtype] = {
    "pcd_dit": torch.bfloat16,
    "lat_dit": torch.bfloat16,
    "pcd_vae": torch.float16,
    "pcd_rep": torch.float16,
    "lat_vae": torch.float16,
}


def stage_autocast(stage: str, device: Union[str, torch.device], enabled: bool = True):
    """Autocast context for a pipeline stage (dtype chosen per stage)."""
    dtype = STAGE_COMPUTE_DTYPES[stage]
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    use_autocast = enabled and device_type == "cuda" and dtype != torch.float32
    return torch.autocast(device_type=device_type, enabled=use_autocast, dtype=dtype)

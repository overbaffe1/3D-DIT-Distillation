"""Ensure repo root is on sys.path before legacy `models` / `base_models` imports."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
# If this directory is on sys.path, top-level `import models` shadows legacy code.
_PACKAGE_DIR_STR = str(_REPO_ROOT / "kandi3d_image23d")
_LEGACY_MODELS_SHADOW = "/kandi3d_image23d/models/"


def repo_root() -> Path:
    return _REPO_ROOT


def ensure_repo_root_on_path() -> Path:
    """Prefer repo-root legacy packages over kandi3d_image23d/models on sys.path."""
    if _PACKAGE_DIR_STR in sys.path:
        sys.path.remove(_PACKAGE_DIR_STR)

    if sys.path and sys.path[0] == _REPO_ROOT_STR:
        pass
    else:
        if _REPO_ROOT_STR in sys.path:
            sys.path.remove(_REPO_ROOT_STR)
        sys.path.insert(0, _REPO_ROOT_STR)

    models_mod = sys.modules.get("models")
    if models_mod is not None:
        mod_file = (getattr(models_mod, "__file__", None) or "").replace("\\", "/")
        if _LEGACY_MODELS_SHADOW in mod_file:
            for key in list(sys.modules):
                if key == "models" or key.startswith("models."):
                    del sys.modules[key]

    return _REPO_ROOT

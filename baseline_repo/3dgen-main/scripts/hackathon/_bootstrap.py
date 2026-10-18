"""Put repo root on sys.path so these scripts can run from any working directory."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_root_on_path() -> Path:
    root = str(REPO_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return REPO_ROOT


ensure_repo_root_on_path()

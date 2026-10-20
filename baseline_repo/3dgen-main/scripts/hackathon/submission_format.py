"""Fixed submission layout for the lat_dit hackathon track.

Participants send a zip. Organizers unpack to a fixed folder and run eval.

Recognized names (anything else is ignored):

    solution.py     # class `Solution` — if missing, baseline LatDitModel
    weights/        # non-empty — if missing, organizer --lat-dit-dir
    lib/            # optional extra Python modules

A single top-level wrapper directory in the zip is allowed.
Forbidden at submission root: `kandi3d_image23d/`, `scripts/` (hard fail).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from zipfile import ZipFile, is_zipfile

SOLUTION_FILE = "solution.py"
SOLUTION_CLASS = "Solution"
WEIGHTS_DIR = "weights"
OPTIONAL_LIB_DIR = "lib"
FORBIDDEN_ROOT_DIRS = ("kandi3d_image23d", "scripts")


@dataclass
class SubmissionLayout:
    root: Path
    solution_py: Optional[Path]
    weights_dir: Optional[Path]
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Fatal layout ok (forbidden paths / not a directory). Missing code/weights is allowed."""
        return not self.errors

    def generator_impl_spec(self) -> str:
        if self.solution_py is None:
            raise FileNotFoundError(SOLUTION_FILE)
        return f"{self.solution_py}:{SOLUTION_CLASS}"

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise FileNotFoundError("Invalid submission layout:\n" + "\n".join(f"- {e}" for e in self.errors))


def _logical_root_from_names(names: Sequence[str]) -> str:
    rels = []
    for raw in names:
        name = raw.replace("\\", "/").strip("/")
        if not name or name.endswith("/"):
            continue
        rels.append(name)
    if not rels:
        return ""
    tops = {p.split("/", 1)[0] for p in rels}
    if len(tops) == 1:
        top = next(iter(tops))
        if all(p == top or p.startswith(top + "/") for p in rels):
            has_nested = any("/" in p for p in rels)
            looks_like_file = "." in top
            if has_nested and not looks_like_file:
                return top + "/"
    return ""


def _has_file(names: Iterable[str], root_prefix: str, filename: str) -> bool:
    target = f"{root_prefix}{filename}"
    return any(n.replace("\\", "/").strip("/") == target.rstrip("/") for n in names)


def _has_dir_with_files(names: Iterable[str], root_prefix: str, dirname: str) -> bool:
    prefix = f"{root_prefix}{dirname}/"
    for raw in names:
        name = raw.replace("\\", "/").strip("/")
        if name.startswith(prefix) and name != prefix.rstrip("/"):
            return True
    return False


def inspect_zip(path: Path) -> SubmissionLayout:
    errors: List[str] = []
    notes: List[str] = []
    with ZipFile(path) as zf:
        names = zf.namelist()
    prefix = _logical_root_from_names(names)
    has_code = _has_file(names, prefix, SOLUTION_FILE)
    has_weights = _has_dir_with_files(names, prefix, WEIGHTS_DIR)
    if not has_code:
        notes.append(f"missing {prefix}{SOLUTION_FILE}; eval will use baseline LatDitModel")
    if not has_weights:
        notes.append(f"missing non-empty {prefix}{WEIGHTS_DIR}/; eval will use organizer lat_dit weights")
    for forbidden in FORBIDDEN_ROOT_DIRS:
        if _has_dir_with_files(names, prefix, forbidden) or _has_file(names, prefix, forbidden):
            errors.append(f"forbidden path {prefix}{forbidden}/ (pipeline is provided by organizers)")
    return SubmissionLayout(
        root=path,
        solution_py=None,
        weights_dir=None,
        errors=errors,
        notes=notes,
    )


def inspect_dir(root: Path) -> SubmissionLayout:
    errors: List[str] = []
    notes: List[str] = []
    if not root.is_dir():
        return SubmissionLayout(
            root=root,
            solution_py=None,
            weights_dir=None,
            errors=[f"not a directory: {root}"],
        )

    solution = root / SOLUTION_FILE
    weights = root / WEIGHTS_DIR
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    files = [p for p in root.iterdir() if p.is_file()]
    if len(subdirs) == 1 and not files:
        child = subdirs[0]
        if (child / SOLUTION_FILE).is_file() or (child / WEIGHTS_DIR).is_dir():
            root = child
            solution = root / SOLUTION_FILE
            weights = root / WEIGHTS_DIR

    if solution.is_file():
        solution_path = solution.resolve()
    else:
        solution_path = None
        notes.append(f"missing {SOLUTION_FILE}; eval will use baseline LatDitModel")

    if weights.is_dir() and any(weights.rglob("*")):
        weights_path = weights.resolve()
    else:
        weights_path = None
        notes.append(f"missing non-empty {WEIGHTS_DIR}/; eval will use organizer lat_dit weights")

    for forbidden in FORBIDDEN_ROOT_DIRS:
        if (root / forbidden).exists():
            errors.append(f"forbidden path {forbidden}/ (pipeline is provided by organizers)")

    return SubmissionLayout(
        root=root.resolve(),
        solution_py=solution_path,
        weights_dir=weights_path,
        errors=errors,
        notes=notes,
    )


def inspect_submission(path: Path) -> SubmissionLayout:
    path = path.expanduser().resolve()
    if path.is_file() and (path.suffix.lower() == ".zip" or is_zipfile(path)):
        return inspect_zip(path)
    return inspect_dir(path)

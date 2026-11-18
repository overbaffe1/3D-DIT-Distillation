#!/usr/bin/env python3
"""Validate the portfolio documentation without third-party packages.

Checks:
  * local Markdown/HTML/CSS/SVG links resolve to files in the repository;
  * generated SVGs are well-formed XML;
  * the asset manifest matches the files on disk;
  * the static page contains no localhost callbacks.
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets" / "portfolio"
MARKDOWN = [ROOT / "README.md", ROOT / ".github" / "pull_request_template.md", *sorted(DOCS.glob("*.md"))]
HTML_FILES = sorted(DOCS.glob("*.html"))
CSS_FILES = sorted(DOCS.glob("*.css"))
SVG_FILES = sorted(ASSETS.glob("*.svg"))

IGNORED_PREFIXES = ("http://", "https://", "mailto:", "data:", "//", "#", "tel:")


def ignored(target: str) -> bool:
    t = target.strip()
    if not t:
        return True
    if t.startswith(IGNORED_PREFIXES):
        return True
    # LaTeX fragments and shell snippets are not paths.
    if "\\" in t or "$" in t or " " in t and not t.startswith("./"):
        return True
    return False


def strip_anchor(target: str) -> str:
    return target.split("#", 1)[0]


def markdown_links(text: str) -> list[str]:
    """Extract link targets, balancing parentheses inside URLs."""
    out: list[str] = []
    i = 0
    while True:
        j = text.find("](", i)
        if j < 0:
            break
        start = j + 2
        depth = 1
        k = start
        while k < len(text) and depth:
            ch = text[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            k += 1
        if depth:
            break
        raw = text[start : k - 1].strip()
        # Optional Markdown title: take the first whitespace-separated token.
        target = raw.split()[0] if raw else ""
        if target:
            out.append(target)
        i = k
    return out


class HtmlLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k: v or "" for k, v in attrs}
        for attr in ("href", "src"):
            if attr in d:
                self.targets.append(d[attr])
        if "srcset" in d:
            for item in d["srcset"].split(","):
                parts = item.strip().split()
                if parts:
                    self.targets.append(parts[0])


def check_path(target: str, source: Path, errors: list[str], seen: set[Path]) -> None:
    if ignored(target):
        return
    clean = strip_anchor(target)
    if not clean:
        return
    path = (source.parent / clean).resolve()
    if not path.is_relative_to(ROOT):
        errors.append(f"{source.relative_to(ROOT)}: link escapes repository: {target}")
        return
    seen.add(path)
    if not path.exists():
        errors.append(f"{source.relative_to(ROOT)}: missing link target: {target} → {path.relative_to(ROOT)}")


SVG_NS = "{http://www.w3.org/2000/svg}"


def check_svg_layout(errors: list[str]) -> None:
    """Catch the two visual regressions that motivated the second portfolio PR."""
    for name in ("cover-light.svg", "cover-dark.svg"):
        path = ASSETS / name
        root = ElementTree.parse(path).getroot()
        rects = root.findall(f".//{SVG_NS}rect")
        panels = [r for r in rects if r.get("x") == "34" and r.get("y") == "30"]
        if not panels:
            errors.append(f"{name}: cover panel rect not found")
            continue
        panel = panels[0]
        panel_bottom = float(panel.get("y", "0")) + float(panel.get("height", "0"))
        for rect in rects:
            if rect.get("rx") != "14":
                continue
            chip_bottom = float(rect.get("y", "0")) + float(rect.get("height", "0"))
            if chip_bottom > panel_bottom - 8:
                errors.append(f"{name}: chip card crosses the panel border ({chip_bottom:.0f} vs {panel_bottom:.0f})")

    for name in ("timeline-light.svg", "timeline-dark.svg"):
        path = ASSETS / name
        root = ElementTree.parse(path).getroot()
        horizontal = [
            line for line in root.findall(f".//{SVG_NS}line")
            if line.get("y1") == line.get("y2") and float(line.get("y1", "0")) > 250
        ]
        left_labels = [
            text for text in root.findall(f".//{SVG_NS}text")
            if float(text.get("x", "0")) < 300 and 230 < float(text.get("y", "0")) < 310
        ]
        for line in horizontal:
            y = float(line.get("y1", "0"))
            for text in left_labels:
                if abs(y - float(text.get("y", "0"))) < 18:
                    errors.append(f"{name}: timeline line at y={y:.0f} overlaps label at y={float(text.get('y', '0')):.0f}")


def main() -> int:
    errors: list[str] = []
    seen: set[Path] = set()

    for md in MARKDOWN:
        if not md.is_file():
            continue
        for target in markdown_links(md.read_text(encoding="utf-8")):
            check_path(target, md, errors, seen)

    for html in HTML_FILES:
        parser = HtmlLinks()
        parser.feed(html.read_text(encoding="utf-8"))
        for target in parser.targets:
            check_path(target, html, errors, seen)
        text = html.read_text(encoding="utf-8")
        if "localhost" in text or "127.0.0.1" in text:
            errors.append(f"{html.relative_to(ROOT)}: browser-facing markup must not contain localhost callbacks")

    for css in CSS_FILES:
        for target in re.findall(r"url\(([^)]+)\)", css.read_text(encoding="utf-8")):
            check_path(target.strip("'\" "), css, errors, seen)
        if "localhost" in css.read_text(encoding="utf-8") or "127.0.0.1" in css.read_text(encoding="utf-8"):
            errors.append(f"{css.relative_to(ROOT)}: CSS must not contain localhost callbacks")

    for svg in SVG_FILES:
        try:
            ElementTree.parse(svg)
        except ElementTree.ParseError as exc:
            errors.append(f"{svg.relative_to(ROOT)}: malformed SVG XML: {exc}")

    try:
        check_svg_layout(errors)
    except ElementTree.ParseError as exc:
        errors.append(f"SVG layout check could not parse assets: {exc}")

    manifest_path = ASSETS / "manifest.json"
    if not manifest_path.is_file():
        errors.append("docs/assets/portfolio/manifest.json is missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = set(manifest.get("assets", []))
        on_disk = {p.name for p in ASSETS.iterdir() if p.is_file() and p.name != "manifest.json"}
        missing = sorted(on_disk - listed)
        stale = sorted(listed - on_disk)
        if missing:
            errors.append(f"manifest.json does not list: {', '.join(missing)}")
        if stale:
            errors.append(f"manifest.json lists missing files: {', '.join(stale)}")
        for source in manifest.get("sources", []):
            if not (ROOT / source).is_file():
                errors.append(f"manifest source missing: {source}")

    if not SVG_FILES:
        errors.append("no SVG assets found; run `make docs-assets`")

    if errors:
        print("docs-check FAILED")
        for err in errors:
            print("  -", err)
        return 1

    print(f"docs-check OK: {len(seen)} local targets, {len(SVG_FILES)} SVGs, {len(HTML_FILES)} HTML pages, layout checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

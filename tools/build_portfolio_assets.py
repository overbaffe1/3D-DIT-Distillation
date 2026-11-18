#!/usr/bin/env python3
"""Regenerate the portfolio case-study artwork from repository evidence.

The script deliberately uses only the Python standard library. All numbers are
read from checked-in artifacts, so the visuals cannot silently drift away from
the protocol they illustrate. ImageMagick is used opportunistically to export a
PNG social preview; the SVGs remain the source of truth.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "portfolio"
EVIDENCE_PATH = ROOT / "fetched_info" / "solver_experiments.json"
CONFIG_PATH = ROOT / "lib" / "solver_config.json"

EVIDENCE = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
EQ = EVIDENCE["equivalence"]
REF_NEG = float(EQ["ref_neg_sec"])
FAST_NEG = float(EQ["fast_neg_1token_sec"])
NEG_SPEEDUP = float(EQ["neg_speedup"])
NEG_DIFF = float(EQ["neg_1token_max_abs_diff"])
POS_DIFF = float(EQ["pos_max_abs_diff"])
STEPS = int(CONFIG.get("num_inference_steps", 4))
GUIDED = int(CONFIG.get("n_guided_steps", 3))
UNGUIDED = int(CONFIG.get("n_unguided_steps", 1))

PALETTES = {
    "dark": {
        "bg": "#0d1513",
        "panel": "#16211e",
        "panel2": "#1a2723",
        "ink": "#f3eee0",
        "dim": "#b3c1b8",
        "muted": "#718376",
        "line": "#3a4740",
        "grid": "#22312c",
        "mint": "#a6dec0",
        "brass": "#e3b57d",
        "blue": "#7fb2cf",
        "rose": "#d99079",
        "white": "#fffaf0",
    },
    "light": {
        "bg": "#f5f1e8",
        "panel": "#fffdf6",
        "panel2": "#eee8db",
        "ink": "#17231f",
        "dim": "#4e635a",
        "muted": "#7b8c82",
        "line": "#cdc5b4",
        "grid": "#e2dbcc",
        "mint": "#267a5c",
        "brass": "#a86422",
        "blue": "#28658a",
        "rose": "#9c4f3c",
        "white": "#fffdf6",
    },
}

FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"


def e(value: object) -> str:
    return escape(str(value))


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def comma(value: float, digits: int = 3) -> str:
    return num(value, digits).replace(".", ",")


def tidy(svg: str) -> str:
    """Remove trailing whitespace so `git diff --check` stays clean."""
    return "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"


def defs() -> str:
    return """
  <defs>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M 32 0 L 0 0 0 32" fill="none" stroke="{grid}" stroke-width="1" opacity="0.55"/>
    </pattern>
    <linearGradient id="cubeTop" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{mint}" stop-opacity="0.34"/>
      <stop offset="1" stop-color="{blue}" stop-opacity="0.14"/>
    </linearGradient>
    <linearGradient id="cubeLeft" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{mint}" stop-opacity="0.22"/>
      <stop offset="1" stop-color="{mint}" stop-opacity="0.05"/>
    </linearGradient>
    <linearGradient id="cubeRight" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{brass}" stop-opacity="0.26"/>
      <stop offset="1" stop-color="{rose}" stop-opacity="0.08"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{brass}"/>
    </marker>
    <marker id="arrowMint" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{mint}"/>
    </marker>
  </defs>
""".strip()


def doc(width: int, height: int, p: dict[str, str], title: str, desc: str, body: str) -> str:
    common = {
        "grid": p["grid"],
        "mint": p["mint"],
        "blue": p["blue"],
        "brass": p["brass"],
        "rose": p["rose"],
    }
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{e(title)}</title>
  <desc id="desc">{e(desc)}</desc>
  {defs().format(**common)}
  <rect width="{width}" height="{height}" fill="{p['bg']}"/>
  <rect width="{width}" height="{height}" fill="url(#grid)" opacity="0.5"/>
{body}
</svg>
"""


def txt(x: float, y: float, value: object, *, size: int = 18, fill: str, weight: int = 400,
        anchor: str = "start", spacing: float = 0.0, opacity: float = 1.0) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'letter-spacing="{spacing:.2f}" opacity="{opacity:.2f}">{e(value)}</text>')


def chip(x: float, y: float, w: float, h: float, label: str, value: str, p: dict[str, str],
         accent: str) -> str:
    return f"""
  <g>
    <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="14" fill="{p['panel']}" stroke="{p['line']}"/>
    <rect x="{x:.1f}" y="{y:.1f}" width="5" height="{h:.1f}" rx="2.5" fill="{accent}"/>
    {txt(x + 18, y + 24, label, size=12, fill=p['dim'], weight=600, spacing=1.2)}
    {txt(x + 18, y + 51, value, size=21, fill=p['ink'], weight=700)}
  </g>""".strip()


def cube(p: dict[str, str], x: float, y: float, scale: float = 1.0) -> str:
    """A small isometric voxel block with a single highlighted token."""
    def pt(px: float, py: float) -> str:
        return f"{x + px * scale:.1f},{y + py * scale:.1f}"

    top = " ".join([pt(0, 42), pt(120, 0), pt(220, 48), pt(100, 90)])
    left = " ".join([pt(0, 42), pt(100, 90), pt(100, 230), pt(0, 182)])
    right = " ".join([pt(100, 90), pt(220, 48), pt(220, 188), pt(100, 230)])
    lines = []
    for i in range(1, 5):
        t = i / 5
        lines.append(f'<line x1="{x + (120 * t) * scale:.1f}" y1="{y + (42 - 42 * t) * scale:.1f}" '
                     f'x2="{x + (100 + 120 * t) * scale:.1f}" y2="{y + (90 - 42 * t) * scale:.1f}" '
                     f'stroke="{p["mint"]}" stroke-opacity="0.22"/>')
        lines.append(f'<line x1="{x + (120 * t) * scale:.1f}" y1="{y + (42 + 6 * t) * scale:.1f}" '
                     f'x2="{x + (220 - 100 * t) * scale:.1f}" y2="{y + (188 - 140 * t) * scale:.1f}" '
                     f'stroke="{p["brass"]}" stroke-opacity="0.20"/>')
        lines.append(f'<line x1="{x:.1f}" y1="{y + (42 + 28 * i) * scale:.1f}" '
                     f'x2="{x + 100 * scale:.1f}" y2="{y + (90 + 28 * i) * scale:.1f}" '
                     f'stroke="{p["mint"]}" stroke-opacity="0.18"/>')
    cx = x + 110 * scale
    cy = y + 45 * scale
    return f"""
  <g>
    <polygon points="{top}" fill="url(#cubeTop)" stroke="{p['mint']}" stroke-opacity="0.60"/>
    <polygon points="{left}" fill="url(#cubeLeft)" stroke="{p['mint']}" stroke-opacity="0.42"/>
    <polygon points="{right}" fill="url(#cubeRight)" stroke="{p['brass']}" stroke-opacity="0.48"/>
    {''.join(lines)}
    <path d="M {cx:.1f} {cy:.1f} L {x + 100 * scale:.1f} {y + 230 * scale:.1f}" stroke="{p['brass']}" stroke-dasharray="5 7" stroke-opacity="0.45"/>
    <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{31 * scale:.1f}" fill="none" stroke="{p['brass']}" stroke-opacity="0.45" stroke-width="2"/>
    <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{18 * scale:.1f}" fill="{p['brass']}" stroke="{p['bg']}" stroke-width="3"/>
    {txt(cx, cy + 7 * scale, '1', size=int(20 * scale), fill=p['bg'], weight=800, anchor='middle')}
  </g>""".strip()


def cover(p: dict[str, str], variant: str) -> str:
    w, h = 1200, 470
    body = f"""
  <rect x="34" y="30" width="{w - 68}" height="{h - 60}" rx="26" fill="{p['panel']}" stroke="{p['line']}" opacity="0.92"/>
  {cube(p, 805, 88, 1.02)}
  {txt(72, 86, 'PORTFOLIO CASE  /  08', size=13, fill=p['brass'], weight=700, spacing=2.4)}
  {txt(72, 150, '3D-DIT', size=64, fill=p['ink'], weight=800, spacing=-1.5)}
  {txt(72, 210, 'DISTILLATION', size=43, fill=p['mint'], weight=700, spacing=1.0)}
  {txt(74, 256, 'Хакатонский трек lat_dit · Kandinsky3D', size=21, fill=p['dim'])}
  {txt(74, 288, 'Один токен вместо 4 101 в безусловной ветви CFG', size=19, fill=p['ink'], weight=600)}
  {txt(74, 314, 'Точная алгебра, Triton sm_80 и калиброванный поток', size=16, fill=p['muted'])}
  <g>
    <rect x="804" y="330" width="306" height="62" rx="31" fill="{p['panel2']}" stroke="{p['brass']}" stroke-opacity="0.65"/>
    {txt(957, 360, '4 101  →  1', size=23, fill=p['brass'], weight=800, spacing=1.0, anchor='middle')}
    {txt(957, 383, 'без аппроксимации', size=13, fill=p['dim'], anchor='middle')}
  </g>
  {chip(70, 330, 164, 62, 'CFG BRANCH', f'{comma(REF_NEG)} → {comma(FAST_NEG)} с', p, p['mint'])}
  {chip(246, 330, 150, 62, 'KERNELS', '6 · sm_80', p, p['blue'])}
  {chip(408, 330, 150, 62, 'TUNED SOLVER', f'50 → {STEPS} NFE', p, p['brass'])}
  {chip(570, 330, 176, 62, 'STACK', 'PyTorch · Triton', p, p['rose'])}
  {txt(w - 72, 86, 'CASE STUDY', size=12, fill=p['muted'], weight=700, spacing=2.0, anchor='end')}
"""
    return doc(w, h, p, "3D-DIT-Distillation — обложка портфолио-кейса", "Изометрический воксельный куб и ключевые показатели хакатонского решения.", body)


def one_token(p: dict[str, str], variant: str) -> str:
    w, h = 1200, 455
    dots = []
    for row in range(9):
        for col in range(13):
            cx = 92 + col * 27
            cy = 128 + row * 22
            dots.append(f'<circle cx="{cx}" cy="{cy}" r="5.4" fill="{p["mint"]}" opacity="0.42"/>')
    dots.append(f'<circle cx="{92 + 6 * 27}" cy="{128 + 4 * 22}" r="8" fill="{p["brass"]}" opacity="0.9"/>')
    bar_w = 420
    fast_w = max(16.0, bar_w * FAST_NEG / REF_NEG)
    body = f"""
  <rect x="30" y="28" width="540" height="398" rx="22" fill="{p['panel']}" stroke="{p['line']}"/>
  <rect x="630" y="28" width="540" height="398" rx="22" fill="{p['panel']}" stroke="{p['brass']}" stroke-opacity="0.62"/>
  {txt(58, 74, 'БЫЛО · БЕЗУСЛОВНАЯ ВЕТВЬ', size=13, fill=p['dim'], weight=700, spacing=1.5)}
  {txt(658, 74, 'СТАЛО · ОДИН НОСИТЕЛЬ', size=13, fill=p['brass'], weight=700, spacing=1.5)}
  {''.join(dots)}
  {txt(58, 336, '4 101 одинаковый zero-токен', size=19, fill=p['ink'], weight=650)}
  <rect x="58" y="356" width="{bar_w}" height="16" rx="8" fill="{p['mint']}" opacity="0.72"/>
  {txt(58, 397, f'{comma(REF_NEG)} с', size=15, fill=p['dim'])}
  {txt(478, 397, 'CPU-стенд', size=12, fill=p['muted'], anchor='end')}
  <circle cx="900" cy="205" r="76" fill="{p['brass']}" opacity="0.14"/>
  <circle cx="900" cy="205" r="48" fill="{p['brass']}" opacity="0.92"/>
  <circle cx="900" cy="205" r="66" fill="none" stroke="{p['brass']}" stroke-opacity="0.6" stroke-width="2"/>
  {txt(900, 214, '1', size=48, fill=p['bg'], weight=800, anchor='middle')}
  {txt(658, 336, '[1, 1, 1536] вместо [1, 4101, 1536]', size=19, fill=p['ink'], weight=650)}
  <rect x="658" y="356" width="{bar_w}" height="16" rx="8" fill="{p['line']}" opacity="0.65"/>
  <rect x="658" y="356" width="{fast_w:.1f}" height="16" rx="8" fill="{p['brass']}"/>
  {txt(658, 397, f'{comma(FAST_NEG)} с · {comma(NEG_SPEEDUP, 2)}× на ветви', size=15, fill=p['dim'])}
  {txt(1078, 397, 'max |Δ| ≈ {:.2e}'.format(NEG_DIFF), size=13, fill=p['muted'], anchor='end')}
  <path d="M 580 220 L 620 220" stroke="{p['brass']}" stroke-width="3" marker-end="url(#arrow)"/>
  {txt(600, 202, '→', size=26, fill=p['brass'], anchor='middle', weight=700)}
"""
    return doc(w, h, p, "Один токен вместо 4101", "Сравнение времени безусловной ветви CFG до и после точной конденсации.", body)


def flow_box(x: float, y: float, w: float, h: float, index: str, title: str,
             lines: list[str], p: dict[str, str], accent: str) -> str:
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="16" fill="{p["panel"]}" stroke="{p["line"]}"/>',
        f'<circle cx="{x + 26:.1f}" cy="{y + 28:.1f}" r="15" fill="{accent}" opacity="0.92"/>',
        txt(x + 26, y + 33, index, size=14, fill=p["bg"], weight=800, anchor="middle"),
        txt(x + 50, y + 34, title, size=16, fill=p["ink"], weight=700),
    ]
    for i, line in enumerate(lines):
        parts.append(txt(x + 20, y + 74 + i * 24, line, size=13, fill=p["dim"]))
    return "\n  " + "\n  ".join(parts)


def architecture(p: dict[str, str], variant: str) -> str:
    w, h = 1200, 500
    boxes = [
        (38, "Вход", ["кеш вокселей", "DINOv3-признаки", " frozen VAE"], p["blue"]),
        (272, "Геометрия", ["1 раз на сэмпл", "Hilbert-порядки", "RoPE + conv-индексы"], p["mint"]),
        (506, "Fast DiT", ["24 блока", "1-token uncond", "6 Triton-ядер"], p["brass"]),
        (740, "Солвер", [f"{STEPS} шага в конфиге", "граница CFG t=0.5", "QAOA-калибровка"], p["rose"]),
        (974, "Выход", ["sparse-латент", "LPIPS 4 вида", "log-time factor"], p["blue"]),
    ]
    parts = [
        f'<rect x="28" y="26" width="1144" height="448" rx="24" fill="{p["panel2"]}" stroke="{p["line"]}" opacity="0.62"/>',
        txt(52, 74, "КОНВЕЙЕР РЕШЕНИЯ", size=13, fill=p["brass"], weight=700, spacing=2.0),
        txt(1148, 74, "хакатонский контракт: generate() не переопределяется", size=13, fill=p["muted"], anchor="end"),
    ]
    for i, (x, title, lines, accent) in enumerate(boxes):
        parts.append(flow_box(x, 112, 188, 158, str(i + 1), title, lines, p, accent))
        if i < len(boxes) - 1:
            parts.append(f'<line x1="{x + 190:.1f}" y1="190" x2="{x + 228:.1f}" y2="190" stroke="{p["brass"]}" stroke-width="2.5" marker-end="url(#arrow)"/>')
    parts.append(f'<line x1="60" y1="320" x2="1140" y2="320" stroke="{p["line"]}" stroke-dasharray="6 8"/>')
    notes = [
        (60, "ТОЧНО", "сокращение ветви без аппроксимации", p["mint"]),
        (350, "ОДИН РАЗ", "статическая геометрия вне цикла диффузии", p["blue"]),
        (650, "A100", "AOT-проверка PTX для sm_80", p["brass"]),
        (930, "ПРОТОКОЛ", "CI-log + solver_experiments.json", p["rose"]),
    ]
    for x, label, note, accent in notes:
        parts.append(f'<circle cx="{x + 8}" cy="368" r="6" fill="{accent}"/>')
        parts.append(txt(x + 24, 373, label, size=13, fill=accent, weight=800, spacing=1.2))
        parts.append(txt(x + 24, 400, note, size=14, fill=p["dim"]))
    parts.append(txt(52, 450, "Время генератора измеряет обёртка LatDitGenerator; декодер и рендер в зачёт скорости не входят.", size=13, fill=p["muted"]))
    return doc(w, h, p, "Архитектура ускоренного lat_dit", "Пять этапов от организаторского кеша до sparse-латента и оценки.", "\n  " + "\n  ".join(parts))


def timeline(p: dict[str, str], variant: str) -> str:
    w, h = 1200, 420
    x0, x1 = 110, 1090
    boundary = x0 + (x1 - x0) * 0.75
    ticks = []
    for i in range(51):
        x = x0 + (x1 - x0) * i / 50
        ticks.append(f'<line x1="{x:.1f}" y1="160" x2="{x:.1f}" y2="172" stroke="{p["mint"]}" stroke-opacity="0.55"/>')
    student_ticks = []
    for i in range(STEPS + 1):
        x = x0 + (x1 - x0) * i / STEPS
        student_ticks.append(f'<line x1="{x:.1f}" y1="286" x2="{x:.1f}" y2="310" stroke="{p["brass"]}" stroke-width="4"/>')
        student_ticks.append(f'<circle cx="{x:.1f}" cy="292" r="7" fill="{p["brass"]}"/>')
    body = f"""
  <rect x="28" y="26" width="1144" height="368" rx="24" fill="{p['panel']}" stroke="{p['line']}"/>
  {txt(54, 70, 'ТРАЕКТОРИЯ ПОТОКА', size=13, fill=p['brass'], weight=700, spacing=2.0)}
  {txt(1146, 70, 't = 1 → 0 · CFG активен до t = 0.5', size=13, fill=p['muted'], anchor='end')}
  {txt(54, 112, 'Бейзлайн', size=17, fill=p['ink'], weight=700)}
  {txt(54, 138, '50 шагов Euler + CFG', size=14, fill=p['dim'])}
  <line x1="{x0}" y1="166" x2="{x1}" y2="166" stroke="{p['mint']}" stroke-width="3" opacity="0.65"/>
  {''.join(ticks)}
  {txt(54, 238, f'Настроенный солвер', size=17, fill=p['ink'], weight=700)}
  {txt(54, 264, f'{STEPS} шага · {GUIDED} guided + {UNGUIDED} unguided', size=14, fill=p['dim'])}
  <line x1="{x0}" y1="292" x2="{x1}" y2="292" stroke="{p['line']}" stroke-width="3"/>
  <line x1="{x0}" y1="292" x2="{boundary:.1f}" y2="292" stroke="{p['blue']}" stroke-width="5"/>
  <line x1="{boundary:.1f}" y1="292" x2="{x1}" y2="292" stroke="{p['rose']}" stroke-width="5"/>
  {''.join(student_ticks)}
  <line x1="{boundary:.1f}" y1="102" x2="{boundary:.1f}" y2="352" stroke="{p['brass']}" stroke-dasharray="7 8" stroke-opacity="0.72"/>
  {txt(boundary + 10, 116, 'ГРАНИЦА CFG · t = 0.5', size=12, fill=p['brass'], weight=700, spacing=1.0)}
  {txt(x0, 344, 'guided', size=13, fill=p['blue'], weight=700)}
  {txt(x1, 344, 'unguided', size=13, fill=p['rose'], weight=700, anchor='end')}
  {txt(x0, 376, f'В CPU-абляции сравнивались 12–20 шагов; {STEPS} — значение текущего lib/solver_config.json, а не результат того прогона.', size=12, fill=p['muted'])}
"""
    return doc(w, h, p, "Расписание солвера", "Сравнение 50-шагового бейзлайна и настроенного расписания с границей CFG.", body)


def social(p: dict[str, str]) -> str:
    w, h = 1280, 640
    body = f"""
  <rect x="40" y="40" width="1200" height="560" rx="30" fill="{p['panel']}" stroke="{p['line']}"/>
  {cube(p, 850, 170, 1.35)}
  {txt(86, 118, 'ХАКАТОН · LAT_DIT · PORTFOLIO CASE', size=18, fill=p['brass'], weight=700, spacing=2.0)}
  {txt(84, 220, 'Один токен', size=68, fill=p['ink'], weight=800)}
  {txt(84, 292, 'вместо 4 101', size=68, fill=p['mint'], weight=800)}
  {txt(88, 352, '3D-DIT-Distillation', size=28, fill=p['dim'], weight=600)}
  {txt(88, 396, 'Kandinsky3D · точная конденсация CFG', size=22, fill=p['ink'])}
  {txt(88, 430, 'Triton sm_80 · калиброванный поток', size=22, fill=p['ink'])}
  <rect x="86" y="486" width="420" height="62" rx="31" fill="{p['panel2']}" stroke="{p['brass']}" stroke-opacity="0.7"/>
  {txt(112, 526, f'{comma(REF_NEG)} с → {comma(FAST_NEG)} с', size=25, fill=p['brass'], weight=800)}
  {txt(530, 526, 'одна ветвь · CPU', size=15, fill=p['dim'])}
  {txt(1200, 568, 'overbaffe1 / GitHub', size=15, fill=p['muted'], anchor='end')}
"""
    return doc(w, h, p, "3D-DIT-Distillation — social preview", "Обложка для превью репозитория с ключевым замером.", body)


def write_assets() -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for variant, p in PALETTES.items():
        assets = {
            f"cover-{variant}.svg": cover(p, variant),
            f"one-token-{variant}.svg": one_token(p, variant),
            f"architecture-{variant}.svg": architecture(p, variant),
            f"timeline-{variant}.svg": timeline(p, variant),
        }
        for name, svg in assets.items():
            (OUT / name).write_text(tidy(svg), encoding="utf-8")
            written.append(name)
    (OUT / "social-preview.svg").write_text(tidy(social(PALETTES["dark"])), encoding="utf-8")
    written.append("social-preview.svg")
    return written


def write_manifest(assets: list[str]) -> None:
    manifest = {
        "generated_by": "tools/build_portfolio_assets.py",
        "sources": [
            "fetched_info/solver_experiments.json",
            "lib/solver_config.json",
        ],
        "evidence": {
            "ref_neg_sec": REF_NEG,
            "fast_neg_1token_sec": FAST_NEG,
            "neg_branch_speedup": NEG_SPEEDUP,
            "neg_1token_max_abs_diff": NEG_DIFF,
            "pos_max_abs_diff": POS_DIFF,
        },
        "solver": {
            "num_inference_steps": STEPS,
            "n_guided_steps": GUIDED,
            "n_unguided_steps": UNGUIDED,
        },
        "assets": sorted(assets),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def maybe_png() -> bool:
    """Render a PNG preview if ImageMagick is available.

    Some ImageMagick builds have no SVG delegate. In that case we still export a
    representative PNG with native MVG drawing commands, using DejaVu font files
    that are present on most Linux runners.
    """
    converter = shutil.which("convert")
    if not converter:
        return False
    src = OUT / "social-preview.svg"
    dst = OUT / "social-preview.png"
    try:
        result = subprocess.run(
            [converter, "-background", "none", str(src), str(dst)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and dst.is_file():
            return True
    except (OSError, subprocess.TimeoutExpired):
        pass

    p = PALETTES["dark"]
    ref = comma(REF_NEG)
    fast = comma(FAST_NEG)
    mvg = f"""push graphic-context
fill '{p['bg']}'
rectangle 0,0 1279,639
fill '{p['panel']}'
stroke '{p['line']}'
stroke-width 2
roundrectangle 40,40 1240,600 30,30
fill '#203a32'
stroke '{p['mint']}'
stroke-width 2
polygon 850,212 1012,170 1112,218 950,260
fill '#162a25'
polygon 850,212 950,260 950,400 850,352
fill '#3d3225'
stroke '{p['brass']}'
polygon 950,260 1112,218 1112,358 950,400
stroke '#5c8a76'
stroke-width 1
line 886,200 986,248
line 922,188 1022,236
line 958,176 1058,224
line 850,240 950,288
line 850,268 950,316
line 850,296 950,344
stroke '{p['brass']}'
stroke-width 1
line 986,248 1086,206
line 1022,236 1092,212
stroke '{p['brass']}'
stroke-width 2
fill none
circle 981,215 1017,215
fill '{p['brass']}'
stroke '{p['bg']}'
stroke-width 3
circle 981,215 1001,215
fill '{p['bg']}'
stroke none
font DejaVu-Sans-Bold
font-size 24
text 974,223 '1'
fill '{p['brass']}'
font DejaVu-Sans-Bold
font-size 18
text 86,118 'ХАКАТОН · LAT_DIT · PORTFOLIO CASE'
font-size 68
text 84,220 'Один токен'
fill '{p['mint']}'
text 84,292 'вместо 4 101'
fill '{p['dim']}'
font DejaVu-Sans
font-size 28
text 88,352 '3D-DIT-Distillation'
fill '{p['ink']}'
font-size 22
text 88,396 'Kandinsky3D · точная конденсация CFG'
text 88,430 'Triton sm_80 · калиброванный поток'
fill '{p['panel2']}'
stroke '{p['brass']}'
stroke-width 2
roundrectangle 86,486 506,548 31,31
fill '{p['brass']}'
stroke none
font DejaVu-Sans-Bold
font-size 25
text 112,526 '{ref} с → {fast} с'
fill '{p['dim']}'
font DejaVu-Sans
font-size 15
text 530,526 'одна ветвь · CPU'
fill '{p['muted']}'
font-size 15
text 900,568 'overbaffe1 / GitHub'
pop graphic-context
"""
    mvg_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".mvg", encoding="utf-8", delete=False) as handle:
            handle.write(mvg)
            mvg_path = handle.name
        result = subprocess.run(
            [converter, "-size", "1280x640", f"xc:{p['bg']}", "-draw", f"@{mvg_path}",
             "-depth", "8", f"PNG24:{dst}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        if mvg_path:
            try:
                os.unlink(mvg_path)
            except OSError:
                pass
    return result.returncode == 0 and dst.is_file()


def main() -> None:
    assets = write_assets()
    png = maybe_png()
    if (OUT / "social-preview.png").is_file():
        assets.append("social-preview.png")
    write_manifest(assets)
    print(f"Wrote {len(assets)} assets + manifest.json to {OUT.relative_to(ROOT)}")
    if png:
        print("Wrote social-preview.png (ImageMagick export)")
    else:
        print("social-preview.png skipped; social-preview.svg is the portable source")


if __name__ == "__main__":
    main()

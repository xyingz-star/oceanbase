#!/usr/bin/env python3
"""
CPU vs GPU K-means sweep — PNG 纯 Pillow 版（无 matplotlib 依赖）。

与 NMBKM megapanel **同款布局**的推荐入口见：
  plot_cpu_gpu_km_megapanel.py  → build_km_nmbkm_charts.plot_cpu_vs_gpu_km_megapanel（matplotlib）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent / "vectordb_bench" / "results" / "OceanBase"
CPU_DIR = BASE / "cpu_km_v1"
GPU_DIR = BASE / "gpu_km_v1"
OUT_PNG = SCRIPT_DIR / "cpu_gpu_km_comparison.png"

# Reference-style palette (white panel, black baseline, green alternate)
BG = (255, 255, 255)
GRID_MAJOR = (220, 220, 220)
GRID_MINOR = (240, 240, 240)
AXIS = (90, 90, 90)
TEXT = (35, 35, 35)
MUTED = (110, 110, 110)
ORANGE_LABEL = (200, 90, 20)  # speedup / relative rows in reference; we use for y-axis titles accent sparingly
CPU_LINE = (25, 25, 25)  # KM baseline black
GPU_LINE = (39, 174, 96)  # green — distinct from black

# log2(mult) axis range (matches 0.25 .. 4 * sqrt(N) sweep)
LOG2_X_LO = -2.5
LOG2_X_HI = 2.5


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _yrange(a: list[float], b: list[float]) -> tuple[float, float]:
    vals = [x for x in a + b if x == x]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        lo -= 0.05 * abs(lo or 1.0)
        hi += 0.05 * abs(hi or 1.0)
    else:
        pad = 0.05 * (hi - lo)
        lo -= pad
        hi += pad
    return lo, hi


def _lx(nlist: int, sqrt_n: float) -> float:
    m = nlist / sqrt_n
    return math.log2(m) if m > 0 else LOG2_X_LO


def _x_pix(lx: float, px0: int, px1: int) -> int:
    return int(px0 + (lx - LOG2_X_LO) / (LOG2_X_HI - LOG2_X_LO) * (px1 - px0))


def _draw_marker_circle(draw: ImageDraw.ImageDraw, x: int, y: int, r: int, fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=1)


def _draw_marker_square(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    hs = s // 2
    draw.rectangle([x - hs, y - hs, x + hs, y + hs], fill=fill, outline=outline, width=1)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    nlist: list[int],
    y_cpu: list[float],
    y_gpu: list[float],
    sqrt_n: float,
    subplot_title: str,
    row_metric: str,
    y_unit: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_small: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    is_bottom_row: bool,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=GRID_MAJOR, width=1)
    # Subplot title (reference: dataset — N · dim)
    draw.text((x0 + 6, y0 + 4), subplot_title, fill=TEXT, font=font_small)
    # Row metric (orange accent like reference secondary rows — subtle)
    draw.text((x0 + 6, y0 + 20), row_metric, fill=ORANGE_LABEL, font=font_small)

    pad_l, pad_r, pad_t, pad_b = 10, 10, 38, 28
    px0, px1 = x0 + pad_l, x1 - pad_r
    py0, py1 = y0 + pad_t, y1 - pad_b
    if px1 <= px0 or py1 <= py0:
        return

    ymin, ymax = _yrange(y_cpu, y_gpu)

    def vy(v: float) -> int:
        return int(py1 - (v - ymin) / (ymax - ymin) * (py1 - py0))

    # Grid: vertical at log2 mult ticks -2..2
    for lv in (-2, -1, 0, 1, 2):
        xx = _x_pix(float(lv), px0, px1)
        draw.line([(xx, py0), (xx, py1)], fill=GRID_MINOR, width=1)
    # Horizontal grid
    for t in (0.25, 0.5, 0.75):
        yy = int(py0 + t * (py1 - py0))
        draw.line([(px0, yy), (px1, yy)], fill=GRID_MINOR, width=1)

    lx_list = [_lx(n, sqrt_n) for n in nlist]
    pts_c = [(_x_pix(lx, px0, px1), vy(v)) for lx, v in zip(lx_list, y_cpu)]
    pts_g = [(_x_pix(lx, px0, px1), vy(v)) for lx, v in zip(lx_list, y_gpu)]

    if len(pts_c) >= 2:
        draw.line(pts_c, fill=CPU_LINE, width=2)
    for p in pts_c:
        _draw_marker_circle(draw, p[0], p[1], 4, fill=(255, 255, 255), outline=CPU_LINE)

    if len(pts_g) >= 2:
        draw.line(pts_g, fill=GPU_LINE, width=2)
    for p in pts_g:
        _draw_marker_square(draw, p[0], p[1], 7, fill=GPU_LINE, outline=(20, 100, 50))

    # Y label (unit)
    draw.text((x1 - 52, y0 + 6), y_unit, fill=MUTED, font=font_small)

    # X tick labels (bottom row only): 0.25 .. 4 as multiples of sqrt(N)
    if is_bottom_row:
        tick_labels = [("0.25", -2), ("0.5", -1), ("1", 0), ("2", 1), ("4", 2)]
        for lab, lv in tick_labels:
            xx = _x_pix(float(lv), px0, px1)
            w = int(draw.textlength(lab, font=font_small)) if hasattr(draw, "textlength") else 20
            draw.text((xx - w // 2, py1 + 4), lab, fill=MUTED, font=font_small)


def main() -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    from build_km_nmbkm_charts import DATASET_META, DATASET_ORDER, collect_cpu_gpu_sweep_payload

    payload, meta = collect_cpu_gpu_sweep_payload(CPU_DIR, GPU_DIR)
    if not payload or not any(payload.get(ds, {}).get("nlist") for ds in payload):
        raise SystemExit(
            "No comparable CPU/GPU pairs (need identical stages + rebuild-only). "
            f"meta={meta!r}"
        )

    datasets = [d for d in DATASET_ORDER if d in payload and payload[d]["nlist"]] + [
        d for d in sorted(payload.keys()) if d not in DATASET_ORDER and payload[d]["nlist"]
    ]

    cols = len(datasets)
    rows_m = 3
    left_margin = 128
    W = max(1680, left_margin + cols * 420)
    H = max(980, rows_m * 300 + 120)
    margin_top = 88
    margin_bot = 88
    title_block = 52
    inner_w = W - left_margin - 24
    inner_h = H - margin_top - margin_bot - title_block
    cell_w = inner_w // cols
    cell_h = inner_h // rows_m

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    font = _font(12)
    font_small = _font(11)
    font_title = _font(17)
    font_left = _font(12)

    draw.text(
        (left_margin, 16),
        "CPU K-means (in-process) vs GPU external K-means — IVF sweep",
        fill=TEXT,
        font=font_title,
    )
    note = (
        f"Fair compare: same task_label, identical stages, rebuild-only (optimize+search). "
        f"points={meta.get('pairs_used')} skipped={meta.get('skipped')}"
    )
    draw.text((left_margin, 40), note, fill=MUTED, font=font_small)
    draw.text(
        (left_margin, 58),
        "X-axis: log₂(nlist / √N)  ·  layout as NMBKM megapanel",
        fill=MUTED,
        font=font_small,
    )

    metrics = [
        ("Create index time (optimize_duration)", "opt_cpu", "opt_gpu", "s"),
        ("Recall@k", "rec_cpu", "rec_gpu", "recall"),
        ("Concurrent QPS", "qps_cpu", "qps_gpu", "QPS"),
    ]
    row_labels_left = [
        "Create index\n time",
        "Recall@k",
        "Concurrent\n QPS",
    ]

    for r, (mname, kc, kg, yu) in enumerate(metrics):
        ly = margin_top + title_block + r * cell_h + cell_h // 2 - 20
        draw.text((12, ly), row_labels_left[r], fill=TEXT, font=font_left)

    for r, (mname, kc, kg, yu) in enumerate(metrics):
        for c, ds in enumerate(datasets):
            d = payload[ds]
            N, dim = DATASET_META.get(ds, (0, 0))
            sqrt_n = math.sqrt(float(N)) if N else 1.0
            subtitle = f"{ds} — N={N:,} · dim={dim}"
            bx0 = left_margin + c * cell_w
            by0 = margin_top + title_block + r * cell_h
            box = (bx0 + 2, by0 + 2, bx0 + cell_w - 4, by0 + cell_h - 4)
            draw_panel(
                draw,
                box,
                d["nlist"],
                d[kc],
                d[kg],
                sqrt_n,
                subtitle,
                mname,
                yu,
                font,
                font_small,
                is_bottom_row=(r == rows_m - 1),
            )

    # Global X-axis caption under panels
    cap_y = H - margin_bot + 8
    draw.text(
        (left_margin, cap_y),
        "IVF nlist as mult of √N  (log₂ axis)   ·   ● CPU (Elkan)    ■ GPU (external worker)",
        fill=TEXT,
        font=font_small,
    )

    # Legend box (reference-style dual series)
    lx = left_margin + inner_w - 400
    ly = 44
    _draw_marker_circle(draw, lx + 6, ly + 6, 4, fill=(255, 255, 255), outline=CPU_LINE)
    draw.text((lx + 18, ly), "CPU (in-process Elkan)", fill=TEXT, font=font_small)
    _draw_marker_square(draw, lx + 200, ly + 6, 7, fill=GPU_LINE, outline=(20, 100, 50))
    draw.text((lx + 218, ly), "GPU (external worker)", fill=TEXT, font=font_small)

    img.save(OUT_PNG, format="PNG", optimize=True)
    print(f"Wrote {OUT_PNG} ({W}x{H}) datasets={datasets} meta={meta}")


if __name__ == "__main__":
    main()

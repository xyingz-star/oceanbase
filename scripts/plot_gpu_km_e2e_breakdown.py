#!/usr/bin/env python3
"""
Stacked bar chart: CREATE VECTOR INDEX full breakdown — CPU vs GPU on the same axes.

For each dataset, one subplot: at each nlist (same sweep params), **two stacked bars**
side by side — left = CPU (Elkan in-process), right = GPU (external worker + flash-kmeans).

CPU segments (cpu_km_v1/kmeans*.log + JSON):
  ob_ddl_rest, kmeans++ (center_init_cost_ms), cpu_elkan_iters, cpu_kmeans_misc (parsed; not drawn)

GPU segments (worker log + gpu_km_v2/kmeans*.log + JSON):
  ob_ddl_rest, center_init (OB), input_cpu_gpu (= write_input + read_bin + cuda_h2d; not drawn separately),
  km_surround (parsed residual; not drawn), import_torch, k-means++ (worker),
  batch_kmeans, finalize, shell_fork

Only **nlist** present in both CPU and GPU results are drawn as pairs; otherwise GPU-only.

Usage:
  cd .../VectorDBBench/scripts
  python3.11 plot_gpu_km_e2e_breakdown.py
  python3.11 plot_gpu_km_e2e_breakdown.py --cpu-kmeans-dir ~/log/cpu_km_v1 --cpu-json-dir ../vectordb_bench/results/OceanBase/cpu_km_v1
  # cpu_km_v1 vs gpu_km_v5 preset:
  python3.11 plot_cpu_km_v1_vs_gpu_km_v5_breakdown.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OCEANBASE_RESULTS = HERE.parent / "vectordb_bench" / "results" / "OceanBase"

DATASET_ORDER = ["1536D50K", "1536D500K", "768D1M"]

# CPU path (standard km) — kmeanspp = k-means++ (center_init_cost_ms in log)
PHASE_KEYS_CPU = (
    "ob_ddl_rest",
    "kmeanspp",
    "cpu_elkan_iters",
    "cpu_kmeans_misc",
)
# GPU path — kmeanspp = worker flash k-means++ ("k-means++ init done in … ms")
PHASE_KEYS_GPU = (
    "ob_ddl_rest",
    "center_init",
    "input_cpu_gpu",
    "km_surround",
    "import_torch",
    "kmeanspp",
    "batch_kmeans",
    "finalize",
    "shell_fork",
)

# Not drawn in stacks / omitted from legend (residual time from logs)
SKIP_STACK_KEYS = frozenset({"cpu_kmeans_misc", "km_surround"})

COLORS: dict[str, str] = {
    "ob_ddl_rest": "#34495e",
    "center_init": "#a93226",
    "ob_write_input": "#8e44ad",
    "input_cpu_gpu": "#5dade2",
    "kmeanspp": "#e74c3c",
    "cpu_elkan_iters": "#2980b9",
    "cpu_kmeans_misc": "#95a5a6",
    "km_surround": "#9b59b6",
    "read_bin": "#7fb3d5",
    "import_torch": "#f4d03f",
    "cuda_h2d": "#bb8fce",
    # Same hue as cpu_elkan_iters — both are Lloyd-style iteration work (CPU Elkan vs GPU flash)
    "batch_kmeans": "#2980b9",
    # Legend-only: same color; bars still use cpu_elkan_iters / batch_kmeans
    "kmeans_iter": "#2980b9",
    "finalize": "#aeb6bf",
    "shell_fork": "#e59866",
}

# Legend text (internal keys → short labels; ASCII so default matplotlib fonts render)
PHASE_DISPLAY: dict[str, str] = {
    # CPU path
    "ob_ddl_rest": "DDL (non-kmeans)",
    "kmeanspp": "k-means++",
    "cpu_elkan_iters": "k-means iter",
    "cpu_kmeans_misc": "Other (CPU km)",
    # GPU — OB
    "center_init": "Center prep (OB)",
    "ob_write_input": "Write input",
    "input_cpu_gpu": "Input (CPU→GPU)",
    "km_surround": "OB glue",
    # GPU — worker
    "read_bin": "Read samples",
    "import_torch": "Import torch (cold)",
    "cuda_h2d": "H2D (tensors)",
    "batch_kmeans": "k-means iter",
    "kmeans_iter": "k-means iter",
    "finalize": "Output(GPU->CPU)",
    "shell_fork": "others",
}


def _gpu_stack_height_ms(row: dict[str, Any], pk: str) -> float:
    """Stack height in ms for one GPU row; `input_cpu_gpu` = OB write + read + H2D (pre-iter)."""
    if pk == "input_cpu_gpu":
        return (
            float(row.get("ob_write_input", 0.0))
            + float(row.get("read_bin", 0.0))
            + float(row.get("cuda_h2d", 0.0))
        )
    return float(row.get(pk, 0.0))


def _line_total_ms(line: str) -> float | None:
    m = re.search(r"([\d.]+)ms\s*\(total\)\s*$", line.strip())
    return float(m.group(1)) if m else None


def _find_total(block: str, keyword: str) -> float | None:
    for line in block.splitlines():
        if keyword in line and "(total)" in line:
            t = _line_total_ms(line)
            if t is not None:
                return t
    return None


def _parse_worker_sessions(text: str) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    blocks = re.split(r"\n======== ob_external_kmeans_worker pid=\d+", text)
    for raw in blocks:
        raw = raw.strip()
        if not raw or "worker start" not in raw:
            continue
        m0 = re.search(r"worker start in=(\S+)\s+out=", raw)
        if not m0:
            continue
        in_path = m0.group(1)

        t_close = _find_total(raw, "input file closed")
        t_import = _find_total(raw, "flash: imports done")
        t_k0 = _find_total(raw, "flash: batch_kmeans start")
        t_k1 = _find_total(raw, "flash: batch_kmeans done")
        t_out = _find_total(raw, "write output ok")
        if None in (t_close, t_import, t_k0, t_k1, t_out):
            continue

        m_kpp = re.search(r"flash: k-means\+\+ init done in ([\d.]+)ms", raw)
        kmeanspp_ms = float(m_kpp.group(1)) if m_kpp else 0.0
        cuda_h2d_ms = max(0.0, (t_k0 - t_import) - kmeanspp_ms)

        sessions.append(
            {
                "in_path": in_path,
                "read_bin": t_close,
                "import_torch": t_import - t_close,
                "kmeanspp": kmeanspp_ms,
                "cuda_h2d": cuda_h2d_ms,
                "batch_kmeans": t_k1 - t_k0,
                "finalize": t_out - t_k1,
                "worker_total_ms": t_out,
            }
        )
    return sessions


def _parse_kmeans_system_elapsed(kmeans_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in sorted(kmeans_dir.glob("kmeans*.log")):
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        cur_in: str | None = None
        for line in lines:
            mi = re.search(r"external_kmeans_invoke[^\n]+in=(\S+)", line)
            if mi:
                cur_in = mi.group(1)
                continue
            ms = re.search(r"external_kmeans_system_return[^\n]+system_elapsed_ms=(\d+)", line)
            if ms and cur_in:
                out[cur_in] = float(ms.group(1))
                cur_in = None
    return out


def _dataset_dim_n(n: int, dim: int) -> str | None:
    if dim == 768 and n >= 500_000:
        return "768D1M"
    if dim == 1536 and n >= 100_000:
        return "1536D500K"
    if dim == 1536:
        return "1536D50K"
    return None


def _load_task_labels(json_dir: Path) -> dict[tuple[str, int], str]:
    m: dict[tuple[str, int], str] = {}
    for p in sorted(json_dir.glob("result_*.json")):
        try:
            root = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        lab = root.get("task_label") or ""
        parts = lab.split("_")
        if len(parts) < 2 or parts[0] != "sweep":
            continue
        ds = parts[1]
        tc = root["results"][0].get("task_config") or {}
        dcc = tc.get("db_case_config") or {}
        nlist = dcc.get("nlist")
        if nlist is None:
            continue
        m[(ds, int(nlist))] = lab
    return m


def _load_optimize_duration_sec_by_task_label(json_dir: Path) -> dict[str, float]:
    m: dict[str, float] = {}
    for p in sorted(json_dir.glob("result_*.json")):
        try:
            root = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        lab = root.get("task_label") or ""
        if not lab or not root.get("results"):
            continue
        metrics = root["results"][0].get("metrics") or {}
        od = metrics.get("optimize_duration")
        if od is None:
            continue
        m[lab] = float(od)
    return m


def _parse_cpu_kmeans_text(text: str) -> dict[str, Any] | None:
    """Parse one cpu_km kmeans log: Elkan iter breakdown + totals."""
    m0 = re.search(r"kmeans build start lists=(\d+) dim=(\d+) count=(\d+)", text)
    if not m0:
        return None
    lists, dim, count = int(m0.group(1)), int(m0.group(2)), int(m0.group(3))
    mic = re.search(r"center_init_finished kmeanspp center_init_cost_ms=(\d+)", text)
    if not mic:
        return None
    center_init = float(mic.group(1))
    iter_sum = 0.0
    for line in text.splitlines():
        mi = re.search(r"kmeans iter iter=\d+ iter_cost_ms=(\d+)", line)
        if mi:
            iter_sum += float(mi.group(1))
    mfin = re.search(r"kmeans build finished ret=\d+ kmeans_cost_ms=(\d+)", text)
    if not mfin:
        return None
    kmeans_cost = float(mfin.group(1))
    misc = max(0.0, kmeans_cost - center_init - iter_sum)
    return {
        "lists": lists,
        "dim": dim,
        "count": count,
        "center_init_ms": center_init,
        "cpu_elkan_iters_ms": iter_sum,
        "cpu_kmeans_misc_ms": misc,
        "kmeans_cost_ms": kmeans_cost,
    }


def _parse_gpu_kmeans_text(text: str) -> dict[str, Any] | None:
    """Parse one gpu_km kmeans log: center_init, kmeans_cost, system_elapsed (external)."""
    m0 = re.search(r"kmeans build start lists=(\d+) dim=(\d+) count=(\d+)", text)
    if not m0:
        return None
    lists, dim, count = int(m0.group(1)), int(m0.group(2)), int(m0.group(3))
    mic = re.search(r"center_init_finished kmeanspp center_init_cost_ms=(\d+)", text)
    center_init = float(mic.group(1)) if mic else 0.0
    ms = re.search(r"external_kmeans_system_return[^\n]+system_elapsed_ms=(\d+)", text)
    system_elapsed = float(ms.group(1)) if ms else 0.0
    mfin = re.search(r"kmeans build finished ret=\d+ kmeans_cost_ms=(\d+)", text)
    if not mfin:
        return None
    kmeans_cost = float(mfin.group(1))
    m_w = re.search(r"external_kmeans_write_input_ms=(\d+)", text)
    write_input_ms = float(m_w.group(1)) if m_w else 0.0
    km_surround_raw = max(0.0, kmeans_cost - center_init - system_elapsed)
    # OB logs fwrite duration; remainder stays in km_surround (old logs: write_input_ms=0)
    km_surround = max(0.0, km_surround_raw - write_input_ms)
    return {
        "lists": lists,
        "dim": dim,
        "count": count,
        "center_init_ms": center_init,
        "kmeans_cost_ms": kmeans_cost,
        "system_elapsed_ms": system_elapsed,
        "write_input_ms": write_input_ms,
        "km_surround_ms": km_surround,
    }


def _scan_kmeans_dir_cpu(kmeans_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not kmeans_dir.is_dir():
        return out
    for p in sorted(kmeans_dir.glob("kmeans*.log")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parsed = _parse_cpu_kmeans_text(text)
        if not parsed:
            continue
        ds = _dataset_dim_n(parsed["count"], parsed["dim"])
        if not ds:
            continue
        key = (ds, parsed["lists"])
        out[key] = parsed
    return out


def _scan_kmeans_dir_gpu(kmeans_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not kmeans_dir.is_dir():
        return out
    for p in sorted(kmeans_dir.glob("kmeans*.log")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parsed = _parse_gpu_kmeans_text(text)
        if not parsed:
            continue
        if parsed.get("system_elapsed_ms", 0) <= 0:
            continue
        ds = _dataset_dim_n(parsed["count"], parsed["dim"])
        if not ds:
            continue
        key = (ds, parsed["lists"])
        out[key] = parsed
    return out


def _match_session_to_label(
    sess: dict[str, Any],
    invoke_by_in: dict[str, tuple[int, int, int]],
    task_map: dict[tuple[str, int], str],
) -> tuple[str, str, str] | None:
    ip = sess["in_path"]
    if ip not in invoke_by_in:
        return None
    n, k_lists, dim = invoke_by_in[ip]
    ds = _dataset_dim_n(n, dim)
    if not ds:
        return None
    key = (ds, k_lists)
    full = task_map.get(key)
    if not full:
        return None
    short = f"nlist={k_lists}"
    return ds, short, full


def _parse_kmeans_invokes(kmeans_dir: Path) -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    for p in sorted(kmeans_dir.glob("kmeans*.log")):
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            m = re.search(
                r"external_kmeans_invoke n=(\d+) k=(\d+) dim=(\d+).*?in=(\S+)",
                line,
            )
            if m:
                out[m.group(4)] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return out


def _build_cpu_rows(
    cpu_scan: dict[tuple[str, int], dict[str, Any]],
    task_map: dict[tuple[str, int], str],
    optimize_sec: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (ds, nlist), km in sorted(cpu_scan.items(), key=lambda x: (x[0][0], x[0][1])):
        full_lab = task_map.get((ds, nlist))
        if not full_lab or full_lab not in optimize_sec:
            continue
        opt_ms = optimize_sec[full_lab] * 1000.0
        kc = km["kmeans_cost_ms"]
        ob_rest = max(0.0, opt_ms - kc)
        rows.append(
            {
                "dataset": ds,
                "xlabel": f"nlist={nlist}",
                "task_label": full_lab,
                "ob_ddl_rest": ob_rest,
                "kmeanspp": km["center_init_ms"],
                "cpu_elkan_iters": km["cpu_elkan_iters_ms"],
                "cpu_kmeans_misc": km["cpu_kmeans_misc_ms"],
                "optimize_duration_ms": opt_ms,
            }
        )
    return rows


def _build_gpu_rows(
    worker_log: Path,
    gpu_kmeans_dir: Path,
    task_map: dict[tuple[str, int], str],
    optimize_sec: dict[str, float],
    gpu_km_summary: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    text = worker_log.read_text(encoding="utf-8", errors="replace")
    sessions = _parse_worker_sessions(text)
    sys_map = _parse_kmeans_system_elapsed(gpu_kmeans_dir)
    invoke_map = _parse_kmeans_invokes(gpu_kmeans_dir)

    rows: list[dict[str, Any]] = []
    for sess in sessions:
        ip = sess["in_path"]
        shell_ms = max(0.0, sys_map.get(ip, 0.0) - sess["worker_total_ms"])
        sess = {**sess, "shell_fork": shell_ms if ip in sys_map else 0.0}

        matched = _match_session_to_label(sess, invoke_map, task_map)
        if not matched:
            continue
        ds, short_lab, full_lab = matched
        nlist = int(re.search(r"nlist=(\d+)", short_lab).group(1))

        opt_ms = optimize_sec.get(full_lab)
        if opt_ms is None:
            continue
        opt_ms *= 1000.0

        km = gpu_km_summary.get((ds, nlist))
        if not km:
            continue

        ob_rest = max(0.0, opt_ms - km["kmeans_cost_ms"])
        row = {
            "dataset": ds,
            "xlabel": short_lab,
            "task_label": full_lab,
            "ob_ddl_rest": ob_rest,
            "center_init": km["center_init_ms"],
            "ob_write_input": km.get("write_input_ms", 0.0),
            "km_surround": km["km_surround_ms"],
            "read_bin": sess["read_bin"],
            "import_torch": sess["import_torch"],
            "kmeanspp": sess.get("kmeanspp", 0.0),
            "cuda_h2d": sess["cuda_h2d"],
            "batch_kmeans": sess["batch_kmeans"],
            "finalize": sess["finalize"],
            "shell_fork": sess["shell_fork"],
            "optimize_duration_ms": opt_ms,
        }
        rows.append(row)
    return rows


def _sort_chunk(chunk: list[dict[str, Any]]) -> None:
    chunk.sort(key=lambda x: int(re.search(r"nlist=(\d+)", x["xlabel"]).group(1)))


def _nlist_of(row: dict[str, Any]) -> int:
    return int(re.search(r"nlist=(\d+)", row["xlabel"]).group(1))


def _ivf_tick_label_sqrt_n(nlist: int, n_train: int) -> str:
    """X tick mathtext: IVF nlist as multiples of √N (aligned with megapanel)."""
    if n_train <= 0:
        return str(nlist)
    m = float(nlist) / math.sqrt(float(n_train))
    standards: list[tuple[float, str]] = [
        (0.25, r"$0.25\sqrt{N}$"),
        (0.5, r"$0.5\sqrt{N}$"),
        (1.0, r"$\sqrt{N}$"),
        (2.0, r"$2\sqrt{N}$"),
        (4.0, r"$4\sqrt{N}$"),
        (8.0, r"$8\sqrt{N}$"),
        (16.0, r"$16\sqrt{N}$"),
    ]
    for m0, lab in standards:
        if abs(m - m0) <= max(1e-9, 0.02 * m0):
            return lab
    return rf"${m:.4g}\sqrt{{N}}$"


def _index_by_nlist(chunk: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {_nlist_of(r): r for r in chunk}


def run_plot(
    worker_log: Path,
    gpu_kmeans_dir: Path,
    gpu_json_dir: Path,
    out_path: Path,
    dpi: int,
    *,
    cpu_kmeans_dir: Path | None = None,
    cpu_json_dir: Path | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    task_map = _load_task_labels(gpu_json_dir)
    optimize_gpu = _load_optimize_duration_sec_by_task_label(gpu_json_dir)

    optimize_cpu: dict[str, float] = {}
    if cpu_json_dir and cpu_json_dir.is_dir():
        optimize_cpu = _load_optimize_duration_sec_by_task_label(cpu_json_dir)

    cpu_scan: dict[tuple[str, int], dict[str, Any]] = {}
    if cpu_kmeans_dir and cpu_kmeans_dir.is_dir():
        cpu_scan = _scan_kmeans_dir_cpu(cpu_kmeans_dir)
    gpu_km_summary = _scan_kmeans_dir_gpu(gpu_kmeans_dir)

    cpu_rows = _build_cpu_rows(cpu_scan, task_map, optimize_cpu) if cpu_scan else []
    gpu_rows = _build_gpu_rows(worker_log, gpu_kmeans_dir, task_map, optimize_gpu, gpu_km_summary)

    if not gpu_rows:
        raise RuntimeError(
            "No GPU rows (check worker log, gpu kmeans logs with external_kmeans_system_return, gpu JSON)."
        )

    by_cpu: dict[str, list[dict[str, Any]]] = {d: [] for d in DATASET_ORDER}
    by_gpu: dict[str, list[dict[str, Any]]] = {d: [] for d in DATASET_ORDER}
    for r in cpu_rows:
        if r["dataset"] in by_cpu:
            by_cpu[r["dataset"]].append(r)
    for r in gpu_rows:
        if r["dataset"] in by_gpu:
            by_gpu[r["dataset"]].append(r)
    for ds in DATASET_ORDER:
        _sort_chunk(by_cpu[ds])
        _sort_chunk(by_gpu[ds])

    show_cpu = bool(cpu_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.1), sharey=False, facecolor="white")
    fig.suptitle(
        "CREATE VECTOR INDEX — CPU vs GPU breakdown (paired by nlist)",
        fontsize=13,
        fontweight="semibold",
        color="#222222",
        y=0.98,
    )

    legend_phase_order: list[str] = []

    def _maybe_label(pk: str) -> str:
        if pk in legend_phase_order:
            return ""
        legend_phase_order.append(pk)
        return PHASE_DISPLAY.get(pk, pk)

    # CPU vs GPU: black vs red outline (no hatch)
    CPU_EDGE = "#000000"
    GPU_EDGE = "#c62828"
    BAR_EDGE_LW = 1.25
    width = 0.36
    y_headroom = 1.12

    for ax, ds in zip(axes, DATASET_ORDER, strict=True):
        gpu_chunk = by_gpu.get(ds) or []
        cpu_chunk = by_cpu.get(ds) or []
        if not gpu_chunk:
            ax.set_visible(False)
            continue

        n0, d0 = {"1536D50K": (50_000, 1536), "1536D500K": (500_000, 1536), "768D1M": (1_000_000, 768)}[ds]

        if show_cpu and cpu_chunk:
            cpu_by_n = _index_by_nlist(cpu_chunk)
            gpu_by_n = _index_by_nlist(gpu_chunk)
            common = sorted(set(cpu_by_n.keys()) & set(gpu_by_n.keys()))
            if not common:
                title_note = " (GPU only — no matching CPU nlist)"
                single_gpu = True
            else:
                title_note = ""
                single_gpu = False
        else:
            common = []
            gpu_by_n = _index_by_nlist(gpu_chunk)
            single_gpu = True
            title_note = " (GPU only)" if not show_cpu else " (GPU only — no CPU data)"

        if not single_gpu:
            n_g = len(common)
            x = np.arange(n_g, dtype=np.float64)
            x_cpu = x - width / 2
            x_gpu = x + width / 2
            bottoms_c = np.zeros(n_g)
            bottoms_g = np.zeros(n_g)

            for pk in PHASE_KEYS_CPU:
                if pk in SKIP_STACK_KEYS:
                    continue
                h = np.array([cpu_by_n[nl].get(pk, 0.0) for nl in common], dtype=np.float64) / 1000.0
                if not h.any():
                    continue
                ax.bar(
                    x_cpu,
                    h,
                    width,
                    bottom=bottoms_c,
                    label=_maybe_label(pk),
                    color=COLORS.get(pk, "#888888"),
                    edgecolor=CPU_EDGE,
                    linewidth=BAR_EDGE_LW,
                )
                bottoms_c += h

            for pk in PHASE_KEYS_GPU:
                if pk in SKIP_STACK_KEYS:
                    continue
                h = (
                    np.array([_gpu_stack_height_ms(gpu_by_n[nl], pk) for nl in common], dtype=np.float64)
                    / 1000.0
                )
                if not h.any():
                    continue
                ax.bar(
                    x_gpu,
                    h,
                    width,
                    bottom=bottoms_g,
                    label=_maybe_label(pk),
                    color=COLORS.get(pk, "#888888"),
                    edgecolor=GPU_EDGE,
                    linewidth=BAR_EDGE_LW,
                )
                bottoms_g += h

            tick_labels = [_ivf_tick_label_sqrt_n(int(nl), n0) for nl in common]
            ax.set_xticks(x)
        else:
            n_g = len(gpu_chunk)
            x = np.arange(n_g, dtype=np.float64)
            bottoms_g = np.zeros(n_g)
            for pk in PHASE_KEYS_GPU:
                if pk in SKIP_STACK_KEYS:
                    continue
                h = np.array([_gpu_stack_height_ms(r, pk) for r in gpu_chunk], dtype=np.float64) / 1000.0
                if not h.any():
                    continue
                ax.bar(
                    x,
                    h,
                    0.72,
                    bottom=bottoms_g,
                    label=_maybe_label(pk),
                    color=COLORS.get(pk, "#888888"),
                    edgecolor=GPU_EDGE,
                    linewidth=BAR_EDGE_LW,
                )
                bottoms_g += h
            tick_labels = [_ivf_tick_label_sqrt_n(_nlist_of(c), n0) for c in gpu_chunk]
            ax.set_xticks(x)

        if not single_gpu:
            ymax_local = max(float(np.max(bottoms_c)), float(np.max(bottoms_g)))
        else:
            ymax_local = float(np.max(bottoms_g))
        y_top = max(ymax_local * y_headroom, 1e-9)

        ax.set_xticklabels(tick_labels, rotation=28, ha="right", fontsize=8, color="#333333")
        ax.set_ylabel("Wall time (s)", fontsize=9, color="#333333")
        ax.set_xlabel(
            r"IVF nlist as mult of $\sqrt{N}$ ($\log_2$ axis)",
            fontsize=8.5,
            color="#333333",
            labelpad=6,
        )
        ax.set_title(f"{ds} — N={n0:,} · dim={d0}{title_note}", fontsize=10, color="#222222")
        ax.set_ylim(0.0, y_top)
        ax.margins(x=0.02)
        ax.set_facecolor("#f8f9fa")
        ax.grid(True, axis="y", linestyle="-", linewidth=0.55, color="#e0e4e8", alpha=1.0)
        ax.set_axisbelow(True)

    def _phase_patch(k: str) -> Patch:
        return Patch(
            facecolor=COLORS[k],
            edgecolor="white",
            label=PHASE_DISPLAY.get(k, k),
        )

    # Legend rows: (1) bar outline (2) shared semantics on both sides (3) CPU-only (4) GPU-only
    # k-means iter: CPU bar = cpu_elkan_iters, GPU bar = batch_kmeans — one legend entry (kmeans_iter)
    cpu_only_keys: tuple[str, ...] = ()
    gpu_only_keys = tuple(
        pk
        for pk in PHASE_KEYS_GPU
        if pk not in ("ob_ddl_rest", "kmeanspp", "batch_kmeans", "km_surround")
    )

    shared_h: list[Patch] = []
    for k in ("ob_ddl_rest", "kmeanspp"):
        if k in legend_phase_order:
            shared_h.append(_phase_patch(k))
    if ("cpu_elkan_iters" in legend_phase_order) or ("batch_kmeans" in legend_phase_order):
        shared_h.append(_phase_patch("kmeans_iter"))

    cpu_only_h = [_phase_patch(k) for k in cpu_only_keys if k in legend_phase_order]
    gpu_only_h = [_phase_patch(k) for k in gpu_only_keys if k in legend_phase_order]

    bar_handles = [
        Patch(
            facecolor="#eceff1",
            edgecolor=CPU_EDGE,
            linewidth=1.6,
            label="CPU",
        ),
        Patch(
            facecolor="#ffebee",
            edgecolor=GPU_EDGE,
            linewidth=1.6,
            label="GPU",
        ),
    ]

    leg_kw = dict(
        loc="upper left",
        frameon=True,
        fontsize=7.0,
        title_fontsize=7.2,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=0.94,
        handletextpad=0.45,
        borderpad=0.35,
    )
    # Groups: each ncol=1 (vertical); groups in a row; anchor to first subplot upper-left
    _legend_groups: list[tuple[str, list]] = [("Bars", bar_handles)]
    if shared_h:
        _legend_groups.append(("Shared (CPU & GPU)", shared_h))
    if cpu_only_h:
        _legend_groups.append(("CPU only", cpu_only_h))
    if gpu_only_h:
        _legend_groups.append(("GPU only", gpu_only_h))

    ax0 = axes[0]
    n_g = len(_legend_groups)
    _x0_ax = 0.02
    _step_ax = min(0.21, (0.97 - _x0_ax) / max(n_g, 1))
    # Shared column is wide; nudge left so it does not overlap GPU only
    _x_nudge: dict[str, float] = {"Shared (CPU & GPU)": -0.08}
    for i, (leg_title, leg_handles) in enumerate(_legend_groups):
        xa = _x0_ax + i * _step_ax + _x_nudge.get(leg_title, 0.0)
        xa = max(0.0, min(0.92, xa))
        _leg = fig.legend(
            handles=leg_handles,
            ncol=1,
            bbox_to_anchor=(xa, 1.0),
            bbox_transform=ax0.transAxes,
            title=leg_title,
            **leg_kw,
        )
        _leg.set_zorder(1000)
        fig.add_artist(_leg)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.14, wspace=0.28)
    fig.savefig(
        out_path,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close(fig)
    return out_path.resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    home = Path(os.environ.get("HOME", "."))
    ap.add_argument(
        "--worker-log",
        type=Path,
        default=home / "log" / "gpu_km_v2" / "ob_external_kmeans_worker.log",
    )
    ap.add_argument(
        "--gpu-kmeans-dir",
        type=Path,
        default=home / "log" / "gpu_km_v2",
        help="kmeans*.log with external_kmeans + kmeans_cost_ms (GPU path)",
    )
    ap.add_argument(
        "--cpu-kmeans-dir",
        type=Path,
        default=home / "log" / "cpu_km_v1",
        help="kmeans*.log with kmeans iter lines (CPU Elkan path); omit if dir missing",
    )
    ap.add_argument(
        "--gpu-json-dir",
        type=Path,
        default=OCEANBASE_RESULTS / "gpu_km_v2",
    )
    ap.add_argument(
        "--cpu-json-dir",
        type=Path,
        default=OCEANBASE_RESULTS / "cpu_km_v1",
        help="result JSON for optimize_duration on CPU row",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=HERE / "gpu_km_e2e_breakdown_stacked.png",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    cpu_km = args.cpu_kmeans_dir if args.cpu_kmeans_dir.is_dir() else None
    cpu_js = args.cpu_json_dir if args.cpu_json_dir.is_dir() else None

    p = run_plot(
        args.worker_log,
        args.gpu_kmeans_dir,
        args.gpu_json_dir,
        args.out,
        args.dpi,
        cpu_kmeans_dir=cpu_km,
        cpu_json_dir=cpu_js,
    )
    print(p)


if __name__ == "__main__":
    main()

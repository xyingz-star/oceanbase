#!/usr/bin/env python3
"""
Shared helpers + megapanel plotting for K-means / NMBKM / IVF result JSONs under
vectordb_bench/results/OceanBase/.

CPU vs GPU 对比图与 km_nmbkm_*_megapanel 一致：白底、浅灰网格、横轴 log₂(nlist/√N)、
刻度 $0.25\sqrt{N}$…$4\sqrt{N}$、横轴标题 IVF nlist as mult of $\sqrt{N}$ ($\log_2$ axis)、
第 2 行为建索引加速比 (CPU time / GPU time)、子图标题 「数据集 — N=… · dim=…」。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent

OCEANBASE_RESULTS = HERE.parent / "vectordb_bench" / "results" / "OceanBase"
CPU_KM_DIR = OCEANBASE_RESULTS / "cpu_km_v1"
GPU_KM_DIR = OCEANBASE_RESULTS / "gpu_km_v1"

# 与 sweep CaseType / megapanel 一致
DATASET_META: dict[str, tuple[int, int]] = {
    "1536D50K": (50_000, 1536),
    "1536D500K": (500_000, 1536),
    "768D1M": (1_000_000, 768),
}

DATASET_ORDER = ["1536D50K", "1536D500K", "768D1M"]


def _stages_tuple(tc: dict[str, Any]) -> tuple[str, ...]:
    st = tc.get("stages")
    if not st:
        return ()
    return tuple(str(x) for x in st)


def _load_result_dir(d: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(d.glob("result_*.json")):
        with open(p, encoding="utf-8") as f:
            root = json.load(f)
        label = root.get("task_label") or ""
        if not root.get("results"):
            continue
        m = root["results"][0].get("metrics") or {}
        tc = root["results"][0].get("task_config") or {}
        dcc = tc.get("db_case_config") or {}
        nlist = dcc.get("nlist")
        if nlist is None:
            continue
        out[label] = {
            "nlist": int(nlist),
            "optimize_duration": float(m.get("optimize_duration") or 0),
            "recall": float(m.get("recall") or 0),
            "qps": float(m.get("qps") or 0),
            "stages": _stages_tuple(tc),
        }
    return out


def _dataset_from_task_label(label: str) -> str | None:
    parts = label.split("_")
    if len(parts) < 2 or parts[0] != "sweep":
        return None
    return parts[1]


def collect_cpu_gpu_sweep_payload(
    cpu_dir: Path | None = None,
    gpu_dir: Path | None = None,
    *,
    require_identical_stages: bool = True,
    require_rebuild_only: bool = True,
) -> tuple[dict[str, dict[str, list[float] | list[int] | list[str]]], dict[str, Any]]:
    """对齐 task_label，且默认要求 CPU/GPU 两次结果的 VectorDBBench stages 一致（否则 optimize 不可比）。

    require_rebuild_only=True 时：仅保留 stages == optimize + search_serial + search_concurrent
    （仅重建向量索引、无本 run 内 load），保证 optimize_duration 均为「建索引」阶段、口径一致。
    """
    cpu = _load_result_dir(cpu_dir or CPU_KM_DIR)
    gpu = _load_result_dir(gpu_dir or GPU_KM_DIR)
    keys = sorted(set(cpu.keys()) & set(gpu.keys()))
    if not keys:
        return {}, {"skipped": 0, "reason": "no common task_label", "pairs_used": 0}

    REBUILD_STAGES = ("optimize", "search_serial", "search_concurrent")
    skipped: list[tuple[str, str]] = []
    by_ds: dict[str, list[tuple[int, float, float, float, float, float, float, str]]] = {}
    for lab in keys:
        c, g = cpu[lab], gpu[lab]
        sc, sg = c["stages"], g["stages"]
        if require_identical_stages and sc != sg:
            skipped.append((lab, f"stages differ: CPU={sc!r} GPU={sg!r}"))
            continue
        if require_rebuild_only:
            if sc != REBUILD_STAGES or sg != REBUILD_STAGES:
                skipped.append((lab, f"not rebuild-only: stages={sc!r}"))
                continue
        ds = _dataset_from_task_label(lab)
        if not ds:
            continue
        by_ds.setdefault(ds, []).append(
            (
                c["nlist"],
                c["optimize_duration"],
                g["optimize_duration"],
                c["recall"],
                g["recall"],
                c["qps"],
                g["qps"],
                lab,
            )
        )

    order = [d for d in DATASET_ORDER if d in by_ds] + [d for d in sorted(by_ds.keys()) if d not in DATASET_ORDER]
    payload: dict[str, dict[str, list[float] | list[int] | list[str]]] = {}
    for ds in order:
        rows = sorted(by_ds[ds], key=lambda r: r[0])
        payload[ds] = {
            "nlist": [r[0] for r in rows],
            "opt_cpu": [r[1] for r in rows],
            "opt_gpu": [r[2] for r in rows],
            "rec_cpu": [r[3] for r in rows],
            "rec_gpu": [r[4] for r in rows],
            "qps_cpu": [r[5] for r in rows],
            "qps_gpu": [r[6] for r in rows],
            "task_labels": [r[7] for r in rows],
        }
    meta = {
        "skipped": len(skipped),
        "skipped_detail": skipped[:20],
        "stages_rule": "identical CPU/GPU stages"
        + ("; rebuild-only (optimize+search_*)" if require_rebuild_only else ""),
        "pairs_used": sum(len(payload[ds]["nlist"]) for ds in payload),
    }
    return payload, meta


def _log2_mult_axis(nlist: list[int], n_train: int) -> np.ndarray:
    sqrt_n = math.sqrt(float(n_train))
    m = np.array(nlist, dtype=np.float64) / sqrt_n
    return np.log2(np.maximum(m, 1e-15))


def plot_cpu_vs_gpu_km_megapanel(
    out_path: Path | None = None,
    *,
    cpu_dir: Path | None = None,
    gpu_dir: Path | None = None,
    dpi: int = 150,
    require_rebuild_only: bool = True,
    suptitle: str | None = None,
    cpu_legend: str = "CPU (in-process Elkan)",
    gpu_legend: str = "GPU (external worker)",
    e2e_breakdown_png: str | None = None,
) -> Path:
    """
    4 行（Create index time / Create index speedup / Recall@k / QPS）× N 列（数据集），样式对齐 NMBKM megapanel。

    默认仅使用 **CPU 与 GPU JSON 中 stages 完全一致** 且 **均为仅重建索引**（含 optimize）的成对结果，
    使 optimize_duration 表示同一类「建索引」口径。
    """
    import matplotlib.pyplot as plt

    payload, pmeta = collect_cpu_gpu_sweep_payload(
        cpu_dir,
        gpu_dir,
        require_identical_stages=True,
        require_rebuild_only=require_rebuild_only,
    )
    if not payload or not any(payload[ds]["nlist"] for ds in payload):
        raise RuntimeError(
            "No comparable pairs: need same task_label, identical stages, and (by default) "
            "rebuild-only runs. See collect_cpu_gpu_sweep_payload skipped_detail. "
            f"skipped={pmeta.get('skipped')!r}"
        )

    datasets = [d for d in DATASET_ORDER if d in payload] + [d for d in payload if d not in DATASET_ORDER]
    ncols = len(datasets)
    nrows = 4

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(12.0, 3.8 * ncols), 8.8),
        sharex=True,
        facecolor="white",
        squeeze=False,
    )
    fig.suptitle(
        suptitle or "End-to-end Comparison -- CPU vs GPU",
        fontsize=14,
        fontweight="semibold",
        color="#222222",
        y=0.98,
    )
    # (y-axis title line, y unit, cpu key, gpu key) — gpu None + cpu "speedup" → single speedup curve
    row_titles = [
        ("Create index time", "s", "opt_cpu", "opt_gpu"),
        ("Create index speed", "×", "speedup", None),
        ("Recall@k", "recall", "rec_cpu", "rec_gpu"),
        ("Concurrent QPS", "QPS", "qps_cpu", "qps_gpu"),
    ]
    xticks = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    xticklabels = [
        r"$0.25\sqrt{N}$",
        r"$0.5\sqrt{N}$",
        r"$\sqrt{N}$",
        r"$2\sqrt{N}$",
        r"$4\sqrt{N}$",
    ]

    for c, ds in enumerate(datasets):
        d = payload[ds]
        n_train, dim = DATASET_META.get(ds, (0, 0))
        x = _log2_mult_axis([int(x) for x in d["nlist"]], n_train)

        for r, (row_name, _yunit, kc, kg) in enumerate(row_titles):
            ax = axes[r, c]
            ax.set_facecolor("white")

            if kg is None and kc == "speedup":
                y_cpu = np.array(d["opt_cpu"], dtype=np.float64)
                y_gpu = np.maximum(np.array(d["opt_gpu"], dtype=np.float64), 1e-15)
                sp = y_cpu / y_gpu
                ax.axhline(
                    1.0,
                    color="#bbbbbb",
                    linestyle="--",
                    linewidth=1.0,
                    zorder=1,
                )
                ax.plot(
                    x,
                    sp,
                    color="#6c3483",
                    linestyle="-",
                    linewidth=1.8,
                    marker="D",
                    markersize=4.5,
                    markerfacecolor="#af7ac5",
                    markeredgecolor="#512e5f",
                    markeredgewidth=1.0,
                    zorder=3,
                )
            else:
                y_cpu = np.array(d[kc], dtype=np.float64)
                y_gpu = np.array(d[kg], dtype=np.float64)

                ax.plot(
                    x,
                    y_cpu,
                    color="#1a1a1a",
                    linestyle="-",
                    linewidth=1.8,
                    marker="o",
                    markersize=5,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    markeredgecolor="#1a1a1a",
                    label=cpu_legend,
                    zorder=3,
                )
                ax.plot(
                    x,
                    y_gpu,
                    color="#1e8449",
                    linestyle="-",
                    linewidth=1.8,
                    marker="s",
                    markersize=5,
                    markerfacecolor="#27ae60",
                    markeredgecolor="#145a32",
                    label=gpu_legend,
                    zorder=3,
                )

            ax.grid(True, which="major", linestyle="-", linewidth=0.6, color="#dddddd", alpha=0.95)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#eeeeee", alpha=0.8)
            ax.set_xlim(-2.35, 2.35)
            ax.set_xticks(xticks)
            ax.set_xticklabels(
                xticklabels,
                fontsize=8,
                color="#444444",
            )

            ax.tick_params(axis="y", labelsize=8, colors="#333333")
            if c == 0:
                if kg is None and kc == "speedup":
                    y_label = "Create index speed\n(CPU time / GPU time)"
                elif row_name in ("Recall@k", "Concurrent QPS"):
                    y_label = row_name
                else:
                    y_label = f"{row_name}\n({_yunit})"
                ax.set_ylabel(
                    y_label,
                    fontsize=9,
                    color="#222222",
                )

            ttl = f"{ds} — N={n_train:,} · dim={dim}"
            ax.set_title(ttl, fontsize=9.5, pad=6, color="#222222")

    # sharex=True hides x tick labels on non-bottom axes by default — force on every panel
    for ax in axes.flat:
        ax.tick_params(axis="x", which="major", labelbottom=True)
        for lb in ax.get_xticklabels():
            lb.set_visible(True)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.17, wspace=0.28, hspace=0.44)

    fig.supxlabel(
        r"IVF nlist as mult of $\sqrt{N}$ ($\log_2$ axis)",
        fontsize=9.5,
        color="#333333",
        y=0.075,
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color="#1a1a1a",
            linestyle="-",
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markeredgecolor="#1a1a1a",
            label=cpu_legend,
        ),
        plt.Line2D(
            [0],
            [0],
            color="#27ae60",
            linestyle="-",
            marker="s",
            markersize=6,
            markerfacecolor="#27ae60",
            markeredgecolor="#145a32",
            label=gpu_legend,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        fontsize=9,
    )

    out = out_path or (HERE / "cpu_gpu_km_comparison.png")
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out.resolve()


if __name__ == "__main__":
    p = plot_cpu_vs_gpu_km_megapanel()
    print(p)

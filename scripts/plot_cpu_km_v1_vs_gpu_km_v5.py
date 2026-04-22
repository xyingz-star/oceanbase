#!/usr/bin/env python3
"""
Compare VectorDBBench result dirs: cpu_km_v1 vs gpu_km_v5 (same task_labels / IVF sweep).

  cd .../VectorDBBench/scripts
  python3 plot_cpu_km_v1_vs_gpu_km_v5.py

Output: cpu_km_v1_vs_gpu_km_v5.png next to this script.

Data sources (JSON only; logs are not read here):
  vectordb_bench/results/OceanBase/cpu_km_v1/
  vectordb_bench/results/OceanBase/gpu_km_v5/
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    sys.path.insert(0, str(HERE))
    from build_km_nmbkm_charts import OCEANBASE_RESULTS, plot_cpu_vs_gpu_km_megapanel

    out = plot_cpu_vs_gpu_km_megapanel(
        out_path=HERE / "cpu_km_v1_vs_gpu_km_v5.png",
        cpu_dir=OCEANBASE_RESULTS / "cpu_km_v1",
        gpu_dir=OCEANBASE_RESULTS / "gpu_km_v5",
        suptitle="End-to-end Comparison -- CPU vs GPU",
        cpu_legend="cpu_km_v1 (Elkan in-process)",
        gpu_legend="gpu_km_v5 (external worker + flash-kmeans)",
        e2e_breakdown_png="gpu_km_e2e_breakdown_stacked_cpu_v1_gpu_v5.png",
    )
    print(out, file=sys.stdout)


if __name__ == "__main__":
    main()

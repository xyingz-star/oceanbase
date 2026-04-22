#!/usr/bin/env python3
"""
Stacked E2E breakdown: cpu_km_v1 vs gpu_km_v5.

  cd .../VectorDBBench/scripts
  python3.11 plot_cpu_km_v1_vs_gpu_km_v5_breakdown.py

Output: gpu_km_e2e_breakdown_stacked_cpu_v1_gpu_v5.png

Needs: $HOME/log/cpu_km_v1/kmeans*.log, $HOME/log/gpu_km_v5/{kmeans*.log,ob_external_kmeans_worker.log}
       + vectordb_bench/results/OceanBase/{cpu_km_v1,gpu_km_v5}/result_*.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path(os.environ.get("HOME", "."))
OCEANBASE_RESULTS = HERE.parent / "vectordb_bench" / "results" / "OceanBase"


def main() -> None:
    sys.path.insert(0, str(HERE))
    from plot_gpu_km_e2e_breakdown import run_plot

    p = run_plot(
        HOME / "log" / "gpu_km_v5" / "ob_external_kmeans_worker.log",
        HOME / "log" / "gpu_km_v5",
        OCEANBASE_RESULTS / "gpu_km_v5",
        HERE / "gpu_km_e2e_breakdown_stacked_cpu_v1_gpu_v5.png",
        150,
        cpu_kmeans_dir=HOME / "log" / "cpu_km_v1",
        cpu_json_dir=OCEANBASE_RESULTS / "cpu_km_v1",
    )
    print(p, file=sys.stdout)


if __name__ == "__main__":
    main()

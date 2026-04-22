#!/usr/bin/env python3
"""
CPU vs GPU 外切 K-means：3×3 megapanel（optimize / recall / QPS），
样式与 km_nmbkm_*_megapanel 一致（白底、log₂ 横轴、子图标题 N·dim）。

用法（与 v4_div vs nmbkm_v9_pca 入口相同：sys.path + build_km_nmbkm_charts）：

  cd .../VectorDBBench/scripts
  python3.11 plot_cpu_gpu_km_megapanel.py

可选环境变量：无（JSON 目录默认为 vectordb_bench/results/OceanBase/cpu_km_v1 与 gpu_km_v1）。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    sys.path.insert(0, str(HERE))
    from build_km_nmbkm_charts import plot_cpu_vs_gpu_km_megapanel

    out = plot_cpu_vs_gpu_km_megapanel()
    print(out, file=sys.stdout)


if __name__ == "__main__":
    main()

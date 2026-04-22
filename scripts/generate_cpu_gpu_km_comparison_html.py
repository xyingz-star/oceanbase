#!/usr/bin/env python3
"""Load cpu_km_v1 / gpu_km_v1 result JSONs and emit comparison HTML (3 datasets x 3 metrics).

仅纳入与 build_km_nmbkm_charts 相同的「公平」成对结果：CPU/GPU JSON stages 一致且均为 rebuild-only。
"""
from __future__ import annotations

import html as html_module
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent / "vectordb_bench" / "results" / "OceanBase"
CPU_DIR = BASE / "cpu_km_v1"
GPU_DIR = BASE / "gpu_km_v1"
OUT_HTML = SCRIPT_DIR / "cpu_gpu_km_comparison.html"


def main() -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    from build_km_nmbkm_charts import DATASET_ORDER, collect_cpu_gpu_sweep_payload

    payload, meta = collect_cpu_gpu_sweep_payload(CPU_DIR, GPU_DIR)
    if not payload or not any(payload.get(ds, {}).get("nlist") for ds in payload):
        raise SystemExit(f"No comparable pairs: {meta!r}")

    datasets = [d for d in DATASET_ORDER if d in payload and payload[d]["nlist"]] + [
        d for d in sorted(payload.keys()) if d not in DATASET_ORDER and payload[d]["nlist"]
    ]
    chart_payload = {
        ds: {k: v for k, v in payload[ds].items() if k != "task_labels"} for ds in datasets
    }
    data_json = json.dumps(chart_payload, ensure_ascii=False)
    _meta_note = html_module.escape(
        f"points={meta.get('pairs_used')} skipped={meta.get('skipped')} "
        f"(stages must match; rebuild-only for fair optimize)"
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CPU vs GPU K-means — IVF sweep 对比</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --cpu: #5eead4;
      --gpu: #f472b6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
      background: var(--bg); color: var(--text); min-height: 100vh;
      padding: 1.25rem 1.5rem 2rem;
    }}
    h1 {{
      font-weight: 600; font-size: 1.35rem; margin: 0 0 0.35rem;
      letter-spacing: -0.02em;
    }}
    .sub {{ color: var(--muted); font-size: 0.88rem; margin-bottom: 1.25rem; max-width: 72ch; line-height: 1.45; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat({len(datasets)}, minmax(220px, 1fr));
      grid-template-rows: repeat(3, minmax(240px, 1fr));
      gap: 1rem 1rem;
      align-items: stretch;
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; grid-template-rows: auto; }}
    }}
    .cell {{
      background: var(--card); border-radius: 12px; padding: 0.65rem 0.75rem 0.4rem;
      border: 1px solid rgba(255,255,255,0.06);
      box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }}
    .cell h2 {{
      margin: 0 0 0.5rem; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.06em; color: var(--muted);
    }}
    .cell .metric {{ font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 0.35rem; }}
    .canvas-wrap {{ position: relative; height: 220px; width: 100%; }}
    .legend-note {{ font-size: 0.75rem; color: var(--muted); margin-top: 0.35rem; }}
  </style>
</head>
<body>
  <h1>CPU K-means vs GPU 外切 K-means</h1>
  <p class="sub">仅对比 CPU/GPU JSON 中 <strong>stages 完全一致</strong> 且均为 <strong>rebuild-only</strong>（optimize+search）的成对结果；optimize_duration 口径一致。元数据：{_meta_note}</p>
  <div class="grid" id="grid"></div>
  <script>
  const DATA = {data_json};
  const datasets = {json.dumps(datasets)};
  const metrics = [
    {{ key: 'optimize', title: '索引创建 / 优化时间', yLabel: '秒 (s)', cpu: 'opt_cpu', gpu: 'opt_gpu' }},
    {{ key: 'recall', title: 'Recall', yLabel: 'recall', cpu: 'rec_cpu', gpu: 'rec_gpu' }},
    {{ key: 'qps', title: 'QPS', yLabel: 'QPS', cpu: 'qps_cpu', gpu: 'qps_gpu' }},
  ];
  const grid = document.getElementById('grid');
  const commonOpts = {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: '#c5d0e0', font: {{ size: 11 }} }} }},
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'nlist', color: '#8b9cb3', font: {{ size: 11 }} }},
        ticks: {{ color: '#8b9cb3', maxRotation: 45 }},
        grid: {{ color: 'rgba(255,255,255,0.06)' }},
      }},
      y: {{
        ticks: {{ color: '#8b9cb3' }},
        grid: {{ color: 'rgba(255,255,255,0.06)' }},
      }},
    }},
  }};

  metrics.forEach((met, row) => {{
    datasets.forEach((ds) => {{
      const d = DATA[ds];
      if (!d) return;
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.innerHTML = '<h2>' + ds + '</h2><div class="metric">' + met.title + '</div><div class="canvas-wrap"><canvas></canvas></div>';
      grid.appendChild(cell);
      const canvas = cell.querySelector('canvas');
      const ctx = canvas.getContext('2d');
      new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: d.nlist.map(String),
          datasets: [
            {{
              label: 'CPU (Elkan / 进程内)',
              data: d[met.cpu],
              borderColor: '#5eead4',
              backgroundColor: 'rgba(94,234,212,0.15)',
              tension: 0.2,
              fill: false,
              pointRadius: 4,
            }},
            {{
              label: 'GPU (外切 worker)',
              data: d[met.gpu],
              borderColor: '#f472b6',
              backgroundColor: 'rgba(244,114,182,0.12)',
              tension: 0.2,
              fill: false,
              pointRadius: 4,
            }},
          ],
        }},
        options: {{
          ...commonOpts,
          scales: {{
            ...commonOpts.scales,
            y: {{
              ...commonOpts.scales.y,
              title: {{ display: true, text: met.yLabel, color: '#8b9cb3', font: {{ size: 11 }} }},
            }},
          }},
        }},
      }});
    }});
  }});
  </script>
</body>
</html>
"""

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML}")
    print(f"Datasets: {datasets}, meta: {meta}")


if __name__ == "__main__":
    main()

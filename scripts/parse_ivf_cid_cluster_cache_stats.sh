#!/usr/bin/env bash
# Parse OB_IVF_CID_CLUSTER_CACHE_STATS from ~/log/ob_ivf_cid_cluster_cache*.log
#
# Usage:
#   ./scripts/parse_ivf_cid_cluster_cache_stats.sh
#   ./scripts/parse_ivf_cid_cluster_cache_stats.sh /path/to/log/dir
#   ./scripts/parse_ivf_cid_cluster_cache_stats.sh --final-only
#   ./scripts/parse_ivf_cid_cluster_cache_stats.sh --since 2026-05-21
#
set -euo pipefail

LOG_DIR="${HOME}/log"
FINAL_ONLY=0
SINCE=""

usage() {
  sed -n '2,8p' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --final-only) FINAL_ONLY=1; shift ;;
    --since) SINCE="${2:-}"; shift 2 ;;
    -*) echo "unknown option: $1" >&2; usage 1 ;;
    *) LOG_DIR="$1"; shift ;;
  esac
done

shopt -s nullglob
if [[ -f "${LOG_DIR}" ]]; then
  FILES=("${LOG_DIR}")
else
  LOG_DIR="$(readlink -f "${LOG_DIR}")"
  mapfile -t FILES < <(ls -1 "${LOG_DIR}"/ob_ivf_cid_cluster_cache*.log 2>/dev/null | sort)
fi
if ((${#FILES[@]} == 0)); then
  echo "no cache stats logs under ${LOG_DIR}" >&2
  exit 1
fi

if [[ -n "${SINCE}" ]]; then
  FILTERED=()
  for f in "${FILES[@]}"; do
    mtime="$(date -r "${f}" +%Y-%m-%d 2>/dev/null || true)"
    if [[ "${mtime}" == "${SINCE}" ]]; then
      FILTERED+=("${f}")
    fi
  done
  FILES=("${FILTERED[@]}")
fi

if ((${#FILES[@]} == 0)); then
  echo "no logs match --since ${SINCE}" >&2
  exit 1
fi

export FINAL_ONLY
export CACHE_LOG_FILES="${FILES[*]}"
python3 <<'PY'
import os, re, sys
from collections import defaultdict

final_only = os.environ.get("FINAL_ONLY") == "1"
files = [f for f in os.environ.get("CACHE_LOG_FILES", "").split(" ") if f]

def _field(line, name, default="0"):
    m = re.search(rf"{name}=([-\w.]+)", line)
    return m.group(1) if m else default

def parse_line(line):
    if "[OB_IVF_CID_CLUSTER_CACHE_STATS]" not in line:
        return None
    phase = _field(line, "phase", "")
    if phase not in ("progress", "final", "session"):
        return None
    return {
        "phase": phase,
        "mode": _field(line, "current_mode", "-"),
        "tid": _field(line, "tid"),
        "n": _field(line, "n"),
        "d": _field(line, "d"),
        "c": _field(line, "c"),
        "cid_sw": _field(line, "cid_switch_cnt"),
        "so": _field(line, "storage_only_cid_cnt"),
        "replay": _field(line, "replay_cid_cnt"),
        "fill": _field(line, "fill_cid_cnt"),
        "cid_hit": _field(line, "cid_hit_rate", "0"),
        "d_hit": _field(line, "cache_d_get_hit"),
        "d_miss": _field(line, "cache_d_get_miss"),
        "d_put": _field(line, "cache_d_put_ok"),
        "put_ok": _field(line, "put_cid_ok"),
        "entries": _field(line, "cur_entry_cnt"),
        "bytes": _field(line, "cur_bytes"),
    }

def parse_file(path):
    finals, last_progress = [], {}
    modes = defaultdict(int)
    with open(path, "r", errors="replace") as f:
        for line in f:
            d = parse_line(line)
            if not d:
                continue
            phase = d["phase"]
            tid = d["tid"]
            if phase == "progress":
                modes[d["mode"]] += 1
                last_progress[tid] = d
            elif phase in ("final", "session"):
                finals.append((phase, d))
    return finals, last_progress, dict(modes)

rows = []
for path in files:
    finals, last_progress, modes = parse_file(path)
    base = os.path.basename(path)
    if finals:
        for phase, d in finals:
            hit = int(d["d_hit"])
            miss = int(d["d_miss"])
            total = hit + miss
            rows.append({
                "file": base, "phase": phase, "tid": d["tid"],
                "n": d["n"], "d": d["d"], "c": d["c"],
                "cid_hit": float(d["cid_hit"]),
                "get_hit_rate": (hit / total) if total else 0.0,
                "d_hit": hit, "d_miss": miss, "d_put": int(d["d_put"]),
                "put_ok": int(d["put_ok"]), "entries": int(d["entries"]),
                "bytes": int(d["bytes"]), "replay": int(d["replay"]),
                "fill": int(d["fill"]), "modes": modes,
            })
    elif not final_only and last_progress:
        # aggregate per-file from last progress line per tid
        agg = defaultdict(lambda: defaultdict(int))
        last = None
        for tid, d in last_progress.items():
            last = d
            for k in ("d_hit", "d_miss", "d_put", "put_ok", "entries", "bytes",
                      "replay", "fill", "so", "cid_sw"):
                agg["sum"][k] += int(d[k])
            agg["max"]["cid_sw"] = max(agg["max"]["cid_sw"], int(d["cid_sw"]))
        hit, miss = agg["sum"]["d_hit"], agg["sum"]["d_miss"]
        total = hit + miss
        rows.append({
            "file": base, "phase": "progress(incomplete)",
            "tid": f"{len(last_progress)}t",
            "n": last["n"], "d": last["d"], "c": last["c"],
            "cid_hit": 0.0,
            "get_hit_rate": (hit / total) if total else 0.0,
            "d_hit": hit, "d_miss": miss, "d_put": agg["sum"]["d_put"],
            "put_ok": agg["sum"]["put_ok"], "entries": agg["sum"]["entries"],
            "bytes": agg["sum"]["bytes"], "replay": agg["sum"]["replay"],
            "fill": agg["sum"]["fill"], "modes": modes,
        })

if not rows:
    print("no parseable stats lines")
    sys.exit(0)

print(f"{'file':<52} {'phase':<18} {'n/d/c':<16} {'cid_hit':>7} {'get_hit%':>8} "
      f"{'hit':>6} {'miss':>6} {'put':>5} {'ent':>4} {'MiB':>6} {'modes'}")
print("-" * 130)
for r in sorted(rows, key=lambda x: (x["phase"], x["file"])):
    ndc = f"{r['n']}/{r['d']}/{r['c']}"
    mib = r["bytes"] / (1024 * 1024)
    mode_s = ",".join(f"{k}:{v}" for k, v in sorted(r["modes"].items()))
    print(f"{r['file']:<52} {r['phase']:<18} {ndc:<16} {r['cid_hit']:7.3f} "
          f"{100*r['get_hit_rate']:8.1f} {r['d_hit']:6} {r['d_miss']:6} "
          f"{r['d_put']:5} {r['entries']:4} {mib:6.1f} {mode_s}")

ok = [r for r in rows if r["phase"] == "final" and r["d_hit"] > 0]
bad = [r for r in rows if r["phase"] != "final" or r["d_hit"] == 0]
print()
if ok:
    print(f"OK (final + hits): {len(ok)} file(s)")
if bad:
    print(f"INCOMPLETE or zero-hit: {len(bad)} file(s) — cache not warmed or query aborted")
PY

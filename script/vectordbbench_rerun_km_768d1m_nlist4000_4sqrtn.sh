#!/usr/bin/env bash
#
# 单独重跑对比图里「KM（黑线）」在 768D1M、nlist=4000（4√N）上的异常点。
# 与历史 sweep 一致：IVF_FLAT，sample_per_nlist=250，ivf_nprobes=56。
#
# 默认只做索引重建 + 串行/并发检索（不重导数据），与 result JSON 里 stages 一致。
#
# 用法（前台）：
#   export OB_USER='root@xxx' OB_DATABASE=test OB_PORT=11000
#   export OB_TABLE_NAME=vdb_768d1m   # 与 km_v3 原跑一致
#   bash vectordbbench_rerun_km_768d1m_nlist4000_4sqrtn.sh
#
# 后台（示例）：
#   mkdir -p ~/log/vdb_ivf_rerun
#   nohup env OB_USER="$OB_USER" OB_DATABASE="${OB_DATABASE:-test}" OB_PORT="${OB_PORT:-11000}" \
#     OB_TABLE_NAME="${OB_TABLE_NAME:-vdb_768d1m}" \
#     bash /path/to/vectordbbench_rerun_km_768d1m_nlist4000_4sqrtn.sh \
#     >> ~/log/vdb_ivf_rerun/km_768d1m_nlist4000_rerun.log 2>&1 &
#
# KM 路径：若曾跑过 NMBKM sweep，observer 可能仍读 /tmp/ob_nmbkm_min_n_scale；设
#   RERUN_KM_CLEAR_NMBKM_DIV_FILE=1（默认 1）会在开跑前删掉该文件，避免误用 div 配置。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="${SCRIPT_DIR}/vectordbbench_oceanbaseivf_single.sh"

if [[ ! -f "${INNER}" ]]; then
  echo "ERROR: missing ${INNER}" >&2
  exit 1
fi

if [[ "${RERUN_KM_CLEAR_NMBKM_DIV_FILE:-1}" == "1" ]]; then
  rm -f "${SWEEP_NMBKM_DIV_FILE:-/tmp/ob_nmbkm_min_n_scale}"
fi

_ts="$(date +%Y%m%dT%H%M%S)"
export CASE_TYPE="${CASE_TYPE:-Performance768D1M}"
export NLIST="${NLIST:-4000}"
export SAMPLE_PER_NLIST="${SAMPLE_PER_NLIST:-250}"
export IVF_NPROBES="${IVF_NPROBES:-56}"
export VDB_TASK_LABEL="${VDB_TASK_LABEL:-sweep_768D1M_sqrtn_m4_nlist4000_spn250_N1000000_rerun_${_ts}}"

# 与 km_v3 单次跑一致：复用表、不重导、重建向量索引
export VDB_SKIP_DROP_OLD="${VDB_SKIP_DROP_OLD:-1}"
export VDB_SKIP_LOAD="${VDB_SKIP_LOAD:-1}"
export VDB_REBUILD_INDEX="${VDB_REBUILD_INDEX:-1}"

# 结果 JSON 写入目录（可被 RESULTS_LOCAL_DIR 覆盖）
export RESULTS_LOCAL_DIR="${RESULTS_LOCAL_DIR:-/home/zhuxueying.zxy/test/VectorDBBench/vectordb_bench/results/OceanBase/km_v3}"
mkdir -p "${RESULTS_LOCAL_DIR}"

echo "### rerun KM 768D1M nlist=4000 (4√N) task_label=${VDB_TASK_LABEL} RESULTS_LOCAL_DIR=${RESULTS_LOCAL_DIR}"

exec bash "${INNER}"
